"""LLM 백엔드. OpenRouter (OpenAI 호환).

- OpenRouterClient : 실제 호출. tools/pilot/run_pilot.py 의 urllib·백오프 패턴을 계승.
- StubClient       : 테스트용. 미리 정해둔 응답 시퀀스를 순서대로 돌려준다.
                     LLM 은 비결정적이라 이게 없으면 합격 기준을 못 쓴다.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# **Gemini 를 직접 부를 때.** Google 이 OpenAI 호환 엔드포인트를 제공하므로 같은 클라이언트를
# 쓴다 — 몸통(messages·tools·tool_choice)이 같고 다른 것은 주소와 열쇠뿐이다.
#
# 다만 **OpenRouter 전용 필드는 실으면 안 된다.** `provider`(프로바이더 라우팅)와
# `reasoning`(OpenRouter 통합 사고 파라미터)은 저쪽에 없는 이름이라 400 을 받는다.
# 아래 `_openrouter_only` 가 그것을 가른다.
BACKENDS = {
    "openrouter": ENDPOINT,
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}
KEY_ENV = {"openrouter": "OPENROUTER_API_KEY", "gemini": "GEMINI_API_KEY"}


class RateLimitGuard:
    """재시도를 다 쓴 429 를 센다. 한도를 넘으면 `RateLimitStorm` 을 던진다.

    **클라이언트 둘이 하나를 나눠 갖는다** — 에이전트가 막히든 번역기가 막히든 그 런의
    데이터가 상하는 것은 같고, 둘을 따로 세면 각각 한도의 절반씩 맞고도 안 걸린다.

    `limit=0` 이면 끄는 것이다 (짧은 테스트·재현용).
    """

    def __init__(self, limit: int = 5):
        self.limit = limit
        self.count = 0
        self.by_model: dict[str, int] = {}
        self._lock = threading.Lock()

    def exhausted(self, model: str) -> None:
        if self.limit <= 0:
            return
        with self._lock:
            self.count += 1
            self.by_model[model] = self.by_model.get(model, 0) + 1
            if self.count >= self.limit:
                raise RateLimitStorm(
                    f"429 로 버린 호출이 {self.count}건 — 한도 {self.limit}. 런을 세웁니다.\n"
                    f"  모델별: {self.by_model}\n"
                    f"  마지막 복원점부터 `--resume` 으로 이어할 수 있습니다.")


class OutOfCredit(Exception):
    """잔액 소진. **`LLMCallError` 가 아니다** — 그 계열이면 「이 에이전트만 이 해를
    포기」 로 삼켜져서 런이 빈 세계로 계속 흘러간다. 위로 터뜨려 런을 세운다.
    (`RateLimitStorm` 이 같은 이유로 별도 계열인 것과 같다.)
    """


class RateLimitStorm(RuntimeError):
    """**429 가 그치지 않는다 — 런을 세운다.** `LLMCallError` 와 **일부러 무관하다.**

    `LLMCallError` 는 상류가 잡아서 그 에이전트의 차례를 끝내거나 메시지 하나를
    미전달로 떨어뜨린다. 그 설계 덕에 런은 안 죽는데, 429 폭풍에서는 그것이 **독**이다:

        260826-002-ai010   번역 429 154건 · 244통 중 23통(9.4%)이 죽었다
                           런은 절뚝이며 50해를 다 갔고, 3시간 33분이 걸렸다

    죽은 23통은 **세계에 뚫린 구멍**이다. 엔진 장애라 수신자에게 흔적도 안 남으니
    (`translate_failed` 는 세계의 사건이 아니다), 사후에는 「말이 안 통했다」 와 구분은
    되지만 **일어나지 않은 대화**는 되돌릴 수 없다. 그럴 바에는 세우고 나중에 이어한다 —
    매해 복원점이 있으므로 이어붙이는 값이 싸다 (8/25).

    그래서 이 예외는 `LLMCallError` 의 자식이 **아니다.** 상류의 그물 둘
    (`agent_loop` 의 차례 종료 · `loop` 의 미전달)이 이것을 잡으면 안 된다.
    """


class LLMCallError(RuntimeError):
    """**API·망 쪽 실패.** 이 예외만 상류에서 잡아 미전달로 떨어뜨린다.

    그전에는 상류가 `except Exception` 이었다. 런이 안 죽는 것이 목적이었는데, 그
    그물이 **우리 코드의 버그까지 삼켰다** — `KeyError`·`TypeError`·`AttributeError` 가
    "번역 실패" 통계 한 줄로 묻히고 크래시로 드러나지 않는다.

    지난주 HTTP 200 에 error 를 실은 응답이 `KeyError` 로 런을 죽였는데, **죽어서
    발견됐으니 고칠 수 있었다.** 삼켜지면 그 기회가 없다.

    그래서 경계가 자기 실패를 **선언**한다. 여기 들어오는 것:

        HTTP 4xx (429 제외)          재시도 없이
        HTTP 5xx · 429 · 타임아웃     재시도를 다 쓴 뒤
        망 오류 · 잘린 JSON 본문      재시도를 다 쓴 뒤
        HTTP 200 인데 choices 없음    프로바이더가 error 를 실어 보낸 경우
        재시도 소진

    나머지는 전부 통과시킨다 — 그것이 버그다.
    """


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None, tool_choice: str | None = None,
             log_tag: dict | None = None) -> dict:
        """OpenAI 호환 응답을 그대로 반환한다.

        반환에서 쓰는 것: choices[0].message (content 또는 tool_calls)
        """
        ...


def load_key(env: str = "OPENROUTER_API_KEY") -> str:
    """`.env.local` → `.env` → 환경변수 순으로 열쇠를 찾는다.

    `env` 로 이름을 고른다 — 백엔드마다 열쇠가 다르다 (`KEY_ENV`).
    """
    import os
    for name in (".env.local", ".env"):
        p = ROOT / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith(env):
                    return line.split("=", 1)[1].strip().strip("\"'")
    if os.environ.get(env):
        return os.environ[env]
    raise RuntimeError(f"{env} 를 .env.local 에 넣으세요")


def key_for(backend: str) -> str:
    if backend not in KEY_ENV:
        raise RuntimeError(f"모르는 백엔드: {backend}. {sorted(BACKENDS)} 중 하나")
    return load_key(KEY_ENV[backend])


class OpenRouterClient:
    """실제 호출. POST /chat/completions.

    ⚠ 모델이 tools 를 지원해야 tool_calls 가 온다. 미지원이면 content 에 JSON 흉내가
      온다 (자주 틀리는 곳 8). supported_parameters 를 사전에 확인할 것.
    """

    def __init__(self, model: str, api_key: str | None = None,
                 # **4 → 10** (8/27 · Eddie). 260827-002 에서 소진된 번역 5건이
                 # **전부 4회 상한에 걸렸다** (`[4,4,4,4,4]`) — 총 대기 56초로는
                 # 429 가 몰리는 구간(10해에 19건)을 못 넘긴다. 번역 유실은 세계의
                 # 사실이 아니라 인프라 사고이므로 「몇 번 만에 포기」 가 아니라
                 # 「끝까지 기다린다」 가 옳다. 10회면 최악 7분이고 한 해가 7.4분이다.
                 temperature: float = 0.7, retries: int = 10, timeout: int = 120,
                 recorder=None, deadline: float = 90.0, max_tokens: int | None = None,
                 reasoning: dict | None = None, provider: dict | None = None,
                 seed: int | None = None, backend: str = "openrouter",
                 rate_guard: "RateLimitGuard | None" = None):
        self.model = model
        if backend not in BACKENDS:
            raise RuntimeError(f"모르는 백엔드: {backend}. {sorted(BACKENDS)} 중 하나")
        self.backend = backend
        self.endpoint = BACKENDS[backend]
        self.api_key = api_key or load_key(KEY_ENV[backend])
        self.temperature = temperature
        self.retries = retries
        self.timeout = timeout
        self.recorder = recorder      # 호출 1회(재시도 각각)를 raw 로 남긴다 (spec 9장)
        self.deadline = deadline      # 호출 1회의 **벽시계** 상한. 아래 설명 참조
        self.max_tokens = max_tokens  # 응답 상한. 없으면 반복 붕괴가 안 잘린다
        self.reasoning = reasoning    # 사고 예산 (OpenRouter 통합 파라미터를 그대로)
        self.provider = provider      # 프로바이더 라우팅. 같은 모델도 업체마다 가격이 다르다
        self.seed = seed              # 샘플링 시드. 프로바이더가 존중할 때만 뜻이 있다
        # **429 폭풍이면 런을 세운다.** 클라이언트 둘(에이전트·번역기)이 **같은** 그릇을
        # 나눠 갖는다 — 어느 쪽이 막히든 그 런의 데이터가 상하는 것은 같다.
        self.rate_guard = rate_guard

    def _call_with_deadline(self, req):
        """urlopen 을 별도 스레드에서 돌리고 `deadline` 초 안에 안 오면 버린다.

        ⚠ `urlopen(timeout=)` 은 **소켓 읽기 하나**의 제한이지 호출 전체가 아니다.
          응답이 찔끔찔끔 오면 타이머가 매번 초기화돼 상한이 아무것도 묶지 못한다.
          50턴 실측에서 이것 때문에 한 호출이 **31분** 걸렸고, 전체 1,395콜 중
          30초 넘는 20건(1.4%)이 벽시계 149분 중 146분을 먹었다. 나머지 1,375건은
          전부 5초 미만이었다.

        버려진 스레드는 데몬이라 프로세스 종료를 막지 않는다. 소켓이 결국 닫히면
        알아서 끝나고, 그 사이 우리는 재시도를 진행한다.
        """
        box: dict = {}

        def _work():
            try:
                box["resp"] = json.load(urllib.request.urlopen(req, timeout=self.timeout))
            except BaseException as e:       # 스레드 안의 예외를 밖으로 옮긴다
                box["exc"] = e

        t = threading.Thread(target=_work, daemon=True)
        t.start()
        t.join(self.deadline)
        if t.is_alive():
            raise TimeoutError(f"deadline {self.deadline:.0f}s 초과 — 호출을 버립니다")
        if "exc" in box:
            raise box["exc"]
        return box["resp"]

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None, tool_choice: str | None = None,
             log_tag: dict | None = None) -> dict:
        """`log_tag` 는 raw_calls 행에 그대로 합쳐진다.

        **누가 언제 건 호출인지가 raw 에 없었다.** kind 는 클라이언트를 만들 때 한 번
        붙는 고정 태그라 turn·agent 를 담을 수 없었고, 그래서 raw_calls 를 events 와
        이어붙일 방법이 없었다 — 호출 단위 분석이 통째로 막혀 있었다.
        """
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        # **OpenRouter 전용 필드는 저쪽에 없는 이름이다** — 실으면 400 이다.
        if self.reasoning and self.backend == "openrouter":
            body["reasoning"] = dict(self.reasoning)
        if self.provider and self.backend == "openrouter":
            body["provider"] = dict(self.provider)
        if self.seed is not None:
            body["seed"] = self.seed
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        for attempt in range(self.retries):
            t0 = time.time()
            last = attempt == self.retries - 1
            # ── 망 구간. **여기서만** 예외를 삼킨다 ────────────────────────────
            try:
                resp = self._call_with_deadline(req)
            except urllib.error.HTTPError as e:
                self._record(body, attempt + 1, t0, error=f"HTTP {e.code}", log_tag=log_tag)
                # **402 는 즉시 세운다** (8/28 · Eddie). 크레딧이 떨어지자 네 런이
                # **멈추지 않고 30해까지 흘러갔다** — 402 는 429 도 망 오류도 아니라
                # 차단기에 안 걸리고, 에이전트마다 「그 해 포기」 로 조용히 처리됐다.
                # 결과는 6해 이후 아무도 행동하지 않는 세계 넷이고, 그것이 정상 완주로
                # 기록됐다 (콜 681~730 · 402 238~257건).
                #
                # 재시도해도 소용없다 — 돈이 생기기 전에는 계속 402 다. 그러니 물러날
                # 것이 아니라 **바로 서야** 한다. 복원점이 있으니 충전 후 이어붙인다.
                if e.code == 402:
                    raise OutOfCredit(
                        f"402 — 크레딧이 떨어졌습니다 ({self.model}).\n"
                        f"  충전 후 마지막 복원점부터 `--from-turn` 으로 이어할 수 있습니다."
                    ) from e
                if e.code == 429:                          # 레이트 리밋: 길게 물러난다
                    if last:
                        # **재시도를 다 쓴 429 만 센다.** 중간에 회수된 것은 시간만
                        # 먹었지 데이터를 안 상하게 한다 (154건 중 23건만 실제 손실).
                        if self.rate_guard is not None:
                            self.rate_guard.exhausted(self.model)
                        raise LLMCallError(f"HTTP 429 (재시도 소진)") from e
                    time.sleep(min(60, 8 * (2 ** attempt)))
                    continue
                if 500 <= e.code < 600 and not last:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise LLMCallError(f"HTTP {e.code} {e.reason}") from e
            except (TimeoutError, urllib.error.URLError, OSError, ValueError) as e:
                # URLError·TimeoutError 는 OSError 계열, 잘린 본문은 JSONDecodeError
                # (ValueError 계열). 여기까지가 **망 쪽 사정**이다.
                self._record(body, attempt + 1, t0,
                             error=f"{type(e).__name__}: {e}", log_tag=log_tag)
                if last:
                    raise LLMCallError(f"{type(e).__name__}: {e}") from e
                time.sleep(3 * (attempt + 1))
                continue
            # ── 응답을 받았다. 여기부터 우리 코드의 문제는 **그대로 터진다** ──
            #
            # ⚠ 프로바이더가 **HTTP 200 에 error 를 실어 보낸다.** gemma :free 에서
            #   22콜 중 5건이 {"error":{"code":504,"message":"Provider timed out"}}
            #   였고, choices 를 그대로 인덱싱하다 KeyError 로 런 전체가 죽었다.
            #   이건 우리 버그가 아니라 프로바이더 사정이므로 재시도를 태운다.
            if "choices" not in resp:
                err = resp.get("error") or {}
                why = f"no choices — {err.get('code', '?')}: {str(err.get('message'))[:120]}"
                self._record(body, attempt + 1, t0, response=resp, error=why, log_tag=log_tag)
                if last:
                    raise LLMCallError(why)
                time.sleep(3 * (attempt + 1))
                continue
            self._record(body, attempt + 1, t0, response=resp, log_tag=log_tag)
            return resp
        raise LLMCallError("재시도 소진")

    def _record(self, body, attempt, t0, response=None, error=None, log_tag=None):
        if self.recorder is None:
            return
        self.recorder({**(log_tag or {}),
                       "attempt": attempt, "latency_ms": round((time.time() - t0) * 1000),
                       "request": body, "response": response, "error": error})


class StubClient:
    """테스트용. 정해둔 assistant 메시지 시퀀스를 chat() 호출마다 하나씩 돌려준다.

    각 script 원소는 choices[0].message 로 그대로 들어갈 dict:
      {"role": "assistant", "content": None,
       "tool_calls": [{"id": "c1", "type": "function",
                       "function": {"name": "speak", "arguments": "{...}"}}]}
    tool_calls 가 없거나 빈 배열이면 에이전트 루프가 종료한다.

    편의를 위해 tool_call(name, **args) 헬퍼로 스크립트를 짧게 쓸 수 있다.
    """

    def __init__(self, script: list[dict], recorder=None):
        self._script = list(script)
        self._i = 0
        self.calls: list[dict] = []          # 검증용: 받은 messages 기록
        self.recorder = recorder             # 실물과 같은 raw 기록 경로 (테스트에서 형식 검증)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None, tool_choice: str | None = None,
             log_tag: dict | None = None) -> dict:
        self.calls.append({"messages": list(messages), "tools": tools,
                           "tool_choice": tool_choice, "log_tag": log_tag})   # 스냅샷
        _t0 = time.time()
        if self._i >= len(self._script):
            # 스크립트 소진 → 도구 없이 종료 신호
            msg = {"role": "assistant", "content": "", "tool_calls": []}
        else:
            msg = self._script[self._i]
            self._i += 1
        resp = {"choices": [{"message": msg}]}
        if self.recorder is not None:
            self.recorder({**(log_tag or {}),
                           "attempt": 1, "latency_ms": round((time.time() - _t0) * 1000),
                           "request": {"model": "stub", "messages": list(messages),
                                       "tools": tools, "temperature": temperature,
                                       "tool_choice": tool_choice},
                           "response": resp, "error": None})
        return resp


def tool_call(name: str, call_id: str = "c", **arguments) -> dict:
    """스크립트용 tool_call 한 건을 만든다."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def assistant_msg(*tool_calls: dict, content: str = "") -> dict:
    """tool_call 들을 담은 assistant 메시지 하나."""
    return {"role": "assistant", "content": content, "tool_calls": list(tool_calls)}
