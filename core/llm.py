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


def load_key() -> str:
    """`.env.local` → `.env` → 환경변수 순으로 OPENROUTER_API_KEY 를 찾는다."""
    import os
    for name in (".env.local", ".env"):
        p = ROOT / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENROUTER_API_KEY"):
                    return line.split("=", 1)[1].strip().strip("\"'")
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    raise RuntimeError("OPENROUTER_API_KEY 를 .env.local 에 넣으세요")


class OpenRouterClient:
    """실제 호출. POST /chat/completions.

    ⚠ 모델이 tools 를 지원해야 tool_calls 가 온다. 미지원이면 content 에 JSON 흉내가
      온다 (자주 틀리는 곳 8). supported_parameters 를 사전에 확인할 것.
    """

    def __init__(self, model: str, api_key: str | None = None,
                 temperature: float = 0.7, retries: int = 4, timeout: int = 120,
                 recorder=None, deadline: float = 90.0, max_tokens: int | None = None,
                 reasoning: dict | None = None, provider: dict | None = None,
                 seed: int | None = None):
        self.model = model
        self.api_key = api_key or load_key()
        self.temperature = temperature
        self.retries = retries
        self.timeout = timeout
        self.recorder = recorder      # 호출 1회(재시도 각각)를 raw 로 남긴다 (spec 9장)
        self.deadline = deadline      # 호출 1회의 **벽시계** 상한. 아래 설명 참조
        self.max_tokens = max_tokens  # 응답 상한. 없으면 반복 붕괴가 안 잘린다
        self.reasoning = reasoning    # 사고 예산 (OpenRouter 통합 파라미터를 그대로)
        self.provider = provider      # 프로바이더 라우팅. 같은 모델도 업체마다 가격이 다르다
        self.seed = seed              # 샘플링 시드. 프로바이더가 존중할 때만 뜻이 있다

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
        if self.reasoning:
            body["reasoning"] = dict(self.reasoning)
        if self.provider:
            body["provider"] = dict(self.provider)
        if self.seed is not None:
            body["seed"] = self.seed
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        req = urllib.request.Request(
            ENDPOINT,
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
                if e.code == 429:                          # 레이트 리밋: 길게 물러난다
                    if last:
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
