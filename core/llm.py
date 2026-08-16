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


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None) -> dict:
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
                 recorder=None, deadline: float = 90.0):
        self.model = model
        self.api_key = api_key or load_key()
        self.temperature = temperature
        self.retries = retries
        self.timeout = timeout
        self.recorder = recorder      # 호출 1회(재시도 각각)를 raw 로 남긴다 (spec 9장)
        self.deadline = deadline      # 호출 1회의 **벽시계** 상한. 아래 설명 참조

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
             temperature: float | None = None) -> dict:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                resp = self._call_with_deadline(req)
                self._record(body, attempt + 1, t0, response=resp)
                return resp
            except urllib.error.HTTPError as e:
                self._record(body, attempt + 1, t0, error=f"HTTP {e.code}")
                if e.code == 429:                          # 레이트 리밋: 길게 물러난다
                    time.sleep(min(60, 8 * (2 ** attempt)))
                    continue
                if 500 <= e.code < 600 and attempt < self.retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise
            except Exception as e:
                self._record(body, attempt + 1, t0, error=f"{type(e).__name__}: {e}")
                if attempt == self.retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("재시도 소진")

    def _record(self, body, attempt, t0, response=None, error=None):
        if self.recorder is None:
            return
        self.recorder({"attempt": attempt, "latency_ms": round((time.time() - t0) * 1000),
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
             temperature: float | None = None) -> dict:
        self.calls.append({"messages": list(messages), "tools": tools})   # 스냅샷
        _t0 = time.time()
        if self._i >= len(self._script):
            # 스크립트 소진 → 도구 없이 종료 신호
            msg = {"role": "assistant", "content": "", "tool_calls": []}
        else:
            msg = self._script[self._i]
            self._i += 1
        resp = {"choices": [{"message": msg}]}
        if self.recorder is not None:
            self.recorder({"attempt": 1, "latency_ms": round((time.time() - _t0) * 1000),
                           "request": {"model": "stub", "messages": list(messages),
                                       "tools": tools, "temperature": temperature},
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
