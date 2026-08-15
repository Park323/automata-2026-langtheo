"""LLM 백엔드. OpenRouter (OpenAI 호환).

- OpenRouterClient : 실제 호출. tools/pilot/run_pilot.py 의 urllib·백오프 패턴을 계승.
- StubClient       : 테스트용. 미리 정해둔 응답 시퀀스를 순서대로 돌려준다.
                     LLM 은 비결정적이라 이게 없으면 합격 기준을 못 쓴다.
"""
from __future__ import annotations

import json
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
                 temperature: float = 0.7, retries: int = 4, timeout: int = 120):
        self.model = model
        self.api_key = api_key or load_key()
        self.temperature = temperature
        self.retries = retries
        self.timeout = timeout

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
            try:
                return json.load(urllib.request.urlopen(req, timeout=self.timeout))
            except urllib.error.HTTPError as e:
                if e.code == 429:                          # 레이트 리밋: 길게 물러난다
                    time.sleep(min(60, 8 * (2 ** attempt)))
                    continue
                if 500 <= e.code < 600 and attempt < self.retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("재시도 소진")


class StubClient:
    """테스트용. 정해둔 assistant 메시지 시퀀스를 chat() 호출마다 하나씩 돌려준다.

    각 script 원소는 choices[0].message 로 그대로 들어갈 dict:
      {"role": "assistant", "content": None,
       "tool_calls": [{"id": "c1", "type": "function",
                       "function": {"name": "speak", "arguments": "{...}"}}]}
    tool_calls 가 없거나 빈 배열이면 에이전트 루프가 종료한다.

    편의를 위해 tool_call(name, **args) 헬퍼로 스크립트를 짧게 쓸 수 있다.
    """

    def __init__(self, script: list[dict]):
        self._script = list(script)
        self._i = 0
        self.calls: list[dict] = []          # 검증용: 받은 messages 기록

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None) -> dict:
        self.calls.append({"messages": list(messages), "tools": tools})   # 스냅샷
        if self._i >= len(self._script):
            # 스크립트 소진 → 도구 없이 종료 신호
            msg = {"role": "assistant", "content": "", "tool_calls": []}
        else:
            msg = self._script[self._i]
            self._i += 1
        return {"choices": [{"message": msg}]}


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
