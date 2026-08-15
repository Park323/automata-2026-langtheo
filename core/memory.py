"""개체 기억의 토큰 회계와 축출. spec 4.5.

에이전트의 `messages` 는 태어나서 죽을 때까지 이어진다. 한계에 닿으면 오래된 턴
블록부터 축출하되 `system`(0번)과 가장 최근 관측은 남긴다.

⚠ 정확한 토크나이저를 두지 않는다 — 축출은 안전장치이지 과금 대상이 아니다. 실제
  압박 판정(warn)은 API 응답의 `usage.prompt_tokens` 를 쓰고(있으면), 그게 없는
  경로(StubClient)와 사전 축출에는 글자 기반 추정을 쓴다. spec 4.5 가 "언어별 편향은
  감수한다" 고 명시한 그 편향이 이 추정에도 그대로 있다.
"""
from __future__ import annotations

import json

# OpenAI 계열 대략치: 영어 ~4자/토큰. CJK 는 이보다 조밀하지만(토큰이 더 많이 나옴)
# 축출은 보수적이어도 무방하므로 단일 계수로 둔다. 편향은 raw_calls 의 실측으로 확인.
_CHARS_PER_TOKEN = 4


def approx_tokens(messages: list[dict], tool_tokens: int = 0) -> int:
    """messages + 도구 스키마의 대략 토큰 수. content 와 tool_calls 직렬화를 함께 센다."""
    total = tool_tokens
    for m in messages:
        total += 4                                   # role·구분자 등 메시지당 고정 오버헤드
        content = m.get("content")
        if content:
            total += len(content) // _CHARS_PER_TOKEN + 1
        tcs = m.get("tool_calls")
        if tcs:                                      # 함수명·인자 JSON 도 프롬프트에 실린다
            total += len(json.dumps(tcs, ensure_ascii=False)) // _CHARS_PER_TOKEN + 1
    return total


def tool_schema_tokens(tools: list[dict] | None) -> int:
    """도구 스키마는 매 호출 프롬프트에 통째로 실린다 — 고정 비용으로 함께 센다."""
    if not tools:
        return 0
    return len(json.dumps(tools, ensure_ascii=False)) // _CHARS_PER_TOKEN + 1


def evict(messages: list[dict], limit: int, tool_tokens: int = 0) -> int:
    """한계 초과 시 가장 오래된 턴 블록부터 축출한다. 반환: 축출한 블록 수.

    턴 블록 = user(관측) 하나 + 그 뒤의 assistant/tool 왕복(다음 user 직전까지).
    블록 통째로만 지운다 — assistant(tool_calls)와 그 tool 응답이 갈라지면 API 가 400 을
    돌려주기 때문이다. system(0번)과 **가장 최근 블록**은 절대 지우지 않는다.

    ⚠ messages 를 제자리에서 수정한다.
    """
    dropped = 0
    while approx_tokens(messages, tool_tokens) > limit:
        starts = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if len(starts) <= 1:
            break                                    # 남은 블록이 하나뿐 → 더 못 지운다
        del messages[starts[0]:starts[1]]            # 가장 오래된 user 블록 전체
        dropped += 1
    return dropped
