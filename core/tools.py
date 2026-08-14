"""도구 정의. spec 4.2 의 행동을 OpenAI function 스키마로 옮긴다.

행동 하나 = 도구 하나. 에이전트는 이 도구들만 호출할 수 있다.
end_turn 은 spec 에 없는 루프 제어용 — 없으면 종료 조건이 애매해진다.
"""
from __future__ import annotations


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_ROUTE = {"type": "string", "enum": ["original", "ai"],
          "description": "국제 발신에만 유효. 자국민이면 무시된다"}
_TEXT = {"type": "string", "description": "본문. 당신의 모국어로 쓴다"}
_INTENT = {"type": "string", "description": "전하려 한 것 한 문장. 상대에게 전달되지 않는다 (로그 전용)"}
_TR_INSTR = {"type": "string",
             "description": "번역기에 줄 지시. 비우면 '번역하라' 만 쓰인다"}


TOOLS: list[dict] = [
    _fn("speak", "한 명에게 메시지를 보낸다. 다음 턴에 도착한다.",
        {"to": {"type": "string", "description": "수신자 id (예: B2)"},
         "route": _ROUTE, "text": _TEXT, "intent": _INTENT,
         "translate_instruction": _TR_INSTR},
        ["to", "text", "intent"]),

    _fn("ask", "이미 받은 메시지에 되묻는다. 비용은 ask_clarification + 경로 비용.",
        {"to": {"type": "string", "description": "수신자 id"},
         "route": _ROUTE, "text": _TEXT, "intent": _INTENT,
         "translate_instruction": _TR_INSTR,
         "reply_to": {"type": "integer", "description": "되묻는 대상 메시지의 msg_id"}},
        ["to", "text", "intent", "reply_to"]),

    _fn("invest", "자원에 투자한다. facility 만 대상 국가(to)를 지정할 수 있다.",
        {"target": {"type": "string", "enum": ["wellness", "national", "facility"]},
         "amount": {"type": "number", "description": "투자액(예산에서 차감)"},
         "to": {"type": "string", "description": "facility 한정. 대상 국가 id (생략 시 자국)"}},
        ["target", "amount"]),

    _fn("learn", "다른 국가의 언어를 배운다. 언어 코드가 아니라 국가 id 를 준다 (spec 3.4).",
        {"country": {"type": "string", "description": "배울 대상 국가 id (예: B)"}},
        ["country"]),

    _fn("propose_vote", "국내 시설 용도 전환을 발의한다. 국내 전용.",
        {"target": {"type": "string", "enum": ["bunker", "interceptor"]}},
        ["target"]),

    _fn("procreate", "아이를 남기고 죽는다. 호출되면 그 턴이 즉시 끝난다.",
        {"testament": {"type": "string", "description": "유언 한 문장 (아이에게 전해진다)"}},
        ["testament"]),

    _fn("end_turn", "이번 턴에 더 할 일이 없다. 루프를 종료한다.", {}, []),
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
