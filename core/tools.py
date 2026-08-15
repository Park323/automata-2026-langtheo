"""Tool definitions. spec 4.2 actions as OpenAI function schemas.

One action = one tool. Descriptions in English (team convention). They state the
mechanic only — no goal, and no hint that AI translation loses anything.
end_turn is not in the spec; it is a loop-control signal (without it the stop
condition is ambiguous).
"""
from __future__ import annotations


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """모든 도구에 reasoning 을 필수로 붙인다.

    spec 4.2 가 reasoning 을 "필수. 사후 분류의 감사 표면" 이라고 한 그대로다.
    행동마다 근거가 남으므로 지표 4(의도 실패율)를 여기서 역추적한다 — 별도의
    understood 수집 도구는 두지 않는다. 세계를 바꾸지 않는 도구는 모델이 부르지
    않는다는 것이 실측으로 확인됐다 (MAX_STEPS 8·20 양쪽에서 0건).
    """
    props = {**properties, "reasoning": _REASONING}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props,
                           "required": [*required, "reasoning"]},
        },
    }


_ROUTE = {"type": "string", "enum": ["original", "ai"],
          "description": "international only; ignored for a recipient in your own nation"}
_TEXT = {"type": "string", "description": "the message body, written in your own language"}
_TR_INSTR = {"type": "string",
             "description": "an instruction for the translator; leave empty for a neutral default"}
# spec 4.2 의 reasoning. **모든** 도구의 필수 인자다 (_fn 이 자동 주입).
# API 응답의 message.reasoning(추론 모델의 사고 과정)과는 다른 것이다.
_REASONING = {"type": "string",
              "description": "one sentence on why you did what you did this turn"}


TOOLS: list[dict] = [
    _fn("speak", "Send a message to one recipient. It arrives next turn.",
        {"to": {"type": "string", "description": "recipient id (e.g. Ranoa2)"},
         "route": _ROUTE, "text": _TEXT,
         "translate_instruction": _TR_INSTR},
        ["to", "text"]),

    _fn("ask", "Ask back about a message you received. Costs ask_clarification plus the route cost.",
        {"to": {"type": "string", "description": "recipient id"},
         "route": _ROUTE, "text": _TEXT,
         "translate_instruction": _TR_INSTR,
         "reply_to": {"type": "integer", "description": "the msg_id you are replying to"}},
        ["to", "text", "reply_to"]),

    _fn("invest", "Invest in a resource. Only facility can name a target nation (to).",
        {"target": {"type": "string", "enum": ["wellness", "national", "facility"]},
         "amount": {"type": "number", "description": "amount taken from your budget"},
         "to": {"type": "string", "description": "facility only: target nation id (defaults to your own)"}},
        ["target", "amount"]),

    _fn("learn", "Learn another nation's language. Give a nation id, not a language code.",
        {"country": {"type": "string", "description": "the nation whose language to learn (e.g. Ranoa)"}},
        ["country"]),

    _fn("propose_vote", "Propose changing your nation's facility. Domestic only.",
        {"target": {"type": "string", "enum": ["bunker", "interceptor"]}},
        ["target"]),

    _fn("procreate", "Leave a child and die. Calling it ends your turn at once.",
        {"testament": {"type": "string", "description": "one sentence passed to the child"}},
        ["testament"]),

    _fn("memory_write",
        "Overwrite your notes. They stay with you next turn; nobody else sees them.",
        {"text": {"type": "string", "description": "your notes, replacing whatever was there"}},
        ["text"]),

    _fn("end_turn", "You have nothing more to do this turn. Ends the loop.", {}, []),
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
