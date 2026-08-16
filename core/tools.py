"""Tool definitions. spec 4.2 actions as OpenAI function schemas.

One action = one tool. Descriptions in English (team convention). They state the
mechanic only — no goal, and no hint that AI translation loses anything.
end_turn is not in the spec; it is a loop-control signal (without it the stop
condition is ambiguous).
"""
from __future__ import annotations


def _fn(name: str, description: str, properties: dict, required: list[str],
        reasoning: bool = True) -> dict:
    """**행동**하는 도구에 reasoning 을 필수로 붙인다.

    spec 4.2 가 reasoning 을 "필수. 사후 분류의 감사 표면" 이라고 한 그대로다.
    행동마다 근거가 남으므로 지표 4(의도 실패율)를 여기서 역추적한다 — 별도의
    understood 수집 도구는 두지 않는다. 세계를 바꾸지 않는 도구는 모델이 부르지
    않는다는 것이 실측으로 확인됐다 (MAX_STEPS 8·20 양쪽에서 0건).

    `end_turn` 만 예외다 (`reasoning=False`). 행동이 아니라 **행동을 그만두는 신호**라
    "행동 하나에 근거 하나" 에 대응하지 않는다. 실측에서 근거가 있는 에이전트턴 407 중
    end_turn 근거 **뿐**인 것은 14건(3%)뿐이라, 빼도 지표 4 의 표본이 3% 줄 뿐이다.
    """
    props = dict(properties)
    req = list(required)
    if reasoning:
        props["reasoning"] = _REASONING
        req.append("reasoning")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": req},
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

    _fn("invest",
        "Invest in a resource. Only facility can name a target nation (to). "
        "Money you put into a facility goes into whatever that nation is currently "
        "building — which may not be what you think it is. Only that nation knows.",
        {"target": {"type": "string", "enum": ["wellness", "national", "facility"]},
         "amount": {"type": "number", "description": "amount taken from your budget"},
         "to": {"type": "string", "description": "facility only: target nation id (defaults to your own)"}},
        ["target", "amount"]),

    _fn("learn", "Learn another nation's language. Give a nation id, not a language code.",
        {"country": {"type": "string", "description": "the nation whose language to learn (e.g. Ranoa)"}},
        ["country"]),

    _fn("propose_vote",
        "Open a proposal to set your nation's facility. Your nation only. "
        "Nothing changes yet: three turns pass so people can talk it over, and on the "
        "fourth turn the ballot is held. **Only people of your own nation may vote on "
        "it** — a foreigner cannot, no matter what they say. It passes if approvals "
        "outnumber rejections. A nation can hold only one proposal at a time.",
        {"target": {"type": "string", "enum": ["bunker", "interceptor"]}},
        ["target"]),

    _fn("vote",
        "Cast your ballot on **your own nation's** open proposal — you cannot vote on "
        "another nation's. Only on the turn the ballot is held; the observation tells "
        "you which turn that is.",
        {"approve": {"type": "boolean", "description": "true to approve, false to reject"}},
        ["approve"]),

    _fn("procreate",
        "Leave a child and die. Calling it ends your turn at once. "
        "The child inherits your remaining budget and your testament, and gets a "
        "discount on learning any language you could read. The child does NOT "
        "inherit the languages themselves, nor your memory of this life.",
        {"testament": {"type": "string", "description": "one sentence passed to the child"}},
        ["testament"]),

    _fn("memory_write",
        "Overwrite your notes. They stay with you next turn; nobody else sees them.",
        {"text": {"type": "string", "description": "your notes, replacing whatever was there"}},
        ["text"]),

    # 유일하게 reasoning 이 없는 도구 — 행동이 아니라 행동을 그만두는 신호다.
    _fn("end_turn", "You have nothing more to do this turn. Ends the loop.", {}, [],
        reasoning=False),
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
