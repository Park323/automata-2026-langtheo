"""Tool definitions. spec 4.2 actions as OpenAI function schemas.

One action = one tool. Descriptions in English (team convention). They state the
mechanic only — no goal, and no hint that AI translation loses anything.
end_turn is not in the spec; it is a loop-control signal (without it the stop
condition is ambiguous).
"""
from __future__ import annotations

import copy


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
          "description": "international only; ignored for a recipient in your own nation. "
                         "`original` sends your words untranslated. If you can handle the "
                         "recipient's national language it always lands, whatever they can "
                         "read. If you cannot, it lands only on someone who reads yours — "
                         "and you are not told beforehand whether they do"}
# **경로가 언어를 정한다** (8/22).
#
#   ai        모국어로 써야 한다. **여기가 측정 채널이다** — 번역이 무엇을 잃는지 재려면
#             입력 언어가 흔들리면 안 된다. 실측에서 zh 에이전트가 `ai` 로 **일본어**를
#             보냈고, 그러면 번역기에 이미 도착 언어를 넣는 셈이라 지표 7 이 죽는다.
#   original  **나라마다 길이 하나씩만 열린다** (8/25 · #44). 그 나라 말을 알면 그 말로,
#             모르면 내 말로 — 관측의 나라별 줄이 그것을 이름으로 말한다.
#             「아는 말 아무거나」 였을 때는 **제3의 언어**로 쓴 글이 아무도 못 읽는 채로
#             전달되고 라벨이 거짓 사유를 댔다. 도구 설명은 정적이라 그 안내를 담을 수
#             없다(목록이 모듈 상수이고 토큰 회계가 id() 로 캐시한다) — 관측을 가리킨다.
_TEXT = {"type": "string",
         "description": "the message body. With `ai`, write it in your own language. "
                        "With `original`, write it in the language the observation names "
                        "for that destination"}
_TR_INSTR = {"type": "string",
             "description": "an instruction for the translator; leave empty for a neutral default"}
# spec 4.2 의 reasoning. **모든** 도구의 필수 인자다 (_fn 이 자동 주입).
# API 응답의 message.reasoning(추론 모델의 사고 과정)과는 다른 것이다.
_REASONING = {"type": "string",
              "description": "one sentence on why you did what you did this year"}


def _build(reasoning_arg: bool) -> list[dict]:
  def fn(name, desc, props, req, reasoning=True):
      return _fn(name, desc, props, req, reasoning and reasoning_arg)
  return [
    fn("speak",
        "Send a message to one recipient. It arrives when that person next acts, and "
        "their reply reaches you when you next act after that. Sending the same thing "
        "again before then does not make it arrive sooner.",
        {"to": {"type": "string", "description": "recipient id (e.g. Ranoa2)"},
         "route": _ROUTE, "text": _TEXT,
         "translate_instruction": _TR_INSTR},
        ["to", "text"]),

    fn("invest",
        "Put one fixed amount into a resource; the observation shows how much it moves "
        "and how much action it takes. How much one call moves differs from person to "
        "person; how much progress a given amount buys differs from nation to nation. "
        "The two are set separately. Your amount is yours; you cannot see anyone "
        "else's, so ask them. "
        "`national` raises your nation's technical level, which "
        "lifts how much progress a facility gets out of what is put into it, and "
        "the precision of observe_risk — for everyone in that nation. "
        "For facility you may name any nation with `to` — your "
        "own or another; leaving `to` out puts it into your own nation's. "
        "What you put into a facility goes into whatever that nation is currently "
        "building — which may not be what you think it is, and a nation that has not "
        "yet decided what to build has nothing to put it into, so it buys "
        "no progress. "
        "Only that nation knows which it is.",
        {"target": {"type": "string", "enum": ["wellness", "national", "facility"]},
         "to": {"type": "string", "description": "facility only: any nation id, yours or another's. "
                               "Defaults to your own nation"}},
        ["target"]),

    fn("learn",
        "Put one fixed amount toward learning another nation's language; one payment "
        "takes the same action as an investment. "
        "Give a nation id, not a language code. "
        "What you put in accumulates, and your observation shows how far along you are "
        "and how much is still needed. You can read and write the language once the "
        "accumulated amount reaches what it costs. That cost can fall while you are "
        "partway through — if someone in your nation comes to speak it, or if your parent "
        "did — and then you may finish sooner than you expected.",
        {"country": {"type": "string", "description": "the nation whose language to learn (e.g. Ranoa)"}},
        ["country"]),

    fn("observe_risk",
        "Measure three things: how many years remain until the meteorite strikes, and "
        "how much progress an interceptor and a bunker each need. **All three readings "
        "are imprecise, and you are not told by how much.** Your nation's accumulated "
        "national investment is what sharpens them. Each reading is a fresh "
        "measurement and costs a large share of your action points — measuring the "
        "world takes most of a year. What you learn is yours alone — nobody else "
        "sees it.",
        {}, []),

    fn("propose_vote",
        "Call a ballot on what your nation should build. Your nation only. You do not "
        "say what to build — the ballot itself decides that. Nothing changes yet: time "
        "passes so people can talk it over, and then everyone casts a choice; your "
        "observation tells you which year that is. **Only people of your own nation may "
        "vote** — a foreigner cannot, no "
        "matter what they say. If a ballot is already called, calling again does nothing. "
        "It costs nothing to fund — nobody is priced out of proposing — but it "
        "takes more than half of your action points for the year.",
        {}, []),

    fn("vote",
        "Choose what **your own nation** builds — `interceptor`, `bunker`, or `abstain`. "
        "You cannot vote in another nation. Only in the year the ballot is held; the "
        "observation tells you which year that is. The choice with the most votes wins; "
        "`abstain` counts for neither. If the two tie, or nobody votes, your nation keeps "
        "what it has and its progress survives. It costs almost no action "
        "points, so voting never takes away your chance to speak on the day it matters most.",
        {"choice": {"type": "string", "enum": ["interceptor", "bunker", "abstain"]}},
        ["choice"]),

    # **`give` 를 없앴다** (8/25 · AP 전면 통일). 양도할 것이 없다 — 돈이 사라졌고 AP 는
    # 「내 올해 주의력」 이라 넘길 수 없다. 나라 사이 이전은 `invest to=<타국>` 이 맡는다.

    fn("memory_write",
        "Overwrite your notes. They stay with you next year; nobody else sees them.",
        {"text": {"type": "string", "description": "your notes, replacing whatever was there"}},
        ["text"]),

    # 유일하게 reasoning 이 없는 도구 — 행동이 아니라 행동을 그만두는 신호다.
    fn("end_turn", "You have nothing more to do this year. Ends the loop.", {}, [],
        reasoning=False),
]


# 두 벌을 미리 만들어 둔다. 사고형 모델에서는 도구마다 reasoning 을 또 받지 않는다
# (spec 12.1) — 모델이 이미 사고를 하고 있고, 그건 api_reasoning 으로 따로 남는다.
TOOLS = _build(True)
TOOLS_NO_REASONING = _build(False)
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


# ── 기억은 자리가 좁아진 뒤에만 ───────────────────────────────────────────────
#
# `memory_write` 는 **잃을 것이 생긴 뒤에** 뜻이 있는 도구다. 대화가 아직 짧으면 적어 둘
# 이유가 없는데, 30해 실측에서 **206번** 불렸다 — 압박이 걸리기 한참 전부터다.
#
# 그 값이 공짜(돈 0 · AP 0)라 무엇도 막지 않고, 순차 라운드로빈은 스텝 단위로 도므로
# 한 번 부르면 그만큼 남들이 먼저 움직인다. 즉 **공짜가 아니라 차례를 쓴다.**
#
# 그래서 압박선(`context_limit × warn_ratio`) 아래에서는 **목록에서 빼고 비용표에서도
# 숨긴다.** 없는 도구를 설명하지 않는 것이 「사실만 적는다」 에 맞고, 압박 경고가 뜨는
# 그때 도구도 함께 나타나므로 **경고가 곧 안내**가 된다.
_NO_MEM = {"memory_write"}


def _drop_memory(tools: list[dict]) -> list[dict]:
    return [t for t in tools if t["function"]["name"] not in _NO_MEM]


TOOLS_NO_MEM = _drop_memory(TOOLS)
TOOLS_NO_REASONING_NO_MEM = _drop_memory(TOOLS_NO_REASONING)


# ── AI 번역이 없는 세계 ───────────────────────────────────────────────────────
#
# **기준 조건이다** (8/25). 가설은 「AI 번역 비용이 **내려가면**」 이므로, 내려가기 전의
# 세계가 있어야 그 변화를 잴 수 있다. 노브를 아주 비싸게 두는 것과는 다르다 — 비싸면
# 에이전트가 그 선택지를 보고 값을 재지만, 없으면 **길이 둘뿐**이다: 배우거나, 내 말로
# 보내고 상대가 읽어주길 걸거나.
#
# 노브 값을 한 해 AP 위로 올려 흉내내지 않는다. `asserts` 가 그것을 막고 있고, 막는 이유가
# 「비싼 것과 없는 것이 구분되지 않는다」 다.
def _drop_ai(tools: list[dict]) -> list[dict]:
    """`speak` 의 route 에서 `ai` 를 뺀다. 없는 선택지는 설명도 하지 않는다."""
    out = []
    for t in tools:
        t = copy.deepcopy(t)
        if t["function"]["name"] == "speak":
            pr = t["function"]["parameters"]["properties"]["route"]
            pr["enum"] = [r for r in pr["enum"] if r != "ai"]
            pr["description"] = (
                "international only; ignored for a recipient in your own nation. "
                "`original` sends your words untranslated. If you can handle the "
                "recipient's national language it always lands, whatever they can "
                "read. If you cannot, it lands only on someone who reads yours — "
                "and you are not told beforehand whether they do")
        out.append(t)
    return out


TOOLS_NO_AI = _drop_ai(TOOLS)
TOOLS_NO_AI_NO_MEM = _drop_memory(TOOLS_NO_AI)
TOOLS_NO_REASONING_NO_AI = _drop_ai(TOOLS_NO_REASONING)
TOOLS_NO_REASONING_NO_AI_NO_MEM = _drop_memory(TOOLS_NO_REASONING_NO_AI)


def tools_for(cfg, memory: bool = True, ai: bool = True) -> list[dict]:
    """모델에게 실어 보낼 도구 목록.

    `memory` 는 「기억을 쓸 수 있는 상태인가」 다 — 호출부가 `under_pressure()` 로 정한다.
    `ai`   는 「이 세계에 AI 번역이 있는가」 다 — `knob_ai is not None` 이 그 뜻이다.

    **여덟 벌을 미리 만들어 둔다.** 토큰 회계가 `id()` 로 캐시하므로 (`_TOOL_TOKENS_BY_ID`)
    호출마다 새 리스트를 만들면 조회가 빗나가고 문맥 예산이 조용히 샌다.
    """
    reasoning = getattr(cfg.llm, "tool_reasoning", True)
    if reasoning:
        if ai:
            return TOOLS if memory else TOOLS_NO_MEM
        return TOOLS_NO_AI if memory else TOOLS_NO_AI_NO_MEM
    if ai:
        return TOOLS_NO_REASONING if memory else TOOLS_NO_REASONING_NO_MEM
    return TOOLS_NO_REASONING_NO_AI if memory else TOOLS_NO_REASONING_NO_AI_NO_MEM
