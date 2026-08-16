"""한 에이전트의 한 턴. spec 4.2.

  messages = [system, user(관측)]
  반복 (종료 조건은 spec 4.5):
      resp = client.chat(messages, tools=TOOLS)
      tool_calls 없으면 종료
      각 tool_call 실행 → 결과를 role="tool" 로 append
  procreate / end_turn 은 루프를 즉시 끝낸다.

⚠ 도구는 세계를 즉시 바꾸지 않는다. 자기 budget/ap 만 즉시 차감하고 효과는 Sink 에
  넣는다. 국토 확정·진척 판정·cap 배분·번역은 전원의 루프가 끝난 뒤 loop.py 5단계에서.
⚠ 도구 결과로 감춰야 할 것을 흘리지 않는다 (진척 증가분·λ 변화·success_prob).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from core import messaging
from core.tools import TOOLS, TOOL_NAMES



@dataclass
class Sink:
    """이번 턴 전원의 '의도'를 모은다. loop.py 5단계에서 agent_id 정렬 순으로 정산한다.

    ⚠ 병렬 실행 시 append 순서는 비결정적이다. 모든 항목이 agent_id 를 담으므로
      정산 때 안정 정렬하면 결정론이 회복된다 (재현성 #1).
    """
    facility: list = field(default_factory=list)      # (to_country, amount, agent_id)
    wellness: list = field(default_factory=list)      # (agent_id, amount)
    national: list = field(default_factory=list)      # (country, amount, agent_id)
    messages: list = field(default_factory=list)      # 발신 dict (5장, 'from' 에 agent_id)
    votes: list = field(default_factory=list)         # 제안 (agent_id, country, target)
    ballots: list = field(default_factory=list)       # 찬반 (agent_id, country, approve)
    learns: list = field(default_factory=list)        # (agent_id, lang) — 다음 턴부터 유효
    procreations: list = field(default_factory=list)  # (agent_id, testament)


# ── 학습 비용 (spec 3.4) ──────────────────────────────────────────────────────

def learn_discounts(agent, country_id: str, world) -> tuple[bool, bool]:
    """(국내 구사자 있음, 부모가 구사함). spec 3.4 — 판정은 **그 순간** 새로 한다.

    국내 구사자가 죽으면 그 뒤의 학습은 다시 2배다. 할인은 상태가 아니라 조건이다.
    """
    target_lang = world.countries[country_id].lang
    domestic = any(
        o.id != agent.id and o.country == agent.country and target_lang in o.known_langs
        for o in world.agents.values()
    )
    return domestic, target_lang in agent.parent_langs


def learn_cost(agent, country_id: str, world, cfg) -> tuple[float, str]:
    """(비용, 할인사유). L × 국내구사자(×0.5) × 부모(×0.5), 중복 시 ×0.25."""
    base = cfg.costs.learn_base
    domestic, parent = learn_discounts(agent, country_id, world)
    mult = (0.5 if domestic else 1.0) * (0.5 if parent else 1.0)
    reasons = []
    if domestic:
        reasons.append("someone in your nation speaks it (x0.5)")
    if parent:
        reasons.append("your parent spoke it (x0.5)")
    return base * mult, " · ".join(reasons) if reasons else "no discount"


# ── 도구 실행 ────────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict, world, agent, cfg, sink: Sink,
                 knob_ai: float) -> tuple[dict, str | None]:
    """(tool_result, control). control="end" 면 턴 종료 (procreate/end_turn)."""

    if name == "end_turn":
        return {"ok": True, "ended": True}, "end"

    if name == "memory_write":
        # 예산이 아니라 AP 로 묶는다 (spec 4.5) — 예산을 물리면 기억이 시설 투자와
        # 경쟁해서 "AI 가 싸지면 기억을 덜 하는가" 관측에 교란이 섞인다.
        if agent.ap < cfg.ap.memory_write:
            return {"ok": False, "error": f"not enough AP; memory_write needs {cfg.ap.memory_write}"}, None
        if "text" not in args:
            # 인자가 잘려 파싱에 실패하면 args 가 {} 로 온다. 그때 덮어쓰면 기억이
            # 통째로 지워진다 — 실측에서 실제로 일어났다 ("saved": 0).
            return {"ok": False, "error": "memory_write needs text"}, None
        agent.ap -= cfg.ap.memory_write
        agent.memory = str(args.get("text", ""))
        return {"ok": True, "saved": len(agent.memory), "ap_left": round(agent.ap, 2)}, None

    if name == "invest":
        target = args.get("target")
        try:
            amount = float(args.get("amount", 0))
        except (TypeError, ValueError):
            return {"ok": False, "error": "amount must be a number"}, None
        if target not in ("wellness", "national", "facility"):
            return {"ok": False, "error": f"unknown invest target: {target}"}, None
        if amount <= 0:
            return {"ok": False, "error": "amount must be positive"}, None
        # facility 대상 국가는 예산 차감 전에 검증한다 (LLM 이 국가 대신 에이전트 id 를 줄 수 있음)
        to = None
        if target == "facility":
            to = args.get("to") or agent.country
            if to not in world.countries:
                return {"ok": False,
                        "error": f"unknown nation: {to} — facility invest takes a nation id (e.g. Ranoa)"}, None
        if agent.budget < amount:
            return {"ok": False, "error": f"not enough budget; need {amount:.0f}, have {agent.budget:.0f}"}, None
        agent.budget -= amount            # invest 는 AP 0
        if target == "facility":
            sink.facility.append((to, amount, agent.id))
            # 접수와 과금만 답한다. **그 나라가 시설을 정했는지는 알려주지 않는다** —
            # 알려주면 10원짜리 조회로 타국 국토를 읽을 수 있고(국제 메시지가 24~48원인데
            # 그보다 싸다), "타국 사정은 소통해야만 안다" 는 전제가 통째로 무너진다.
            # 정해지지 않았으면 돈은 나가고 아무 일도 일어나지 않는다 — route=original 과
            # 같은 도박이다 (spec 4.1 은닉 목록: 타국의 진척·예산·국토·언어 능력).
            return {"ok": True, "accepted": f"{to} facility investment accepted", "charged": amount,
                    "budget_left": round(agent.budget, 1)}, None
        if target == "wellness":
            sink.wellness.append((agent.id, amount))
            return {"ok": True, "accepted": "wellness investment accepted", "charged": amount,
                    "budget_left": round(agent.budget, 1)}, None    # λ 변화 비공개
        sink.national.append((agent.country, amount, agent.id))
        return {"ok": True, "accepted": "national investment accepted", "charged": amount,
                "budget_left": round(agent.budget, 1)}, None

    if name == "learn":
        country_id = args.get("country")
        if country_id not in world.countries:
            return {"ok": False, "error": f"unknown nation: {country_id}"}, None
        if country_id == agent.country:
            return {"ok": False, "error": "you already know your own language"}, None
        c, reason = learn_cost(agent, country_id, world, cfg)
        if agent.ap < cfg.ap.learn:
            return {"ok": False, "error": f"not enough AP; learn needs {cfg.ap.learn}"}, None
        if agent.budget < c:
            return {"ok": False, "error": f"not enough budget; need {c:.0f}, have {agent.budget:.0f}"}, None
        agent.budget -= c
        agent.ap -= cfg.ap.learn
        # known_langs 는 다른 에이전트가 읽으므로(국내 구사자 판정) 즉시 바꾸지 않는다.
        # sink 에 넣어 정산 때(정렬 순) 반영한다 — 병렬 레이스·재현성 방지.
        domestic, parent = learn_discounts(agent, country_id, world)
        sink.learns.append({
            "agent": agent.id, "country": agent.country,
            "target": country_id, "lang": world.countries[country_id].lang,
            # ★ x̂ 의 전제. 어느 눈금이었는지가 없으면 "학습이 일어났다" 만 남고
            #   x 를 구간으로 좁힐 수 없다 (spec 6.1 · 8.4).
            "charged": c, "rung": round(c / cfg.costs.learn_base, 4),
            "discount_domestic": domestic, "discount_parent": parent,
            # 나이는 x̂ 의 최대 노이즈원이다 — 늙으면 회수 기간이 없어 같은 눈금도
            #   사실상 더 비싸다. 층화하지 않으면 "안 배운 것" 이 x 가 작아서인지
            #   늙어서인지 구분되지 않는다 (spec 8.4).
            "age": agent.age, "budget_after": round(agent.budget, 2),
            "lam": round(agent.lam, 4),
        })
        return {"ok": True, "learned": country_id, "charged": c, "discount": reason,
                "effect": "you can read it from next turn",
                "budget_left": round(agent.budget, 1), "ap_left": round(agent.ap, 1)}, None

    if name == "speak":
        to = args.get("to")
        if to not in world.agents:
            return {"ok": False, "error": f"unknown recipient: {to}"}, None
        if to == agent.id:
            return {"ok": False, "error": "you cannot send a message to yourself"}, None
        recipient = world.agents[to]
        kind = messaging.classify(agent.country, recipient.country, args.get("route"))
        c = messaging.cost(kind, cfg, knob_ai)
        ap_cost = cfg.ap.speak
        if agent.ap < ap_cost:
            return {"ok": False, "error": f"not enough AP; speak needs {ap_cost}"}, None
        if agent.budget < c:
            return {"ok": False, "error": f"not enough budget; need {c:.0f}, have {agent.budget:.0f}"}, None
        agent.budget -= c
        agent.ap -= ap_cost
        ti = args.get("translate_instruction")
        sink.messages.append({
            "kind": "speak", "from": agent.id, "from_country": agent.country,
            "from_lang": agent.native_lang, "to": to, "to_country": recipient.country,
            "to_lang": recipient.native_lang, "route": args.get("route"),
            # LLM 이 문자열 아닌 값을 줄 수 있어 강제 문자열화 (truncate·translate 크래시 방지)
            "text": str(args.get("text", "")),
            "translate_instruction": None if ti is None else str(ti),
            "reply_to": args.get("reply_to"),
        })
        # 전달 성공/실패는 알리지 않는다 (original 은 도박). 접수·과금만.
        return {"ok": True, "queued": f"will arrive at {to} next turn", "charged": c,
                "budget_left": round(agent.budget, 1), "ap_left": round(agent.ap, 1)}, None

    if name == "propose_vote":
        target = args.get("target")
        if target not in ("bunker", "interceptor"):
            return {"ok": False, "error": "target must be bunker or interceptor"}, None
        c = world.countries[agent.country]
        if c.proposal is not None:
            return {"ok": False, "error":
                    f"your nation already has an open proposal ({c.proposal['target']}); "
                    f"the ballot is on turn {c.proposal['vote_turn']}"}, None
        if c.land == target:
            return {"ok": False, "error": f"your nation is already building {target}"}, None
        if agent.ap < cfg.ap.propose_vote:
            return {"ok": False, "error": f"not enough AP; propose_vote needs {cfg.ap.propose_vote}"}, None
        if agent.budget < cfg.costs.propose_vote:
            return {"ok": False, "error": f"not enough budget; need {cfg.costs.propose_vote}"}, None
        agent.budget -= cfg.costs.propose_vote
        agent.ap -= cfg.ap.propose_vote
        sink.votes.append((agent.id, agent.country, target))
        return {"ok": True, "proposed": target, "charged": cfg.costs.propose_vote,
                "effect": "nothing changes yet; the ballot is held after three turns",
                "ap_left": round(agent.ap, 1)}, None

    if name == "vote":
        c = world.countries[agent.country]
        if c.proposal is None:
            return {"ok": False, "error": "your nation has no open proposal"}, None
        if world.turn != c.proposal["vote_turn"]:
            return {"ok": False, "error":
                    f"the ballot is on turn {c.proposal['vote_turn']}, not now"}, None
        if "approve" not in args:
            return {"ok": False, "error": "vote needs approve (true or false)"}, None
        if agent.ap < cfg.ap.propose_vote:
            return {"ok": False, "error": f"not enough AP; vote needs {cfg.ap.propose_vote}"}, None
        agent.ap -= cfg.ap.propose_vote          # 표는 무료다. 돈을 물리면 참여가 재산이 된다
        sink.ballots.append((agent.id, agent.country, bool(args["approve"])))
        return {"ok": True, "voted": bool(args["approve"]),
                "on": c.proposal["target"], "ap_left": round(agent.ap, 1)}, None

    if name == "procreate":
        if agent.ap < cfg.ap.procreate:
            return {"ok": False, "error": f"not enough AP; procreate needs {cfg.ap.procreate}"}, None
        agent.ap -= cfg.ap.procreate
        sink.procreations.append((agent.id, args.get("testament", "")))
        return {"ok": True, "done": "you leave a child and die"}, "end"

    return {"ok": False, "error": f"unknown tool: {name}"}, None



# ── 개체 기억 (spec 4.5) ──────────────────────────────────────────────────────

# 폭주 보험. 설계 파라미터가 아니다 — 정상 턴은 도구 5~15회라 여기 닿지 않는다.
# 옛 MAX_STEPS=8 이 정상 행동까지 자르던 것과 다르다.
RUNAWAY_CAP = 64

# CJK 기준 대략치. 영어는 ~4자/토큰이지만 우리 프롬프트는 ja/zh/fr 이라 훨씬 조밀하다.
#
# 실측 150콜로 계수를 골랐다 (추정/실측 중앙값, 1.0 이 정확):
#     본문//3 스키마//3  →  1.010   ← 채택
#     본문//3 스키마//4  →  0.893
#     본문//2 스키마//4  →  1.182
#
# 도구 스키마는 영어 JSON 이라 //3 이 53% 과대추정하지만(1389 vs 실측 909),
# 그것이 CJK 본문의 과소추정과 상쇄되어 전체가 가장 정확해진다. 계수를 나누면
# 오히려 나빠지므로 단일 계수로 둔다.
_CHARS_PER_TOKEN = 3


def estimate_tokens(messages: list[dict], tool_tokens: int = 0) -> int:
    """대략치. 압박 판정은 응답의 usage.prompt_tokens(실측)를 쓰고, 축출 회계와
    Stub 경로에만 이걸 쓴다.

    ⚠ 도구 스키마를 반드시 함께 센다. 매 호출 프롬프트에 통째로 실리는 909 토큰이라,
      빼면 실질 한계가 8192 가 아니라 9100 쯤으로 느슨해진다.
    """
    n = tool_tokens
    for m in messages:
        n += 4                                   # role·구분자 등 메시지당 고정 오버헤드
        n += len(str(m.get("content") or "")) // _CHARS_PER_TOKEN
        for tc in m.get("tool_calls") or []:
            n += len(json.dumps(tc, ensure_ascii=False)) // _CHARS_PER_TOKEN
    return n


def tool_schema_tokens(tools) -> int:
    """도구 스키마는 매 호출 프롬프트에 실린다 — 고정비로 센다."""
    return (len(json.dumps(tools, ensure_ascii=False)) // _CHARS_PER_TOKEN + 1) if tools else 0


_TOOL_TOKENS = tool_schema_tokens(TOOLS)


def _turn_blocks(convo: list[dict]) -> list[tuple[int, int]]:
    """대화를 턴 블록으로 나눈다. 블록 = user(관측) 하나 + 뒤따르는 assistant/tool."""
    idx = [i for i, m in enumerate(convo) if m.get("role") == "user"]
    return [(a, b) for a, b in zip(idx, idx[1:] + [len(convo)])]


def evict(convo: list[dict], limit_tokens: int, tool_tokens: int = 0) -> tuple[list[dict], int]:
    """한계를 넘으면 오래된 턴 블록부터 버린다. 최근 한 턴은 반드시 남긴다.

    system 은 convo 에 없다 (매 호출 앞에 붙인다). 반환 (남은 대화, 버린 블록 수).
    """
    dropped = 0
    blocks = _turn_blocks(convo)
    while len(blocks) > 1 and estimate_tokens(convo, tool_tokens) > limit_tokens:
        convo = convo[blocks[0][1]:]
        dropped += 1
        blocks = _turn_blocks(convo)
    return convo, dropped


def under_pressure(agent, cfg) -> bool:
    """직전 호출의 실측 토큰이 경고 임계를 넘었나."""
    return agent.last_prompt_tokens >= cfg.llm.context_limit * cfg.llm.warn_ratio


def can_act(agent, cfg, knob_ai: float) -> bool:
    """남은 예산·AP 로 실행 가능한 도구가 하나라도 있나 (종료 조건 ②, spec 4.5).

    end_turn 은 세지 않는다 — 그건 종료이지 행동이 아니다.
    """
    cheapest_budget = min(cfg.costs.comm_domestic, cfg.costs.propose_vote)
    if agent.budget >= cheapest_budget and agent.ap >= min(cfg.ap.speak, cfg.ap.propose_vote):
        return True
    if agent.budget > 0:                      # invest 는 AP 0
        return True
    return agent.ap >= cfg.ap.memory_write    # 기억은 예산을 안 쓴다


# ── 에이전트 한 턴 ────────────────────────────────────────────────────────────

def run_agent_turn(world, agent, cfg, client, sink: Sink, knob_ai: float,
                   system_prompt: str, user_prompt: str) -> dict:
    """한 에이전트의 한 턴. 대화는 태어나서 죽을 때까지 이어진다 (spec 4.5).

    한 에이전트의 chat 호출이 실패해도(400/네트워크 등) 그 에이전트만 이번 턴을 접고
    전체 시뮬레이션은 계속된다 — 단일 API 실패가 50턴 런을 죽이면 안 된다.

    종료 조건 (임의 상한을 두지 않는다):
      ① end_turn / procreate
      ② 남은 예산으로도 AP 로도 실행 가능한 도구가 없다
      ③ 동일한 (도구, 인자) 호출이 repeat_guard 회 반복
    """
    # 압박 경고는 관측 **앞**에 붙인다. 사실 통지이지 지시가 아니다 (spec 4.5).
    if under_pressure(agent, cfg):
        from domains.meteor.prompts import T          # 도메인 문구 (모국어)
        user_prompt = T[agent.native_lang]["warn"] + "\n\n" + user_prompt
        pressured = True
    else:
        pressured = False
    agent.convo.append({"role": "user", "content": user_prompt})
    messages = [{"role": "system", "content": system_prompt}, *agent.convo]
    actions: list[dict] = []
    reasonings: list[dict] = []   # spec 4.2 — 행동마다의 근거. 지표 4 를 여기서 역추적한다
    api_reasoning = ""      # API 의 message.reasoning — 추론 모델의 사고 과정. 다른 것이다
    error = None
    evicted = 0
    ended_by = "exhausted"  # ended | exhausted | error | repeat_guard | runaway
    seen: dict[str, int] = {}       # (도구,인자) 반복 카운터 — 실패는 자원을 안 쓴다
    steps = 0

    while True:
        if steps >= RUNAWAY_CAP:
            ended_by = "runaway"
            break
        if not can_act(agent, cfg, knob_ai):
            ended_by = "exhausted"
            break
        steps += 1
        # 한계를 넘으면 오래된 턴 블록부터 버린다. system 은 convo 밖이라 안전하다.
        agent.convo, dropped = evict(agent.convo, cfg.llm.context_limit, _TOOL_TOKENS)
        evicted += dropped
        messages = [{"role": "system", "content": system_prompt}, *agent.convo]
        try:
            resp = client.chat(messages, tools=TOOLS)
        except Exception as e:                          # 이 에이전트만 턴 종료
            error = f"{type(e).__name__}: {str(e)[:200]}"
            break
        # 압박 판정은 실측 토큰으로 한다. 없으면(Stub) 추정치.
        usage = resp.get("usage") or {}
        agent.last_prompt_tokens = int(usage.get("prompt_tokens")
                                       or estimate_tokens(messages, _TOOL_TOKENS))
        msg = resp["choices"][0]["message"]
        # ⚠ message.reasoning 은 추론 모델의 사고 과정이고 spec 의 reasoning 과 다르다.
        #    섞지 않는다 (spec 9장). 원본은 raw_calls.jsonl 에 그대로 남는다.
        if msg.get("reasoning"):
            api_reasoning = str(msg["reasoning"])
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break
        # tool_call 에 id 를 보장한다 (없으면 echo 한 assistant 와 tool 응답의 짝이 어긋나 400)
        # 그리고 arguments 를 **정규화**한다. 모델이 출력 상한에 걸려 잘린 JSON 을 주면
        # 그대로 되돌려줄 때 프로바이더가 400 을 낸다 (실측 218콜 중 8건).
        for i, tc in enumerate(tool_calls):
            if not tc.get("id"):
                tc["id"] = f"call_{i}"
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            if isinstance(raw_args, dict):
                fn["arguments"] = json.dumps(raw_args, ensure_ascii=False)
            elif isinstance(raw_args, str):
                try:
                    json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    fn["arguments"] = "{}"          # 잘린 것은 빈 인자로 되돌린다
            else:
                fn["arguments"] = "{}"
        # content 는 tool_calls 와 함께면 None 이어야 한다 (빈 문자열은 일부 프로바이더가 거부→400)
        agent.convo.append({"role": "assistant", "content": msg.get("content") or None,
                            "tool_calls": tool_calls})
        stop = False
        for tc in tool_calls:
            fn = tc.get("function") or {}           # 모델이 malformed 를 줄 수 있어 방어
            name = fn.get("name")
            raw = fn.get("arguments")
            if isinstance(raw, dict):
                args = raw                          # 일부 모델은 이미 파싱된 dict 를 준다
            else:
                try:
                    args = json.loads(raw or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
            if name not in TOOL_NAMES:
                result = {"ok": False, "error": f"unknown tool: {name}"}
                control = None
            else:
                result, control = execute_tool(name, args, world, agent, cfg, sink, knob_ai)
            why = str(args.get("reasoning", ""))
            reasonings.append({"tool": name, "ok": bool(result.get("ok")), "reasoning": why})
            if name != "end_turn" and result.get("ok"):
                actions.append({"type": name, **args})
            agent.convo.append({"role": "tool", "tool_call_id": tc["id"],
                                "content": json.dumps(result, ensure_ascii=False)})
            # ③ 실패한 호출만 센다. 성공은 자원을 쓰므로 ②가 이미 막는다 —
            #    성공까지 세면 정상 행동(같은 상대에게 3번 말하기)이 끊긴다.
            if not result.get("ok"):
                key = f"{name}|{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
                seen[key] = seen.get(key, 0) + 1
                if seen[key] >= cfg.llm.repeat_guard:
                    stop = True
                    ended_by = "repeat_guard"
                    break
            if control == "end":
                stop = True
                ended_by = "ended"
                break     # procreate 뒤쪽 tool_call 은 버린다
        if stop:
            break

    if error:
        ended_by = "error"
    return {"reasonings": reasonings, "api_reasoning": api_reasoning,
            "actions": actions, "error": error, "ended_by": ended_by,
            "reasoning_missing": not any(r["reasoning"] for r in reasonings),
            "steps": steps, "prompt_tokens": agent.last_prompt_tokens,
            "pressured": pressured, "evicted_blocks": evicted,
            "memory_len": len(agent.memory)}
