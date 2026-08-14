"""한 에이전트의 한 턴. spec 4.2.

  messages = [system, user(관측)]
  반복(MAX_STEPS 이하):
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

MAX_STEPS = 8


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
    votes: list = field(default_factory=list)         # (agent_id, country, target)
    learns: list = field(default_factory=list)        # (agent_id, lang) — 다음 턴부터 유효
    procreations: list = field(default_factory=list)  # (agent_id, testament)


# ── 학습 비용 (spec 3.4) ──────────────────────────────────────────────────────

def learn_cost(agent, country_id: str, world, cfg) -> tuple[float, str]:
    """(비용, 할인사유). L × 국내구사자(×0.5) × 부모(×0.5), 중복 시 ×0.25."""
    target_lang = world.countries[country_id].lang
    base = cfg.costs.learn_base
    domestic = any(
        o.id != agent.id and o.country == agent.country and target_lang in o.known_langs
        for o in world.agents.values()
    )
    parent = target_lang in agent.parent_langs
    mult = (0.5 if domestic else 1.0) * (0.5 if parent else 1.0)
    reasons = []
    if domestic:
        reasons.append("국내에 구사자가 있습니다 (×0.5)")
    if parent:
        reasons.append("부모가 구사했습니다 (×0.5)")
    return base * mult, " · ".join(reasons) if reasons else "할인 없음"


# ── 도구 실행 ────────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict, world, agent, cfg, sink: Sink,
                 knob_ai: float) -> tuple[dict, str | None]:
    """(tool_result, control). control="end" 면 턴 종료 (procreate/end_turn)."""

    if name == "end_turn":
        return {"ok": True, "ended": True}, "end"

    if name == "invest":
        target = args.get("target")
        try:
            amount = float(args.get("amount", 0))
        except (TypeError, ValueError):
            return {"ok": False, "error": "amount 는 숫자여야 합니다"}, None
        if target not in ("wellness", "national", "facility"):
            return {"ok": False, "error": f"알 수 없는 투자 대상: {target}"}, None
        if amount <= 0:
            return {"ok": False, "error": "투자액은 양수여야 합니다"}, None
        # facility 대상 국가는 예산 차감 전에 검증한다 (LLM 이 국가 대신 에이전트 id 를 줄 수 있음)
        to = None
        if target == "facility":
            to = args.get("to") or agent.country
            if to not in world.countries:
                return {"ok": False,
                        "error": f"알 수 없는 국가: {to} — facility 투자는 국가 id 를 준다 (예: B)"}, None
        if agent.budget < amount:
            return {"ok": False, "error": f"예산이 부족합니다. 필요 {amount:.0f}, 보유 {agent.budget:.0f}"}, None
        agent.budget -= amount            # invest 는 AP 0
        if target == "facility":
            sink.facility.append((to, amount, agent.id))
            # 진척 증가분은 절대 여기서 답하지 않는다 (success_prob 역산 방지)
            return {"ok": True, "accepted": f"{to} 시설 투자 접수", "charged": amount,
                    "budget_left": round(agent.budget, 1)}, None
        if target == "wellness":
            sink.wellness.append((agent.id, amount))
            return {"ok": True, "accepted": "wellness 투자 접수", "charged": amount,
                    "budget_left": round(agent.budget, 1)}, None    # λ 변화 비공개
        sink.national.append((agent.country, amount, agent.id))
        return {"ok": True, "accepted": "national 투자 접수", "charged": amount,
                "budget_left": round(agent.budget, 1)}, None

    if name == "learn":
        country_id = args.get("country")
        if country_id not in world.countries:
            return {"ok": False, "error": f"알 수 없는 국가: {country_id}"}, None
        if country_id == agent.country:
            return {"ok": False, "error": "자국어는 이미 압니다"}, None
        c, reason = learn_cost(agent, country_id, world, cfg)
        if agent.ap < cfg.ap.learn:
            return {"ok": False, "error": f"AP 가 부족합니다. learn 은 {cfg.ap.learn} 이 필요합니다"}, None
        if agent.budget < c:
            return {"ok": False, "error": f"예산이 부족합니다. 필요 {c:.0f}, 보유 {agent.budget:.0f}"}, None
        agent.budget -= c
        agent.ap -= cfg.ap.learn
        # known_langs 는 다른 에이전트가 읽으므로(국내 구사자 판정) 즉시 바꾸지 않는다.
        # sink 에 넣어 정산 때(정렬 순) 반영한다 — 병렬 레이스·재현성 방지.
        sink.learns.append((agent.id, world.countries[country_id].lang))
        return {"ok": True, "learned": country_id, "charged": c, "discount": reason,
                "effect": "다음 턴부터 읽을 수 있습니다",
                "budget_left": round(agent.budget, 1), "ap_left": round(agent.ap, 1)}, None

    if name in ("speak", "ask"):
        to = args.get("to")
        if to not in world.agents:
            return {"ok": False, "error": f"알 수 없는 수신자: {to}"}, None
        recipient = world.agents[to]
        kind = messaging.classify(agent.country, recipient.country, args.get("route"))
        c = messaging.cost(kind, cfg, knob_ai)
        if name == "ask":
            if args.get("reply_to") is None:
                return {"ok": False, "error": "ask 는 reply_to(메시지 id)가 필요합니다"}, None
            c += cfg.costs.ask_clarification
        ap_cost = cfg.ap.ask if name == "ask" else cfg.ap.speak
        if agent.ap < ap_cost:
            return {"ok": False, "error": f"AP 가 부족합니다. {name} 은 {ap_cost} 이 필요합니다"}, None
        if agent.budget < c:
            return {"ok": False, "error": f"예산이 부족합니다. 필요 {c:.0f}, 보유 {agent.budget:.0f}"}, None
        agent.budget -= c
        agent.ap -= ap_cost
        ti = args.get("translate_instruction")
        sink.messages.append({
            "kind": name, "from": agent.id, "from_country": agent.country,
            "from_lang": agent.native_lang, "to": to, "to_country": recipient.country,
            "to_lang": recipient.native_lang, "route": args.get("route"),
            # LLM 이 문자열 아닌 값을 줄 수 있어 강제 문자열화 (truncate·translate 크래시 방지)
            "text": str(args.get("text", "")), "intent": str(args.get("intent", "")),
            "translate_instruction": None if ti is None else str(ti),
            "reply_to": args.get("reply_to"),
        })
        # 전달 성공/실패는 알리지 않는다 (original 은 도박). 접수·과금만.
        return {"ok": True, "queued": f"{to} 에게 다음 턴 도착합니다", "charged": c,
                "budget_left": round(agent.budget, 1), "ap_left": round(agent.ap, 1)}, None

    if name == "propose_vote":
        target = args.get("target")
        if target not in ("bunker", "interceptor"):
            return {"ok": False, "error": "target 은 bunker 또는 interceptor"}, None
        if agent.ap < cfg.ap.propose_vote:
            return {"ok": False, "error": f"AP 가 부족합니다. propose_vote 는 {cfg.ap.propose_vote}"}, None
        if agent.budget < cfg.costs.propose_vote:
            return {"ok": False, "error": f"예산이 부족합니다. 필요 {cfg.costs.propose_vote}"}, None
        agent.budget -= cfg.costs.propose_vote
        agent.ap -= cfg.ap.propose_vote
        sink.votes.append((agent.id, agent.country, target))
        return {"ok": True, "proposed": target, "charged": cfg.costs.propose_vote,
                "ap_left": round(agent.ap, 1)}, None

    if name == "procreate":
        if agent.ap < cfg.ap.procreate:
            return {"ok": False, "error": f"AP 가 부족합니다. procreate 는 {cfg.ap.procreate}"}, None
        agent.ap -= cfg.ap.procreate
        sink.procreations.append((agent.id, args.get("testament", "")))
        return {"ok": True, "done": "아이를 남기고 죽습니다"}, "end"

    return {"ok": False, "error": f"알 수 없는 도구: {name}"}, None


# ── 에이전트 한 턴 ────────────────────────────────────────────────────────────

def run_agent_turn(world, agent, cfg, client, sink: Sink, knob_ai: float,
                   system_prompt: str, user_prompt: str) -> dict:
    """반환 = 로그용 논리 형식 {"reasoning","actions","received"} (spec 4.2).

    한 에이전트의 chat 호출이 실패해도(400/네트워크 등) 그 에이전트만 이번 턴을 접고
    전체 시뮬레이션은 계속된다 — 단일 API 실패가 50턴 런을 죽이면 안 된다.
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    actions: list[dict] = []
    reasoning = ""
    error = None

    for _ in range(MAX_STEPS):
        try:
            resp = client.chat(messages, tools=TOOLS)
        except Exception as e:                          # 이 에이전트만 턴 종료
            error = f"{type(e).__name__}: {str(e)[:200]}"
            break
        msg = resp["choices"][0]["message"]
        if msg.get("content"):
            reasoning = msg["content"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break
        # tool_call 에 id 를 보장한다 (없으면 echo 한 assistant 와 tool 응답의 짝이 어긋나 400)
        for i, tc in enumerate(tool_calls):
            if not tc.get("id"):
                tc["id"] = f"call_{i}"
        # content 는 tool_calls 와 함께면 None 이어야 한다 (빈 문자열은 일부 프로바이더가 거부→400)
        messages.append({"role": "assistant", "content": msg.get("content") or None,
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
                result = {"ok": False, "error": f"알 수 없는 도구: {name}"}
                control = None
            else:
                result, control = execute_tool(name, args, world, agent, cfg, sink, knob_ai)
            if name != "end_turn" and result.get("ok"):
                actions.append({"type": name, **args})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result, ensure_ascii=False)})
            if control == "end":
                stop = True
                break     # procreate 뒤쪽 tool_call 은 버린다
        if stop:
            break

    return {"reasoning": reasoning, "actions": actions, "received": [], "error": error}
