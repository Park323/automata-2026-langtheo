"""한 에이전트의 한 턴. spec 4.2 · 4.5.

  기억은 태어나서 죽을 때까지 이어진다 (spec 4.5). agent.messages 를 누적한다:
      첫 턴          messages = [system]
      매 턴 시작     (한계 초과면) 오래된 블록 축출 → user(관측) append
      반복           resp = client.chat(messages, tools=TOOLS)
                     tool_calls 없으면 종료, 각 tool_call 실행 → role="tool" append
      종료 조건      ① end_turn/procreate  ② 실행 가능한 도구 없음  ③ (도구,인자) 반복 실패

⚠ 도구는 세계를 즉시 바꾸지 않는다. 자기 budget/ap 만 즉시 차감하고 효과는 Sink 에
  넣는다. 국토 확정·진척 판정·cap 배분·번역은 전원의 루프가 끝난 뒤 loop.py 5단계에서.
⚠ 도구 결과로 감춰야 할 것을 흘리지 않는다 (진척 증가분·λ 변화·success_prob).
⚠ 발신자 컨텍스트 누수 금지 (spec 4.5): 자기 메시지의 번역 결과·절단 후 텍스트·수신자
  understood·전달 성공 여부는 도구 결과에 담기지 않는다. speak 는 접수·과금만 답한다.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from core import memory, messaging
from core.tools import TOOLS, TOOL_NAMES

# 도구 스키마는 매 호출 프롬프트에 통째로 실린다 — 축출 회계에 고정비로 함께 센다.
_TOOL_TOKENS = memory.tool_schema_tokens(TOOLS)

# 비정상 폭주에 대한 비용 backstop. spec 4.5 의 종료 3조건이 정상 종료를 지배하고, 이건
# 그 아래에서만 발동한다 — 정상 턴은 도구 5~15회라 여기 닿지 않는다. 닿으면 end_reason
# "step_cap" 으로 구분해 로그(예: 서로 다른 인자의 실패를 무한히 뱉는 약한 모델). MAX_STEPS=8
# 이 정상 행동까지 잘랐던 것과 달리 여긴 훨씬 높다.
_STEP_CAP = 64


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
    understandings: list = field(default_factory=list)  # (agent_id, msg_id, understood) — 6.1


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
            # 진척 증가분은 절대 여기서 답하지 않는다 (success_prob 역산 방지)
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
        sink.learns.append((agent.id, world.countries[country_id].lang))
        return {"ok": True, "learned": country_id, "charged": c, "discount": reason,
                "effect": "you can read it from next turn",
                "budget_left": round(agent.budget, 1), "ap_left": round(agent.ap, 1)}, None

    if name in ("speak", "ask"):
        to = args.get("to")
        if to not in world.agents:
            return {"ok": False, "error": f"unknown recipient: {to}"}, None
        if to == agent.id:
            return {"ok": False, "error": "you cannot send a message to yourself"}, None
        recipient = world.agents[to]
        kind = messaging.classify(agent.country, recipient.country, args.get("route"))
        c = messaging.cost(kind, cfg, knob_ai)
        if name == "ask":
            if args.get("reply_to") is None:
                return {"ok": False, "error": "ask needs reply_to (a message id)"}, None
            c += cfg.costs.ask_clarification
        ap_cost = cfg.ap.ask if name == "ask" else cfg.ap.speak
        if agent.ap < ap_cost:
            return {"ok": False, "error": f"not enough AP; {name} needs {ap_cost}"}, None
        if agent.budget < c:
            return {"ok": False, "error": f"not enough budget; need {c:.0f}, have {agent.budget:.0f}"}, None
        agent.budget -= c
        agent.ap -= ap_cost
        ti = args.get("translate_instruction")
        sink.messages.append({
            "kind": name, "from": agent.id, "from_country": agent.country,
            "from_lang": agent.native_lang, "to": to, "to_country": recipient.country,
            "to_lang": recipient.native_lang, "route": args.get("route"),
            # LLM 이 문자열 아닌 값을 줄 수 있어 강제 문자열화 (truncate·translate 크래시 방지)
            "text": str(args.get("text", "")),
            "translate_instruction": None if ti is None else str(ti),
            "reply_to": args.get("reply_to"),
        })
        # 전달 성공/실패는 알리지 않는다 (original 은 도박). 접수·과금만. (누수 금지, spec 4.5)
        return {"ok": True, "queued": f"will arrive at {to} next turn", "charged": c,
                "budget_left": round(agent.budget, 1), "ap_left": round(agent.ap, 1)}, None

    if name == "memory_write":
        if agent.ap < cfg.ap.memory_write:
            return {"ok": False, "error": f"not enough AP; memory_write needs {cfg.ap.memory_write}"}, None
        agent.ap -= cfg.ap.memory_write
        agent.memory = str(args.get("text", ""))       # 통째로 덮어쓰기 (spec 4.5)
        return {"ok": True, "saved": "memory updated", "ap_left": round(agent.ap, 1)}, None

    if name == "report_understanding":
        try:
            mid = int(args.get("msg_id"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "msg_id must be an integer"}, None
        # 관측 장치이지 행동이 아니다 — AP·예산 0 (spec 4.5). 세계 효과는 아무것도 답하지 않는다.
        sink.understandings.append((agent.id, mid, str(args.get("understood", ""))))
        return {"ok": True, "noted": True}, None

    if name == "propose_vote":
        target = args.get("target")
        if target not in ("bunker", "interceptor"):
            return {"ok": False, "error": "target must be bunker or interceptor"}, None
        if agent.ap < cfg.ap.propose_vote:
            return {"ok": False, "error": f"not enough AP; propose_vote needs {cfg.ap.propose_vote}"}, None
        if agent.budget < cfg.costs.propose_vote:
            return {"ok": False, "error": f"not enough budget; need {cfg.costs.propose_vote}"}, None
        agent.budget -= cfg.costs.propose_vote
        agent.ap -= cfg.ap.propose_vote
        sink.votes.append((agent.id, agent.country, target))
        return {"ok": True, "proposed": target, "charged": cfg.costs.propose_vote,
                "ap_left": round(agent.ap, 1)}, None

    if name == "procreate":
        if agent.ap < cfg.ap.procreate:
            return {"ok": False, "error": f"not enough AP; procreate needs {cfg.ap.procreate}"}, None
        agent.ap -= cfg.ap.procreate
        sink.procreations.append((agent.id, args.get("testament", "")))
        return {"ok": True, "done": "you leave a child and die"}, "end"

    return {"ok": False, "error": f"unknown tool: {name}"}, None


# ── 에이전트 한 턴 ────────────────────────────────────────────────────────────

def _any_affordable(agent, cfg, knob_ai: float) -> bool:
    """남은 예산·AP 로 실행 가능한 행동 도구가 하나라도 있는가 (종료 조건 ②, spec 4.5).

    end_turn 은 세지 않는다 — 그건 언제나 가능하지만 '할 일'이 아니다. 하나라도 있으면
    True. 전부 불가면(대개 budget<=0 이고 ap<memory_write) 자연 종료(exhausted)한다.
    """
    b, ap = agent.budget, agent.ap
    if b > 0 and ap >= cfg.ap.invest:                       # invest: AP 0, 양의 예산만
        return True
    if ap >= cfg.ap.memory_write:                           # memory_write: 예산 0
        return True
    if b >= cfg.costs.comm_domestic and ap >= cfg.ap.speak:  # 가장 싼 경로의 speak
        return True
    if b >= cfg.costs.learn_base * 0.25 and ap >= cfg.ap.learn:  # 최대 할인 시 learn
        return True
    if b >= cfg.costs.propose_vote and ap >= cfg.ap.propose_vote:
        return True
    if ap >= cfg.ap.procreate:
        return True
    return False


def run_agent_turn(world, agent, cfg, client, sink: Sink, knob_ai: float,
                   system_prompt: str, user_prompt: str, turn: int = 0) -> dict:
    """반환 = 로그용 논리 형식 (spec 4.2):
        {"reasoning", "reasoning_missing", "actions", "received", "error", "end_reason"}

    기억이 누적된다 (spec 4.5): agent.messages 를 이어 쓰고 매 턴 관측을 뒤에 붙인다.
    한 에이전트의 chat 호출이 실패해도(400/네트워크 등) 그 에이전트만 이번 턴을 접고
    전체 시뮬레이션은 계속된다 — 단일 API 실패가 50턴 런을 죽이면 안 된다.
    """
    # 첫 턴이면 system 을 깐다. 이후 턴은 system(0번) 위에 관측이 누적돼 있다.
    if not agent.messages:
        agent.messages.append({"role": "system", "content": system_prompt})
    # 관측을 붙이기 전에 한계 초과분을 축출한다 (오래된 블록부터, system·최근 블록 보존).
    memory.evict(agent.messages, cfg.llm.context_limit, _TOOL_TOKENS)
    agent.messages.append({"role": "user", "content": user_prompt})
    messages = agent.messages

    actions: list[dict] = []
    reasoning = ""
    reasoning_missing = True          # end_turn/procreate 가 비-빈 reasoning 을 줄 때만 False
    error = None
    end_reason = None
    fail_counts: Counter = Counter()  # (도구, 인자) → ok:False 반복 횟수 (종료 조건 ③)
    guard = cfg.turn.repeat_guard
    step = 0

    while True:
        step += 1
        if step > _STEP_CAP:                            # 비용 backstop (정상 턴은 안 닿음)
            end_reason = "step_cap"
            break
        try:
            resp = client.chat(messages, tools=TOOLS,
                               meta={"kind": "agent", "agent": agent.id, "turn": turn, "step": step})
        except Exception as e:                          # 이 에이전트만 턴 종료
            error = f"{type(e).__name__}: {str(e)[:200]}"
            end_reason = "error"
            break
        usage = resp.get("usage") or {}
        pt = usage.get("prompt_tokens")
        if pt:                                          # API 가 주면 압박 판정에 실측을 쓴다
            agent.last_prompt_tokens = pt
        msg = resp["choices"][0]["message"]
        # ⚠ msg.get("content") 를 reasoning 으로 쓰지 않는다 — 그건 API 의 message.reasoning
        #   자리이고 spec 의 reasoning 과 다르다 (spec 9). reasoning 은 end_turn/procreate 인자에서만.
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:                              # 모델이 도구 없이 멈춤 → 종료로 본다
            end_reason = "ended"
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
                result = {"ok": False, "error": f"unknown tool: {name}"}
                control = None
            else:
                result, control = execute_tool(name, args, world, agent, cfg, sink, knob_ai)
            # reasoning 은 턴을 끝내는 두 도구의 인자에서만 취한다 (턴당 정확히 한 번, spec 9)
            if name in ("end_turn", "procreate"):
                r = str(args.get("reasoning", "")).strip()
                if r:
                    reasoning, reasoning_missing = r, False
            if name != "end_turn" and result.get("ok"):
                actions.append({"type": name, **args})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result, ensure_ascii=False)})
            if control == "end":
                end_reason = "ended"
                stop = True
                break     # procreate 뒤쪽 tool_call 은 버린다
            # ③ 실패한 호출은 자원을 안 쓰므로 ②로 못 막는다. 동일 실패가 guard 회면 종료.
            if not result.get("ok"):
                key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str))
                fail_counts[key] += 1
                if fail_counts[key] >= guard:
                    end_reason = "repeat_guard"
                    stop = True
                    break
        if stop:
            break
        # ② 남은 자원으로 할 수 있는 행동이 없으면 자연 종료 (예산 소진 등)
        if not _any_affordable(agent, cfg, knob_ai):
            end_reason = "exhausted"
            break

    # 다음 턴 압박 통지: 직전 프롬프트가 warn 임계를 넘겼는가 (실측 없으면 추정으로 대체)
    tokens = agent.last_prompt_tokens or memory.approx_tokens(messages, _TOOL_TOKENS)
    agent.mem_pressure = tokens >= cfg.llm.context_limit * cfg.llm.warn_ratio

    return {"reasoning": reasoning, "reasoning_missing": reasoning_missing,
            "actions": actions, "received": [], "error": error, "end_reason": end_reason}
