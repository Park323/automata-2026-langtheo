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
import random
import re
import time
from dataclasses import dataclass, field

from core import messaging
from core.llm import LLMCallError
from core.tools import TOOLS, TOOL_NAMES, TOOLS_NO_REASONING, tools_for



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
    ballots: list = field(default_factory=list)       # 표 (agent_id, country, choice)
    learns: list = field(default_factory=list)        # (agent_id, lang) — 다음 턴부터 유효
    procreations: list = field(default_factory=list)  # (agent_id, testament)
    observations: list = field(default_factory=list)  # 위험 관측 (진실·관측치·오차)
    observations_by: dict = field(default_factory=dict)   # 이번 턴 개체별 관측 횟수


# ── 학습 비용 (spec 3.4) ──────────────────────────────────────────────────────

def invest_per_ap(agent, world, cfg) -> float:
    """**AP 1.0 어치의 투자량.** 국가 기술력이 넓힌다.

        AP 1.0 당 = invest_per_ap × 자국 생산배수
        소모 AP   = 금액 ÷ 그 값          (턴당 AP 가 1.0 이라 그것이 곧 천장)

    `invest` 는 AP 를 안 쓰고 있었다 — 돈만 들고 **아무것도 포기하지 않았다.**
    실측에서 invest 211건으로 speak 176건보다 많았다. AP 에 연동하면 상한이 저절로
    생기고 **말하기·배우기·관측과 경쟁하게 된다.**

    그리고 돈을 일로 바꾸는 **속도**가 정해지므로, 몰아붓기가 막히고(★A) 늙어서 다 못
    쓸 돈이 생겨 `procreate` 가 처음으로 이득이 된다. 그전에는 예산이 내 손에 있는 편이
    언제나 나아서 죽을 이유가 없었다 (실측: 21명 전원 자연사, procreate 0건).

    `wellness` 는 걸지 않는다 — 사적 재화이고, 막으면 수명이 예산에 안 반응한다.
    """
    return cfg.facility.invest_per_ap * world.countries[agent.country].multiplier(cfg)


def risk_sigma(country, cfg) -> float:
    """관측의 **상대 표준편차**. 국가 자본(기술력)이 좁힌다.

        σ비율 = sigma_ratio / (1 + √(national_capital / growth_scale))

    남은 턴과 임계 둘 다 이 하나에서 유도한다. 절대 턴 수로 두면 total_turns 를 줄인
    런에서 깨진다 — 20턴 런에서 ±50턴, 임계 ±250% 가 나왔다.

    **정규분포다.** 꼬리에서는 크게 빗나간다 — 의도된 것이다. 관측이 대체로 맞되
    가끔 크게 틀리는 쪽이, 늘 일정 폭 안에서 틀리는 것보다 실제 계측에 가깝다.

    `national` 투자에 두 번째 쓸모를 준다 — 그전에는 생산 배수뿐이었다.
    """
    import math
    return cfg.risk.sigma_ratio / (
        1.0 + math.sqrt(country.national_capital / cfg.growth.growth_scale))


def risk_error(country, cfg) -> float:
    """남은 턴의 σ (턴 단위)."""
    return risk_sigma(country, cfg) * cfg.world.total_turns


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


# ── 새어나온 도구 호출 회수 ──────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _json_objects(text: str):
    """중괄호 짝을 세어 최상위 JSON 객체들을 뽑는다. 문자열 안의 괄호는 건너뛴다."""
    depth = start = 0
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]
            elif depth < 0:
                depth = 0


def recover_tool_calls(content: str | None) -> list[dict]:
    """모델이 `tool_calls` 대신 `content` 에 넣은 도구 호출을 주워담는다.

    **전송 장애이지 세계의 사건이 아니다.** 8턴 실측에서 도구를 안 부른 응답 19건이
    **전부** content 안의 도구 호출이었고 (learn 6 · vote 6 · invest 3 · end_turn 2 …),
    그대로 버려지고 있었다. "학습 0건" 의 원인이 여기 있을 수 있다 — 모델은 배우려
    했는데 호출이 도구 채널로 안 나갔다.

    스펙이 "자주 틀리는 곳 8" 로 적어둔 그것이다. qwen-2.5-7b 는 tools 를 지원하는데도
    8% 가 샌다.

    ⚠ **조건 간에 균등하게 새지 않을 수 있다** — 노브가 비싸면 learn 을 더 자주
      시도할 텐데 그게 더 많이 새면 학습률이 조건 의존적으로 왜곡된다. 그래서 버리지
      않고 줍는다. 주운 건수는 `recovered` 로 따로 센다.
    """
    if not content:
        return []
    # 모델이 마지막을 `}` 대신 `)` 로 닫는 일이 있다 — 실측 2건, **둘 다 learn** 이었다.
    repaired = re.sub(r"\)\s*$", "}", content.strip())
    texts = [content, repaired, *(m.group(1) for m in _FENCE.finditer(content))]
    out: list[dict] = []
    for t in texts:
        for blob in _json_objects(t):
            try:
                o = json.loads(blob)
            except json.JSONDecodeError:
                continue
            calls = o.get("tool_calls") if isinstance(o, dict) else None
            for c in (calls if isinstance(calls, list) else [o]):
                if not isinstance(c, dict):
                    continue
                fn = c.get("function") if isinstance(c.get("function"), dict) else c
                name = fn.get("name")
                if name not in TOOL_NAMES:
                    continue
                args = fn.get("arguments", fn.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                out.append({"id": f"rec_{len(out)}", "type": "function",
                            "function": {"name": name,
                                         "arguments": json.dumps(args, ensure_ascii=False)}})
        if out:
            break                     # 원문에서 건졌으면 코드펜스는 같은 것의 중복이다
    return out


# ── 도구 실행 ────────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict, world, agent, cfg, sink: Sink,
                 knob_ai: float) -> tuple[dict, str | None]:
    """(tool_result, control). control="end" 면 턴 종료 (procreate/end_turn)."""

    if name == "end_turn":
        return {"ok": True}, "end"

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
        return {"ok": True}, None      # 돈도 AP 도 안 든다 — 돌려줄 것이 없다

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
        asked = amount        # 절삭 전 요청액. 다르면 그것만 돌려준다 (_clamped)
        # facility 대상 국가는 예산 차감 전에 검증한다 (LLM 이 국가 대신 에이전트 id 를 줄 수 있음)
        to = None
        if target == "facility":
            to = args.get("to") or agent.country
            if to not in world.countries:
                return {"ok": False,
                        "error": f"unknown nation: {to} — facility invest takes a nation id (e.g. Ranoa)"}, None
        # **남은 AP 로 자르고 나서 예산을 본다.** 순서를 바꾸면 AP 가 잘라줬을 금액을
        # 그대로 들고 "예산 부족" 으로 거절한다 — 9,999 를 내려다 300 만 냈어야 할 것이
        # 통째로 실패한다.
        per_ap = None
        if target == "wellness":
            # 사적 재화라 금액에 비례해 묶지 않는다 (묶으면 수명이 예산에 반응하지
            # 않게 되고, 지표 11 이 관측하려는 것이 바로 그 반응이다). 다만 공짜도 아니다.
            ap_used = cfg.ap.invest_wellness
            if agent.ap < ap_used:
                return {"ok": False,
                        "error": f"not enough AP; invest(wellness) needs {ap_used}"}, None
        else:
            per_ap = invest_per_ap(agent, world, cfg)
            affordable = agent.ap * per_ap
            if affordable <= 0:
                return {"ok": False, "error":
                        f"no action points left; {per_ap:.0f} of investment costs 1.0 AP "
                        f"and your nation's technical level sets that rate"}, None
            # round — 0.667 AP × 300 이 200.00000000000003 로 나와 그대로 청구된다.
            amount = round(min(amount, affordable), 6)   # 넘치게 내면 AP 가 닿는 데까지만
            ap_used = min(agent.ap, amount / per_ap)     # 부동소수로 AP 가 음수가 되지 않게
        if agent.budget < amount:
            return {"ok": False, "error": f"not enough budget; need {amount:.0f}, have {agent.budget:.0f}"}, None
        agent.budget -= amount
        agent.ap -= ap_used
        if target == "facility":
            sink.facility.append((to, amount, agent.id))
            # 접수와 과금만 답한다. **그 나라가 시설을 정했는지는 알려주지 않는다** —
            # 알려주면 10원짜리 조회로 타국 국토를 읽을 수 있고(국제 메시지가 24~48원인데
            # 그보다 싸다), "타국 사정은 소통해야만 안다" 는 전제가 통째로 무너진다.
            # 정해지지 않았으면 돈은 나가고 아무 일도 일어나지 않는다 — route=original 과
            # 같은 도박이다 (spec 4.1 은닉 목록: 타국의 진척·예산·국토·언어 능력).
            # **내가 그 나라에 낸 누적**을 함께 돌려준다. learn 이 그러는데 여기만
            # 안 그러고 있었다 (state.Agent.facility_invested).
            agent.facility_invested[to] = agent.facility_invested.get(to, 0.0) + amount
            return {"ok": True, **_clamped(asked, amount),
                    "your_total_into": round(agent.facility_invested[to], 1),
                    "budget_left": round(agent.budget, 1),
                    "ap_left": round(agent.ap, 3)}, None
        if target == "wellness":
            sink.wellness.append((agent.id, amount))
            return {"ok": True, **_clamped(asked, amount),        # λ 변화 비공개
                    "budget_left": round(agent.budget, 1),
                    "ap_left": round(agent.ap, 3)}, None
        sink.national.append((agent.country, amount, agent.id))
        return {"ok": True, **_clamped(asked, amount),
                "budget_left": round(agent.budget, 1),
                "ap_left": round(agent.ap, 3)}, None

    if name == "learn":
        country_id = args.get("country")
        if country_id not in world.countries:
            return {"ok": False, "error": f"unknown nation: {country_id}"}, None
        if country_id == agent.country:
            return {"ok": False, "error": "you already know your own language"}, None
        lang = world.countries[country_id].lang
        if lang in agent.known_langs:
            return {"ok": False, "error": f"you already read {country_id}'s language"}, None
        try:
            amount = float(args.get("amount", 0))
        except (TypeError, ValueError):
            return {"ok": False, "error": "amount must be a number"}, None
        if amount <= 0:
            return {"ok": False, "error": "amount must be positive"}, None
        asked = amount        # 절삭 전 요청액. 다르면 그것만 돌려준다 (_clamped)
        need, reason = learn_cost(agent, country_id, world, cfg)
        done_before = agent.lang_progress.get(lang, 0.0)
        # 넘치게 내면 필요한 만큼만 받는다 — 남는 돈이 조용히 사라지면 안 된다
        amount = min(amount, max(0.0, need - done_before))
        if amount <= 0:
            return {"ok": False, "error": f"{country_id}'s language is already paid for"}, None
        # **AP 도 금액에 비례한다.** 정액이면 분할이 손해다 — 600 을 여섯 번에 나눠 내면
        # AP 1.8, 한 번에 내면 0.3. 분할을 넣어놓고 분할에 벌을 주게 된다.
        # 비례로 두면 나눠 내든 몰아 내든 합계가 같고, 정가 전액이 딱 한 턴이 된다.
        per_ap = cfg.costs.learn_base / cfg.ap.learn_full
        affordable = agent.ap * per_ap
        if affordable <= 0:
            return {"ok": False, "error":
                    f"no action points left; {per_ap:.0f} of learning costs 1.0 AP"}, None
        amount = round(min(amount, affordable), 6)   # AP 가 닿는 데까지만 (invest 와 같다)
        if agent.budget < amount:
            return {"ok": False,
                    "error": f"not enough budget; need {amount:.0f}, have {agent.budget:.0f}"}, None
        ap_used = min(agent.ap, amount / per_ap)
        agent.budget -= amount
        agent.ap -= ap_used
        # known_langs 는 다른 에이전트가 읽으므로(국내 구사자 판정) 즉시 바꾸지 않는다.
        # sink 에 넣어 정산 때(정렬 순) 반영한다 — 병렬 레이스·재현성 방지.
        domestic, parent = learn_discounts(agent, country_id, world)
        sink.learns.append({
            "agent": agent.id, "country": agent.country,
            "target": country_id, "lang": lang,
            "charged": amount, "progress_before": round(done_before, 2),
            "required": need, "rung": round(need / cfg.costs.learn_base, 4),
            "discount_domestic": domestic, "discount_parent": parent,
            "age": agent.age, "budget_after": round(agent.budget, 2),
            "lam": round(agent.lam, 4),
        })
        done = done_before + amount
        # 남는 것은 **내가 몰랐던 것**뿐이다. 누적 진척과 그때그때의 필요액은 턴을
        # 넘나들며 바뀌고(국내 구사자가 생기면 절반이 된다), 계산으로 알 수 없다.
        return {"ok": True, **_clamped(asked, amount),
                "progress": round(done, 1), "required": need,
                "remaining": round(max(0.0, need - done), 1),
                # **일정을 말하지 않는다 — 다 냈는지만 적는다.**
                #
                # `can_read_next_turn` 이었는데, 순차 라운드로빈이 학습을 **차례마다**
                # 반영하게 되면서(_settle_step) 거짓이 됐다. 다 낸 순간부터 그 턴에 바로
                # 쓸 수 있는데 "다음 턴부터" 라고 말하면, **막 배운 말을 그 턴에 안 쓰게
                # 만든다** — 하필 학습이 살아나기를 바라는 지점이다.
                #
                # 병렬 경로는 여전히 턴 끝 정산이라 다음 턴부터다. execute_tool 은 어느
                # 루프인지 모르므로, 언제부터인지는 관측의 「읽을 수 있는 언어」 가 답한다.
                "complete": done >= need,
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
        # 받는 이·다음 턴 도착은 내가 방금 말한 것이고 규칙이다. 남은 자원만 돌려준다.
        return {"ok": True,
                "budget_left": round(agent.budget, 1), "ap_left": round(agent.ap, 1)}, None

    if name == "observe_risk":
        if agent.ap < cfg.ap.observe_risk:
            return {"ok": False, "error": f"not enough AP; observe_risk needs {cfg.ap.observe_risk}"}, None
        if agent.budget < cfg.costs.observe_risk:
            return {"ok": False,
                    "error": f"not enough budget; need {cfg.costs.observe_risk:.0f}, "
                             f"have {agent.budget:.0f}"}, None
        agent.budget -= cfg.costs.observe_risk
        agent.ap -= cfg.ap.observe_risk
        truth = cfg.world.total_turns - world.turn        # 남은 턴 (마지막 턴에 판정)
        err = risk_error(world.countries[agent.country], cfg)
        # 잡음은 **매 관측마다 새로** 뽑는다. 여러 번 재면 평균으로 좁혀지지만 관측이
        # 비싸서 공짜가 아니다 — 그 값이 곧 국가 자본(기술력)과 겨루는 가격이다.
        # 시드·턴·개체·회차로 결정론적으로 뽑는다 (병렬이라 전역 rng 는 재현을 깬다).
        n = sink.observations_by.get(agent.id, 0)
        sink.observations_by[agent.id] = n + 1
        rng = random.Random(f"{cfg.run.seed}|{world.turn}|{agent.uid}|{n}")
        seen = max(0, round(truth + rng.gauss(0, err)))
        # 임계도 같은 기술력으로 잰다. 턴 오차를 전체 기간으로 나눠 **비율 오차**로 옮긴다
        # (자본 0 이면 ±25%, 자본이 쌓이면 같은 비율로 좁아진다).
        rel = risk_sigma(world.countries[agent.country], cfg)
        thr_truth = cfg.thresholds.interceptor
        thr_seen = max(1, round(thr_truth * (1 + rng.gauss(0, rel))))
        sink.observations.append({
            "agent": agent.id, "country": agent.country, "nth": n,
            "truth": truth, "observed": seen, "error": round(err, 2),
            "threshold_truth": thr_truth, "threshold_observed": thr_seen,
            "threshold_sigma": round(rel, 4),
            "national_capital": round(world.countries[agent.country].national_capital, 1),
        })
        # 전부 내가 몰랐던 것이다. "당신만의 것" 은 도구 설명에 이미 있다.
        return {"ok": True,
                "turns_until_impact": seen, "typical_error": round(err, 1),
                "interceptor_needs": thr_seen,
                "interceptor_typical_error_pct": round(rel * 100, 1),
                "budget_left": round(agent.budget, 1), "ap_left": round(agent.ap, 1)}, None

    if name == "propose_vote":
        # **무엇을 지을지는 여기서 정하지 않는다 — 採決을 소집하기만 한다.**
        #
        # 전에는 `target` 을 들고 「이것으로 하자」 를 열었고, `vote` 는 찬/반이었다.
        # 그래서 같은 턴에 둘이 제안하면 둘 다 도구를 통과하는데 하나만 열렸고,
        # 밀린 쪽은 AP 0.6 을 내고 **아무 일도 안 일어난 것을 알 방법이 없었다.**
        #
        # 소집에 내용이 없으면 겹칠 것이 없다. 둘이 소집해도 같은 採決이다.
        c = world.countries[agent.country]
        if c.proposal is not None:
            return {"ok": False, "error":
                    f"a ballot is already called for turn {c.proposal['vote_turn']}"}, None
        if agent.ap < cfg.ap.propose_vote:
            return {"ok": False, "error": f"not enough AP; propose_vote needs {cfg.ap.propose_vote}"}, None
        # **돈은 안 받는다.** 가난이 제안을 막으면 국토가 돈으로 정해진다. 무게는 AP 로만
        # 준다 — 국가의 용도를 여는 행위라 한 턴의 절반이 넘는다.
        if cfg.costs.propose_vote and agent.budget < cfg.costs.propose_vote:
            return {"ok": False, "error": f"not enough budget; need {cfg.costs.propose_vote}"}, None
        agent.budget -= cfg.costs.propose_vote
        agent.ap -= cfg.ap.propose_vote
        sink.votes.append((agent.id, agent.country))
        # **이제 날짜를 돌려줄 수 있다.** 소집에 내용이 없으니 둘이 소집해도 같은
        # 採決이고, 밀려서 안 열리는 일이 없다.
        return {"ok": True, "ballot_turn": world.turn + loop_vote_delay(),
                "ap_left": round(agent.ap, 1)}, None

    if name == "vote":
        c = world.countries[agent.country]
        if c.proposal is None:
            return {"ok": False, "error": "your nation has no open proposal"}, None
        if world.turn != c.proposal["vote_turn"]:
            return {"ok": False, "error":
                    f"the ballot is on turn {c.proposal['vote_turn']}, not now"}, None
        choice = args.get("choice")
        if choice not in ("interceptor", "bunker", "abstain"):
            return {"ok": False, "error":
                    "choice must be interceptor, bunker or abstain"}, None
        # **표는 돈도 AP 도 거의 안 받는다.** 돈을 물리면 참여가 재산이 되고, AP 를 크게
        # 물리면 採決 당일 — 설득이 가장 필요한 날 — 말할 기회가 줄어든다.
        if agent.ap < cfg.ap.vote:
            return {"ok": False, "error": f"not enough AP; vote needs {cfg.ap.vote}"}, None
        agent.ap -= cfg.ap.vote
        sink.ballots.append((agent.id, agent.country, choice))
        return {"ok": True, "ap_left": round(agent.ap, 1)}, None

    if name == "procreate":
        if agent.ap < cfg.ap.procreate:
            return {"ok": False, "error": f"not enough AP; procreate needs {cfg.ap.procreate}"}, None
        agent.ap -= cfg.ap.procreate
        sink.procreations.append((agent.id, args.get("testament", "")))
        return {"ok": True}, "end"

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
_TOOL_TOKENS_NR = tool_schema_tokens(TOOLS_NO_REASONING)


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


def loop_vote_delay() -> int:
    """`core.loop.VOTE_DELAY`. 여기서 import 하면 순환이 되므로 호출 시점에 읽는다."""
    from core.loop import VOTE_DELAY
    return VOTE_DELAY


def _clamped(asked: float, charged: float) -> dict:
    """**청구액은 요청과 다를 때만 돌려준다.** 그 존재 자체가 「잘렸다」 는 신호다.

    같으면 이미 바로 위 `assistant` 메시지에 그 숫자가 있다 — 되돌려주면 군더더기이고,
    필드가 일곱 개쯤 되면 **정작 잘렸을 때 그것이 묻힌다.**
    """
    return {} if abs(charged - asked) < 1e-9 else {"charged": round(charged, 1)}


def _redact_args(name: str, args: dict) -> dict:
    """호출 인자에서 **다른 곳에 온전히 있는 것만** 뺀다.

    `reasoning` 은 같은 이벤트의 `reasonings` 에, `speak` 의 `text` 는 `messages.jsonl` 에
    원문·도착문이 함께 있다. 그 둘 말고는 아무것도 버리지 않는다 — `memory_write` 의
    본문과 `procreate` 의 유언이 바로 그렇게 사라지고 있었다.
    """
    drop = {"reasoning"} | ({"text"} if name == "speak" else set())
    return {k: v for k, v in args.items() if k not in drop}


def can_act(agent, cfg, knob_ai: float) -> bool:
    """남은 예산·AP 로 실행 가능한 도구가 하나라도 있나 (종료 조건 ②, spec 4.5).

    `end_turn` 은 세지 않는다 — 그건 종료이지 행동이 아니다.

    > **자유 행동이 생기면서 이 조건은 사실상 죽었습니다** (8/17). `memory_write` 와
    > `procreate` 가 돈도 AP 도 안 쓰므로 자원이 바닥나도 고를 것이 남습니다. 정상
    > 종료는 `end_turn` 이고, 폭주는 `RUNAWAY_CAP`(64) 이 막습니다. 그래도 **정직하게
    > 계산합니다** — 여기서 거짓으로 False 를 돌려주면 합법적인 행동을 잘라내게 됩니다.
    """
    free_ap = min(cfg.ap.memory_write, cfg.ap.procreate)
    if agent.ap >= free_ap and free_ap <= 0:          # 공짜 행동이 하나라도 있으면 참
        return True
    if agent.budget >= cfg.costs.comm_domestic and agent.ap >= cfg.ap.speak:
        return True
    if agent.ap >= cfg.ap.propose_vote and agent.budget >= cfg.costs.propose_vote:
        return True
    if agent.budget > 0 and agent.ap >= cfg.ap.invest_wellness:
        return True
    return agent.ap >= min(cfg.ap.memory_write, cfg.ap.vote)


# ── 에이전트 한 턴 ────────────────────────────────────────────────────────────

class _StepAcc:
    """한 턴(병렬 경로) 또는 한 차례(순차 라운드로빈)가 스텝마다 쌓는 로그.

    `run_agent_turn`(병렬·1회정산)과 순차 라운드로빈이 **같은 스텝 실행기**
    (`_agent_one_call`)를 공유하려고 누적 상태를 밖으로 뺐다.
    """
    __slots__ = ("actions", "reasonings", "calls", "seen", "api_reasoning", "steps",
                 "evicted", "error", "recovered", "no_tool_content", "llm_ms", "pressured")

    def __init__(self):
        self.actions = []
        self.reasonings = []
        self.calls = []
        self.seen = {}
        self.api_reasoning = ""
        self.steps = 0
        self.evicted = 0
        self.error = None
        self.recovered = 0
        self.no_tool_content = ""
        self.llm_ms = 0.0
        self.pressured = False


def _agent_one_call(world, agent, cfg, client, sink: "Sink", knob_ai: float,
                    system_prompt: str, tool_list, tool_tokens, st: "_StepAcc") -> str | None:
    """LLM 한 콜 + 그 응답의 도구들을 실행한다. 의도는 sink 에 적는다(정산은 밖).

    반환: 이 콜로 턴/차례가 끝나면 그 사유("ended"·"repeat_guard"·"no_tool_call"·
    "error"), 계속하면 None. steps·actions·reasonings·calls 는 st 에 누적된다.
    """
    st.steps += 1
    # 한계를 넘으면 오래된 턴 블록부터 버린다. system 은 convo 밖이라 안전하다.
    agent.convo, dropped = evict(agent.convo, cfg.llm.context_limit, tool_tokens)
    st.evicted += dropped
    messages = [{"role": "system", "content": system_prompt}, *agent.convo]
    t_call = time.time()
    try:
        # 모든 스텝에서 도구 호출을 강제한다 (end_turn 도 도구라 "할 게 없다"는 표현됨).
        resp = client.chat(messages, tools=tool_list, tool_choice="required",
                           log_tag={"turn": world.turn, "agent": agent.id,
                                    "step": st.steps + 1, "age": agent.age,
                                    "country": agent.country})
    except LLMCallError as e:                       # 이 에이전트만 턴/차례 종료
        st.llm_ms += (time.time() - t_call) * 1000
        st.error = f"{type(e).__name__}: {str(e)[:200]}"
        return "error"
    st.llm_ms += (time.time() - t_call) * 1000
    usage = resp.get("usage") or {}
    agent.last_prompt_tokens = int(usage.get("prompt_tokens")
                                   or estimate_tokens(messages, tool_tokens))
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        st.error = f"malformed response: {str(resp)[:150]}"
        return "error"
    # message.reasoning 은 추론 모델의 사고이고 spec 의 reasoning 과 다르다 (섞지 않는다).
    think = str(msg.get("reasoning") or "").strip()
    if think:
        st.api_reasoning = think
        if tool_list is not TOOLS:
            st.reasonings.append({"tool": None, "ok": True, "step": st.steps,
                                  "source": "thinking", "reasoning": think})
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        tool_calls = recover_tool_calls(msg.get("content"))   # content 로 샌 호출 회수
        if tool_calls:
            st.recovered += len(tool_calls)
            msg = {**msg, "content": None}
        else:
            st.no_tool_content = (msg.get("content") or "")[:400]
            return "no_tool_call"
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
                fn["arguments"] = "{}"
        else:
            fn["arguments"] = "{}"
    agent.convo.append({"role": "assistant", "content": msg.get("content") or None,
                        "tool_calls": tool_calls})
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name")
        raw = fn.get("arguments")
        if isinstance(raw, dict):
            args = raw
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
        st.calls.append({"step": st.steps, "tool": name, "args": _redact_args(name, args),
                         "ok": bool(result.get("ok")),
                         "error": result.get("error"),
                         "result": {k: v for k, v in result.items() if k != "error"}})
        why = str(args.get("reasoning", ""))
        if tool_list is TOOLS:
            st.reasonings.append({"tool": name, "ok": bool(result.get("ok")),
                                  "source": "tool", "reasoning": why})
        else:
            st.reasonings.append({"tool": name, "ok": bool(result.get("ok")),
                                  "source": "tool", "reasoning": ""})
        if name != "end_turn" and result.get("ok"):
            st.actions.append({"type": name, **args})
        agent.convo.append({"role": "tool", "tool_call_id": tc["id"],
                            "content": json.dumps(result, ensure_ascii=False)})
        # 실패한 호출만 센다 (성공은 자원을 쓰므로 can_act 가 이미 막는다).
        if not result.get("ok"):
            key = f"{name}|{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
            st.seen[key] = st.seen.get(key, 0) + 1
            if st.seen[key] >= cfg.llm.repeat_guard:
                return "repeat_guard"
        if control == "end":
            return "ended"     # procreate/end_turn 뒤쪽 tool_call 은 버린다
    return None


def _turn_log(agent, st: "_StepAcc", ended_by: str, t_turn: float) -> dict:
    """스텝 누적(st)을 이벤트 로그 dict 로. run_agent_turn 과 라운드로빈이 공유한다."""
    return {"reasonings": st.reasonings, "api_reasoning": st.api_reasoning,
            "calls": st.calls,
            "actions": st.actions, "error": st.error, "ended_by": ended_by,
            "reasoning_missing": not any(r["reasoning"] for r in st.reasonings),
            "steps": st.steps, "prompt_tokens": agent.last_prompt_tokens,
            "pressured": st.pressured, "evicted_blocks": st.evicted,
            "memory_len": len(agent.memory),
            "recovered_calls": st.recovered, "no_tool_content": st.no_tool_content,
            "elapsed_ms": round((time.time() - t_turn) * 1000),
            "llm_ms": round(st.llm_ms),
            "ms_per_step": round(st.llm_ms / st.steps) if st.steps else None}


def run_agent_step(world, agent, cfg, client, sink: Sink, knob_ai: float,
                   system_prompt: str, user_prompt: str | None, st: "_StepAcc") -> str | None:
    """순차 라운드로빈의 **한 차례**. `_agent_one_call` 1회.

    `st` 는 이 에이전트의 **이번 턴 누적**이라 차례를 거듭해도 유지된다(스텝·행동·근거를
    모아 턴당 로그 하나로). 반환은 종료 사유 또는 None(계속).

    `user_prompt` 가 None 이면 **아무것도 붙이지 않는다.** 지금 그러한 것은 system 이 매
    콜 새로 담으므로, 차례마다 user 를 쌓을 이유가 없다 — 붙일 것은 턴을 여는 한 마디와
    새로 도착한 메시지뿐이다.
    """
    if under_pressure(agent, cfg):
        from domains.meteor.prompts import T          # 도메인 문구 (모국어)
        warn = T[agent.native_lang]["warn"]
        # 압박 경고는 사실 통지다. 붙일 메시지가 없으면 경고만으로 한 줄을 만든다 —
        # 안 그러면 한계에 부딪히고 있다는 사실이 전달되지 않는다.
        user_prompt = warn if user_prompt is None else warn + "\n\n" + user_prompt
        st.pressured = True
    if user_prompt is not None:
        agent.convo.append({"role": "user", "content": user_prompt})
    tool_list = tools_for(cfg)
    tool_tokens = _TOOL_TOKENS if tool_list is TOOLS else _TOOL_TOKENS_NR
    return _agent_one_call(world, agent, cfg, client, sink, knob_ai,
                           system_prompt, tool_list, tool_tokens, st)


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
    # 사고형 모델이면 도구마다 reasoning 을 또 받지 않는다 (spec 12.1).
    # 그 대신 **모델 자신의 사고를 reasonings 스트림에 넣는다** — 안 그러면
    # 지표 4(2단계 판정)가 읽을 근거가 통째로 사라진다.
    tool_list = tools_for(cfg)
    tool_tokens = _TOOL_TOKENS if tool_list is TOOLS else _TOOL_TOKENS_NR
    error = None
    evicted = 0
    ended_by = "exhausted"  # ended | exhausted | error | repeat_guard | runaway
    calls: list[dict] = []  # 도구 호출 전문 (인자·결과·실패 사유)
    seen: dict[str, int] = {}       # (도구,인자) 반복 카운터 — 실패는 자원을 안 쓴다
    steps = 0
    # 한 사람이 한 턴을 사는 데 걸린 시간. llm_ms 를 따로 재는 이유 — 벽시계의 거의
    # 전부가 API 대기라서, 둘이 갈리면 우리 코드가 병목이라는 뜻이다.
    t_turn = time.time()
    llm_ms = 0.0
    recovered = 0                   # content 로 새어 회수한 도구 호출 수
    no_tool_content = ""            # 끝내 회수 못 한 응답 본문 (진단용)

    while True:
        if steps >= RUNAWAY_CAP:
            ended_by = "runaway"
            break
        if not can_act(agent, cfg, knob_ai):
            ended_by = "exhausted"
            break
        steps += 1
        # 한계를 넘으면 오래된 턴 블록부터 버린다. system 은 convo 밖이라 안전하다.
        agent.convo, dropped = evict(agent.convo, cfg.llm.context_limit, tool_tokens)
        evicted += dropped
        messages = [{"role": "system", "content": system_prompt}, *agent.convo]
        t_call = time.time()
        try:
            # **모든 스텝에서 도구 호출을 강제한다.** 사고를 끈 뒤로 모델이 content 에
            # 숙고를 쏟고 그대로 끝내는 일이 잦다 — 실측에서 턴의 2~7% 가 통째로
            # 날아갔고, JSON 이 아니라 회수기도 못 잡았다 (계획만 적거나, 메시지 본문을
            # 산문으로 씀). **`end_turn` 도 도구이므로 "할 게 없다" 는 여전히 표현된다** —
            # 강제해도 잃는 선택지가 없다.
            resp = client.chat(messages, tools=tool_list, tool_choice="required",
                               log_tag={"turn": world.turn, "agent": agent.id,
                                        "step": steps + 1, "age": agent.age,
                                        "country": agent.country})
        except LLMCallError as e:                       # 이 에이전트만 턴 종료
            # **`except Exception` 이었다.** 한 에이전트의 API 실패로 50턴 런이 죽으면
            # 안 된다는 것이 목적이었는데, 그 그물이 프롬프트 렌더링·도구 실행의 버그까지
            # 삼켰다 — 그러면 그 에이전트가 매 턴 조용히 아무것도 못 하고, 로그에는
            # `error` 한 줄만 남는다. 원인을 찾을 방법이 없다.
            #
            # 이제 경계가 선언한 실패만 잡는다. 나머지는 런을 죽이고 **그게 낫다.**
            llm_ms += (time.time() - t_call) * 1000
            error = f"{type(e).__name__}: {str(e)[:200]}"
            break
        llm_ms += (time.time() - t_call) * 1000
        # 압박 판정은 실측 토큰으로 한다. 없으면(Stub) 추정치.
        usage = resp.get("usage") or {}
        agent.last_prompt_tokens = int(usage.get("prompt_tokens")
                                       or estimate_tokens(messages, tool_tokens))
        # 응답 모양이 예상과 달라도 **이 에이전트만** 턴을 접는다. 인덱싱하다 터지면
        # 스레드 풀을 타고 올라가 런 전체가 죽는다 (실측에서 실제로 죽었다).
        try:
            msg = resp["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            error = f"malformed response: {type(e).__name__} {str(resp)[:150]}"
            break
        # ⚠ message.reasoning 은 추론 모델의 사고 과정이고 spec 의 reasoning 과 다르다.
        #    섞지 않는다 (spec 9장). 원본은 raw_calls.jsonl 에 그대로 남는다.
        think = str(msg.get("reasoning") or "").strip()
        if think:
            api_reasoning = think          # 마지막 스텝의 사고 (하위 호환)
            if tool_list is not TOOLS:
                # 도구 인자가 없으니 이것이 유일한 근거다. **스텝 단위**라 어느 근거가
                # 어느 행동인지는 확정되지 않는다 (spec 12.1 이 경고한 그 지점).
                reasonings.append({"tool": None, "ok": True, "step": steps,
                                   "source": "thinking", "reasoning": think})
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # 도구 채널이 아니라 content 로 샌 호출을 줍는다 (전송 장애)
            tool_calls = recover_tool_calls(msg.get("content"))
            if tool_calls:
                recovered += len(tool_calls)
                msg = {**msg, "content": None}     # 회수했으니 본문은 비운다
            else:
                ended_by = "no_tool_call"          # **"exhausted" 로 뭉뚱그리지 않는다**
                no_tool_content = (msg.get("content") or "")[:400]
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
            # **호출 하나를 통째로 남긴다** — 인자·결과·실패 사유까지.
            #
            # 그전에는 성공한 호출만 `actions` 에 인자와 함께 남고, 실패는 `reasonings` 에
            # 이름과 ok=False 로만 남았다. 그래서 **왜 실패했는지가 어디에도 없었다** —
            # AP 가 모자랐는지, 국가 이름을 틀렸는지, 이미 아는 언어였는지 구분이 안 됐다.
            # 성공한 호출도 `actions` 는 **요청한 값**이라 실제 과금·절삭이 안 남는다
            # (9,999 를 냈는데 AP 가 300 으로 잘라도 로그에는 9,999 로 남았다).
            calls.append({"step": steps, "tool": name, "args": _redact_args(name, args),
                          "ok": bool(result.get("ok")),
                          "error": result.get("error"),
                          "result": {k: v for k, v in result.items() if k != "error"}})
            why = str(args.get("reasoning", ""))
            if tool_list is TOOLS:
                reasonings.append({"tool": name, "ok": bool(result.get("ok")),
                                   "source": "tool", "reasoning": why})
            else:
                reasonings.append({"tool": name, "ok": bool(result.get("ok")),
                                   "source": "tool", "reasoning": ""})
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
            "calls": calls,
            "actions": actions, "error": error, "ended_by": ended_by,
            "reasoning_missing": not any(r["reasoning"] for r in reasonings),
            "steps": steps, "prompt_tokens": agent.last_prompt_tokens,
            "pressured": pressured, "evicted_blocks": evicted,
            "memory_len": len(agent.memory),
            "recovered_calls": recovered, "no_tool_content": no_tool_content,
            "elapsed_ms": round((time.time() - t_turn) * 1000),
            "llm_ms": round(llm_ms),
            "ms_per_step": round(llm_ms / steps) if steps else None}
