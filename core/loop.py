"""턴 루프. spec 3.1.

LLM 없이 더미 정책으로 total_turns 턴을 결정론적으로 돌린다. 과제 1 의 목표는
'세계가 정확히 도는 것'을 확인하는 것이지 창발을 관측하는 것이 아니다.
"""
from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from core import messaging, survival
from core.agent_loop import Sink, run_agent_turn
from core.policy import PROCREATE_AGE, dummy_policy
from core.state import Agent, Country, World

# 더미 세계에서 시설 최초 확정 시의 기본 용도. 과제 2 에서 에이전트가 정한다.
DEFAULT_FACILITY_TYPE = "interceptor"


@dataclass
class RunResult:
    world: World
    deaths: int = 0                                   # 총 교체(=출생) 횟수
    death_ages: list[int] = field(default_factory=list)
    alive_counts: list[int] = field(default_factory=list)   # 턴별 생존 수
    births: list[dict] = field(default_factory=list)  # {turn, id, born_by, budget}
    acted: list[set] = field(default_factory=list)    # 턴별 이번 턴 행동한 id 집합
    state_lines: list[str] = field(default_factory=list)
    final: dict = field(default_factory=dict)         # 생존 판정 결과 (spec 2.5)
    messages_log: list = field(default_factory=list)  # 처리된 발신 메시지 (과제2)
    votes_log: list = field(default_factory=list)     # propose_vote 기록 (과제2)
    agent_logs: list = field(default_factory=list)    # 턴별 {aid: {reasoning,actions,received}}

    @property
    def state_log(self) -> str:
        return "\n".join(self.state_lines)

    @property
    def interceptor_best(self) -> float:
        """요격기는 부지별로 독립이다 — 두 곳에 반씩 지으면 둘 다 미완성.

        합산(sum)이 아니라 최댓값(max)으로 판정한다. 합산하면 3국이 각자 자기
        요격기를 따로 지어도 합계가 임계를 넘어 조율이 무의미해진다 (spec 4.4).
        """
        return max(
            (c.progress for c in self.world.countries.values() if c.land == "interceptor"),
            default=0.0,
        )


# ─────────────────────────────────────────── 초기화 ───────────────────────────────

def init_world(cfg, counter: "itertools.count") -> World:
    """spec 2.1. 국가 3 × 3명 = 9명, 전원 동일 초기값."""
    countries: dict[str, Country] = {}
    agents: dict[str, Agent] = {}
    testaments: dict[str, list[str]] = {}
    for cdef in cfg.world.countries:
        countries[cdef.id] = Country(id=cdef.id, lang=cdef.lang)
        for i in range(1, cfg.world.agents_per_country + 1):
            aid = f"{cdef.id}{i}"
            agents[aid] = _newborn(
                aid, cdef.id, cdef.lang, cfg.income.initial_budget, set(),
                turn=0, born_by="natural", cfg=cfg, counter=counter,
            )
            testaments[aid] = []
    return World(turn=0, countries=countries, agents=agents, testaments=testaments)


def _newborn(aid: str, country: str, lang: str, budget: float, parent_langs: set,
             turn: int, born_by: str, cfg, counter: "itertools.count") -> Agent:
    return Agent(
        id=aid,
        country=country,
        native_lang=lang,
        known_langs={lang},                 # 능력은 상속되지 않는다 (spec 3.3)
        parent_langs=set(parent_langs),     # 할인 자격만 넘어간다 (3.4)
        budget=budget,
        age=0,
        lam=cfg.survival.lambda_base,
        born_turn=turn,
        born_by=born_by,
        uid=next(counter),
    )


def _state_line(world: World) -> str:
    """재현성 검증용 정규 상태 한 줄. 키 정렬로 바이트 단위 비교가 가능하게 한다."""
    agents = {
        # lam·known_langs 도 포함해 학습·wellness 정산의 비결정성까지 재현성 검사가 잡게 한다
        aid: [a.country, a.age, round(a.budget, 6), a.alive, a.born_turn, a.born_by,
              round(a.lam, 6), sorted(a.known_langs)]
        for aid, a in sorted(world.agents.items())
    }
    countries = {
        cid: [c.land, round(c.progress, 6), round(c.national_capital, 6)]
        for cid, c in sorted(world.countries.items())
    }
    return json.dumps(
        {"turn": world.turn, "agents": agents, "countries": countries},
        sort_keys=True, ensure_ascii=False,
    )


# ─────────────────────────────────────────── 사망·출생·상속 ───────────────────────

def _death_birth(world: World, cfg, rng: random.Random, snapshot_ids, procreated,
                 counter: "itertools.count", result: RunResult) -> None:
    """spec 2.2 · 3.2. snapshot 에이전트에 hazard 를 굴려 자연사·출생을 처리한다."""
    for aid in snapshot_ids:
        if aid in procreated:
            continue                         # 이미 이번 턴 procreate 로 교체됨
        a = world.agents[aid]
        if not a.alive:
            continue
        if rng.random() < survival.hazard(a.age, a.lam, cfg.survival.k):
            result.deaths += 1
            result.death_ages.append(a.age)   # 마지막 생존 나이 = 죽는 턴의 age (spec 2.2)
            # 자연사는 계보와 무관한 '자연발생한 뒷세대' (spec 3.2). 개인에 속한 것은 전부
            # 소실(예산·언어·부모 할인 자격·쌓인 유언), 국가·세계는 유지(국토·진척·national_capital).
            child = _newborn(
                aid, a.country, a.native_lang, cfg.income.initial_budget,
                set(),                        # parent_langs: 자연사에는 부모가 없다 → 빈 집합
                world.turn, "natural", cfg, counter,
            )
            world.agents[aid] = child
            world.testaments[aid] = []        # 쌓인 유언도 계보와 함께 소실
            result.births.append(
                {"turn": world.turn, "id": aid, "uid": child.uid,
                 "born_by": "natural", "budget": child.budget}
            )
        else:
            a.age += 1


def _procreate_child(world: World, aid: str, testament: str, cfg,
                     counter: "itertools.count", result: RunResult) -> None:
    """spec 3.3. procreate 로 죽고, 예산·유언·부모 언어 할인 자격을 아이에게 넘긴다."""
    a = world.agents[aid]
    carry = ([testament] + world.testaments.get(aid, []))[: cfg.inheritance.testament_carry]
    world.testaments[aid] = carry
    child = _newborn(aid, a.country, a.native_lang, a.budget, a.known_langs,
                     world.turn, "procreate", cfg, counter)
    world.agents[aid] = child
    result.deaths += 1
    result.death_ages.append(a.age)
    result.births.append({"turn": world.turn, "id": aid, "uid": child.uid,
                          "born_by": "procreate", "budget": child.budget})


# ─────────────────────────────────────────── 턴 (더미) ─────────────────────────────

def run_turn(world: World, cfg, rng: random.Random, result: RunResult,
             counter: "itertools.count", is_last: bool = False,
             procreate_age: int | None = PROCREATE_AGE) -> None:
    """한 턴. 7단계 순서를 바꾸지 말 것. (spec 3.1)"""
    # 1. 소득 지급 + AP 리셋 (이월: 예산은 남고, AP 는 리셋)
    for a in world.agents.values():
        mult = world.countries[a.country].multiplier(cfg)
        a.budget += cfg.income.per_turn * mult
        a.ap = cfg.turn.action_points

    # 2. 관측 스냅샷 — 이번 턴 행동하는 인스턴스(uid)를 고정
    snapshot_ids = sorted(world.agents.keys())
    snapshot_uids = {world.agents[aid].uid for aid in snapshot_ids}

    # 3. 정책 호출 (지금은 더미)
    decisions = {
        aid: dummy_policy(world, world.agents[aid], cfg, procreate_age)
        for aid in snapshot_ids
    }

    # 4. 검증 — 배열 순서가 우선순위. 예산·AP 확인. procreate 즉시 사망 처리.
    facility_invest: dict[str, float] = {cid: 0.0 for cid in world.countries}
    procreated: set[str] = set()
    for aid in snapshot_ids:
        a = world.agents[aid]
        for act in decisions[aid]["actions"]:
            t = act["type"]
            if t == "invest":
                if act["target"] != "facility":
                    continue  # 더미는 facility 만. wellness/national 은 과제 2
                amount = act["amount"]
                # 남은 시설 투자 상한(국가 단위)
                room = cfg.facility.cap_per_turn - facility_invest[act["to"]]
                amount = max(0.0, min(amount, a.budget, room))
                if a.ap < cfg.ap.invest or amount <= 0:
                    continue
                a.budget -= amount
                facility_invest[act["to"]] += amount
                # 최초 시설 투자로 국토 용도 확정 (선착순)
                land = world.countries[act["to"]]
                if land.land is None and amount > 0:
                    land.land = DEFAULT_FACILITY_TYPE
            elif t == "procreate":
                if a.ap < cfg.ap.procreate:
                    continue
                a.ap -= cfg.ap.procreate
                # 유언 계승 (구전의 감쇠: 최근 testament_carry 개)
                carry = world.testaments.get(aid, [])
                carry = ([act.get("testament", "")] + carry)[: cfg.inheritance.testament_carry]
                world.testaments[aid] = carry
                child = _newborn(
                    aid, a.country, a.native_lang, a.budget, a.known_langs,
                    world.turn, "procreate", cfg, counter,
                )
                world.agents[aid] = child
                procreated.add(aid)
                result.deaths += 1
                result.death_ages.append(a.age)   # procreate: 현재 나이에 죽음
                result.births.append(
                    {"turn": world.turn, "id": aid, "uid": child.uid,
                     "born_by": "procreate", "budget": child.budget}
                )
                break  # procreate 뒤쪽 행동은 전부 버림 (이미 죽었다)

    # 5. 환경 갱신 — 투자 집계 → 확률 판정 → 진척, 국토 확정, national_capital
    #    증가분 = Binomial(n = 투자량 × facility_eff, p = success_prob)
    # TODO(과제2): cap_per_turn 은 국가 단위 상한인데 지금은 sorted(id) 순으로 소진해
    #   같은 국가의 A1 이 A2·A3 보다 유리하다 (spec 3.1 이 금지한 순서 편향).
    #   현재 config 에선 국가당 투자 ~150 < 상한 500 이라 상한에 안 닿아 무해하지만,
    #   과제 2 에서 LLM 이 크게 투자하면 실제 편향이 된다. 비례 배분 또는 라운드로빈으로 교체.
    for cid, invested in facility_invest.items():
        c = world.countries[cid]
        eff = cfg.facility.eff * c.multiplier(cfg)   # 국가 투자로 갱신 (더미는 mult=1)
        n = int(invested * eff)
        gained = sum(1 for _ in range(n) if rng.random() < cfg.world.success_prob)
        c.progress += gained
    # national_capital 은 더미가 투자하지 않으므로 변화 없음 (mult = 1 유지)

    # 6. 메시지 큐잉 — 과제 2. 지금은 비어 있음.

    # 7. 생사 판정 — 마지막 턴은 생략 (곧바로 생존 판정 2.5)
    if not is_last:
        _death_birth(world, cfg, rng, snapshot_ids, procreated, counter, result)

    # 기록 — acted 는 이번 턴에 실제로 행동한 인스턴스(uid) 집합
    result.acted.append(snapshot_uids)
    result.alive_counts.append(sum(1 for a in world.agents.values() if a.alive))
    result.state_lines.append(_state_line(world))


# ─────────────────────────────────────────── 실행 ─────────────────────────────────

def final_survival(world: World, cfg, rng: random.Random) -> dict:
    """생존 판정 (spec 2.5). 국가당 한 번의 주사위 — 개인별로 굴리지 않는다."""
    import math

    # 요격기는 부지별 독립. 최댓값 하나가 임계에 닿아야 성공 (합산 아님, spec 4.4)
    intc_best = max(
        (c.progress for c in world.countries.values() if c.land == "interceptor"),
        default=0.0,
    )
    if intc_best >= cfg.thresholds.interceptor:
        return {"outcome": "all_survive", "interceptor_best": intc_best,
                "survivors": list(world.countries)}
    survivors: list[str] = []
    for cid, c in world.countries.items():
        if c.land == "bunker":
            p = 1.0 - math.exp(-c.progress / cfg.thresholds.bunker_scale)
            if rng.random() < p:               # 국가당 한 번
                survivors.append(cid)
        # interceptor 유치국·미확정국은 확률조차 없이 전원 사망
    return {"outcome": "intercept_failed", "interceptor_best": intc_best,
            "survivors": survivors}


def run(cfg, rng: random.Random, procreate_age: int | None = PROCREATE_AGE) -> RunResult:
    """total_turns 만큼 돌리고 마지막에 생존 판정(spec 2.5).

    procreate_age=None 이면 procreate 를 끈다 (수명 모델만 격리 측정 — 캘리브레이션용).
    """
    counter = itertools.count(1)
    world = init_world(cfg, counter)
    result = RunResult(world=world)
    for t in range(1, cfg.world.total_turns + 1):
        world.turn = t
        run_turn(world, cfg, rng, result, counter,
                 is_last=(t == cfg.world.total_turns), procreate_age=procreate_age)
    result.final = final_survival(world, cfg, rng)
    return result


# ─────────────────────────────────────────── 턴 (에이전트 · 과제2) ─────────────────

def _dequeue_inbox(world: World, aid: str) -> list[dict]:
    """이 턴 도착 메시지를 큐에서 꺼내 msg_id 를 붙인다.
    수신 슬롯 점유자가 바뀌었으면(수신자 사망·교체) 폐기한다 (spec 4.2 경계)."""
    current = world.agents.get(aid)
    out = []
    for e in world.inbox_queue:
        if e["deliver_turn"] == world.turn and e["to"] == aid:
            if current and e.get("to_uid") is not None and e["to_uid"] != current.uid:
                continue                     # recipient_dead → 폐기
            out.append(e["msg"])
    return out          # msg_id 는 발신 시점에 전역으로 부여됨 (spec 6.1)


def _settle_agentic(world: World, cfg, rng: random.Random, sink: Sink, translator,
                    knob_ai: float, counter: "itertools.count", result: RunResult,
                    msg_ids: "itertools.count") -> set:
    """전원의 의도(sink)를 정산한다. 모든 반복은 agent_id 정렬 순 → 결정론(재현성 #1).
    반환: 이번 턴 procreate 로 죽은 id 집합."""
    # a. 학습 반영 (다음 턴 관측부터 유효). known_langs 변경은 여기서 처음 일어난다.
    for aid, lang in sink.learns:
        if aid in world.agents:
            world.agents[aid].known_langs.add(lang)

    # b. 시설 투자 — 국가별 집계, cap 초과분은 비례 환급(순서 무관, #12), 진척 판정
    by_country: dict[str, list] = defaultdict(list)
    for to_country, amount, agent_id in sink.facility:
        by_country[to_country].append((amount, agent_id))
    for cid in sorted(by_country):
        entries = by_country[cid]
        total = sum(a for a, _ in entries)
        cap = cfg.facility.cap_per_turn
        effective = min(total, cap)
        if total > cap:
            for amount, agent_id in entries:      # 초과분 비례 환급
                if agent_id in world.agents:
                    world.agents[agent_id].budget += (amount / total) * (total - cap)
        c = world.countries[cid]
        if c.land is None and effective > 0:
            c.land = DEFAULT_FACILITY_TYPE        # 용도 미지정 → 기본 요격기 (과제3에서 투표)
        eff = cfg.facility.eff * c.multiplier(cfg)
        n = int(effective * eff)
        c.progress += sum(1 for _ in range(n) if rng.random() < cfg.world.success_prob)

    # c. wellness (수명), d. national (자본)
    for aid, amount in sink.wellness:
        if aid in world.agents:
            world.agents[aid].lam += amount * cfg.wellness.gain
    for cid, amount, _ in sink.national:
        world.countries[cid].national_capital += amount

    # e. 메시지 → 번역 → 다음 턴 도착
    for sent in sink.messages:
        recipient = world.agents.get(sent["to"])
        reck = recipient.known_langs if recipient else set()
        to_uid = recipient.uid if recipient else None
        p = messaging.process_message(sent, reck, cfg, translator, knob_ai)
        world.inbox_queue.append({"deliver_turn": world.turn + 1, "to": sent["to"],
                                  "to_uid": to_uid, "msg": p["inbox"]})
        gid = next(msg_ids)
        p["inbox"]["msg_id"] = gid          # 전역 id — understood 의 조인 키 (spec 6.1)
        result.messages_log.append({"turn": world.turn, "msg_id": gid,
                                    "from": sent["from"], "to": sent["to"],
                                    "route": p["kind"], "delivered": p["delivered"],
                                    "understood": None,   # T+1 에 수신자가 채운다
                                    "meta": p["meta"]})
        if p["sender_notice"]:
            su = world.agents[sent["from"]].uid if sent["from"] in world.agents else None
            world.inbox_queue.append({"deliver_turn": world.turn + 1, "to": sent["from"],
                                      "to_uid": su, "msg": {"from": None, "text": None,
                                      "label": None, "original": None, "msg_id": next(msg_ids),
                                      "delivery_failed_to": sent["to"]}})

    # f. 투표 기록 (정식 집계는 이후 과제)
    for by, country, target in sink.votes:
        result.votes_log.append({"turn": world.turn, "by": by, "country": country, "target": target})

    # g. procreate (예산 환급까지 반영된 뒤라 자식 예산이 정확)
    procreated: set = set()
    for aid, testament in sink.procreations:
        _procreate_child(world, aid, testament, cfg, counter, result)
        procreated.add(aid)
    return procreated


def run_turn_agentic(world: World, cfg, rng: random.Random, result: RunResult,
                     counter: "itertools.count", client_for, translator, knob_ai: float,
                     render_obs, system_prompt, msg_ids, is_last: bool = False,
                     parallel: bool = True, on_turn_end=None) -> None:
    """한 턴 (에이전트). spec 3.1 순서를 지키되 3단계는 9명 병렬, 5단계는 정렬 정산."""
    # 1. 소득 + AP 리셋
    for a in world.agents.values():
        a.budget += cfg.income.per_turn * world.countries[a.country].multiplier(cfg)
        a.ap = cfg.turn.action_points

    # 2. 관측 스냅샷 (도착 메시지·프롬프트를 스레드 시작 전에 고정)
    snapshot_ids = sorted(world.agents.keys())
    snapshot_uids = {world.agents[aid].uid for aid in snapshot_ids}
    inboxes = {aid: _dequeue_inbox(world, aid) for aid in snapshot_ids}
    world.inbox_queue = [e for e in world.inbox_queue if e["deliver_turn"] > world.turn]
    user_prompts = {aid: render_obs(world, world.agents[aid], cfg, knob_ai, inboxes[aid])
                    for aid in snapshot_ids}

    # 3. 정책 호출 — 병렬. 각자 자기 Sink 에만 쓴다 (공유 상태 미변경 → 스레드 안전)
    sinks = {aid: Sink() for aid in snapshot_ids}

    def run_one(aid):
        agent = world.agents[aid]
        # system_prompt 는 문자열이거나 (agent)->str 콜러블. 후자는 모국어 프롬프트용
        sp = system_prompt(agent) if callable(system_prompt) else system_prompt
        return aid, run_agent_turn(world, agent, cfg, client_for(aid), sinks[aid],
                                   knob_ai, sp, user_prompts[aid])

    if parallel:
        with ThreadPoolExecutor(max_workers=len(snapshot_ids)) as ex:
            logs = dict(ex.map(run_one, snapshot_ids))
    else:
        logs = dict(run_one(aid) for aid in snapshot_ids)
    result.agent_logs.append({aid: logs[aid] for aid in snapshot_ids})

    # 4·5. sink 를 정렬 순으로 합치고 정산 (결정론)
    merged = Sink()
    for aid in snapshot_ids:
        s = sinks[aid]
        merged.facility += s.facility
        merged.wellness += s.wellness
        merged.national += s.national
        merged.messages += s.messages
        merged.votes += s.votes
        merged.learns += s.learns
        merged.procreations += s.procreations
    procreated = _settle_agentic(world, cfg, rng, merged, translator, knob_ai, counter,
                                 result, msg_ids)

    # 7. 생사 판정 (마지막 턴 생략)
    if not is_last:
        _death_birth(world, cfg, rng, snapshot_ids, procreated, counter, result)

    result.acted.append(snapshot_uids)
    result.alive_counts.append(sum(1 for a in world.agents.values() if a.alive))
    result.state_lines.append(_state_line(world))
    if on_turn_end is not None:
        on_turn_end(world.turn, result)


def run_agentic(cfg, rng: random.Random, client_for, translator, knob_ai: float,
                render_obs, system_prompt, parallel: bool = True,
                on_turn_end=None) -> RunResult:
    """LLM(또는 StubClient) 에이전트로 total_turns 턴을 돌린다.

    client_for(aid) : 에이전트별 클라이언트 (병렬이라 상태 있는 Stub 은 에이전트마다 별개여야).
                      실제 API 는 stateless OpenRouterClient 를 공유해도 안전.
    translator      : 번역 전용 클라이언트 (정산은 단일 스레드라 공유 가능).
    """
    counter = itertools.count(1)
    msg_ids = itertools.count(1)      # 전역 메시지 id — understood 의 조인 키 (spec 6.1)
    world = init_world(cfg, counter)
    result = RunResult(world=world)
    for t in range(1, cfg.world.total_turns + 1):
        world.turn = t
        run_turn_agentic(world, cfg, rng, result, counter, client_for, translator, knob_ai,
                         render_obs, system_prompt, msg_ids,
                         is_last=(t == cfg.world.total_turns),
                         parallel=parallel, on_turn_end=on_turn_end)
    result.final = final_survival(world, cfg, rng)
    return result
