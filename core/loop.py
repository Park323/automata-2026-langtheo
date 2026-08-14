"""턴 루프. spec 3.1.

LLM 없이 더미 정책으로 total_turns 턴을 결정론적으로 돌린다. 과제 1 의 목표는
'세계가 정확히 도는 것'을 확인하는 것이지 창발을 관측하는 것이 아니다.
"""
from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field

from core import survival
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
        aid: [a.country, a.age, round(a.budget, 6), a.alive, a.born_turn, a.born_by]
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


# ─────────────────────────────────────────── 턴 ───────────────────────────────────

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
        for aid in snapshot_ids:
            if aid in procreated:
                continue                     # 이미 이번 턴 procreate 로 교체됨
            a = world.agents[aid]
            if not a.alive:
                continue
            if rng.random() < survival.hazard(a.age, a.lam, cfg.survival.k):
                result.deaths += 1
                result.death_ages.append(a.age)   # 마지막 생존 나이 = 죽는 턴의 age (spec 2.2)
                # 자연사는 계보와 무관한 '자연발생한 뒷세대'다 (spec 3.2). 개인에 속한 것은
                # 전부 소실(예산·언어·부모 할인 자격·쌓인 유언), 국가·세계에 속한 것은 유지
                # (국토·진척·national_capital 은 Country 객체에 그대로 남는다).
                child = _newborn(
                    aid, a.country, a.native_lang, cfg.income.initial_budget,
                    set(),                    # parent_langs: 자연사에는 부모가 없다 → 빈 집합
                    world.turn, "natural", cfg, counter,
                )
                world.agents[aid] = child
                world.testaments[aid] = []    # 쌓인 유언도 계보와 함께 소실
                result.births.append(
                    {"turn": world.turn, "id": aid, "uid": child.uid,
                     "born_by": "natural", "budget": child.budget}
                )
            else:
                a.age += 1

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
