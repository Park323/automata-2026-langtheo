"""턴 루프. spec 3.1.

LLM 없이 더미 정책으로 total_turns 턴을 결정론적으로 돌린다. 과제 1 의 목표는
'세계가 정확히 도는 것'을 확인하는 것이지 창발을 관측하는 것이 아니다.
"""
from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from core import messaging, survival, visibility
from core import agent_loop
from core.agent_loop import Sink, run_agent_turn
from core.policy import PROCREATE_AGE, dummy_policy
from core.state import Agent, Country, World

# 더미 세계(과제 1)에서만 쓰는 기본 용도. 에이전트 세계에서는 **투표로만** 정해진다.
DEFAULT_FACILITY_TYPE = "interceptor"

# 소집 → 유예 → 採決. 소집한 해 t 의 t+1 이 상의 기간이고 t+VOTE_DELAY 가 採決일.
#
# **4 에서 2 로 줄였다** (8/20). 4 였을 때는 t+1·t+2·t+3 이 유예라 소집부터 개표까지 다섯
# 해가 걸렸다. 그 길이는 **메시지 왕복에 두 해가 들던 때** 정해진 것이다 — 보내면 다음
# 해에 닿고 답은 그 다음 해였으니, 한 번 주고받는 데 두 해가 필요했고 세 해는 겨우
# 한 왕복 반이었다.
#
# 순차 라운드로빈은 **같은 해에 왕복이 된다** (#20). 소집한 해에 이미 이야기가 오가고,
# t+1 한 해가 더 있으면 충분하다. 기대수명이 16해인데 다섯 해를 절차에 쓰면 한 사람이
# 겪는 採決이 세 번뿐이다.
VOTE_DELAY = 2


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
    learns_log: list = field(default_factory=list)    # 학습 1건 = x̂ 관측 1건 (spec 6.1)
    risk_log: list = field(default_factory=list)      # 위험 관측 (진실·관측치·오차)
    land_changes: list = field(default_factory=list)  # 국토 전환 = 진척 파괴 (SYSTEM 규칙 5)
    deaths_log: list = field(default_factory=list)    # 부고 — 같은 나라 사람에게 알린다
    facility_gains: list = field(default_factory=list)  # 출자 → 진척 기여 (행위 후 공개)
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

def init_world(cfg, counter: "itertools.count", rng: random.Random | None = None) -> World:
    """spec 2.1. 국가 3 × 3명 = 9명.

    **전원 동일 초기값이 아니다.** 둘을 흩어 놓는다.

    ① 나이 — `1 ~ init_age_max` 에서 뽑는다. 전원 0살로 시작하면 **한꺼번에 죽는다.**
       20턴 실측에서 t14~19 에 9명이 전부 교체됐고, 그 6턴 사이에 쌓아둔 기억·관계·예산이
       통째로 사라졌다. 세계가 주기적으로 백지가 되면 구전 감쇠를 관측할 수 없다.

    ② 언어 — 나라마다 **한 명**이 다른 나라 말을 이미 안다 (순환: Asla→Ranoa→Miris→Asla).
       그래야 국내 구사자 할인(×0.5)이 처음부터 살아 있다. 그전에는 국내에 구사자가
       아무도 없어 학습이 **늘 정가 600** 이었고, 20턴 동안 학습 시도가 **0건**이었다.
       `x̂` 는 L/2 눈금이 존재해야 구간으로 좁혀진다 (spec 7장).
    """
    rng = rng or random.Random(cfg.run.seed)
    countries: dict[str, Country] = {}
    agents: dict[str, Agent] = {}
    testaments: dict[str, list[str]] = {}
    defs = list(cfg.world.countries)
    for n, cdef in enumerate(defs):
        countries[cdef.id] = Country(id=cdef.id, lang=cdef.lang)
        # 순환으로 이웃 나라 말을 하나 심는다. 어느 나라도 고립되지 않고,
        # 어느 나라도 두 개를 갖지 않는다.
        seeded = defs[(n + 1) % len(defs)].lang
        for i in range(1, cfg.world.agents_per_country + 1):
            aid = f"{cdef.id}{i}"
            a = _newborn(
                aid, cdef.id, cdef.lang, cfg.income.initial_budget, set(),
                turn=0, born_by="natural", cfg=cfg, counter=counter, rng=rng,
            )
            # **1 ~ init_age_max 로 되돌렸다** (8/22).
            #
            # 8/21 에 성인 범위(10~13)로 좁혔다 — 소득을 「성인부터」 로 바꾼 순간 첫 해
            # 사람들이 빈손인데 줄 부모가 없었기 때문이다. 그런데 그 다음 날 소득 조건을
            # **「부모가 살아 있는가」** 로 바꿨고, 그 뒤 재생산 행위를 없애면서 미성년
            # 무소득 규칙 자체가 사라졌다 (8/22) — 좁힐 이유가 두 번 없어졌다.
            #
            # 그리고 좁은 구간은 나이를 흩는 목적 자체를 무력화한다 — 전원이 같은 시기에
            # 몰려 죽고, 그 뒤 성인 공백기가 온다. 넓게 흩으면 **첫 해부터 세대 사다리가**
            # 생긴다: 성인 5~6명이 바로 낳을 수 있고, 미성년 3~4명이 차례로 성인이 된다.
            a.age = rng.randint(1, cfg.world.init_age_max)
            if i == 1:
                a.known_langs.add(seeded)
            agents[aid] = a
            testaments[aid] = []
    return World(turn=0, countries=countries, agents=agents, testaments=testaments,
                 next_idx={c.id: cfg.world.agents_per_country + 1 for c in countries.values()})


def _next_id(world: World, country: str) -> str:
    """그 나라의 다음 이름. **재사용하지 않는다** (spec 2.2)."""
    n = world.next_idx.get(country, 1)
    world.next_idx[country] = n + 1
    return f"{country}{n}"


def _replace(world: World, old: str, child: Agent, carry: list[str]) -> None:
    """죽은 사람을 새 이름의 아이로 갈아 끼운다. 큐에 남은 메시지는 uid 로 걸러진다."""
    del world.agents[old]
    world.testaments.pop(old, None)
    world.agents[child.id] = child
    world.testaments[child.id] = carry


def income_for(a, world: World, cfg) -> float:
    """이 사람이 이 해에 받는 소득. **한 곳에서만 센다.**

    세 갈래가 곱해진다.

        기본       `income.per_turn`
        국가       그 나라 기술력의 생산배수 (`Country.multiplier`)
        나이       성인 나이 이후 한 해마다 `age_growth` 씩

    **미성년 무소득 규칙은 없어졌다** (8/22). 그 규칙은 「부모가 용돈을 준다」 가 성립해서
    있었고, 재생산 행위(`bear_child`)를 없애면서 **살아 있는 부모가 존재하지 않게** 됐다 —
    후손은 자연사한 사람의 자리에 온다. 줄 사람이 없는 규칙은 그냥 사람을 굶긴다.

    나이 배수의 목적은 **말년에 소비가 못 따라가게** 하는 것이다. 한 해에 쓸 수 있는 돈은
    행동력이 묶으므로(invest 40원·0.2AP → 상한 200) 나이가 들면 잉여가 강제로 쌓이고,
    그 잉여의 용처가 `give` 다 — 이제 그것이 **유일한** 용처다.
    """
    grown = 1.0 + cfg.income.age_growth * max(0, a.age - cfg.world.adult_age)
    return (cfg.income.per_turn * world.countries[a.country].multiplier(cfg)
            * grown * a.income_mult)




def _newborn(aid: str, country: str, lang: str, budget: float, parent_langs: set,
             turn: int, born_by: str, cfg, counter: "itertools.count",
             rng: random.Random | None = None) -> Agent:
    """새 사람 하나. **개체 차이를 여기서 뽑는다** (8/22).

    소득 배수와 처리량 배수를 **독립으로** 뽑는다. 부모의 값과도 무관하다 — 물려받으면 한
    계보가 누적 우위를 갖고, spec 3.3 의 「능력은 상속되지 않는다」 와도 어긋난다.

    `rng` 가 없으면 배수는 1.0 이다 (테스트에서 세계를 손으로 짤 때 흔들리지 않게).
    """
    a = Agent(
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
    if rng is not None:
        a.income_mult = rng.choice(cfg.income.spread)
        a.invest_mult = rng.choice(cfg.facility.throughput_spread)
    return a


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

def _last_words(world: World, agent, cfg, client_for, system_prompt) -> str:
    """죽는 사람에게 한 마디를 청한다. 실패하면 빈 문자열 (**런을 죽이지 않는다**).

    자연사는 예고가 없다. 그래서 「죽을 때 유언을 남긴다」 를 도구로 두면 아무도 못 쓴다 —
    `procreate` 가 30해에 1건이었던 이유다. 대신 죽는 그 순간에 **우리가 묻는다.**

    도구를 안 싣는다 — 행동이 아니라 말이다. 그리고 이 호출은 **세계를 바꾸지 않는다.**

    한 번만 시도한다. 마지막 말을 못 받는 것보다 런이 죽는 것이 나쁘다.
    """
    if client_for is None or not callable(system_prompt):
        return ""
    from domains.meteor import prompts
    try:
        sp = _system(system_prompt, agent, world, cfg, 0.0, same_year=True)
        resp = client_for(agent.id).chat(
            [{"role": "system", "content": sp},
             {"role": "user", "content": prompts.render_last_words(agent, cfg)}],
            log_tag={"turn": world.turn, "agent": agent.id, "step": 0,
                     "age": agent.age, "country": agent.country, "kind_note": "last_words"})
        txt = (resp["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        # 마지막 말은 있으면 좋은 것이다. 못 받으면 그냥 없다.
        return ""
    return txt[: cfg.length.message_max_chars[agent.native_lang]]


def _death_birth(world: World, cfg, rng: random.Random, snapshot_ids, procreated,
                 counter: "itertools.count", result: RunResult,
                 client_for=None, system_prompt=None) -> None:
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
            # **자연사가 후손을 남긴다** (8/22 개정). 재생산 행위(`bear_child`)를 없앴다 —
            # 10해 실측 네 번에서 0건이었고, 원인이 유인이 아니라 인구 구조였다.
            #
            # 그리고 이제 **물려준다.** 예산과 부모 할인 자격이 넘어간다:
            #
            #   예산      죽음이 돈을 태우지 않는다. 다만 한 해에 쓸 수 있는 돈은 행동력이
            #             묶으므로, 물려받은 돈은 쓸 손이 있어야 쓰인다 — `give` 의 자리다
            #   할인 자격  `parent_langs`. 앞사람이 알던 말이 뒷사람에게 싸진다 (3.4).
            #             **언어 자체는 안 넘어간다** (3.3) — 이중언어자 멸종은 그대로 관측된다
            #
            # 그리고 죽는 그 자리에서 **한 마디를 청한다.** 자연사는 예고가 없어서 도구로는
            # 남길 수 없다.
            last = _last_words(world, a, cfg, client_for, system_prompt)
            child = _newborn(
                _next_id(world, a.country), a.country, a.native_lang,
                a.budget,                     # 예산을 물려받는다
                a.known_langs,                # 부모 할인 자격 (언어 자체는 아니다)
                world.turn, "natural", cfg, counter, rng,
            )
            _replace(world, aid, child, [])
            # 남긴 말은 **뒷사람에게만** 간다 (PRIVATE). 기억에 심지 않는다 — 들은 말로
            # 오고, 옮겨 적을지는 본인이 고른다. 그 선택이 구전의 감쇠다 (3.3).
            if last:
                _notify(world, "testament", {"testament": [last]},
                        world.turn, actor=child.id)
            # 부고는 **같은 나라 사람에게만** (spec 4.1). 누가 죽고 누가 그 자리에 왔는지를
            # 한 쌍으로 알린다 — 이름이 안 이어지면 명단만 보고는 짝지을 수 없다.
            # 국내 구사자 할인이 사라진 이유를 알 수 있게 하는 정보이기도 하다.
            # **몇 살에 죽었는지 함께 알린다.** 수명 곡선은 은닉 목록이지만(4.1)
            # 부고에 찍힌 나이는 사실이고, 그것이 쌓이면 인구가 경험으로 배운다.
            # 자기 수명을 모르면 `procreate`(죽고 물려주기)를 고를 시점을 알 수 없다 —
            # 세 런 21명이 전부 자연사했고 procreate 는 0건이었다.
            result.deaths_log.append({"turn": world.turn, "who": aid, "born": child.id,
                                      "age": a.age, "country": a.country, "by": "natural",
                                      # **남긴 말을 로그에 둔다.** 뒷사람이 옮겨 적지 않으면
                                      # 대화에서 사라지고, 그러면 무엇을 남겼는지 알 방법이
                                      # 없다 — 하필 그 사라짐이 관측 대상이다 (3.3).
                                      "testament": last,
                                      "budget_passed": round(a.budget, 1)})
            result.births.append(
                {"turn": world.turn, "id": child.id, "replaces": aid, "uid": child.uid,
                 "born_by": "natural", "budget": child.budget}
            )
        else:
            a.age += 1



# ─────────────────────────────────────────── 턴 (더미) ─────────────────────────────

def run_turn(world: World, cfg, rng: random.Random, result: RunResult,
             counter: "itertools.count", is_last: bool = False,
             procreate_age: int | None = PROCREATE_AGE) -> None:
    """한 턴. 7단계 순서를 바꾸지 말 것. (spec 3.1)"""
    # 1. 소득 지급 + AP 리셋 (이월: 예산은 남고, AP 는 리셋)
    for a in world.agents.values():
        a.budget += income_for(a, world, cfg)
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
                # 더미 경로는 AP 비례 과금을 흉내내지 않는다 — 과제 1 의 배관 검증용이다.
                # 실제 규칙은 agent_loop.execute_tool 에 있다 (4.4).
                if amount <= 0:
                    continue
                a.budget -= amount
                facility_invest[act["to"]] += amount
                # 최초 시설 투자로 국토 용도 확정 (선착순)
                land = world.countries[act["to"]]
                if land.land is None and amount > 0:
                    land.land = DEFAULT_FACILITY_TYPE

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
    world = init_world(cfg, counter, rng)
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
    #    정렬 순으로 도는 이유 — 같은 턴에 둘이 배우면 국내 구사자 판정이 순서를 탄다.
    for rec in sorted(sink.learns, key=lambda r: r["agent"]):
        a = world.agents.get(rec["agent"])
        result.learns_log.append({"turn": world.turn, **rec})
        if a is None:
            continue
        # 진척은 execute_tool 이 **이미** 쌓았다 (한 해에 여러 번 내는 것이 정상
        # 경로가 되면서 즉시 반영이 필요해졌다). 여기서 또 더하면 두 번 센다.
    # 완료 판정은 **그 순간의** 학습가로 한다 (3.4). 같은 턴에 국내 구사자가 생기면
    # 필요액이 절반이 되어, 반쯤 낸 사람이 그 자리에서 끝날 수 있다.
    for aid in sorted(world.agents):
        a = world.agents[aid]
        for cid in sorted(world.countries):
            c = world.countries[cid]
            if cid == a.country or c.lang in a.known_langs:
                continue
            done = a.lang_progress.get(c.lang, 0.0)
            if done <= 0:
                continue
            need, _ = agent_loop.learn_cost(a, cid, world, cfg)
            if done >= need:
                a.known_langs.add(c.lang)
                result.learns_log.append({
                    "turn": world.turn, "kind": "acquired", "agent": aid,
                    "country": a.country, "target": cid, "lang": c.lang,
                    "charged": round(done, 2), "required": need,
                    "rung": round(need / cfg.costs.learn_base, 4),
                    "age": a.age, "budget_after": round(a.budget, 2),
                    "discount_domestic": agent_loop.learn_discounts(a, cid, world)[0],
                    "discount_parent": agent_loop.learn_discounts(a, cid, world)[1],
                })
    for o in sorted(sink.observations, key=lambda x: (x["agent"], x["nth"])):
        result.risk_log.append({"turn": world.turn, **o})

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
        eff = cfg.facility.eff * c.multiplier(cfg)
        # 출자자별로 따로 굴린다 — 각자 자기 출자가 얼마나 진척으로 바뀌었는지 알아야
        # 하기 때문이다(아래 통지). 합쳐서 한 번 굴리는 것과 분포는 같다.
        for amount, agent_id in sorted(entries, key=lambda x: x[1]):   # id 순 → 결정론
            share = amount if total <= cap else amount * (cap / total)
            if c.land is None:
                # 지을 것이 없으면 돈만 나가고 아무 일도 일어나지 않는다. **통지는
                # 똑같이 간다** — 통지가 없으면 그 부재가 곧 "아직 안 정했다" 가 된다.
                gain = 0
            else:
                n_i = int(share * eff)
                gain = sum(1 for _ in range(n_i) if rng.random() < cfg.world.success_prob)
                c.progress += gain
            # 행위 **후에는** 공개한다. 확률적이라 한 건으로는 success_prob 을 못 읽고,
            # 모르면 "얼마를 더 내야 하는가" 를 판단할 근거가 아예 없다.
            result.facility_gains.append({"turn": world.turn, "agent": agent_id,
                                          "to": cid, "amount": round(share, 2),
                                          "gain": gain})

    # c. wellness (수명), d. national (자본)
    for aid, amount in sink.wellness:
        if aid in world.agents:
            world.agents[aid].lam += amount * cfg.wellness.gain
            world.agents[aid].wellness_spent += amount
    for cid, amount, _ in sink.national:
        world.countries[cid].national_capital += amount

    # **받는 쪽 예산은 정산에서 넣는다.** 남의 상태이므로 즉시 바꾸면 병렬이 깨진다
    # (보내는 쪽 차감만 즉시다). 받는 이에게 알린다 — 예산은 PRIVATE 이고, 갑자기 늘어난
    # 이유를 본인이 모르면 그 돈을 쓸 판단을 못 한다.
    for frm, to, amount in sorted(sink.gifts):
        if to not in world.agents:          # 그 사이에 죽었다면 돈은 사라진다
            continue
        world.agents[to].budget += amount
        _notify(world, "gift", {"gift_from": frm, "gift": round(amount, 1)},
                world.turn, actor=to)

    # e. 메시지 → 번역 → 다음 턴 도착
    for sent in sink.messages:
        recipient = world.agents.get(sent["to"])
        reck = recipient.known_langs if recipient else set()
        to_uid = recipient.uid if recipient else None
        sender = world.agents.get(sent["from"])
        # **id 를 번역보다 먼저 뽑는다.** 나중에 뽑으면 번역 호출의 raw 기록에 msg_id 가
        # null 로 남아, 원문·도착문(messages.jsonl)과 실제 API 왕복을 이어붙일 수 없다.
        gid = next(msg_ids)
        try:
            p = messaging.process_message(sent, reck, cfg, translator, knob_ai,
                                          sender_known_langs=(sender.known_langs if sender
                                                              else frozenset()),
                                          log_tag={"turn": world.turn, "msg_id": gid})
        except BaseException as e:
            # 정산은 단일 스레드라 프레임은 남지만 **어느 메시지였는지는 안 남는다.**
            # 기록하고 그대로 던진다 — 삼키지 않는다.
            e.add_note(f"[msg {gid} · {sent['from']} → {sent['to']} · "
                       f"route {sent.get('route')} · turn {world.turn}]")
            raise
        # inbox 가 None 인 경우가 있다 — 번역 엔진 장애. 그때는 **세계에 흔적을
        # 남기지 않는다** (messaging 참조). 수신자에게 「읽을 수 없는 메시지가 왔다」 를
        # 보내면 엔진 장애를 언어 사실로 심게 된다.
        if p["inbox"] is not None:
            p["inbox"]["msg_id"] = gid      # 전역 id — understood 의 조인 키 (spec 6.1)
            # 주고받은 말 — **보낸 이와 받는 이만** (visibility: message PRIVATE)
            _notify(world, "message", p["inbox"], world.turn + 1, actor=sent["to"])
        result.messages_log.append({"turn": world.turn, "msg_id": gid,
                                    "from": sent["from"], "to": sent["to"],
                                    "action": sent.get("kind", "speak"),
                                    "route": p["kind"], "delivered": p["delivered"],
                                    "meta": p["meta"]})
        if p["sender_notice"]:
            su = world.agents[sent["from"]].uid if sent["from"] in world.agents else None
            # 내가 보낸 말이 닿지 않았다 — **나만 안다** (visibility: delivery_failed)
            _notify(world, "delivery_failed",
                    {"from": None, "text": None, "label": None, "original": None,
                     "msg_id": next(msg_ids), "delivery_failed_to": sent["to"],
                     # 언어 때문인지 엔진 장애인지 — 원인을 섞으면 거짓을 심는다
                     "delivery_failed_reason": p["sender_notice"].get("reason")},
                    world.turn + 1, actor=sent["from"])

    # f. 투표 기록 (정식 집계는 이후 과제)
    # ★ 투표는 로그만 남고 아무 일도 하지 않았다. 국토는 첫 시설 투자로
    #   DEFAULT_FACILITY_TYPE 이 되는 게 전부였고, 43턴 실측에서 세 나라가 모두
    #   interceptor 였던 것은 **고른 게 아니라 기본값**이었다.
    #
    #   이제 국토는 **투표로만** 정해진다. 소집 → 3턴 유예(상의할 시간) → 네 번째 턴에
    #   무엇을 지을지 고른다. 유예가 있는 이유 — 그 사이에 설득하지 않으면 한 사람의 표가
    #   그대로 나라를 정한다. 막는 것은 규칙이 아니라 사람들이어야 한다.
    #
    #   **소집에는 내용이 없다.** 전에는 `target` 을 들고 「이것으로 하자」 를 열었고 같은
    #   턴에 둘이 제안하면 하나만 열렸다 — 밀린 쪽은 AP 0.6 을 내고 아무 일도 안 일어난
    #   것을 알 방법이 없었다. 지금은 둘이 소집해도 같은 採決이라 겹칠 것이 없다.
    for by, country in sorted(sink.votes):
        c = world.countries.get(country)
        opened = c is not None and c.proposal is None
        rec = {"by": by, "opened_turn": world.turn, "vote_turn": world.turn + VOTE_DELAY}
        result.votes_log.append({"turn": world.turn, "kind": "propose",
                                 "by": by, "country": country,
                                 "vote_turn": rec["vote_turn"], "opened": opened})
        if opened:
            c.proposal = rec

    # 개표 — **최다득표.** 표는 interceptor / bunker / abstain 중 하나다.
    #
    #   `abstain` 은 어느 쪽으로도 안 센다. 표를 아예 안 낸 것과 개표상 같지만 **로그에는
    #   다르게 남는다** — 「생각해봤지만 정하지 않았다」 가 근거와 함께 남아 지표가 읽는다.
    #
    #   동수거나 아무도 안 내면 **현 상태 그대로다.** 진척도 살아 있다. 합의 실패의 대가를
    #   진척 파괴로 물리면, 소집 한 번이 남의 나라 진척을 지우는 무기가 된다.
    #
    #   결과가 지금 국토와 같아도 그대로다 — 착수를 다시 하는 것이 아니라 유지다.
    ballots_by: dict[str, list] = defaultdict(list)
    for by, country, choice in sorted(sink.ballots):
        ballots_by[country].append((by, choice))
        result.votes_log.append({"turn": world.turn, "kind": "ballot",
                                 "by": by, "country": country, "choice": choice})
    for cid in sorted(world.countries):
        c = world.countries[cid]
        if c.proposal is None or c.proposal["vote_turn"] != world.turn:
            continue
        cast = _one_vote_each(ballots_by.get(cid, []))
        counts = {k: sum(1 for _, ch in cast if ch == k)
                  for k in ("interceptor", "bunker", "abstain")}
        top = max(counts["interceptor"], counts["bunker"])
        tie = counts["interceptor"] == counts["bunker"]
        chosen = None if (top == 0 or tie) else (
            "interceptor" if counts["interceptor"] > counts["bunker"] else "bunker")
        rec = {"turn": world.turn, "country": cid, "called_by": c.proposal["by"],
               "counts": counts, "chosen": chosen, "from": c.land,
               "changed": False, "progress_lost": 0.0}
        if chosen is not None and chosen != c.land:
            # 다른 시설을 착수하면 기존 시설은 파괴된다 — 진척 0 (SYSTEM 규칙 5)
            rec["changed"] = True
            rec["progress_lost"] = round(c.progress, 3)
            c.land, c.progress = chosen, 0.0
        c.proposal = None                       # 바뀌든 안 바뀌든 採決은 닫힌다
        result.land_changes.append(rec)
        # **採決 결과와 그때 사라진 진척을 그 나라에 알린다** (ballot_result PUBLIC).
        # 전에는 아무도 통지받지 않았다 — 다음 해에 진척이 0 인 것을 보고 추론해야 했고,
        # 국토도 같이 바뀌어 「내가 낸 것이 다 날아갔다」 를 알아차릴 단서가 약했다.
        _notify(world, "ballot_result",
                {"ballot": "changed" if rec["changed"] else (
                     "kept" if rec["chosen"] else "none"),
                 "land": c.land, "lost": rec["progress_lost"]},
                world.turn, nation=cid)

    # f-2. 출자자에게 자기 몫의 진척 기여를 다음 턴에 알린다 (행위 후 공개)
    #
    # **타국에는 액수를 알려주지 않는다.** 알려주면 상대국 생산배수가 새어 나온다:
    #
    #     E[gain] / amount = facility.eff × success_prob × multiplier(받는 나라)
    #
    # 상수가 모든 나라에 같으므로 **두 나라를 비교하면 상수가 지워지고 배수 비율만
    # 남는다.** 실측에서 통지를 쌓아 배수 1.13 을 복원했다 (실제 1.13~1.15).
    # 자국 배수보다 나쁜 누출이다 — 자국은 수입에서 추론하는 정당한 경로가 있는데
    # (그래서 관측에서 배수 자체를 뺐다), 타국은 4.1 이 "소통으로만" 이라고 못 박았다.
    #
    # 「늘었다 / 늘지 않았다」 는 살린다. 두 가지가 그것에 걸려 있다.
    #   ① 없으면 남의 땅에 내는 선택이 영영 깜깜이가 된다
    #   ② gain 0 이 곧 "그 나라가 아직 국토를 안 정했다" 다. 통지 자체가 없으면
    #      그 부재가 같은 말을 하게 되므로, 통지는 어느 쪽이든 똑같이 가야 한다
    #
    # 곁들여 **소액 출자의 난수를 신호로 읽는 일**도 막힌다. 80원 출자의 비율은
    # 상대편차가 16%, 20원은 33% 다. 실측에서 자국에 10~40원·타국에 50~80원을 내던
    # 에이전트가 자국 비율이 널뛰는 것을 보고 "타국이 더 효율적" 이라 읽었고,
    # 885원을 남의 **벙커** 로 보냈다.
    for g in result.facility_gains:
        if g["turn"] != world.turn or g["agent"] not in world.agents:
            continue
        msg = {"msg_id": next(msg_ids), "amount": g["amount"], "to": g["to"]}
        if world.agents[g["agent"]].country == g["to"]:
            msg["fac_gain"] = g["gain"]              # 자국은 그대로 (진척 델타로 어차피 보인다)
        else:
            msg["fac_moved"] = g["gain"] > 0         # 타국은 늘었는지 여부만
        # 내 출자가 얼마를 올렸나 — **나만 안다** (visibility: fac_gain PRIVATE)
        _notify(world, "fac_gain", msg, world.turn + 1, actor=g["agent"])

    # g. 재생산 행위는 없다 (8/22) — 자연사가 후손을 남긴다 (`_death_birth`).
    return set()


def _system(system_prompt, agent, world, cfg, knob_ai, *, same_year: bool):
    """SYSTEM 을 만든다. **`same_year` 는 루프가 정한다.**

    전에는 러너가 `functools.partial(system_for, same_year=args.sequential)` 로 기억해야
    했다. 그러면 다른 경로로 돌릴 때 **문구가 조용히 거짓이 된다** — 순차 라운드로빈은
    메시지가 같은 해에 도착하는데 「翌年に届きます」 라고 적히고, 그것을 믿고 계획한다.
    이미 한 번 고친 거짓말이고, 그 진실이 호출자의 기억에 걸려 있었다.

    루프는 자기가 순차인지 안다. 그러니 루프가 말한다.
    """
    if not callable(system_prompt):
        return system_prompt
    try:
        return system_prompt(agent, world, cfg, knob_ai, same_year=same_year)
    except TypeError:
        # same_year 를 안 받는 렌더러도 있다 (테스트의 더미)
        return system_prompt(agent, world, cfg, knob_ai)


def run_turn_agentic(world: World, cfg, rng: random.Random, result: RunResult,
                     counter: "itertools.count", client_for, translator, knob_ai: float,
                     render_obs, system_prompt, msg_ids, is_last: bool = False,
                     parallel: bool = True, on_turn_end=None, render_events=None,
                     render_arrivals=None) -> None:
    """한 턴 (에이전트). spec 3.1 순서를 지키되 3단계는 9명 병렬, 5단계는 정렬 정산."""
    # 1. 소득 + AP 리셋
    for a in world.agents.values():
        inc = income_for(a, world, cfg)
        a.budget += inc
        # **받은 값을 적어 둔다.** 해 시작 문구가 렌더 때 다시 계산하면, 순차에서 나중에
        # 차례가 온 사람은 남들이 national 에 넣은 뒤의 값을 보게 된다 — 실제로 100 을
        # 받았는데 「+102」 라고 적혔다.
        a.income_this_year = inc
        a.ap = cfg.turn.action_points
        # ★ x̂ 의 분모. "그 눈금을 감당할 수 있었는데도 안 배웠다" 를 세려면 **결정
        #   시점의** 예산이 필요하다. 턴 끝 예산으로 대신하면 다른 데 써버린 사람이
        #   기회 자체가 없었던 것으로 집계돼 분모가 조용히 줄어든다.
        a.budget_start = a.budget

    # 2. 관측 스냅샷 (도착 메시지·프롬프트를 스레드 시작 전에 고정)
    snapshot_ids = sorted(world.agents.keys())
    snapshot_uids = {world.agents[aid].uid for aid in snapshot_ids}
    inboxes = {aid: _dequeue_inbox(world, aid) for aid in snapshot_ids}
    world.inbox_queue = [e for e in world.inbox_queue if e["deliver_turn"] > world.turn]
    for aid in inboxes:                     # 새해가 먼저, 그 다음 사건
        _push(world.agents[aid],
              render_obs(world, world.agents[aid], cfg, knob_ai, None,
                         income_this_turn=world.agents[aid].income_this_year,
                         opening=True))
        _push_events(world.agents[aid], inboxes[aid], render_events)
    user_prompts = {aid: (render_arrivals(world.agents[aid], inboxes[aid])
                          if render_arrivals else None)
                    for aid in snapshot_ids}

    # 3. 정책 호출 — 병렬. 각자 자기 Sink 에만 쓴다 (공유 상태 미변경 → 스레드 안전)
    sinks = {aid: Sink() for aid in snapshot_ids}

    def run_one(aid):
        agent = world.agents[aid]
        # **system 은 매 콜 새로 만든다** — 규칙 + 지금 그러한 것. 상태가 대화에 쌓이면
        # 낡은 사본이 남아 모순이 되고(예산이 여러 개), 그 부피가 대화를 방출시킨다.
        sp = _system(system_prompt, agent, world, cfg, knob_ai, same_year=False)
        try:
            return aid, run_agent_turn(world, agent, cfg, client_for(aid), sinks[aid],
                                       knob_ai, sp, user_prompts[aid])
        except BaseException as e:
            # **기록하고 그대로 던진다.** 삼키지 않는다 — 예외를 좁힌 뜻이 사라진다.
            #
            # 병렬이면 ThreadPoolExecutor 가 예외를 주 스레드로 옮기는데, 그때 **어느
            # 에이전트였는지가 사라진다.** 프레임에는 run_agent_turn 만 남고 aid 값은
            # 안 보인다. add_note 로 트레이스백에 새겨 두면 로그만 보고 찾을 수 있다.
            e.add_note(f"[agent {aid} · turn {world.turn} · age {agent.age}]")
            raise

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
        merged.ballots += s.ballots
        merged.observations += s.observations
        merged.learns += s.learns
    procreated = _settle_agentic(world, cfg, rng, merged, translator, knob_ai, counter,
                                 result, msg_ids)

    # 7. 생사 판정 (마지막 턴 생략)
    if not is_last:
        _death_birth(world, cfg, rng, snapshot_ids, procreated, counter, result)
        _queue_obituaries(world, result, msg_ids)

    result.acted.append(snapshot_uids)
    result.alive_counts.append(sum(1 for a in world.agents.values() if a.alive))
    result.state_lines.append(_state_line(world))
    if on_turn_end is not None:
        on_turn_end(world.turn, result)


# ── 순차 라운드로빈 (spec — 한 턴 안에서 서로 반영·대화. issue #20) ──────────────

def _notify(world: World, fact: str, msg: dict, deliver_turn: int,
            actor: str | None = None, nation: str | None = None) -> int:
    """**공개 등급이 정한 사람들에게** 알린다. 넣은 사람 수를 돌려준다.

    청중을 호출부가 정하지 않는다 — `visibility.FACTS` 의 한 줄이 정한다. 그 줄이
    없으면 `KeyError` 로 막힌다.

    `to_uid` 를 함께 넣는 이유 — 그 사이에 죽고 교체됐으면 그 자리에 온 아이는 **다른
    사람**이므로(3.2) 받을 것이 아니다. `_dequeue_inbox_pop` 이 uid 로 걸러 폐기한다.
    """
    n = 0
    for aid in visibility.audience(world, fact, actor=actor, nation=nation):
        world.inbox_queue.append({"deliver_turn": deliver_turn, "to": aid,
                                  "to_uid": world.agents[aid].uid, "msg": dict(msg)})
        n += 1
    return n


def _queue_obituaries(world: World, result: RunResult, msg_ids) -> None:
    """부고를 **공개 등급이 정한 사람들**에게 보낸다 (`visibility.FACTS["obituary"]`).

    청중을 여기서 즉흥으로 정하지 않는다 — 전에는 같은 나라를 훑는 루프가 두 곳에 복사돼
    있었고(병렬·순차), 등급을 바꾸려면 두 곳을 같이 고쳐야 했다.

    **8/20 에 PUBLIC → GLOBAL 로 올렸다.** `roster` 가 이미 교체를 드러내므로(누가
    사라지고 누가 왔는지 명단으로 보인다) 새로 새는 것은 **나이**뿐이고, 그것이 수명을
    배우는 유일한 경로다 — 곡선은 여전히 SECRET 이고 평균만 SYSTEM 에 있다.

    죽은 사람 자신에게는 보내지 않는다. 그 자리에 태어난 아이는 **다른 사람**이라(3.2)
    uid 로 걸러진다.
    """
    for d in result.deaths_log:
        if d["turn"] != world.turn:
            continue
        mid = next(msg_ids)                   # **한 사건에 하나의 id.** 사람마다 다른
                                              # 번호를 주면 같은 죽음이 여럿처럼 남는다
        for aid in visibility.audience(world, "obituary",
                                       actor=d["who"], nation=d["country"]):
            if aid == d["who"]:
                continue                      # 자기 부고를 받지 않는다
            world.inbox_queue.append({
                "deliver_turn": world.turn + 1, "to": aid,
                "to_uid": world.agents[aid].uid,
                "msg": {"msg_id": mid, "died": d["who"],
                        "born": d.get("born"), "age": d.get("age")}})


def _push(agent, text: str | None) -> None:
    """대화에 한 항목을 붙인다. **빈 것은 붙이지 않는다.**

    빈 `user` 가 실제로 들어간 적이 있다 — 사건만 온 차례에 도착분 렌더가 빈 문자열을
    돌려주는데 그것을 그대로 append 했다.
    """
    if text:
        agent.convo.append({"role": "user", "content": text})


def _push_events(agent, inbox: list[dict], render_events) -> None:
    """**세계의 사건을 자기 자리에 놓는다.** 해 오프닝과 섞지 않는다.

    죽음·출자 결과·전달 실패는 「올해가 시작됐다」 와 성질이 다르다. 오프닝에 묶으면
    새해 인사에 부고가 딸려 오고, 무엇이 언제 일어났는지가 한 덩어리로 뭉개진다.

    **대화에서 사건이 앞에 온다** — 해 끝에 죽은 사람 소식이 다음 해가 열리기 전에
    놓인다. 모델은 대화를 볼 때만 무엇을 알게 되므로, 그 순서가 곧 「그 시점」 이다.
    """
    if render_events is not None:
        _push(agent, render_events(agent, inbox))


def _has_inbox(world: World, aid: str) -> bool:
    """이 차례에 받을 것이 있나. **꺼내지 않고 본다** (깨울지 판단하는 데만 쓴다)."""
    current = world.agents.get(aid)
    for e in world.inbox_queue:
        if e["deliver_turn"] <= world.turn and e["to"] == aid:
            if current and e.get("to_uid") is not None and e["to_uid"] != current.uid:
                continue                     # 수신 슬롯이 바뀌었다 — 폐기될 것
            return True
    return False


def _dequeue_inbox_pop(world: World, aid: str) -> list[dict]:
    """이 차례에 받을 메시지를 큐에서 **꺼내며 제거**한다 (라운드로빈).

    같은 턴 도착분(deliver_turn ≤ turn)을 수령한다 — A 가 이번 턴에 보낸 것을 B 가
    다음 차례에 본다. 수신 슬롯 점유자가 바뀌었으면(사망·교체) 폐기한다 (spec 4.2)."""
    current = world.agents.get(aid)
    out, keep = [], []
    for e in world.inbox_queue:
        if e["deliver_turn"] <= world.turn and e["to"] == aid:
            if current and e.get("to_uid") is not None and e["to_uid"] != current.uid:
                continue                     # recipient_dead → 폐기 (큐에서도 제거)
            out.append(e["msg"])
        else:
            keep.append(e)
    world.inbox_queue = keep
    return out


def _settle_step(world: World, cfg, rng: random.Random, sink: Sink, translator,
                 knob_ai: float, msg_ids: "itertools.count", result: RunResult,
                 turn_facility: dict, ballots_acc: list,
                 said_this_year: set | None = None) -> None:
    """한 차례(sink)를 세계에 **즉시** 반영. 개표·procreate·부고는 턴 끝에서 (누적만)."""
    # 학습 — 즉시 반영 + 완료 즉시 판정 (그 순간의 학습가로)
    for rec in sink.learns:
        result.learns_log.append({"turn": world.turn, **rec})
        a = world.agents.get(rec["agent"])
        if a is None:
            continue
        # 진척은 execute_tool 이 **이미** 쌓았다 (한 해에 여러 번 내는 것이 정상
        # 경로가 되면서 즉시 반영이 필요해졌다). 여기서 또 더하면 두 번 센다.
        cid = rec["target"]
        c = world.countries.get(cid)
        if c is not None and c.lang not in a.known_langs:
            need, _ = agent_loop.learn_cost(a, cid, world, cfg)
            if a.lang_progress.get(c.lang, 0.0) >= need:
                a.known_langs.add(c.lang)
                result.learns_log.append({
                    "turn": world.turn, "kind": "acquired", "agent": rec["agent"],
                    "country": a.country, "target": cid, "lang": c.lang,
                    "charged": round(a.lang_progress[c.lang], 2), "required": need,
                    "rung": round(need / cfg.costs.learn_base, 4),
                    "age": a.age, "budget_after": round(a.budget, 2),
                    "discount_domestic": agent_loop.learn_discounts(a, cid, world)[0],
                    "discount_parent": agent_loop.learn_discounts(a, cid, world)[1]})
    for o in sink.observations:
        result.risk_log.append({"turn": world.turn, **o})
    # 시설 — 이번 턴 국가별 누적(turn_facility) 기준 **선착순 cap**, 즉시 진척 + 같은 턴 통지
    #
    # **진척 변화는 그 나라에 일괄로 알린다** (visibility: progress_change PUBLIC).
    # 출자자별로 알리면 부피가 3배가 된다 — 실측에서 해당 2.8건이고 각각 3명에게 가면
    # 해마다 8항목이 대화에 쌓인다. 차례 단위로 묶으면 5항목이다.
    #
    # 누가 냈는지는 담지 않는다. 자국민은 자기가 낸 것을 알므로 차이에서 타국 출자를
    # 짐작할 수 있고, 그 짐작은 흘려도 되는 것이다.
    prog_delta: dict = {}
    for to_country, amount, agent_id in sink.facility:
        cap = cfg.facility.cap_per_turn
        used = turn_facility.get(to_country, 0.0)
        share = min(amount, max(0.0, cap - used))
        excess = amount - share
        if excess > 0 and agent_id in world.agents:
            world.agents[agent_id].budget += excess       # 선착순 초과분 즉시 환급
        turn_facility[to_country] = used + share
        c = world.countries[to_country]
        eff = cfg.facility.eff * c.multiplier(cfg)
        if c.land is None:
            gain = 0
        else:
            n_i = int(share * eff)
            gain = sum(1 for _ in range(n_i) if rng.random() < cfg.world.success_prob)
            c.progress += gain
        result.facility_gains.append({"turn": world.turn, "agent": agent_id,
                                      "to": to_country, "amount": round(share, 2),
                                      "gain": gain})
        if agent_id in world.agents:                       # 자기 몫 통지 (같은 턴)
            note = {"msg_id": next(msg_ids), "amount": round(share, 2), "to": to_country}
            if world.agents[agent_id].country == to_country:
                note["fac_gain"] = gain                     # 자국은 그대로
            else:
                note["fac_moved"] = gain > 0                # 타국은 늘었는지 여부만
            _notify(world, "fac_gain", note, world.turn, actor=agent_id)
        if gain:
            prog_delta[to_country] = prog_delta.get(to_country, 0.0) + gain
    for cid, gain in sorted(prog_delta.items()):
        _notify(world, "progress_change",
                {"prog_up": gain, "now": world.countries[cid].progress},
                world.turn, nation=cid)
    # wellness / national
    for aid, amount in sink.wellness:
        if aid in world.agents:
            world.agents[aid].lam += amount * cfg.wellness.gain
            world.agents[aid].wellness_spent += amount
    # **받는 쪽 예산은 정산에서 넣는다.** 남의 상태이므로 즉시 바꾸면 병렬이 깨진다
    # (보내는 쪽 차감만 즉시다). 받는 이에게 알린다 — 예산은 PRIVATE 이고, 갑자기 늘어난
    # 이유를 본인이 모르면 그 돈을 쓸 판단을 못 한다.
    for frm, to, amount in sorted(sink.gifts):
        if to not in world.agents:          # 그 사이에 죽었다면 돈은 사라진다
            continue
        world.agents[to].budget += amount
        _notify(world, "gift", {"gift_from": frm, "gift": round(amount, 1)},
                world.turn, actor=to)

    capped: set = set()
    for cid, amount, _ in sink.national:
        world.countries[cid].national_capital += amount
        capped.add(cid)
    for cid in sorted(capped):
        # 수입·시설 전환율·관측 정확도가 다 여기 걸려 있어 국민 전원의 일이다.
        # **얼마나 올랐는지는 담지 않는다** — 배수 함수는 SECRET 이다.
        #
        # 그래서 **해마다 한 번만** 말한다. 값이 없는 사실이라 두 번 적어도 더 알려주는
        # 것이 없다 — 실측에서 「자국의 기술력이 올랐습니다」 가 한 해에 세 번 붙었다.
        # 진척은 값이 달라지므로 차례마다 말한다.
        if said_this_year is not None:
            if ("cap", cid) in said_this_year:
                continue
            said_this_year.add(("cap", cid))
        _notify(world, "capital_change", {"cap_up": True}, world.turn, nation=cid)
    # 메시지 — 번역 후 **같은 턴** 배달
    for sent in sink.messages:
        recipient = world.agents.get(sent["to"])
        reck = recipient.known_langs if recipient else set()
        to_uid = recipient.uid if recipient else None
        sender = world.agents.get(sent["from"])
        gid = next(msg_ids)
        try:
            p = messaging.process_message(sent, reck, cfg, translator, knob_ai,
                                          sender_known_langs=(sender.known_langs if sender
                                                              else frozenset()),
                                          log_tag={"turn": world.turn, "msg_id": gid})
        except BaseException as e:
            e.add_note(f"[msg {gid} · {sent['from']} → {sent['to']} · "
                       f"route {sent.get('route')} · turn {world.turn}]")
            raise
        if p["inbox"] is not None:
            p["inbox"]["msg_id"] = gid
            _notify(world, "message", p["inbox"], world.turn, actor=sent["to"])
        result.messages_log.append({"turn": world.turn, "msg_id": gid,
                                    "from": sent["from"], "to": sent["to"],
                                    "action": sent.get("kind", "speak"),
                                    "route": p["kind"], "delivered": p["delivered"],
                                    "meta": p["meta"]})
        if p["sender_notice"]:
            su = world.agents[sent["from"]].uid if sent["from"] in world.agents else None
            _notify(world, "delivery_failed",
                    {"from": None, "text": None, "label": None, "original": None,
                     "msg_id": next(msg_ids), "delivery_failed_to": sent["to"],
                     "delivery_failed_reason": p["sender_notice"].get("reason"),
                     "ref_msg_id": gid}, world.turn, actor=sent["from"])
    # 투표 — 소집은 즉시 열고, ballot 은 누적(턴 끝 개표)
    for by, country in sink.votes:
        c = world.countries.get(country)
        opened = c is not None and c.proposal is None
        rec = {"by": by, "opened_turn": world.turn, "vote_turn": world.turn + VOTE_DELAY}
        result.votes_log.append({"turn": world.turn, "kind": "propose", "by": by,
                                 "country": country, "vote_turn": rec["vote_turn"],
                                 "opened": opened})
        if opened:
            c.proposal = rec
    for by, country, choice in sink.ballots:
        ballots_acc.append((by, country, choice))
        result.votes_log.append({"turn": world.turn, "kind": "ballot",
                                 "by": by, "country": country, "choice": choice})


def _one_vote_each(cast: list) -> list:
    """(사람, 선택) 목록에서 **사람마다 한 표만** 남긴다. 처음 던진 것이 그 사람의 표다.

    도구가 이미 두 번째를 거절하므로 여기까지 오지 않는 것이 정상이다. 그래도 둔다 —
    막지 않았을 때 실제로 두 표가 집계됐고(3해 실측), 집계는 **국토를 정하는 자리**라
    한 경로가 새면 나라의 용도가 한 사람 손에 두 번 실린다.
    """
    seen, out = set(), []
    for by, choice in cast:
        if by in seen:
            continue
        seen.add(by)
        out.append((by, choice))
    return out


def _roundrobin_tally(world: World, cfg, result: RunResult, ballots_acc: list) -> None:
    """턴 끝 개표 — 최다득표(interceptor/bunker/abstain). _settle_agentic 과 같은 규칙."""
    ballots_by: dict[str, list] = defaultdict(list)
    for by, country, choice in sorted(ballots_acc):
        ballots_by[country].append((by, choice))
    for cid in sorted(world.countries):
        c = world.countries[cid]
        if c.proposal is None or c.proposal["vote_turn"] != world.turn:
            continue
        cast = _one_vote_each(ballots_by.get(cid, []))
        counts = {k: sum(1 for _, ch in cast if ch == k)
                  for k in ("interceptor", "bunker", "abstain")}
        tie = counts["interceptor"] == counts["bunker"]
        top = max(counts["interceptor"], counts["bunker"])
        chosen = None if (top == 0 or tie) else (
            "interceptor" if counts["interceptor"] > counts["bunker"] else "bunker")
        rec = {"turn": world.turn, "country": cid, "called_by": c.proposal["by"],
               "counts": counts, "chosen": chosen, "from": c.land,
               "changed": False, "progress_lost": 0.0}
        if chosen is not None and chosen != c.land:
            rec["changed"] = True
            rec["progress_lost"] = round(c.progress, 3)
            c.land, c.progress = chosen, 0.0
        c.proposal = None
        result.land_changes.append(rec)
        # **採決 결과와 그때 사라진 진척을 그 나라에 알린다** (ballot_result PUBLIC).
        # 전에는 아무도 통지받지 않았다 — 다음 해에 진척이 0 인 것을 보고 추론해야 했고,
        # 국토도 같이 바뀌어 「내가 낸 것이 다 날아갔다」 를 알아차릴 단서가 약했다.
        _notify(world, "ballot_result",
                {"ballot": "changed" if rec["changed"] else (
                     "kept" if rec["chosen"] else "none"),
                 "land": c.land, "lost": rec["progress_lost"]},
                world.turn, nation=cid)


def run_turn_roundrobin(world: World, cfg, rng: random.Random, result: RunResult,
                        counter: "itertools.count", client_for, translator, knob_ai: float,
                        render_obs, system_prompt, msg_ids, is_last: bool = False,
                        on_turn_end=None, render_events=None,
                        render_arrivals=None) -> None:
    """한 턴 — 순차 라운드로빈. 임의 순서로 한 명씩 한 차례(1콜)씩, AP 남은 사람끼리
    전원 소진까지 돈다. 차례마다 관측을 새로 렌더하고 액션을 즉시 반영한다 (issue #20)."""
    # 1. 소득 + AP 리셋
    for a in world.agents.values():
        inc = income_for(a, world, cfg)
        a.budget += inc
        # **받은 값을 적어 둔다.** 해 시작 문구가 렌더 때 다시 계산하면, 순차에서 나중에
        # 차례가 온 사람은 남들이 national 에 넣은 뒤의 값을 보게 된다 — 실제로 100 을
        # 받았는데 「+102」 라고 적혔다.
        a.income_this_year = inc
        a.ap = cfg.turn.action_points
        a.budget_start = a.budget                       # x̂ 분모 (결정 시점 예산)
    snapshot_ids = sorted(world.agents.keys())
    snapshot_uids = {world.agents[aid].uid for aid in snapshot_ids}
    order = list(snapshot_ids)
    rng.shuffle(order)                                  # 임의 순서 (시드로 결정론)

    turn_facility: dict = {}
    said_this_year: set = set()      # 값 없는 사실은 해마다 한 번만 (capital_change)
    ballots_acc: list = []
    accs = {aid: agent_loop._StepAcc() for aid in snapshot_ids}
    t_turns = {aid: time.time() for aid in snapshot_ids}
    ended: dict = {aid: None for aid in snapshot_ids}
    first_seen: set = set()   # 이 턴 첫 차례엔 풀 관측, 이후엔 델타 (issue #22)

    active = True
    while active:
        active = False
        for aid in order:
            if ended[aid] is not None:
                # **끝냈다고 했는데 그 뒤에 말이 왔다면 다시 깨운다.**
                #
                # `end_turn` 은 「지금 더 할 일이 없다」 는 판단이다. 그 뒤에 도착한
                # 메시지는 **그 판단의 근거를 무너뜨리는 새 정보**다 — 누가 협력을 청했는데
                # 이미 끝냈다고 그 해가 통째로 지나가면, 같은 해 왕복 대화라는 순차
                # 라운드로빈의 취지가 절반만 산다.
                #
                # 세 조건을 다 만족해야 깨운다.
                #   ① 스스로 끝낸 것("ended")만. exhausted·error·repeat_guard·runaway 는
                #      깨우면 같은 실패를 되풀이한다
                #   ② 행동력이 남아 있어야 한다. 0 이면 깨워도 할 수 있는 것이
                #      memory_write 뿐이다
                #   ③ 받을 것이 실제로 있어야 한다 (꺼내지 않고 본다)
                #
                # 무한 왕복은 AP 가 막는다 — speak 이 0.3 이라 한 해에 세 번이 끝이다.
                if not (ended[aid] == "ended"
                        and aid in world.agents
                        and world.agents[aid].uid in snapshot_uids
                        and world.agents[aid].ap > 0
                        and _has_inbox(world, aid)):
                    continue
                ended[aid] = None            # 다시 차례를 준다
            agent = world.agents[aid]
            st = accs[aid]
            if st.steps >= agent_loop.RUNAWAY_CAP:
                ended[aid] = "runaway"
                continue
            if not agent_loop.can_act(agent, cfg, knob_ai):
                ended[aid] = "exhausted"
                continue
            active = True
            inbox = _dequeue_inbox_pop(world, aid)      # 이 차례 도착분 (같은 턴 포함)
            # 턴을 여는 한 마디 + 이번 차례에 도착한 것. **첫 차례이거나 새로 온 것이
            # 있을 때만** 붙인다 — 관측은 system 이 매 콜 새로 담으므로, 차례마다
            # user 를 쌓을 이유가 없어졌다 (델타 렌더가 필요 없어진 이유다).
            fresh = aid not in first_seen
            first_seen.add(aid)
            # **해 오프닝은 그 해 첫 차례에만.** 재방문에 다시 붙이면 같은 해가 여러 번
            # 열린 것처럼 보이고, 안의 예산이 흔들린다 (실측 100 → 97).
            # **새해가 밝은 것이 그 해의 사건보다 앞에 온다.**
            #
            # 전에는 사건을 먼저 붙여서, 나중에 차례가 온 사람은 이런 대화를 받았다:
            #     user: 起きたこと: 自国の技術力が上がりました。
            #     user: 42 年になりました。…
            # 그 기술력 상승은 42년에 일어난 일이다. 해는 모두에게 같은 때 밝는다 —
            # 소득도 AP 도 턴 시작에 한꺼번에 주어진다.
            if fresh:
                _push(agent, render_obs(world, agent, cfg, knob_ai, None,
                                        income_this_turn=agent.income_this_year,
                                        opening=True))
            _push_events(agent, inbox, render_events)
            obs = render_arrivals(agent, inbox) if render_arrivals else None
            # **순차는 같은 해에 도착한다** — 문구가 그것을 말해야 한다
            sp = _system(system_prompt, agent, world, cfg, knob_ai, same_year=True)
            sink = Sink()
            try:
                done = agent_loop.run_agent_step(world, agent, cfg, client_for(aid), sink,
                                                 knob_ai, sp, obs, st)
            except BaseException as e:
                e.add_note(f"[agent {aid} · turn {world.turn} · age {agent.age}]")
                raise
            _settle_step(world, cfg, rng, sink, translator, knob_ai, msg_ids, result,
                         turn_facility, ballots_acc, said_this_year)
            if done is not None:
                ended[aid] = done

    logs = {aid: agent_loop._turn_log(world.agents[aid], accs[aid],
                                      ended[aid] or "exhausted", t_turns[aid])
            for aid in snapshot_ids}
    result.agent_logs.append(logs)

    # 턴 끝 — 개표 → 생사판정. **재생산 행위는 없다** (8/22): 자연사가 후손을 남긴다.
    _roundrobin_tally(world, cfg, result, ballots_acc)

    if not is_last:
        _death_birth(world, cfg, rng, snapshot_ids, set(), counter, result,
                     client_for=client_for, system_prompt=system_prompt)
        _queue_obituaries(world, result, msg_ids)

    result.acted.append(snapshot_uids)
    result.alive_counts.append(sum(1 for a in world.agents.values() if a.alive))
    result.state_lines.append(_state_line(world))
    if on_turn_end is not None:
        on_turn_end(world.turn, result)


def run_agentic(cfg, rng: random.Random, client_for, translator, knob_ai: float,
                render_obs, system_prompt, parallel: bool = True, sequential: bool = False,
                on_turn_end=None, sim_turns: int | None = None,
                resume_from: "Path | None" = None,
                checkpoint_to: "Path | None" = None, render_events=None,
                render_arrivals=None) -> RunResult:
    """LLM(또는 StubClient) 에이전트로 total_turns 턴을 돌린다.

    client_for(aid) : 에이전트별 클라이언트 (병렬이라 상태 있는 Stub 은 에이전트마다 별개여야).
                      실제 API 는 stateless OpenRouterClient 를 공유해도 안전.
    translator      : 번역 전용 클라이언트 (정산은 단일 스레드라 공유 가능).
    """
    from core import checkpoint
    done = 0
    if resume_from and Path(resume_from).exists():
        # **세계 전부**를 되살린다 — 난수 상태와 카운터까지. 하나라도 빠지면
        # 이어붙인 뒤가 원래 런과 다른 세계가 된다.
        world, rng, counter, msg_ids, done = checkpoint.load(resume_from)
    else:
        counter = itertools.count(1)
        msg_ids = itertools.count(1)  # 전역 메시지 id — 로그 조인 키 (spec 6.1)
        world = init_world(cfg, counter, rng)
    result = RunResult(world=world)
    # **운석 시점과 시뮬 길이는 다른 것이다.** `cfg.world.total_turns` 가 운석이 떨어지는
    # 해이고 경제(창·임계)도 수명도 거기서 유도된다. `sim_turns` 는 **몇 턴까지 돌려볼
    # 것인가** 다. 붙여 두면 40턴 테스트가 "40턴짜리 세계" 가 되어 남은 턴·임계·수명이
    # 전부 달라지고, 짧은 테스트의 관측이 본실험과 다른 세계를 재게 된다.
    last = min(sim_turns or cfg.world.total_turns, cfg.world.total_turns)
    for t in range(done + 1, last + 1):
        world.turn = t
        if sequential:
            run_turn_roundrobin(world, cfg, rng, result, counter, client_for, translator,
                                knob_ai, render_obs, system_prompt, msg_ids,
                                is_last=(t == cfg.world.total_turns), on_turn_end=on_turn_end,
                                render_events=render_events,
                                render_arrivals=render_arrivals)
        else:
            run_turn_agentic(world, cfg, rng, result, counter, client_for, translator, knob_ai,
                             render_obs, system_prompt, msg_ids,
                             is_last=(t == cfg.world.total_turns),
                             parallel=parallel, on_turn_end=on_turn_end,
                             render_events=render_events,
                             render_arrivals=render_arrivals)
        if checkpoint_to is not None:
            # 매 턴 적는다. 한 턴이 12~40초인데 체크포인트는 밀리초라 값이 싸다.
            # `itertools.count` 는 현재 값을 못 읽으므로 하나 꺼내 보고 **그 값부터
            # 다시 시작**한다 — 꺼낸 것을 버리면 턴마다 id 가 하나씩 새어 나간다.
            nu, nm = next(counter), next(msg_ids)
            counter, msg_ids = itertools.count(nu), itertools.count(nm)
            checkpoint.save(checkpoint_to, world, rng, nu, nm)
    if last >= cfg.world.total_turns:
        result.final = final_survival(world, cfg, rng)
    else:
        # 운석이 아직 안 떨어졌다. 생존 판정을 돌리면 "요격 실패" 로 기록되어
        # 중간에 끊은 테스트가 멸망한 세계처럼 보인다.
        result.final = {"outcome": "truncated", "simulated_turns": last,
                        "impact_turn": cfg.world.total_turns,
                        "interceptor_best": result.interceptor_best}
    return result
