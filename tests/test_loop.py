"""턴 루프와 경계 조건. 과제 1 Part B. 전부 LLM 없이 검증된다."""
from __future__ import annotations

import random
import statistics
from pathlib import Path

import pytest

from core import config, survival
from core.loop import run

BASE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


@pytest.fixture(scope="module")
def cfg():
    return config.load(BASE)


def _run(cfg, seed):
    return run(cfg, random.Random(seed))


# ── survival 단위 검증 (B-1) ─────────────────────────────────────────────────

def test_survival_numbers(cfg):
    lam, k = cfg.survival.lambda_base, cfg.survival.k
    # 수명 2배 (lambda 8.26 → 16.52). **소통 왕복 하나에 두 턴이 든다** —
    # 기대수명 8턴이면 왕복 4회가 생애 전부라 조율을 배울 시간이 구조적으로 없다.
    assert survival.expected_life(lam, k) == pytest.approx(16.06, abs=0.02)
    assert survival.survival(20, lam, k) == pytest.approx(0.0099, abs=0.001)
    haz = [round(survival.hazard(a, lam, k), 2) for a in range(0, 20, 2)]
    assert haz == [0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.07, 0.18, 0.38, 0.66]


# ── 합격 기준 표 ─────────────────────────────────────────────────────────────

def test_population_only_grows_and_never_dips(cfg):
    """**인구는 이제 늘어난다** (8/21).

    전에는 늘 9명이었다 — `procreate` 가 부모를 죽이고 그 자리를 아이가 대신했으므로
    재생산이 **죽음의 형식**이었다. `bear_child` 는 부모를 죽이지 않으므로 사람이 늘고,
    자연사는 그대로 자리를 채운다(교체). 그래서 **바닥은 지켜지고 위로만 열린다.**

    바닥이 지켜지는 것이 중요하다 — 우연히 수명이 짧게 나온 세대가 겹치면 나라가 비어
    버릴 수 있고, 그러면 조율을 관측할 상대가 사라진다.
    """
    n0 = cfg.world.agents_per_country * len(cfg.world.countries)
    r = _run(cfg, 1)
    assert r.alive_counts, "턴별 생존 수가 기록돼야 한다"
    assert min(r.alive_counts) >= n0, f"바닥이 깨졌다: {min(r.alive_counts)} < {n0}"
    assert r.alive_counts == sorted(r.alive_counts), "줄어드는 구간이 없어야 한다"


def test_reproducibility(cfg):
    """5. 같은 seed 로 두 번 돌려 state 로그가 바이트 단위로 동일."""
    a = _run(cfg, 42).state_log
    b = _run(cfg, 42).state_log
    assert a == b
    assert a.encode("utf-8") == b.encode("utf-8")


def test_budget_nonneg(cfg):
    """6. 어느 턴에서도 budget < 0 이 없다. (state 로그로 확인)"""
    r = _run(cfg, 7)
    import json
    for line in r.state_lines:
        rec = json.loads(line)
        for _, fields in rec["agents"].items():
            assert fields[2] >= 0        # [country, age, budget, alive, born_turn, born_by]


def test_newborn_no_action(cfg):
    """7. 태어난 턴에 행동 기록이 없다 (관측 스냅샷에 없었으므로).

    슬롯 id 는 세대마다 재사용되므로 인스턴스 uid 로 확인한다.
    """
    r = _run(cfg, 3)
    for b in r.births:
        acted_this_turn = r.acted[b["turn"] - 1]   # 이번 턴 행동한 uid 집합
        assert b["uid"] not in acted_this_turn



def test_dummy_intercept_fails(cfg):
    """요격기 판정이 max(부지별 독립)이므로, 더미(3국 각자 자국 요격기)는 실패해야 한다.

    합산(sum)으로 되돌아가면 이 테스트가 깨진다 — 조율 무의미화 회귀 방지.
    """
    r = _run(cfg, 1)
    assert r.interceptor_best < cfg.thresholds.interceptor
    assert r.final["outcome"] == "intercept_failed"
    assert r.final["survivors"] == []


def test_land_exclusivity(cfg):
    """9. land 가 한번 정해지면 투표 없이 바뀌지 않는다."""
    r = _run(cfg, 1)
    import json
    last_land = {}
    for line in r.state_lines:
        rec = json.loads(line)
        for cid, fields in rec["countries"].items():
            land = fields[0]             # [land, progress, national_capital]
            if land is not None:
                if cid in last_land:
                    assert land == last_land[cid]
                last_land[cid] = land


# ── 캘리브레이션 (여러 시드 평균) ─────────────────────────────────────────────
# 참고: 더미 정책이 age≥7 에서 procreate 하므로, 자연사만 가정한 기대치(사망 46~55,
# 수명 8.2~8.4)와 어긋날 수 있다. 어긋나면 아래가 실패하고, 그 사실 자체가 보고 대상.

@pytest.mark.calibration
def test_death_count(cfg):
    """2. 런당 사망 수 45 ~ 55 (자연사 격리).

    수명 모델을 격리해 재려면 더미의 procreate 를 꺼야 한다(procreate_age=None) —
    procreate(age≥7)를 켜면 수명이 잘려 사망이 더 잦아진다.
    """
    # **60턴으로 줄였다** (8/21) — 사망 기대치도 그만큼 줄어든다. 턴당 사망률로 재서
    # total_turns 를 바꿀 때 이 테스트가 낡지 않게 한다.
    counts = [run(cfg, random.Random(s), procreate_age=None).deaths for s in range(30)]
    per_turn = statistics.mean(counts) / cfg.world.total_turns
    n0 = cfg.world.agents_per_country * len(cfg.world.countries)
    # 기대수명 ~16해이므로 9명이면 턴당 9/16 ≈ 0.56 명이 죽는다
    assert 0.45 <= per_turn <= 0.65, (
        f"턴당 사망 {per_turn:.2f} (평균 {statistics.mean(counts):.1f} / "
        f"{cfg.world.total_turns}턴, 초기 {n0}명)")


@pytest.mark.calibration
def test_lifespan(cfg):
    """3. 마지막 생존 나이 평균 14.6 ~ 15.5, 최대 24 이하 (자연사 격리).

    death_ages 는 '마지막 생존 나이'(죽는 턴의 age) 규약이다(spec 2.2).
    Σ_(a≥1) S(a) = 15.06. '살아낸 턴 수'(=기대수명 16.06)와는 1 만큼 다르다 — 후자는
    survival.expected_life() 가 담당한다.
    """
    ages = []
    for s in range(30):
        ages += run(cfg, random.Random(s), procreate_age=None).death_ages
    assert max(ages) <= 24
    assert 14.6 <= statistics.mean(ages) <= 15.5, f"평균 수명 {statistics.mean(ages):.2f}"


def test_a_natural_death_passes_the_budget_and_the_discount(cfg):
    """**자연사가 후손을 남기고, 물려준다** (8/22 개정).

    재생산 행위(`bear_child`)를 없앴다 — 10해 실측 네 번에서 0건이었고, 원인이 유인이 아니라
    인구 구조였다. 성인 기간이 성장 기간보다 짧아 세대가 이어지지 않았다.

    넘어가는 것은 둘이다.

        예산       죽음이 돈을 태우지 않는다
        할인 자격   `parent_langs` — 앞사람이 알던 말이 뒷사람에게 싸진다 (3.4)

    **언어 자체는 안 넘어간다** (3.3). 이중언어자 멸종은 그대로 관측된다 — 그것이 이 실험의
    관심사다.
    """
    import itertools
    import random

    from core import loop
    w = loop.init_world(cfg, itertools.count(1), random.Random(1))
    w.turn = 5                                   # 후손을 born_turn 으로 찾을 수 있게
    before = set(w.agents)
    a = w.agents["Asla1"]
    a.known_langs, a.lang_progress = {"ja", "zh"}, {"fr": 100.0}
    a.memory = "내 메모"

    r = loop.RunResult(world=w)
    # hazard 를 1 로 만들어 반드시 죽게 한다
    a.lam = 0.001
    loop._death_birth(w, cfg, random.Random(0), ["Asla1"], set(),
                      itertools.count(900), r)
    assert "Asla1" not in w.agents
    (heir,) = [w.agents[k] for k in set(w.agents) - before]

    assert heir.parent_langs == {"ja", "zh"}     # 할인 자격도
    assert heir.known_langs == {"ja"}            # **언어 자체는 아니다**
    assert heir.lang_progress == {}              # 진척도 아니다
    assert heir.memory == ""                     # 메모도 아니다
    assert heir.age == 0

    (d,) = r.deaths_log
    assert d["by"] == "natural"
