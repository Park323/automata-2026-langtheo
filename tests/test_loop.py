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
    assert survival.expected_life(lam, k) == pytest.approx(8.28, abs=0.02)
    assert survival.survival(10, lam, k) == pytest.approx(0.0099, abs=0.001)
    haz = [round(survival.hazard(a, lam, k), 2) for a in range(10)]
    assert haz == [0.00, 0.00, 0.00, 0.00, 0.01, 0.06, 0.17, 0.40, 0.70, 0.93]


# ── 합격 기준 표 ─────────────────────────────────────────────────────────────

def test_population_invariant(cfg):
    """1. 모든 턴에서 살아있는 에이전트가 정확히 9명."""
    r = _run(cfg, 1)
    assert all(n == 9 for n in r.alive_counts)


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


def test_procreate_child_budget(cfg):
    """8. procreate 자식 예산 = 부모의 남은 예산 / 자연사 자식 예산 = initial_budget."""
    r = _run(cfg, 5)
    saw_procreate = False
    for b in r.births:
        if b["born_by"] == "natural":
            assert b["budget"] == cfg.income.initial_budget
        else:
            saw_procreate = True
            assert b["budget"] >= 0      # 부모의 남은 예산 (음수 아님)
    assert saw_procreate                 # 더미는 age≥7 에서 procreate 하므로 반드시 발생


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
    counts = [run(cfg, random.Random(s), procreate_age=None).deaths for s in range(30)]
    assert 45 <= statistics.mean(counts) <= 55, f"평균 사망 {statistics.mean(counts):.1f}"


@pytest.mark.calibration
def test_lifespan(cfg):
    """3. 마지막 생존 나이 평균 7.1 ~ 7.4, 최대 12 이하 (자연사 격리).

    death_ages 는 '마지막 생존 나이'(죽는 턴의 age) 규약이다(spec 2.2).
    Σ_(a≥1) S(a) = 7.28. '살아낸 턴 수'(=기대수명 8.28)와는 1 만큼 다르다 — 후자는
    survival.expected_life() 가 담당한다.
    """
    ages = []
    for s in range(30):
        ages += run(cfg, random.Random(s), procreate_age=None).death_ages
    assert max(ages) <= 12
    assert 7.1 <= statistics.mean(ages) <= 7.4, f"평균 수명 {statistics.mean(ages):.2f}"
