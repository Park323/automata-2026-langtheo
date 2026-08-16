"""확정 config 가 Phase 0 의 창 안에 있는가. spec 7 · 12.4.

`core/asserts.py` 가 런 시작 때 같은 부등식을 검사하지만, 그건 **런을 돌려야** 걸립니다.
여기서 잡으면 config 를 건드린 커밋에서 바로 걸립니다 — 창 밖으로 나간 값으로 밤새
배치를 돌리고 아침에 아는 것이 가장 비쌉니다.

`w*` 는 몬테카를로라 느려서 테스트에 넣지 않습니다. `tools/balance/verify_config.py`
가 그것까지 봅니다.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.balance.sweep import POLICY_COEF, bounds, expected_life, passes_asserts
from tools.balance.verify_config import cfg_from_yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def c():
    return cfg_from_yaml(ROOT / "configs" / "base.yaml")


def test_config_passes_every_bound(c):
    good, why = passes_asserts(c)
    assert good, why


def test_interceptor_sits_inside_the_window(c):
    """창이 열려 있어야 하고, 임계는 그 안에 있어야 한다."""
    A, B, C, E = bounds(c)
    lo, hi = max(A, B, E), C * POLICY_COEF
    assert lo < hi, f"창이 닫혔다 [{lo:.0f}, {hi:.0f}] — 어떤 임계도 조건을 다 만족 못 한다"
    assert lo < c.interceptor < hi


def test_bunker_is_a_trap_not_a_bargain(c):
    """1인부담이 요격기보다 싸면 벙커가 함정이 아니라 정답이 된다 (spec 3.5)."""
    b1 = c.bunker / (c.agents * c.epoch_turns)
    i1 = c.interceptor / (3 * c.agents * c.total_turns)
    assert b1 > i1


def test_newborns_do_not_mint_money(c):
    """`initial_budget > 0` 이면 사망이 돈을 찍어낸다 — 런당 사망이 약 40회다.

    수명이 짧을수록 나라가 부유해지고 wellness(수명 연장)가 국가 소득을 깎는
    역설이 생긴다 (spec 7장).
    """
    assert c.initial_budget == 0


def test_expected_life_is_about_one_epoch(c):
    """주기(10턴)는 기대수명의 반올림이다. 둘이 어긋나면 계보 회전이 설계와 달라진다."""
    assert expected_life(c.surv_lambda, c.surv_k) == pytest.approx(8.28, abs=0.1)
    assert c.epoch_turns == 10
