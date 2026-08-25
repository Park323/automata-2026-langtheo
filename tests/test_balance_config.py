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


def test_the_bunker_costs_more_per_head_or_nobody_coordinates(c):
    """**벙커가 더 비싸야 협력할 이유가 생긴다** (spec 3.5).

    8/25 에 「정확히 같게」 로 바꿨다가 되돌렸다 — 같으면 개인에게 비용도 결과(내가 산다)도
    같은데 협력에는 추가 비용(말하기·학습 AP)과 배신 위험이 얹힌다. 그러면 벙커가 지배한다.

    벙커의 유혹은 「싸다」 가 아니라 **「남이 필요 없다」** 다. 초과분이 불신의 가격이고,
    그 크기는 요격기를 골랐을 때 남는 AP 가 국제 발신을 살 수 있는지로 정해진다 —
    7,200 에서 그 경계가 노브 범위 안에 놓인다 (0.2 → 연 1.03통 · 0.5 → 0.41통).
    """
    # **같은 기간으로 나눈다** (#51). 벙커만 한 주기로 나누고 있었다 — 도구는 8/25 에
    # 고쳤는데 이 테스트가 남아 있었다.
    b1 = c.bunker / (c.agents * c.total_turns)
    i1 = c.interceptor / (3 * c.agents * c.total_turns)
    # **정확히 같다** (8/25 · Eddie). 한쪽이 싸면 그쪽이 정답이 되어 선택이 사라진다 —
    # 벙커가 임계가 된 뒤로 「벙커가 더 비싸야」 는 함정을 함정이 아니게 만든다.
    assert b1 > i1, (b1, i1)


def test_newborns_do_not_mint_money(c):
    """`initial_budget > 0` 이면 사망이 돈을 찍어낸다 — 런당 사망이 약 40회다.

    수명이 짧을수록 나라가 부유해지고 wellness(수명 연장)가 국가 소득을 깎는
    역설이 생긴다 (spec 7장).
    """
    assert c.initial_budget == 0


def test_expected_life_is_about_one_epoch(c):
    """주기(20턴)는 기대수명의 반올림이다. 둘이 어긋나면 계보 회전이 설계와 달라진다."""
    assert expected_life(c.surv_lambda, c.surv_k) == pytest.approx(16.06, abs=0.1)
    assert c.epoch_turns == 20


def test_a_lifetime_holds_several_conversation_round_trips(c):
    """**소통 왕복 하나에 두 턴이 든다** — 보내면 다음 턴에 도착한다.

    기대수명 8턴이던 시절엔 왕복 4회가 생애 전부라, 상대를 파악하고 조율에 이르는 것이
    구조적으로 불가능에 가까웠다. 이 부등식이 100턴·수명 2배 전환의 이유다.
    """
    round_trips = expected_life(c.surv_lambda, c.surv_k) / 2
    assert round_trips >= 8, f"생애 왕복 {round_trips:.1f}회 — 조율을 배우기엔 짧다"
