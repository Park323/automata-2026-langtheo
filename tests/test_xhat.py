"""`x̂` 추정. spec 7 · 8.4.

`x̂` 는 지표가 아니라 **사후 추정치**다. 그래서 여기서 지키는 것은 값이 아니라
**추정을 망치는 방식들**이다 — 분모를 잘못 잡는 것, 없는 정밀도를 지어내는 것,
같은 턴에 배운 사람을 그 턴의 할인 근거로 쓰는 것.
"""
from __future__ import annotations

import json

import pytest

from tools.score import xhat

BASE = 300.0
SNAP = ("config:\n  costs:\n    learn_base: 300\n    learn_speedup: 0.5\n"
        "  countries:\n"
        "    - {id: Asla, lang: ja}\n    - {id: Ranoa, lang: zh}\n"
        "    - {id: Miris, lang: fr}\nknob_ai: 24\nseed: 1\n")


def st(turn, agent, country, known, budget_start, age=0, parent=(), alive=True):
    return {"turn": turn, "agent": agent, "country": country, "age": age,
            "known_langs": list(known), "parent_langs": list(parent),
            "budget": budget_start, "budget_start": budget_start, "alive": alive,
            "born_turn": 0, "uid": 1}


def pay_ev(turn, agent, country, lang, charged, speed=1.0, age=0):
    """납부 1건. `charged` 는 **이번에 낸 액**이고 눈금은 `speed` 에서 읽는다.

    **`required` 가 아니다** (8/23). 가속 모델에서 필요액은 늘 `learn_base` 이고,
    달라지는 것은 배속이다 — 총 지출은 `learn_base / speed` 다.
    """
    return {"turn": turn, "type": "learn", "agent": agent, "country": country,
            "target": "Ranoa", "lang": lang, "charged": charged, "required": BASE,
            "progress_before": 0.0, "speed": speed, "discount_domestic": False,
            "discount_parent": False, "age": age, "budget_after": 0.0, "lam": 8.26}


def acq_ev(turn, agent, country, lang, total, speed=1.0, age=0):
    """습득 1건. `charged` 는 **누적 진척**이다 (총 지불액이 아니다)."""
    return {"turn": turn, "type": "learn", "kind": "acquired", "agent": agent,
            "country": country, "target": "Ranoa", "lang": lang, "charged": total,
            "required": BASE, "speed": speed, "age": age,
            "discount_domestic": False, "discount_parent": False, "budget_after": 0.0}


def write(tmp_path, state, events=()):
    (tmp_path / "config_snapshot.yaml").write_text(SNAP, encoding="utf-8")
    (tmp_path / "state.jsonl").write_text(
        "\n".join(json.dumps(r) for r in state), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in events), encoding="utf-8")
    return tmp_path


# ── 분모 ────────────────────────────────────────────────────────────────────────


# **`test_no_budget_is_not_an_opportunity` 를 지웠다** (8/25 · AP 전면 통일) — 분모가 AP 다 — 살아 있으면 늘 참.

def test_having_money_and_not_paying_counts(tmp_path):
    """전액에 못 미쳐도 기회다 — 50원으로도 시작할 수 있다."""
    d = write(tmp_path, [st(1, "A1", "Asla", ["ja"], budget_start=50)])
    r = xhat.take_rates(xhat.observations(d)[0])
    assert r[(1.0, "all")] == {"n": 1, "put_in": 0, "paid": 0.0, "rate": 0.0}


def test_rung_comes_from_the_speed_not_from_the_instalment(tmp_path):
    """**눈금은 배속에서 읽는다** (8/23). 낸 액으로 읽으면 분할 납부가 없는 눈금을 만든다.

    `required` 로 읽던 시절의 테스트였는데, 8/22 에 할인이 가속으로 바뀌면서 `required`
    가 **늘 `learn_base`** 가 됐다 — 그날부터 눈금이 1.0 하나로 붕괴했고, 이 테스트는
    픽스처가 `required=150` 을 손으로 써서 그 붕괴를 못 봤다.
    """
    d = write(tmp_path,
              [st(1, "A1", "Asla", ["ja"], budget_start=400, age=3)],
              [pay_ev(1, "A1", "Asla", "zh", charged=200, speed=2.0, age=3)])
    (o,) = xhat.observations(d)[0]
    assert o["put_in"] is True and o["rung"] == 0.5 and o["paid"] == 200 and o["age"] == 3
    assert o["cost"] == BASE / 2                      # 총 지출은 L/배속


def test_acquisitions_are_counted_separately(tmp_path):
    """납부율은 '시작했는가', 습득은 '끝냈는가'. 분할 납부에서는 둘이 갈린다."""
    d = write(tmp_path,
              [st(1, "A1", "Asla", ["ja"], budget_start=400),
               st(2, "A1", "Asla", ["ja", "zh"], budget_start=400)],
              [pay_ev(1, "A1", "Asla", "zh", charged=150, speed=2.0),
               acq_ev(2, "A1", "Asla", "zh", total=BASE, speed=2.0)])
    obs, diag, acq = xhat.observations(d)
    assert diag["learns"] == 1 and diag["acquired"] == 1
    assert xhat.acquisitions(acq, BASE) == {(0.5, "all"): 1}


def test_trilingual_agent_has_no_opportunity(tmp_path):
    """배울 게 남아 있지 않으면 관측이 아니다."""
    d = write(tmp_path, [st(1, "A1", "Asla", ["ja", "zh", "fr"], budget_start=9999)])
    assert xhat.observations(d)[0] == []



# **`test_missing_budget_start_is_reported_not_silently_dropped` 를 지웠다** (8/25 · AP 전면 통일) — 같다.

def test_domestic_speaker_speeds_up_the_rung(tmp_path):
    """같은 나라에 구사자가 있으면 배속 1.5 → 총 지출 L/1.5 다 (spec 3.4 · 가속 8/22).

    **곱이 아니라 합이다.** 전에는 `×0.5` 였고 사유 둘이면 `×0.25` 였다. 지금은 사유가
    배속에 더해져 1 + 0.5×사유 이고, 눈금은 그 역수다.
    """
    d = write(tmp_path, [
        st(1, "A1", "Asla", ["ja", "zh"], budget_start=500),   # zh 구사자
        st(1, "A2", "Asla", ["ja"], budget_start=500),
        st(2, "A1", "Asla", ["ja", "zh"], budget_start=500),
        st(2, "A2", "Asla", ["ja"], budget_start=500),
    ])
    o = [x for x in xhat.observations(d)[0] if x["turn"] == 2 and x["agent"] == "A2"]
    assert o[0]["rung"] == pytest.approx(1 / 1.5, abs=1e-3)


def test_same_turn_learner_is_not_a_discount_source(tmp_path):
    """state 는 턴 끝 상태다. 그 턴에 배운 사람을 할인 근거로 쓰면 없던 할인이 생긴다.

    A1 이 턴 2에 zh 를 배웠다면, 턴 2의 A2 는 아직 정가(L)를 마주하고 있었다.
    """
    d = write(tmp_path, [
        st(1, "A1", "Asla", ["ja"], budget_start=500),
        st(1, "A2", "Asla", ["ja"], budget_start=500),
        st(2, "A1", "Asla", ["ja", "zh"], budget_start=500),   # 이번 턴에 배움
        st(2, "A2", "Asla", ["ja"], budget_start=500),
    ], [pay_ev(2, "A1", "Asla", "zh", charged=300)])
    o = [x for x in xhat.observations(d)[0] if x["turn"] == 2 and x["agent"] == "A2"]
    assert o[0]["rung"] == 1.0


def test_parent_language_speeds_up_the_rung(tmp_path):
    d = write(tmp_path, [st(1, "A1", "Asla", ["ja"], budget_start=500, parent=["zh"])])
    (o,) = xhat.observations(d)[0]
    assert o["rung"] == pytest.approx(1 / 1.5, abs=1e-3)


def test_two_reasons_add_up_they_do_not_multiply(tmp_path):
    """**사유 둘이면 배속 2.0 · 눈금 L/2 다.** 할인 시절엔 곱해서 L/4 였다 (8/22 폐지).

    이 구별이 x̂ 의 가장 싼 눈금을 정하므로, 틀리면 구간의 하한이 통째로 어긋난다.
    """
    d = write(tmp_path, [
        st(1, "A1", "Asla", ["ja", "zh"], budget_start=500),          # 국내 zh 구사자
        st(1, "A2", "Asla", ["ja"], budget_start=500, parent=["zh"]),  # + 부모도 zh
        st(2, "A1", "Asla", ["ja", "zh"], budget_start=500),
        st(2, "A2", "Asla", ["ja"], budget_start=500, parent=["zh"]),
    ])
    o = [x for x in xhat.observations(d)[0] if x["turn"] == 2 and x["agent"] == "A2"]
    assert o[0]["rung"] == 0.5 and o[0]["cost"] == BASE / 2


def test_cheapest_rung_wins_when_two_targets_differ(tmp_path):
    """둘 다 배울 수 있으면 **가장 싼** 눈금이 그가 마주한 가격이다."""
    d = write(tmp_path, [st(1, "A1", "Asla", ["ja"], budget_start=500, parent=["fr"])])
    (o,) = xhat.observations(d)[0]
    assert o["rung"] == pytest.approx(1 / 1.5, abs=1e-3)
    assert o["cost"] == pytest.approx(BASE / 1.5, abs=1e-2)


# ── 구간 ────────────────────────────────────────────────────────────────────────

def test_bracket_closes_when_a_rung_turns_off():
    rates = {(0.5, "all"): {"n": 10, "learned": 8, "rate": 0.8},
             (1.0, "all"): {"n": 10, "learned": 1, "rate": 0.1}}
    b = xhat.bracket(rates)
    assert (b["lower"], b["upper"]) == (0.5, 1.0)
    assert b["label"] == "L/2 ≤ x < L"


def test_bracket_stays_open_when_no_rung_is_off():
    rates = {(1.0, "all"): {"n": 10, "learned": 9, "rate": 0.9}}
    assert xhat.bracket(rates)["label"] == "x ≥ L"


def test_bracket_stays_open_when_nothing_is_on():
    rates = {(1.0, "all"): {"n": 10, "learned": 0, "rate": 0.0}}
    b = xhat.bracket(rates)
    assert b["lower"] is None and b["label"] == "x < L"


def test_absent_rung_is_not_a_switched_off_rung():
    """`L/4` 는 procreate 로 태어나야 나온다 (3.4). 표본 없음을 '꺼짐' 으로 읽으면
    닫히지 않은 구간을 닫힌 것처럼 적게 된다 — 없는 정밀도를 지어내는 것이다."""
    rates = {(1.0, "all"): {"n": 10, "learned": 9, "rate": 0.9}}
    b = xhat.bracket(rates)
    assert b["rungs_seen"] == ["L"] and b["upper"] is None


def test_bracket_is_empty_without_samples():
    assert xhat.bracket({})["label"] == "표본 없음"


# ── 층화 · Δx ───────────────────────────────────────────────────────────────────

def test_age_bands_split_the_estimate(tmp_path):
    """늙으면 회수 기간이 없어 같은 눈금이 사실상 더 비싸다. 섞으면 구분이 안 된다."""
    d = write(tmp_path, [
        st(1, "A1", "Asla", ["ja", "zh"], budget_start=500, age=1),
        st(1, "A2", "Asla", ["ja"], budget_start=500, age=8),
    ], [pay_ev(1, "A1", "Asla", "zh", charged=300, age=1)])
    est = xhat.estimate([d], by_age=True)
    assert est["brackets"]["0-2"]["lower"] == 1.0        # 젊은 쪽은 켜짐
    assert est["brackets"]["6+"]["lower"] is None        # 늙은 쪽은 꺼짐


def test_delta_x_needs_both_bounds():
    """한쪽 하한이 열려 있으면 뺄 수 없다. 없는 수를 지어내지 않는다."""
    lo = {"brackets": {"all": {"lower": 1.0}}}
    hi = {"brackets": {"all": {"lower": None}}}
    assert xhat.delta_x({6: lo, 48: hi})["delta"] is None
    assert xhat.delta_x({6: lo, 48: {"brackets": {"all": {"lower": 0.5}}}}) == {
        "delta": 0.5, "knob_low": 6, "knob_high": 48, "unit": "L 배수",
        "note": "양수면 AI 가 쌀수록 학습의 암묵효용이 낮다는 뜻입니다"}


def test_delta_x_needs_two_knobs():
    assert xhat.delta_x({6: {"brackets": {"all": {"lower": 1.0}}}})["delta"] is None


def test_format_does_not_crash_on_empty(tmp_path):
    d = write(tmp_path, [])
    assert xhat.format_estimate(xhat.estimate([d]), knob=24)


def test_rungs_come_from_the_speedup():
    """눈금표를 상수로 두면 규칙이 바뀔 때 조용히 거짓이 된다 — 8/22 에 그랬다."""
    assert xhat.rungs_for(0.5) == (0.5, 0.6667, 1.0)
    assert xhat.rungs_for(1.0) == (0.3333, 0.5, 1.0)
    assert [xhat.rung_name(r) for r in xhat.rungs_for(0.5)] == ["L/2", "L/1.5", "L"]


@pytest.mark.parametrize("age,band", [(0, "0-2"), (2, "0-2"), (3, "3-5"), (9, "6+")])
def test_band_of(age, band):
    assert xhat.BAND_NAME[xhat.band_of(age)] == band
