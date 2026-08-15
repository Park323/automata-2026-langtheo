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
SNAP = ("config:\n  costs:\n    learn_base: 300\n"
        "  countries:\n"
        "    - {id: Asla, lang: ja}\n    - {id: Ranoa, lang: zh}\n"
        "    - {id: Miris, lang: fr}\nknob_ai: 24\nseed: 1\n")


def st(turn, agent, country, known, budget_start, age=0, parent=(), alive=True):
    return {"turn": turn, "agent": agent, "country": country, "age": age,
            "known_langs": list(known), "parent_langs": list(parent),
            "budget": budget_start, "budget_start": budget_start, "alive": alive,
            "born_turn": 0, "uid": 1}


def learn_ev(turn, agent, country, lang, charged, age=0):
    return {"turn": turn, "type": "learn", "agent": agent, "country": country,
            "target": "Ranoa", "lang": lang, "charged": charged,
            "rung": charged / BASE, "discount_domestic": False,
            "discount_parent": False, "age": age, "budget_after": 0.0, "lam": 8.26}


def write(tmp_path, state, events=()):
    (tmp_path / "config_snapshot.yaml").write_text(SNAP, encoding="utf-8")
    (tmp_path / "state.jsonl").write_text(
        "\n".join(json.dumps(r) for r in state), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in events), encoding="utf-8")
    return tmp_path


# ── 분모 ────────────────────────────────────────────────────────────────────────

def test_unaffordable_rung_is_not_an_opportunity(tmp_path):
    """감당할 수 없었으면 '안 배운 것' 이 아니다. 분모에 넣으면 x̂ 가 낮게 나온다."""
    d = write(tmp_path, [st(1, "A1", "Asla", ["ja"], budget_start=50)])
    obs, _ = xhat.observations(d)
    assert obs and obs[0]["cost"] == 300 and obs[0]["learned"] is False
    assert xhat.take_rates(obs) == {}          # 기회가 0건


def test_affordable_and_declined_counts(tmp_path):
    d = write(tmp_path, [st(1, "A1", "Asla", ["ja"], budget_start=500)])
    r = xhat.take_rates(xhat.observations(d)[0])
    assert r[(1.0, "all")] == {"n": 1, "learned": 0, "rate": 0.0}


def test_learned_rung_comes_from_the_log_not_a_guess(tmp_path):
    """실제 지불액이 곧 눈금이다. 할인 판정을 재현하다 어긋나면 x̂ 가 통째로 틀린다."""
    d = write(tmp_path,
              [st(1, "A1", "Asla", ["ja", "zh"], budget_start=400, age=3)],
              [learn_ev(1, "A1", "Asla", "zh", charged=150, age=3)])
    (o,) = xhat.observations(d)[0]
    assert o["learned"] is True and o["rung"] == 0.5 and o["age"] == 3


def test_trilingual_agent_has_no_opportunity(tmp_path):
    """배울 게 남아 있지 않으면 관측이 아니다."""
    d = write(tmp_path, [st(1, "A1", "Asla", ["ja", "zh", "fr"], budget_start=9999)])
    assert xhat.observations(d)[0] == []


def test_missing_budget_start_is_reported_not_silently_dropped(tmp_path):
    """이 필드 이전의 런은 x̂ 를 낼 수 없다. 조용히 빼면 n 이 작아진 줄만 안다."""
    row = st(1, "A1", "Asla", ["ja"], budget_start=500)
    del row["budget_start"]
    d = write(tmp_path, [row])
    obs, diag = xhat.observations(d)
    assert obs == [] and diag["no_budget_start"] == 1


# ── 할인 재현 ───────────────────────────────────────────────────────────────────

def test_domestic_speaker_halves_the_rung(tmp_path):
    """같은 나라에 구사자가 있으면 그 사람의 눈금은 L/2 다 (spec 3.4)."""
    d = write(tmp_path, [
        st(1, "A1", "Asla", ["ja", "zh"], budget_start=500),   # zh 구사자
        st(1, "A2", "Asla", ["ja"], budget_start=500),
        st(2, "A1", "Asla", ["ja", "zh"], budget_start=500),
        st(2, "A2", "Asla", ["ja"], budget_start=500),
    ])
    o = [x for x in xhat.observations(d)[0] if x["turn"] == 2 and x["agent"] == "A2"]
    assert o[0]["rung"] == 0.5


def test_same_turn_learner_is_not_a_discount_source(tmp_path):
    """state 는 턴 끝 상태다. 그 턴에 배운 사람을 할인 근거로 쓰면 없던 할인이 생긴다.

    A1 이 턴 2에 zh 를 배웠다면, 턴 2의 A2 는 아직 정가(L)를 마주하고 있었다.
    """
    d = write(tmp_path, [
        st(1, "A1", "Asla", ["ja"], budget_start=500),
        st(1, "A2", "Asla", ["ja"], budget_start=500),
        st(2, "A1", "Asla", ["ja", "zh"], budget_start=500),   # 이번 턴에 배움
        st(2, "A2", "Asla", ["ja"], budget_start=500),
    ], [learn_ev(2, "A1", "Asla", "zh", charged=300)])
    o = [x for x in xhat.observations(d)[0] if x["turn"] == 2 and x["agent"] == "A2"]
    assert o[0]["rung"] == 1.0


def test_parent_language_halves_the_rung(tmp_path):
    d = write(tmp_path, [st(1, "A1", "Asla", ["ja"], budget_start=500, parent=["zh"])])
    (o,) = xhat.observations(d)[0]
    assert o["rung"] == 0.5


def test_cheapest_rung_wins_when_two_targets_differ(tmp_path):
    """둘 다 배울 수 있으면 **가장 싼** 눈금이 그가 마주한 가격이다."""
    d = write(tmp_path, [st(1, "A1", "Asla", ["ja"], budget_start=500, parent=["fr"])])
    (o,) = xhat.observations(d)[0]
    assert o["rung"] == 0.5 and o["cost"] == 150


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
    ], [learn_ev(1, "A1", "Asla", "zh", charged=300, age=1)])
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


def test_snap_rung_absorbs_float_error():
    assert xhat._snap_rung(74.999999, BASE) == 0.25
    assert xhat._snap_rung(150.0000001, BASE) == 0.5


@pytest.mark.parametrize("age,band", [(0, "0-2"), (2, "0-2"), (3, "3-5"), (9, "6+")])
def test_band_of(age, band):
    assert xhat.BAND_NAME[xhat.band_of(age)] == band
