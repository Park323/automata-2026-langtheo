"""지표 산출. spec 8.2.

여기서 고정하는 것은 **분모**다. 지표가 틀리는 방식은 거의 항상 분모가 틀리는 것이고,
분모는 틀려도 그럴듯한 수가 나와서 로그를 봐도 티가 나지 않는다.
"""
from __future__ import annotations

import json

import pytest

from tools.score import metrics


def msg(msg_id, turn, frm, to, route, src="zh", dst="fr", ok=True,
        sent="A", delivered="B", truncated=False, cut=0):
    return {"msg_id": msg_id, "turn": turn, "from": frm, "to": to, "route": route,
            "delivered": ok,
            "meta": {"src_lang": src, "dst_lang": dst, "text_sent": sent,
                     "text_delivered": delivered, "reader": True,
                     "truncated": truncated, "chars_cut": cut}}


def turn_event(turn, agent, *actions):
    return {"turn": turn, "type": "agent_turn", "agent": agent, "actions": list(actions)}


# ── 표본 없음 ≠ 0 ───────────────────────────────────────────────────────────────

def test_rate_is_none_when_no_sample():
    """0.0 을 돌려주면 '실패가 없었다' 로 읽힌다. 표본이 없는 것과는 다르다."""
    assert metrics._rate(0, 0) is None
    assert metrics._rate(0, 5) == 0.0


def test_intent_metrics_without_judging_is_none():
    """judge.py 를 안 돌렸는데 4a 가 0% 로 나오면 '오해가 없었다' 로 오독된다."""
    m = metrics.intent_metrics([])
    assert m["4a"]["fail_rate"] is None and m["judged"] is False
    assert m["4a_minus_4c"] is None


# ── 지표 3 의 분모 ──────────────────────────────────────────────────────────────

def test_policy_shift_denominator_is_person_turn_pairs():
    """메시지 수로 나누면 AI 가 싼 조건에서 분모만 부풀어 지표가 기계적으로 내려간다.

    한 사람이 한 턴에 국제 메시지를 3건 받아도 그건 **쌍 1개**다.
    """
    ms = [msg(i, 1, "Ranoa1", "Miris1", "ai") for i in (1, 2, 3)]
    ev = [turn_event(2, "Miris1", "propose_vote")]
    r = metrics.policy_shift(ms, ev)
    assert r["n_pairs"] == 1 and r["3"] == 1.0


def test_policy_shift_counts_receive_turn_not_send_turn():
    """도착은 발신 다음 턴이다. 발신 턴에서 세면 한 턴씩 어긋난다."""
    ms = [msg(1, 3, "Ranoa1", "Miris1", "ai")]
    assert metrics.policy_shift(ms, [turn_event(3, "Miris1", "propose_vote")])["3"] == 0.0
    assert metrics.policy_shift(ms, [turn_event(4, "Miris1", "propose_vote")])["3"] == 1.0
    assert metrics.policy_shift(ms, [turn_event(5, "Miris1", "propose_vote")])["3_lag"] == 1.0


def test_policy_shift_ignores_domestic():
    """지표 3 은 **국제** 메시지가 정책을 움직였는가다."""
    ms = [msg(1, 1, "Miris2", "Miris1", "domestic", src="fr")]
    assert metrics.policy_shift(ms, [turn_event(2, "Miris1", "propose_vote")])["n_pairs"] == 0


def test_undelivered_message_is_not_in_denominator():
    """못 받은 메시지가 정책을 움직일 수는 없다."""
    ms = [msg(1, 1, "Ranoa1", "Miris1", "original", ok=False)]
    assert metrics.policy_shift(ms, [])["n_pairs"] == 0


# ── 메시지 구성 ─────────────────────────────────────────────────────────────────

def test_pair_dist_covers_international_only():
    """언어쌍 분포는 6방향이다. 국내(자국어→자국어)를 섞으면 구성비가 희석된다."""
    ms = [msg(1, 1, "Ranoa1", "Miris1", "ai", src="zh", dst="fr"),
          msg(2, 1, "Miris1", "Ranoa1", "ai", src="fr", dst="zh"),
          msg(3, 1, "Miris2", "Miris1", "domestic", src="fr", dst="fr")]
    s = metrics.message_shape(ms)
    assert s["pair_dist"] == {"zh→fr": 0.5, "fr→zh": 0.5}
    assert s["n"] == {"total": 3, "ai": 2, "domestic": 1}


def test_original_attempt_and_delivery_failure():
    """지표 10 의 분모는 국제 발신, 지표 9 의 분모는 원문 직통 시도다. 서로 다르다."""
    ms = [msg(1, 1, "Ranoa1", "Miris1", "ai"),
          msg(2, 1, "Ranoa1", "Miris1", "original", ok=False),
          msg(3, 1, "Ranoa1", "Miris1", "original", ok=True)]
    s = metrics.message_shape(ms)
    assert s["10_original_attempt"] == {"n_intl": 3, "rate": pytest.approx(0.6667, abs=1e-4)}
    assert s["9_delivery_failure"] == {"n": 2, "rate": 0.5}


def test_self_censor_split_by_route_and_lang():
    """절단은 경로별·언어별로 봐야 한다 — fr 은 상한 400자, zh 는 90자다."""
    ms = [msg(1, 1, "Ranoa1", "Miris1", "ai", src="zh", truncated=True, cut=20),
          msg(2, 1, "Miris2", "Miris1", "domestic", src="fr")]
    s = metrics.message_shape(ms)["8_self_censor"]
    assert s["by_route"]["ai"]["rate"] == 1.0
    assert s["by_route"]["domestic"]["rate"] == 0.0
    assert s["by_lang"]["zh"]["mean_chars_cut"] == 20.0
    assert s["by_lang"]["ja"]["rate"] is None       # 표본 없음


# ── 학습자 ──────────────────────────────────────────────────────────────────────

def _state(turn, agent, langs, age=0, alive=True):
    return {"turn": turn, "agent": agent, "known_langs": langs, "age": age, "alive": alive}


def test_learner_rate_needs_two_languages():
    """`known_langs` 는 모국어를 포함한다. 1개는 학습자가 아니다."""
    st = [_state(1, "A1", ["zh"]), _state(1, "A2", ["zh", "fr"])]
    assert metrics.learner_rate(st)["rate"] == 0.5


def test_learner_rate_uses_last_window_and_living_only():
    st = [_state(1, "A1", ["zh", "fr"]), _state(9, "A1", ["zh"]),
          _state(10, "A1", ["zh"]), _state(10, "A2", ["zh", "fr"], alive=False)]
    r = metrics.learner_rate(st, window=2)
    assert r["turns"] == "9-10" and r["n_agent_turns"] == 2 and r["rate"] == 0.0


def test_learner_rate_stratifies_by_age():
    """학습은 누적이라 늙을수록 높다. 층화 없이는 인구 구성 변화가 지표로 위장된다."""
    st = [_state(1, "A1", ["zh"], age=0), _state(1, "A2", ["zh", "fr"], age=7)]
    by = metrics.learner_rate(st)["by_age"]
    assert by["0-2"]["rate"] == 0.0 and by["6-8"]["rate"] == 1.0


# ── 조건 식별 ───────────────────────────────────────────────────────────────────

def test_knob_comes_from_run_arg_not_config_list(tmp_path):
    """config 의 knob 은 **스윕 목록**이다. 그것만 보면 조건을 특정할 수 없다."""
    (tmp_path / "messages.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "config_snapshot.yaml").write_text(
        "config:\n  knob:\n    comm_intl_ai: [6, 12, 24, 48]\n  run:\n    seed: 1\n"
        "knob_ai: 24\nseed: 71\n", encoding="utf-8")
    r = metrics.score_run(tmp_path)
    assert r["knob"] == 24 and r["seed"] == 71


def test_outcome_metrics():
    assert metrics.outcome_metrics(
        {"final": {"outcome": "all_survive", "interceptor_best": 9000}}
    ) == {"2_avoided": 1, "2p_survivor_share": 1.0,
          "5_interceptor_best": 9000, "outcome": "all_survive"}
    r = metrics.outcome_metrics(
        {"final": {"outcome": "intercept_failed", "interceptor_best": 40,
                   "survivors": ["Ranoa"]}})
    assert r["2_avoided"] == 0 and r["2p_survivor_share"] == pytest.approx(0.3333, abs=1e-4)


def test_score_run_survives_missing_files(tmp_path):
    """산출물이 일부만 있어도 있는 것까지는 내야 한다 — 45턴에서 죽은 런도 읽는다."""
    (tmp_path / "messages.jsonl").write_text(
        json.dumps(msg(1, 1, "Ranoa1", "Miris1", "ai")), encoding="utf-8")
    r = metrics.score_run(tmp_path)
    assert r["n"]["total"] == 1
    assert r["4"]["judged"] is False and r["2_avoided"] == 0
    assert metrics.format_run(r)          # 표 렌더가 터지지 않는다
