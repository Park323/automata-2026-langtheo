"""의도 전달 2단계 판정. spec 6.2 · 지표 4.

판정 자체는 LLM 이라 비결정적이다. 여기서 고정하는 것은 **판정 주변의 규칙**이다 —
무엇이 분모에 들어가고 무엇이 빠지는가, 판정자가 지어낸 근거를 어떻게 막는가,
그리고 실패율을 어떻게 층으로 나누는가. 이것들이 틀리면 4a 는 조용히 틀린 수를 낸다.
"""
from __future__ import annotations

import json

import pytest

from core.llm import StubClient
from tools.score import judge


def msg(msg_id, turn, frm, to, route, sent, delivered, src="zh", dst="fr",
        ok=True, reader=True):
    return {"msg_id": msg_id, "turn": turn, "from": frm, "to": to, "route": route,
            "delivered": ok,
            "meta": {"src_lang": src, "dst_lang": dst, "text_sent": sent,
                     "text_delivered": delivered, "reader": reader}}


def turn_event(turn, agent, *reasons):
    return {"turn": turn, "type": "agent_turn", "agent": agent,
            "reasonings": [{"tool": "speak", "reasoning": r} for r in reasons],
            "actions": ["speak"]}


def reply(obj: dict) -> dict:
    return {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)}


# ── 링크 ────────────────────────────────────────────────────────────────────────

def test_link_uses_next_turn():
    """턴 t 에 보낸 것은 t+1 에 도착한다. 근거도 t+1 에서 찾아야 한다."""
    ms = [msg(1, 3, "Ranoa1", "Miris1", "ai", "A", "B")]
    ev = [turn_event(3, "Miris1", "이전 턴 근거"),
          turn_event(4, "Miris1", "메시지를 읽고 투표했다")]
    (r,) = judge.link(ms, ev)
    assert r["skip"] is None
    assert r["reasonings"] == ["메시지를 읽고 투표했다"]


def test_link_skips_unreadable():
    """못 읽은 메시지는 4 의 분모가 아니다 — 지표 9(전달 실패율)의 몫이다."""
    ms = [msg(1, 1, "Ranoa1", "Miris1", "original", "A", None, ok=False, reader=False)]
    ev = [turn_event(2, "Miris1", "뭔가 했다")]
    (r,) = judge.link(ms, ev)
    assert r["skip"] == judge.SKIP_UNREADABLE


def test_ai_route_is_judgeable_even_though_reader_is_false():
    """`reader` 는 전달 여부가 아니라 **발신 언어를 읽을 수 있는가** 다.

    AI 경로는 거의 항상 reader=False 다 — 못 읽으니까 번역을 쓴 것이다. 이걸 전달
    실패로 오독하면 4a 의 표본이 통째로 사라진다 (실측 22/22 가 사라졌다).
    """
    ms = [msg(1, 1, "Ranoa1", "Miris1", "ai", "我们需要拦截器",
              "Nous avons besoin", reader=False)]
    ev = [turn_event(2, "Miris1", "j'ai voté")]
    (r,) = judge.link(ms, ev)
    assert r["skip"] is None and r["saw_original"] is False


def test_learner_sees_original_alongside_translation():
    """학습자는 번역문 옆에 원문을 함께 받는다 (spec 5.1).

    판정자에게 번역문만 주면 수신자보다 **적게 본 상태**로 판단하게 된다.
    """
    ms = [msg(1, 1, "Ranoa1", "Miris1", "ai", "我们需要拦截器",
              "Nous avons besoin", reader=True)]
    ev = [turn_event(2, "Miris1", "j'ai voté")]
    (r,) = judge.link(ms, ev)
    assert r["saw_original"] is True
    assert "我们需要拦截器" in judge.stage1_prompt(r)


def test_link_skips_dead_or_last_turn():
    """수신자가 다음 턴에 없다 — 죽었거나 마지막 턴이다."""
    ms = [msg(1, 6, "Ranoa1", "Miris1", "ai", "A", "B")]
    (r,) = judge.link(ms, [])
    assert r["skip"] == judge.SKIP_NO_TURN


def test_link_skips_empty_reasoning():
    """턴은 있었으나 근거를 한 줄도 안 썼다. 추출할 것이 없다."""
    ms = [msg(1, 1, "Ranoa1", "Miris1", "ai", "A", "B")]
    ev = [{"turn": 2, "type": "agent_turn", "agent": "Miris1",
           "reasonings": [{"tool": "invest", "reasoning": "  "}]}]
    (r,) = judge.link(ms, ev)
    assert r["skip"] == judge.SKIP_NO_REASONING


# ── 파싱 ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    '{"mentioned": true}',
    '```json\n{"mentioned": true}\n```',
    'Sure! Here is the answer:\n{"mentioned": true}\nHope that helps.',
])
def test_parse_json_survives_wrapping(raw):
    assert judge.parse_json(raw) == {"mentioned": True}


def test_parse_json_gives_up_cleanly():
    assert judge.parse_json("no json here") is None
    assert judge.parse_json("") is None


# ── 지어낸 근거 막기 ────────────────────────────────────────────────────────────

def _rec(reasonings=("我要投票支持拦截器",)):
    return {"msg_id": 1, "turn": 1, "from": "Ranoa1", "to": "Miris1", "route": "ai",
            "src_lang": "zh", "dst_lang": "fr", "text_sent": "我们需要拦截器",
            "text_delivered": "Nous avons besoin d'un intercepteur",
            "reasonings": list(reasonings), "skip": None}


def test_evidence_must_come_from_reasoning():
    """근거가 reasoning 에 없으면 버린다.

    판정자에게 메시지 본문을 함께 보여주므로 그걸 베껴 '이해했다' 고 답할 수 있다.
    그러면 4a 가 실제보다 낮게 나온다 — 조용히 틀리는 쪽이다.
    """
    j = judge.Judge(StubClient([reply({
        "mentioned": True,
        "evidence": "Nous avons besoin d'un intercepteur",   # 메시지에서 베낌
        "understood": "They want an interceptor.",
    })]))
    out = j.stage1(_rec())
    assert out["mentioned"] is False
    assert out["error"] == "evidence_missing"


def test_evidence_ignores_whitespace_and_quotes():
    """서식 차이로 정당한 판정이 죽으면 안 된다."""
    j = judge.Judge(StubClient([reply({
        "mentioned": True, "evidence": "「我要 投票 支持拦截器」",
        "understood": "They will vote for the interceptor.",
    })]))
    out = j.stage1(_rec())
    assert out["mentioned"] is True and out["error"] is None


def test_stage2_not_called_when_not_mentioned():
    """①이 실패하면 ②는 호출하지 않는다. 분모 밖이고, 호출은 돈이다."""
    stub = StubClient([reply({"mentioned": False, "evidence": "", "understood": ""})])
    out = judge.Judge(stub).judge(_rec())
    assert len(stub.calls) == 1
    assert out["mentioned"] is False and out["same"] is None


def test_two_stages_run_in_order():
    stub = StubClient([
        reply({"mentioned": True, "evidence": "投票支持拦截器",
               "understood": "They will vote for the interceptor."}),
        reply({"same": False, "why": "sender asked for funding, not a vote"}),
    ])
    out = judge.Judge(stub).judge(_rec())
    assert len(stub.calls) == 2
    assert out["mentioned"] is True and out["same"] is False
    # ②는 원문과 이해만 본다. 번역문을 주면 '번역이 잘 됐는가' 를 재게 된다.
    p2 = stub.calls[1]["messages"][1]["content"]
    assert "我们需要拦截器" in p2
    assert "Nous avons besoin" not in p2


def test_skip_never_calls_llm():
    stub = StubClient([])
    out = judge.Judge(stub).judge({**_rec(), "skip": judge.SKIP_UNREADABLE})
    assert stub.calls == [] and out["mentioned"] is None


def test_unparsed_stage1_is_not_a_success():
    j = judge.Judge(StubClient([{"role": "assistant", "content": "글쎄요"}]))
    out = j.stage1(_rec())
    assert out["mentioned"] is False and out["error"] == "unparsed"


# ── 층 나누기 ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("route,frm,to,expect", [
    ("ai", "Ranoa1", "Miris1", "4a"),
    ("domestic", "Miris2", "Miris1", "4c"),
    ("original", "Ranoa1", "Miris1", "4d"),      # 번역 없이 국적만 다르다
    ("original", "Miris2", "Miris1", None),      # 자국민끼리면 domestic 으로 분류됐어야
])
def test_layer(route, frm, to, expect):
    assert judge.layer({"route": route, "from": frm, "to": to}) == expect


def test_aggregate_rates_and_core_contrast():
    judged = [
        {"skip": None, "route": "ai", "from": "R1", "to": "M1",
         "mentioned": True, "same": False},
        {"skip": None, "route": "ai", "from": "R1", "to": "M1",
         "mentioned": True, "same": True},
        {"skip": None, "route": "ai", "from": "R1", "to": "M1",
         "mentioned": False, "same": None},          # 언급률 분모에만
        {"skip": None, "route": "domestic", "from": "M2", "to": "M1",
         "mentioned": True, "same": True},
        {"skip": judge.SKIP_UNREADABLE},
    ]
    agg = judge.aggregate(judged)
    assert agg["4a"] == {"n": 2, "fail_rate": 0.5}       # 언급 안 된 것은 분모 밖
    assert agg["4c"] == {"n": 1, "fail_rate": 0.0}
    assert agg["4b"]["n"] == 3
    assert agg["mention_rate"]["4a"] == {"n": 3, "rate": pytest.approx(0.6667, abs=1e-4)}
    assert agg["4a_minus_4c"] == 0.5                    # ★ 핵심 수치
    assert agg["skipped"] == {judge.SKIP_UNREADABLE: 1}


def test_aggregate_reports_none_not_zero():
    """표본이 없으면 0% 가 아니라 '없음' 이다. 0 으로 적으면 '실패가 없었다' 로 읽힌다."""
    agg = judge.aggregate([{"skip": judge.SKIP_NO_TURN}])
    assert agg["4a"]["fail_rate"] is None
    assert agg["4a_minus_4c"] is None


# ── 이어서 하기 ─────────────────────────────────────────────────────────────────

def test_judge_run_resumes(tmp_path):
    """판정은 비싸다. 이미 한 것은 다시 부르지 않는다."""
    (tmp_path / "messages.jsonl").write_text(
        "\n".join(json.dumps(m) for m in [
            msg(1, 1, "Ranoa1", "Miris1", "ai", "我们需要拦截器", "Nous avons besoin"),
            msg(2, 1, "Ranoa1", "Miris1", "ai", "第二条", "Deuxième"),
        ]), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        json.dumps(turn_event(2, "Miris1", "投票支持拦截器")), encoding="utf-8")

    ok = [reply({"mentioned": True, "evidence": "投票支持拦截器",
                 "understood": "They want an interceptor."}),
          reply({"same": True, "why": "same"})]
    judge.judge_run(tmp_path, StubClient(ok * 2))
    assert len(judge.read_jsonl(tmp_path / "judged.jsonl")) == 2

    stub2 = StubClient(ok * 2)
    judge.judge_run(tmp_path, stub2)
    assert stub2.calls == []                       # 전부 건너뛴다


def test_judge_run_limit_keeps_skips(tmp_path):
    """--limit 은 판정 대상만 자른다. 제외 사유 집계는 온전해야 n 을 믿을 수 있다."""
    (tmp_path / "messages.jsonl").write_text(
        "\n".join(json.dumps(m) for m in [
            msg(1, 1, "Ranoa1", "Miris1", "ai", "A", "B"),
            msg(2, 1, "Ranoa1", "Miris1", "ai", "C", "D"),
            msg(3, 9, "Ranoa1", "Miris1", "ai", "E", "F"),      # 다음 턴 없음
        ]), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        json.dumps(turn_event(2, "Miris1", "投票支持拦截器")), encoding="utf-8")

    judge.judge_run(tmp_path, StubClient([
        reply({"mentioned": True, "evidence": "投票支持拦截器", "understood": "X"}),
        reply({"same": True, "why": ""}),
    ]), limit=1)
    rows = judge.read_jsonl(tmp_path / "judged.jsonl")
    assert len(rows) == 2                                   # 판정 1 + 제외 1
    assert sum(1 for r in rows if r.get("skip")) == 1
