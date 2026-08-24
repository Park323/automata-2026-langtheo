"""면담 도구 — 세계를 건드리지 않는가.

이 도구는 **실험 자료 옆에서 돈다.** 한 줄이라도 raw_calls 나 agent.convo 에 새면 그 런은
못 쓴다. 그래서 「안 새는가」 를 소스와 동작 양쪽에서 지킨다.
"""
from __future__ import annotations

import inspect
import json
import pathlib

import pytest

from tools import interview


@pytest.fixture()
def run(tmp_path):
    """작은 런 하나. state · events · raw_calls 세 파일만 있으면 된다."""
    d = tmp_path / "r1"
    d.mkdir()
    st = []
    for turn in (1, 2, 3):
        for aid, ctry, lang, age in (("Ranoa1", "Ranoa", "zh", 5 + turn),
                                     ("Asla1", "Asla", "ja", 9 + turn)):
            if aid == "Asla1" and turn == 3:
                continue                      # 3턴 정산에서 죽었다 → state 에 없다
            st.append({"turn": turn, "agent": aid, "country": ctry, "age": age,
                       "alive": True, "known_langs": [lang], "born_by": "natural",
                       "native_lang": lang})
    (d / "state.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in st) + "\n", encoding="utf-8")
    (d / "events.jsonl").write_text(json.dumps(
        {"turn": 3, "type": "death", "who": "Asla1", "born": "Asla2", "age": 12,
         "country": "Asla"}, ensure_ascii=False) + "\n", encoding="utf-8")
    raw = [{"kind": "agent", "turn": t, "agent": "Ranoa1", "step": 2, "age": 5 + t,
            "country": "Ranoa", "call_id": f"c{t:05d}",
            "request": {"model": "m/x", "messages": [
                {"role": "system", "content": f"세계 t{t}"},
                {"role": "user", "content": f"{t}년"}]},
            "response": {"usage": {}}} for t in (1, 2, 3)]
    (d / "raw_calls.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in raw) + "\n", encoding="utf-8")
    return d


def test_lifespan_counts_the_year_they_died(run):
    """**죽은 해는 state 에 없다.** 죽음은 그 해의 정산에서 일어나고 state 행은 그 뒤에
    쓰이므로, 마지막 행은 죽기 한 해 전이다. 사망 이벤트로 되돌려주지 않으면 가장 오래 산
    사람의 수명을 **한 해 적게** 말한다 (t20a 에서 Ranoa1 을 17해로 말하고 있었다)."""
    by = {d["agent"]: d for d in interview.roster(run)}
    assert by["Asla1"]["died_turn"] == 3
    assert by["Asla1"]["turns"] == 3 and by["Asla1"]["alive"] is False
    assert by["Ranoa1"]["turns"] == 3 and by["Ranoa1"]["alive"] is True


def test_context_is_taken_verbatim_not_re_rendered(run):
    """**다시 렌더링하지 않는다.** 그때의 예산·진척·명단을 복원해야 하고, 복원이 조금이라도
    틀리면 다른 사람에게 묻는 것이 된다."""
    msgs, meta = interview.context_of(run, "Ranoa1", None)
    assert msgs[0]["content"] == "세계 t3" and meta["call_id"] == "c00003"
    msgs, meta = interview.context_of(run, "Ranoa1", 2)
    assert msgs[0]["content"] == "세계 t2" and meta["turn"] == 2

    with pytest.raises(SystemExit):
        interview.context_of(run, "Ranoa1", 99)      # 없는 해는 조용히 넘기지 않는다
    with pytest.raises(SystemExit):
        interview.context_of(run, "Nobody1", None)


def test_the_question_goes_in_the_agents_own_language(run):
    """프롬프트가 그 사람의 말인데 질문만 다른 말이면 답도 그 말로 나온다 — 그러면 그
    사람이 쓰는 말이 아니게 된다."""
    assert set(interview.ASIDE) == {"ja", "zh", "fr"}
    assert interview.native_lang(run, "Ranoa1") == "zh"
    assert "中文" in interview.ASIDE["zh"]
    for lang, aside in interview.ASIDE.items():
        # 도구를 못 쓴다는 것을 그 말로 알린다 — 안 알리면 도구를 부르려 한다
        assert any(k in aside for k in ("道具", "工具", "outils")), lang


def test_nothing_reaches_the_experiment_logs(run):
    """**한 줄이라도 raw_calls 나 agent.convo 에 새면 그 런은 못 쓴다.**

    소스를 읽어 확인한다 — 새는지는 실제 API 호출 없이는 동작으로 못 보고, 그렇다고
    확인을 건너뛰면 이 도구의 유일한 위험이 무검사로 남는다.
    """
    src = inspect.getsource(interview.ask)
    assert "recorder=None" in src                 # raw_calls 에 안 남는다
    assert "tools=None" in src                    # 도구를 안 싣는다
    assert "interviews.jsonl" in src              # 기록은 여기만
    assert "agent.convo" not in src               # 세계의 대화를 건드리지 않는다
    assert "raw(" not in src

    whole = pathlib.Path(interview.__file__).read_text(encoding="utf-8")
    assert "raw_calls" in whole                   # 읽기는 한다
    assert '"raw_calls.jsonl"' in whole
    # 쓰기는 interviews.jsonl 하나뿐이다
    assert whole.count('.open("a"') == 1
    assert 'run / "interviews.jsonl"' in whole


def test_a_running_run_is_refused_by_default(run, monkeypatch):
    """돌고 있는 런에 물으면 우리가 흘린 것이 그 런의 결과에 남는다."""
    monkeypatch.setattr(interview, "_running", lambda p: True)
    monkeypatch.setattr("sys.argv", ["x", "--run", str(run), "--agent", "Ranoa1",
                                     "--ask", "?"])
    with pytest.raises(SystemExit) as e:
        interview.main()
    assert "돌고 있는" in str(e.value)
