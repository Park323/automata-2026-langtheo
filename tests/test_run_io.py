"""산출물 기록. spec 9장.

파생 로그는 전부 raw_calls.jsonl 에서 재생성할 수 있어야 한다 — 정의는 나중에 바뀌고,
원본이 없으면 그때 다시 돌려야 한다.
"""
from __future__ import annotations

import json
import random

import pytest

from core import config, loop, run_io
from core.llm import StubClient, assistant_msg, tool_call
from domains.meteor import prompts


@pytest.fixture
def cfg():
    return config.load("configs/base.yaml")


def _run_with_writer(cfg, tmp_path, turns=2):
    object.__setattr__(cfg.world, "total_turns", turns)
    w = run_io.RunWriter("test_run", cfg_raw={"world": {"total_turns": turns}},
                         root=tmp_path)
    ids = [f"{c}{i}" for c in ("Asla", "Ranoa", "Miris") for i in (1, 2, 3)]
    end = assistant_msg(tool_call("end_turn", "e", reasoning="이래서"))
    script = {"Asla1": [assistant_msg(tool_call("speak", "1", to="Ranoa1",
                                                route="ai", text="隕石")), end]}
    clients = {a: StubClient(list(script.get(a, [])), recorder=w.recorder(kind="agent", agent=a))
               for a in ids}
    tr = StubClient([{"role": "assistant", "content": "陨石", "tool_calls": []}] * 40,
                    recorder=w.recorder(kind="translate"))
    res = loop.run_agentic(cfg, random.Random(1), lambda a: clients[a], tr, 48.0,
                           prompts.render_observation, prompts.system_for,
                           parallel=False, on_turn_end=w.on_turn_end)
    w.close({"final": res.final, "deaths": res.deaths})
    return w, res


def _lines(w, name):
    p = w.dir / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def test_all_artifacts_written(cfg, tmp_path):
    w, _ = _run_with_writer(cfg, tmp_path)
    for f in ("raw_calls.jsonl", "state.jsonl", "messages.jsonl",
              "events.jsonl", "metrics.jsonl", "summary.json", "config_snapshot.yaml"):
        assert (w.dir / f).exists(), f"{f} 가 없다"


def test_raw_keeps_request_and_response(cfg, tmp_path):
    """요청·응답을 가공 없이. 이게 있어야 파생 로그를 재생성할 수 있다."""
    w, _ = _run_with_writer(cfg, tmp_path)
    raw = _lines(w, "raw_calls")
    assert raw, "raw_calls 가 비었다"
    r = raw[0]
    for k in ("run_id", "kind", "attempt", "latency_ms", "request", "response"):
        assert k in r, f"raw 에 {k} 가 없다"
    assert r["request"]["messages"], "요청 messages 가 비었다"
    assert {x["kind"] for x in raw} >= {"agent"}


def test_config_snapshot_has_commit(cfg, tmp_path):
    """재현에는 설정만으로 부족하다 — 코드 커밋도 남긴다."""
    import yaml
    w, _ = _run_with_writer(cfg, tmp_path)
    snap = yaml.safe_load((w.dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert "code_commit" in snap and "config" in snap


def test_state_has_fields_for_x_hat(cfg, tmp_path):
    """x̂ 를 나이로 층화하려면 age 가 있어야 한다 (spec 8.2)."""
    w, _ = _run_with_writer(cfg, tmp_path)
    st = _lines(w, "state")
    assert st
    for k in ("turn", "agent", "age", "lambda", "known_langs", "parent_langs", "born_by"):
        assert k in st[0], f"state 에 {k} 가 없다"


def test_metrics_have_failure_rate(cfg, tmp_path):
    """실패율이 조건 간에 다르면 교란 신호 — 집계 가능해야 한다 (지표 16)."""
    w, _ = _run_with_writer(cfg, tmp_path)
    m = _lines(w, "metrics")
    assert m and "llm_failure_rate" in m[0] and "ended_by" in m[0]


def test_appended_per_turn(cfg, tmp_path):
    """턴마다 append — 45턴에서 죽어도 거기까지는 남아야 한다."""
    w, _ = _run_with_writer(cfg, tmp_path, turns=3)
    turns = [m["turn"] for m in _lines(w, "metrics")]
    assert turns == [1, 2, 3]


def test_messages_written_once(cfg, tmp_path):
    """메시지가 턴마다 중복 기록되지 않는다."""
    w, _ = _run_with_writer(cfg, tmp_path, turns=3)
    ids = [m["msg_id"] for m in _lines(w, "messages")]
    assert len(ids) == len(set(ids))


def test_refuses_to_append_to_existing_run(cfg, tmp_path):
    """같은 run_id 로 다시 돌리면 두 런이 한 파일에 섞여 지표가 조용히 오염된다."""
    import pytest as _pytest
    w = run_io.RunWriter("dup", root=tmp_path)
    w.raw({"kind": "agent", "request": {}, "response": {}})
    w.close()
    with _pytest.raises(FileExistsError):
        run_io.RunWriter("dup", root=tmp_path)
    run_io.RunWriter("dup", root=tmp_path, overwrite=True)      # 명시하면 허용
