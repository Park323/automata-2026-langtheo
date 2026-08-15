"""실험 산출물 — raw_calls.jsonl 형식 + RunWriter 파일. spec 9 · #8.

StubClient 에 raw_sink 를 주입해 실제 API 없이 형식을 검증한다.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import yaml

from core import config
from core.artifacts import RunWriter
from core.llm import StubClient, assistant_msg, tool_call
from core.loop import run_agentic
from domains.meteor import prompts

BASE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
IDS = [f"{n}{i}" for n in ("Asla", "Ranoa", "Miris") for i in (1, 2, 3)]

RAW_KEYS = {"run_id", "kind", "turn", "agent", "step",
            "attempt", "latency_ms", "request", "response", "error"}


def _cfg_turns(turns):
    d = yaml.safe_load(open(BASE, encoding="utf-8"))
    d["world"]["total_turns"] = turns
    return config.from_dict(d)


@pytest.fixture()
def run_dir(tmp_path):
    """2턴 run 을 raw_sink·on_turn_end 배선으로 돌리고 산출물 디렉토리를 돌려준다."""
    cfg = _cfg_turns(2)
    raw_cfg = yaml.safe_load(open(BASE, encoding="utf-8"))
    writer = RunWriter("test_run", raw_cfg, knob=48, seed=1, runs_dir=tmp_path)

    clients = {aid: StubClient([], raw_sink=writer.raw_sink) for aid in IDS}
    clients["Asla1"] = StubClient(
        [assistant_msg(tool_call("speak", "1", to="Ranoa2", route="ai", text="HELLO")),
         assistant_msg(tool_call("end_turn", "2", reasoning="sent a note"))],
        raw_sink=writer.raw_sink)
    translator = StubClient(
        [{"role": "assistant", "content": "译文", "tool_calls": []}] * 50,
        raw_sink=writer.raw_sink)

    res = run_agentic(cfg, random.Random(1), clients.__getitem__, translator, 48,
                      prompts.render_observation, prompts.system_for, parallel=False,
                      on_turn_end=writer.on_turn_end)
    writer.finish(res)
    writer.close()
    return tmp_path / "test_run"


def test_all_artifact_files_exist(run_dir):
    for name in ("config_snapshot.yaml", "raw_calls.jsonl", "state.jsonl",
                 "messages.jsonl", "events.jsonl", "metrics.jsonl", "summary.json"):
        assert (run_dir / name).exists(), f"{name} 이 없다"


def test_raw_calls_shape(run_dir):
    """모든 호출이 필수 키를 갖고, request 에 messages 가 통째로 남는다."""
    lines = [json.loads(x) for x in
             (run_dir / "raw_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines
    for rec in lines:
        assert RAW_KEYS <= set(rec), f"누락 키: {RAW_KEYS - set(rec)}"
        assert rec["run_id"] == "test_run"
        assert "messages" in rec["request"]              # 파생 로그 재생성의 원천
        assert "choices" in rec["response"]              # 응답 전문 보존


def test_raw_calls_cover_agent_and_translate(run_dir):
    """에이전트 호출과 번역 호출이 둘 다 남는다 (kind 로 구분)."""
    lines = [json.loads(x) for x in
             (run_dir / "raw_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = {r["kind"] for r in lines}
    assert "agent" in kinds and "translate" in kinds
    agent_recs = [r for r in lines if r["kind"] == "agent"]
    assert any(r["agent"] == "Asla1" and r["step"] >= 1 for r in agent_recs)
    tr = [r for r in lines if r["kind"] == "translate"]
    assert tr and all(r["turn"] is not None for r in tr)   # 번역에도 turn 문맥


def test_config_snapshot_has_commit(run_dir):
    snap = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert "_meta" in snap and "commit" in snap["_meta"]
    assert snap["_meta"]["run_id"] == "test_run"


def test_messages_and_metrics_written(run_dir):
    msgs = [json.loads(x) for x in
            (run_dir / "messages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert msgs and "understood" in msgs[0] and "msg_id" in msgs[0]

    metrics = [json.loads(x) for x in
               (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert metrics
    for m in metrics:
        assert "end_reasons" in m and "llm_failure_rate" in m


def test_summary(run_dir):
    summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summ["run_id"] == "test_run"
    assert "end_reasons" in summ and "final" in summ
