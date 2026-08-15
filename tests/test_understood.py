"""understood 기록·조인·불노출 + report_understanding 도구. spec 4.2 · 6.1 · #7.

intent 는 폐지됐다 (자기보고 오염). 지표 4 는 text_sent vs understood 로 낸다.
understood 는 수신자가 T+1 에 직접 기록하고, 전역 msg_id 로 원 메시지에 조인한다.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import yaml

from core import config
from core.agent_loop import Sink, run_agent_turn
from core.llm import StubClient, assistant_msg, tool_call
from core.loop import init_world, run_agentic
from core import tools
from domains.meteor import prompts

BASE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
IDS = [f"{n}{i}" for n in ("Asla", "Ranoa", "Miris") for i in (1, 2, 3)]


def _cfg_turns(turns):
    d = yaml.safe_load(open(BASE, encoding="utf-8"))
    d["world"]["total_turns"] = turns
    return config.from_dict(d)


def _clients(overrides=None):
    d = {aid: StubClient([]) for aid in IDS}
    for aid, script in (overrides or {}).items():
        d[aid] = StubClient(script)
    return d


def _run(cfg, clients):
    translator = StubClient([{"role": "assistant", "content": "译", "tool_calls": []}] * 50)
    return run_agentic(cfg, __import__("random").Random(1), clients.__getitem__, translator,
                       48, prompts.render_observation, prompts.system_for, parallel=False)


# ── 도구 스키마: intent 폐지 확인 ────────────────────────────────────────────

def test_intent_removed_from_schema():
    """speak/ask 에 intent 필드가 없다 (필수에서도)."""
    by_name = {t["function"]["name"]: t["function"] for t in tools.TOOLS}
    for name in ("speak", "ask"):
        props = by_name[name]["parameters"]["properties"]
        assert "intent" not in props
        assert "intent" not in by_name[name]["parameters"]["required"]
    assert "report_understanding" in by_name


def test_reasoning_required_on_terminators():
    """reasoning 은 턴을 끝내는 두 도구의 필수 인자다 (spec 9)."""
    by_name = {t["function"]["name"]: t["function"] for t in tools.TOOLS}
    assert "reasoning" in by_name["end_turn"]["parameters"]["required"]
    assert "reasoning" in by_name["procreate"]["parameters"]["required"]


# ── report_understanding 는 공짜 (관측 장치) ─────────────────────────────────

def test_report_understanding_is_free():
    """AP·예산 0. sink 에 (agent, msg_id, understood) 로 쌓인다."""
    cfg = config.load(BASE)
    world = init_world(cfg, itertools.count(1))
    agent = world.agents["Asla2"]
    agent.ap = cfg.turn.action_points
    agent.budget = 100
    sink = Sink()
    client = StubClient([
        assistant_msg(tool_call("report_understanding", "1", msg_id=1, understood="I read it as X")),
        assistant_msg(tool_call("end_turn", "2", reasoning="done")),
    ])
    sp = prompts.system_for(agent)
    up = prompts.render_observation(world, agent, cfg, 48)
    run_agent_turn(world, agent, cfg, client, sink, 48, sp, up, turn=1)
    assert agent.ap == cfg.turn.action_points and agent.budget == 100     # 공짜
    assert sink.understandings == [("Asla2", 1, "I read it as X")]


def test_report_understanding_bad_msg_id():
    """msg_id 가 정수가 아니면 ok:False, sink 미반영 (크래시 없음)."""
    cfg = config.load(BASE)
    world = init_world(cfg, itertools.count(1))
    agent = world.agents["Asla2"]
    agent.ap = cfg.turn.action_points
    sink = Sink()
    client = StubClient([
        assistant_msg(tool_call("report_understanding", "1", msg_id="abc", understood="x")),
        assistant_msg(tool_call("end_turn", "2", reasoning="r")),
    ])
    sp = prompts.system_for(agent)
    up = prompts.render_observation(world, agent, cfg, 48)
    run_agent_turn(world, agent, cfg, client, sink, 48, sp, up, turn=1)
    assert sink.understandings == []


# ── 전역 msg_id 조인 + 절대 불노출 ──────────────────────────────────────────

def test_understood_joins_by_global_msg_id_and_never_leaks():
    """Asla1→Asla2 발신(msg_id 전역). T+1 에 Asla2 가 보고 → 그 메시지에 조인.
    understood 는 어떤 에이전트의 컨텍스트에도 절대 나타나지 않는다."""
    cfg = _cfg_turns(2)
    clients = _clients({
        "Asla1": [assistant_msg(tool_call("speak", "1", to="Asla2", text="HI_THERE")),
                  assistant_msg(tool_call("end_turn", "2", reasoning="sent"))],
        # 턴1: 아무것도 안 함(도착 전). 턴2: 도착한 msg_id=1 을 이해했다고 보고.
        "Asla2": [{"role": "assistant", "content": "", "tool_calls": []},
                  assistant_msg(tool_call("report_understanding", "1", msg_id=1,
                                          understood="SECRET_UNDERSTOOD")),
                  assistant_msg(tool_call("end_turn", "2", reasoning="read"))],
    })
    res = _run(cfg, clients)

    # 조인: 유일한 메시지에 understood 가 채워졌다
    assert len(res.messages_log) == 1
    entry = res.messages_log[0]
    assert entry["msg_id"] == 1 and entry["delivered"] is True
    assert entry["understood"] == "SECRET_UNDERSTOOD"

    # 🔴 어떤 에이전트의 프롬프트에도 understood 가 새지 않는다 (채점 기준선 노출 금지)
    for c in clients.values():
        for call in c.calls:
            blob = "".join(m.get("content") or "" for m in call["messages"])
            assert "SECRET_UNDERSTOOD" not in blob


def test_unreported_understood_stays_null():
    """수신자가 보고하지 않으면 understood 는 None (지표 4 분모에서 제외됨)."""
    cfg = _cfg_turns(2)
    clients = _clients({
        "Asla1": [assistant_msg(tool_call("speak", "1", to="Asla2", text="HI")),
                  assistant_msg(tool_call("end_turn", "2", reasoning="r"))],
    })
    res = _run(cfg, clients)
    assert len(res.messages_log) == 1
    assert res.messages_log[0]["understood"] is None


def test_non_recipient_report_is_ignored():
    """엉뚱한 에이전트가 남의 msg_id 를 보고해도 그 메시지 understood 는 오염되지 않는다.
    (약한 모델이 지어낸 msg_id 로 지표 4 기준선을 망치는 것을 막는다.)"""
    cfg = _cfg_turns(2)
    clients = _clients({
        # Asla1 → Asla2 (msg_id 1). 수신자는 Asla2 다.
        "Asla1": [assistant_msg(tool_call("speak", "1", to="Asla2", text="HI")),
                  assistant_msg(tool_call("end_turn", "2", reasoning="r"))],
        # Ranoa1 은 수신자가 아닌데 msg_id=1 을 보고한다 → 무시돼야 한다.
        "Ranoa1": [{"role": "assistant", "content": "", "tool_calls": []},
                   assistant_msg(tool_call("report_understanding", "1", msg_id=1,
                                           understood="IMPOSTER")),
                   assistant_msg(tool_call("end_turn", "2", reasoning="x"))],
    })
    res = _run(cfg, clients)
    assert res.messages_log[0]["understood"] is None       # 오염 안 됨
