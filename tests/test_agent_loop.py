"""에이전트 루프. 과제 2 Part A. StubClient 로 검증 (API 안 씀)."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from core import config
from core.agent_loop import MAX_STEPS, Sink, learn_cost, run_agent_turn
from core.llm import StubClient, assistant_msg, tool_call
from core.loop import init_world
from domains.meteor import prompts

BASE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


@pytest.fixture()
def cfg():
    return config.load(BASE)


@pytest.fixture()
def world(cfg):
    return init_world(cfg, itertools.count(1))


def _run(world, cfg, agent_id, script, knob_ai=48, budget=None):
    agent = world.agents[agent_id]
    agent.ap = cfg.turn.action_points        # 실제 루프의 1단계(소득·AP 리셋)를 대신
    if budget is not None:
        agent.budget = budget
    sink = Sink()
    client = StubClient(script)
    sys_p = prompts.SYSTEM
    usr_p = prompts.render_observation(world, agent, cfg, knob_ai)
    log = run_agent_turn(world, agent, cfg, client, sink, knob_ai, sys_p, usr_p)
    return agent, sink, client, log


def _results(client):
    """마지막 chat 호출 시점의 messages 스냅샷에서 tool 응답을 훑는다.
    (스크립트 끝에 end_turn 을 두면 그 직전까지의 모든 tool 결과가 담긴다.)"""
    last = client.calls[-1]["messages"]
    return [json.loads(m["content"]) for m in last if m.get("role") == "tool"]


# ── #2 AP 상한 ───────────────────────────────────────────────────────────────

def test_ap_cap_fourth_speak_fails(cfg, world):
    """speak 4번째가 ok:False (AP 0.3 × 3 = 0.9, 4번째는 1.2 > 1.0)."""
    script = [assistant_msg(
        tool_call("speak", "1", to="A2", text="x", intent="i"),
        tool_call("speak", "2", to="A2", text="x", intent="i"),
        tool_call("speak", "3", to="A2", text="x", intent="i"),
        tool_call("speak", "4", to="A2", text="x", intent="i"),
    ), assistant_msg(tool_call("end_turn", "5"))]
    agent, sink, client, log = _run(world, cfg, "A1", script, budget=10000)
    results = _results(client)
    oks = [r for r in results if "ok" in r]
    assert sum(1 for r in oks if r["ok"]) == 3          # 앞 3건 성공
    assert any((not r["ok"]) and "AP" in r.get("error", "") for r in oks)


# ── #3 예산 고갈 ─────────────────────────────────────────────────────────────

def test_budget_never_negative(cfg, world):
    script = [assistant_msg(tool_call("invest", "1", target="facility", amount=999999)),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "A1", script, budget=100)
    results = _results(client)
    assert any((not r["ok"]) and "예산" in r.get("error", "") for r in results)
    assert agent.budget >= 0                            # 음수 안 됨
    assert sink.facility == []                          # 실패했으니 sink 에 안 들어감


# ── #4 procreate 즉시 종료 ───────────────────────────────────────────────────

def test_procreate_ends_turn(cfg, world):
    """procreate 뒤의 tool_call 은 실행되지 않는다."""
    script = [assistant_msg(
        tool_call("procreate", "1", testament="믿지 마라"),
        tool_call("invest", "2", target="facility", amount=10),   # 버려져야 함
    )]
    agent, sink, client, log = _run(world, cfg, "A1", script, budget=10000)
    assert len(sink.procreations) == 1
    assert sink.facility == []                          # procreate 뒤 invest 무시


# ── #10 학습 할인 ────────────────────────────────────────────────────────────

def test_learn_discount_levels(cfg, world):
    """국내 구사자 없음/있음/부모까지 → 300 / 150 / 75."""
    a1 = world.agents["A1"]           # 국가 A, ja
    # 아무 할인 없음
    cost, _ = learn_cost(a1, "B", world, cfg)       # B = zh
    assert cost == 300
    # 국내 구사자: A2 가 zh 를 앎
    world.agents["A2"].known_langs.add("zh")
    cost, reason = learn_cost(a1, "B", world, cfg)
    assert cost == 150 and "국내" in reason
    # 부모까지: a1 의 부모가 zh
    a1.parent_langs.add("zh")
    cost, reason = learn_cost(a1, "B", world, cfg)
    assert cost == 75


def test_learn_self_not_counted(cfg, world):
    """자기 자신은 국내 구사자로 세지 않는다."""
    a1 = world.agents["A1"]
    a1.known_langs.add("zh")          # 자기가 zh 를 알아도
    cost, _ = learn_cost(a1, "B", world, cfg)
    assert cost == 300                # 할인 안 됨


def test_learn_defers_known_langs(cfg, world):
    """learn 은 known_langs 를 즉시 바꾸지 않고 sink 에 넣는다 (병렬 레이스 방지)."""
    script = [assistant_msg(tool_call("learn", "1", country="B")),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "A1", script, budget=10000)
    assert "zh" not in agent.known_langs          # 즉시 반영 안 됨
    assert sink.learns == [("A1", "zh")]          # sink 로 이연
    assert agent.budget == 10000 - 300            # 예산은 즉시 차감
    assert agent.ap == cfg.turn.action_points - cfg.ap.learn


# ── #11 정보 은닉 (가장 중요) ────────────────────────────────────────────────

FORBIDDEN = ["success_prob", "lambda", "λ", "hazard", "사망 확률", "사망확률",
             "재앙까지", "남은 턴", "수명 증가율"]


def test_prompt_hides_secrets(cfg, world):
    """프롬프트(system·관측)에 success_prob·λ·하자드·재앙까지 남은 턴이 없다."""
    p = prompts.SYSTEM + "\n" + prompts.render_observation(world, world.agents["A1"], cfg, knob_ai=48)
    for bad in FORBIDDEN:
        assert bad not in p, f"프롬프트에 금지어 '{bad}' 노출"


def test_tool_results_hide_progress(cfg, world):
    """invest(facility) 결과에 진척 증가분이 없다 (success_prob 역산 방지)."""
    script = [assistant_msg(tool_call("invest", "1", target="facility", amount=50)),
              assistant_msg(tool_call("invest", "2", target="wellness", amount=30)),
              assistant_msg(tool_call("end_turn", "3"))]
    agent, sink, client, log = _run(world, cfg, "A1", script, budget=10000)
    for r in _results(client):
        blob = json.dumps(r, ensure_ascii=False)
        for bad in ["success_prob", "lambda", "λ", "gained", "증가분", "progress"]:
            assert bad not in blob, f"도구 결과에 금지어 '{bad}' 노출: {blob}"
