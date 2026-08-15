"""개체 기억 — 컨텍스트 누적·압박 경고·축출·종료 조건. spec 4.5.

이전 판은 턴마다 컨텍스트를 새로 만들었다. 그러면 어제 누구와 무엇을 합의했는지도
상대가 내 말을 읽는지도 배울 수 없고, 조율이 구조적으로 실패하며, 그 실패를
번역 왜곡 탓으로 오독하게 된다.
"""
from __future__ import annotations

import itertools
import json
import random

import pytest

from core import config, loop, tools
from core.agent_loop import Sink, can_act, estimate_tokens, evict, run_agent_turn
from core.llm import StubClient, assistant_msg, tool_call
from domains.meteor import prompts


@pytest.fixture()
def cfg():
    return config.load("configs/base.yaml")


@pytest.fixture()
def world(cfg):
    return loop.init_world(cfg, itertools.count(1))


def _turn(world, cfg, aid, script, warn=False):
    a = world.agents[aid]
    a.ap = cfg.turn.action_points
    cl = StubClient(script)
    log = run_agent_turn(world, a, cfg, cl, Sink(), 48.0, prompts.system_for(a),
                         prompts.render_observation(world, a, cfg, 48.0))
    return a, cl, log


# ── 누적 ──────────────────────────────────────────────────────────────────────

def test_conversation_persists_across_turns(cfg, world):
    """대화는 태어나서 죽을 때까지 이어진다 — 턴마다 버리지 않는다."""
    a, _, _ = _turn(world, cfg, "Asla1", [assistant_msg(tool_call("end_turn", "1", reasoning="r"))])
    n1 = len(a.convo)
    _turn(world, cfg, "Asla1", [assistant_msg(tool_call("end_turn", "2", reasoning="r"))])
    assert len(a.convo) > n1, "턴을 넘기며 대화가 이어져야 한다"
    assert sum(1 for m in a.convo if m["role"] == "user") == 2


def test_memory_survives_and_shows_in_observation(cfg, world):
    """memory_write 로 적은 것이 다음 턴 관측에 보인다."""
    a, _, _ = _turn(world, cfg, "Asla1", [
        assistant_msg(tool_call("memory_write", "1", text="Ranoa2 は我々の言語を読める", reasoning="r")),
        assistant_msg(tool_call("end_turn", "2", reasoning="r"))])
    assert a.memory == "Ranoa2 は我々の言語を読める"
    assert a.memory in prompts.render_observation(world, a, cfg, 48.0)


def test_memory_costs_ap_not_budget(cfg, world):
    """예산을 물리면 기억이 시설 투자와 경쟁해 관측에 교란이 섞인다 (spec 4.5)."""
    a = world.agents["Asla1"]
    a.budget = 500.0
    before = a.budget
    _turn(world, cfg, "Asla1", [
        assistant_msg(tool_call("memory_write", "1", text="x", reasoning="r")),
        assistant_msg(tool_call("end_turn", "2", reasoning="r"))])
    assert a.budget == before                       # 예산 무관
    assert a.ap == pytest.approx(1.0 - cfg.ap.memory_write)


def test_memory_dies_with_agent(cfg, world):
    """개인에 속한 것은 전부 소실 — procreate 로도 안 넘어간다 (spec 3.2·4.5)."""
    a = world.agents["Asla1"]
    a.memory = "남기고 싶은 것"
    a.convo.append({"role": "user", "content": "옛 기억"})
    loop._procreate_child(world, "Asla1", "유언만 넘어간다", cfg,
                          itertools.count(9000), loop.RunResult(world=world))
    child = world.agents["Asla1"]
    assert child.memory == "" and child.convo == []
    assert world.testaments["Asla1"][0] == "유언만 넘어간다"


# ── 압박·축출 ─────────────────────────────────────────────────────────────────

def test_pressure_warning_prepended(cfg, world):
    """임계를 넘으면 관측 **앞**에 통지가 붙는다. 사실 통지이지 지시가 아니다."""
    a = world.agents["Asla1"]
    a.last_prompt_tokens = int(cfg.llm.context_limit * cfg.llm.warn_ratio) + 1
    _, cl, log = _turn(world, cfg, "Asla1", [assistant_msg(tool_call("end_turn", "1", reasoning="r"))])
    first_user = [m for m in cl.calls[0]["messages"] if m["role"] == "user"][0]["content"]
    assert first_user.startswith(prompts.T["ja"]["warn"])
    assert log["pressured"] is True


def test_no_warning_below_threshold(cfg, world):
    a = world.agents["Asla1"]
    a.last_prompt_tokens = 10
    _, _, log = _turn(world, cfg, "Asla1", [assistant_msg(tool_call("end_turn", "1", reasoning="r"))])
    assert log["pressured"] is False


def test_evict_drops_oldest_keeps_one(cfg):
    """오래된 턴 블록부터. 최근 한 턴은 반드시 남는다."""
    convo = []
    for i in range(5):
        convo.append({"role": "user", "content": "x" * 3000})
        convo.append({"role": "assistant", "content": "y" * 300, "tool_calls": []})
    kept, dropped = evict(list(convo), 1000)
    assert dropped > 0
    assert [m["role"] for m in kept][0] == "user"
    assert len([m for m in kept if m["role"] == "user"]) >= 1


def test_evict_never_empties(cfg):
    convo = [{"role": "user", "content": "x" * 100000}]
    kept, _ = evict(list(convo), 10)
    assert kept, "최근 한 턴은 남겨야 한다"


# ── 종료 조건 (spec 4.5) ──────────────────────────────────────────────────────

def test_repeat_guard_stops_loop(cfg, world):
    """실패 호출은 자원을 안 쓰므로 예산으로는 못 막는다 — 반복 차단이 필요하다."""
    same = tool_call("speak", "1", to="NOBODY", text="x", reasoning="r")
    _, _, log = _turn(world, cfg, "Asla1", [assistant_msg(same, same, same, same)])
    assert log["ended_by"] == "repeat_guard"
    assert len(log["reasonings"]) == cfg.llm.repeat_guard


def test_exhausted_when_nothing_affordable(cfg, world):
    """남은 예산·AP 로 실행 가능한 도구가 없으면 종료 (임의 상한 없이)."""
    a = world.agents["Asla1"]
    a.budget, a.ap = 0.0, 0.0
    assert can_act(a, cfg, 48.0) is False
    _, cl, log = _turn(world, cfg, "Asla1", [])
    # AP 는 _turn 이 리셋하므로 직접 다시 0 으로 두고 확인
    a.budget, a.ap = 0.0, 0.0
    assert can_act(a, cfg, 48.0) is False


def test_no_max_steps_constant():
    """임의 상한은 없다. 폭주 보험만 남는다."""
    import core.agent_loop as al
    assert not hasattr(al, "MAX_STEPS")
    assert al.RUNAWAY_CAP >= 100


# ── 🔴 누수 불변식 (spec 4.5) ─────────────────────────────────────────────────

def test_sender_context_has_no_translation(cfg):
    """발신자 컨텍스트에 번역 결과·절단본이 새면 왜곡을 즉시 알아챈다."""
    object.__setattr__(cfg.world, "total_turns", 2)
    ids = [f"{c}{i}" for c in ("Asla", "Ranoa", "Miris") for i in (1, 2, 3)]
    end = assistant_msg(tool_call("end_turn", "e", reasoning="r"))
    long_text = "隕石が接近しています。" * 30          # 130자 상한을 넘겨 절단시킨다
    scripts = {"Asla1": [assistant_msg(tool_call("speak", "1", to="Ranoa1", route="ai",
                                                 text=long_text, reasoning="r")), end]}
    clients = {a: StubClient(list(scripts.get(a, []))) for a in ids}
    tr = StubClient([{"role": "assistant", "content": "TRANSLATED_MARK", "tool_calls": []}] * 30)
    res = loop.run_agentic(cfg, random.Random(1), lambda a: clients[a], tr, 48.0,
                           prompts.render_observation, prompts.system_for, parallel=False)
    sender = res.world.agents["Asla1"]
    blob = json.dumps(sender.convo, ensure_ascii=False)
    assert "TRANSLATED_MARK" not in blob, "번역 결과가 발신자에게 샜다"
    assert long_text in blob                      # 자기가 쓴 원문은 남아도 된다

    # 절단본은 발신자 원문의 앞부분이라 문자열 포함으로는 구분할 수 없다.
    # 대신 **도구 결과**가 절단 사실을 흘리지 않는지 본다 — 무엇이 잘렸는지 알면
    # 다음부터 안 넘기게 되고 왜곡 출처 (d) 가 측정에서 사라진다 (spec 4.5).
    results = [json.loads(m["content"]) for m in sender.convo if m["role"] == "tool"]
    assert res.messages_log[0]["meta"]["truncated"] is True      # 실제로 잘렸는데
    for r in results:
        for banned in ("truncated", "chars_cut", "text_sent", "delivered", "understood"):
            assert banned not in r, f"도구 결과가 {banned} 를 흘린다: {r}"


def test_agents_have_separate_contexts(cfg):
    """축출·누적이 남의 컨텍스트를 섞지 않는다."""
    object.__setattr__(cfg.world, "total_turns", 2)
    ids = [f"{c}{i}" for c in ("Asla", "Ranoa", "Miris") for i in (1, 2, 3)]
    end = assistant_msg(tool_call("end_turn", "e", reasoning="r"))
    clients = {a: StubClient([end, end]) for a in ids}
    res = loop.run_agentic(cfg, random.Random(1), lambda a: clients[a],
                           StubClient([{"role": "assistant", "content": "t", "tool_calls": []}] * 30),
                           48.0, prompts.render_observation, prompts.system_for, parallel=False)
    for aid in ids:
        blob = json.dumps(res.world.agents[aid].convo, ensure_ascii=False)
        others = [o for o in ids if o != aid and not blob.count(f"You are {o}")]
        assert others, "자기 관측만 있어야 한다"
        assert f"あなたは {aid}" in blob or f"你是 {aid}" in blob or f"Vous êtes {aid}" in blob


def test_memory_not_wiped_by_truncated_args(cfg, world):
    """모델이 출력 상한에 걸려 인자가 잘리면 args 가 {} 로 온다.
    그때 덮어쓰면 기억이 통째로 지워진다 — 실측에서 실제로 일어났다."""
    a = world.agents["Asla1"]
    a.memory = "지켜야 할 기억"
    a.ap = cfg.turn.action_points
    bad = {"id": "1", "type": "function",
           "function": {"name": "memory_write", "arguments": '{"text": "잘린'}}   # 깨진 JSON
    cl = StubClient([{"role": "assistant", "content": None, "tool_calls": [bad]},
                     assistant_msg(tool_call("end_turn", "2", reasoning="r"))])
    run_agent_turn(world, a, cfg, cl, Sink(), 48.0, prompts.system_for(a),
                   prompts.render_observation(world, a, cfg, 48.0))
    assert a.memory == "지켜야 할 기억", "잘린 인자로 기억이 지워졌다"


def test_broken_arguments_normalized_before_echo(cfg, world):
    """잘린 JSON 을 그대로 되돌려주면 프로바이더가 400 을 낸다 (실측 218콜 중 8건)."""
    a = world.agents["Asla1"]
    a.ap = cfg.turn.action_points
    bad = {"id": "1", "type": "function",
           "function": {"name": "speak", "arguments": '{"to": "Asla2", "text": "잘린'}}
    cl = StubClient([{"role": "assistant", "content": None, "tool_calls": [bad]},
                     assistant_msg(tool_call("end_turn", "2", reasoning="r"))])
    run_agent_turn(world, a, cfg, cl, Sink(), 48.0, prompts.system_for(a),
                   prompts.render_observation(world, a, cfg, 48.0))
    echoed = [m for m in a.convo if m["role"] == "assistant" and m.get("tool_calls")]
    for m in echoed:
        for tc in m["tool_calls"]:
            json.loads(tc["function"]["arguments"])      # 파싱되어야 한다 (안 되면 400)
