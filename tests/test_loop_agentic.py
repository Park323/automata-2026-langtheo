"""에이전트 루프 통합. 과제 2 Part B-3. StubClient 로 검증 (API 안 씀)."""
from __future__ import annotations

import itertools
import random
from pathlib import Path

import pytest
import yaml

from core import config
from core.agent_loop import Sink
from core.llm import StubClient, assistant_msg, tool_call
from core.loop import RunResult, _settle_agentic, init_world, run_agentic
from domains.meteor import prompts

BASE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
IDS = [f"{n}{i}" for n in ("Asla", "Ranoa", "Miris") for i in (1, 2, 3)]


def _cfg(turns=2):
    d = yaml.safe_load(open(BASE, encoding="utf-8"))
    d["world"]["total_turns"] = turns
    return config.from_dict(d)


def _clients(overrides=None):
    d = {aid: StubClient([]) for aid in IDS}
    for aid, script in (overrides or {}).items():
        d[aid] = StubClient(script)
    return d


def _run(cfg, clients, translator=None, knob_ai=48, seed=1, parallel=True, sequential=False):
    translator = translator or StubClient([{"role": "assistant", "content": "译", "tool_calls": []}] * 50)

    def client_for(aid):                       # 신생아(사망→출생)도 빈 스크립트로 받는다
        c = clients.get(aid)
        if c is None:
            c = clients[aid] = StubClient([])
        return c

    return run_agentic(cfg, random.Random(seed), client_for, translator, knob_ai,
                       prompts.render_turn_open, prompts.system_for, parallel=parallel,
                       sequential=sequential)


# ── #5 도착 지연 ─────────────────────────────────────────────────────────────

def test_message_delivery_delayed():
    """이번 턴 발신은 다음 턴 관측에 나타난다 (같은 턴엔 없음)."""
    cfg = _cfg(2)
    clients = _clients({"Asla1": [assistant_msg(
        tool_call("speak", "1", to="Asla2", text="HELLO_MARK"))]})
    _run(cfg, clients, parallel=False)
    a2 = clients["Asla2"]                       # 빈 스크립트 → 턴당 chat 1회
    # 대화가 누적되므로 "그 턴의 관측" 은 마지막 user 메시지다 (spec 4.5)
    last_user = lambda ms: [m for m in ms if m["role"] == "user"][-1]["content"]
    turn1 = last_user(a2.calls[0]["messages"])
    turn2 = last_user(a2.calls[1]["messages"])
    assert "HELLO_MARK" not in turn1         # 같은 턴엔 안 옴
    assert "HELLO_MARK" in turn2             # 다음 턴에 도착


# ── #1 재현성 ────────────────────────────────────────────────────────────────

def test_agentic_reproducible():
    """같은 seed + 같은 스크립트 → state 로그 바이트 동일."""
    cfg = _cfg(3)
    def once():
        c = _clients({"Asla1": [assistant_msg(
            tool_call("invest", "1", target="facility", amount=50))]})
        return _run(cfg, c, seed=3, parallel=True).state_log
    a, b = once(), once()
    assert a == b and a.encode() == b.encode()


# ── #13 병렬 == 순차 ─────────────────────────────────────────────────────────

def test_parallel_equals_sequential():
    """병렬과 순차의 결과(state 로그)가 동일하다."""
    cfg = _cfg(3)
    def once(par):
        c = _clients({"Asla1": [assistant_msg(
            tool_call("speak", "1", to="Ranoa2", route="ai", text="X"))]})
        return _run(cfg, c, seed=7, parallel=par).state_log
    assert once(True) == once(False)


# ── #12 cap 순서 편향 없음 ───────────────────────────────────────────────────

def test_cap_proportional_order_independent():
    """cap 초과 투자 배분이 호출(=리스트) 순서와 무관하다."""
    cfg = _cfg(1)

    def settle_with(order):
        world = init_world(cfg, itertools.count(1))
        world.agents["Asla1"].budget = 1000
        world.agents["Asla2"].budget = 1000
        sink = Sink()
        sink.facility = order            # cap=500, 총 800 → 초과 300 비례 환급
        _settle_agentic(world, cfg, random.Random(42), sink, StubClient([]), 48,
                        itertools.count(1000), RunResult(world=world), itertools.count(1))
        return (round(world.countries["Asla"].progress, 6),
                round(world.agents["Asla1"].budget, 6),
                round(world.agents["Asla2"].budget, 6))

    forward = settle_with([("Asla", 400, "Asla1"), ("Asla", 400, "Asla2")])
    reverse = settle_with([("Asla", 400, "Asla2"), ("Asla", 400, "Asla1")])
    assert forward == reverse
    # 각자 400 중 150 환급 (400/800 × 300) → 예산 1150
    assert forward[1] == 1150 and forward[2] == 1150


# ── 정보 은닉 재확인 (통합 프롬프트) ─────────────────────────────────────────

def test_full_prompt_hides_secrets():
    cfg = _cfg(2)
    clients = _clients()
    _run(cfg, clients, parallel=False)
    forbidden = ["success_prob", "lambda", "hazard", "distort", "inaccurate", "turns until"]
    for c in clients.values():
        for call in c.calls:
            blob = "".join(m.get("content") or "" for m in call["messages"])
            for bad in forbidden:
                assert bad not in blob, f"프롬프트 금지어 '{bad}'"


# ── 순차 라운드로빈 (issue #20) ──────────────────────────────────────────────

def test_roundrobin_reproducible():
    """순차 라운드로빈도 결정론이다 — 같은 seed 두 번 → state 로그 바이트 동일."""
    cfg = _cfg(3)

    def once():
        c = _clients({"Asla1": [assistant_msg(
            tool_call("invest", "1", target="facility", amount=50))]})
        return _run(cfg, c, seed=5, sequential=True).state_log

    a, b = once(), once()
    assert a == b and a.encode() == b.encode()


def test_roundrobin_population_invariant():
    """순차 경로도 매 턴 9명 생존을 지킨다 (턴 루프 정상 작동)."""
    cfg = _cfg(3)
    r = _run(cfg, _clients(), seed=1, sequential=True)
    assert all(n == 9 for n in r.alive_counts)


def test_roundrobin_same_turn_delivery():
    """핵심 (issue #20): 라운드로빈에서 보낸 메시지는 **같은 턴**에 도착한다.

    `_settle_step` 이 deliver_turn=turn 으로 넣고 `_dequeue_inbox_pop` 이 같은 턴에
    꺼내며 제거한다. 옛 경로(무조건 다음 턴 배달)와 갈리는 지점."""
    from core.loop import _settle_step, _dequeue_inbox_pop
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    world.turn = 1
    sink = Sink()
    sink.messages.append({
        "kind": "speak", "from": "Asla1", "from_country": "Asla", "from_lang": "ja",
        "to": "Asla2", "to_country": "Asla", "to_lang": "ja", "route": None,
        "text": "HELLO_SAME_TURN", "translate_instruction": None})
    _settle_step(world, cfg, random.Random(1), sink,
                 StubClient([{"role": "assistant", "content": "x", "tool_calls": []}] * 5),
                 48, itertools.count(1000), RunResult(world=world), {}, [], [])
    got = _dequeue_inbox_pop(world, "Asla2")
    assert "HELLO_SAME_TURN" in str(got)                    # 같은 턴에 도착
    assert _dequeue_inbox_pop(world, "Asla2") == []         # 두 번 안 옴 (제거됨)


def test_delta_observation_drops_static_scaffold():
    """델타 관측(#22): 안 변하는 골격(비용표)은 빼서 짧고, 상태·inbox 는 유지."""
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    world.turn = 3
    agent = world.agents["Asla1"]
    full = prompts.render_observation(world, agent, cfg, 48, [])
    delta = prompts.render_observation(world, agent, cfg, 48, [], delta=True)
    costs = prompts.render_costs(world, agent, cfg, 48)
    assert costs in full                # 풀엔 비용표(골격)가 있고
    assert costs not in delta           # 델타엔 없다 (반복 제거)
    # 실측 5%. 0.6 으로 두면 골격이 절반쯤 다시 새어들어도 통과한다 —
    # 이 테스트가 지키려는 것이 바로 그 재유입이다.
    assert 0 < len(delta) < len(full) * 0.15


def test_delta_keeps_the_two_resources_that_change_within_a_turn():
    """**델타에 예산은 있는데 행동력이 없었다.**

    AP 는 이 세계에서 예산보다 더 묶는 자원이다 (`invest`·`learn` 이 금액 비례로 먹고
    `propose_vote` 는 0.6). 그런데 자기 AP 를 아는 유일한 경로가 직전 도구 응답의
    `ap_left` 였고, 컨텍스트가 밀려 그것이 방출되면 몇 번 더 움직일 수 있는지 모르는
    채로 차례를 받는다 — **하필 이 델타가 막으려는 상황이다.**
    """
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    world.turn = 3
    agent = world.agents["Asla1"]
    agent.budget, agent.ap = 137.0, 0.35
    delta = prompts.render_observation(world, agent, cfg, 48, [], delta=True)
    assert "137" in delta and "0.35" in delta
    # 골격은 여전히 빠져 있다
    assert prompts.render_costs(world, agent, cfg, 48) not in delta


# ── 상태는 system, 사건은 대화 (8/20) ───────────────────────────────────────

def test_state_lives_in_system_and_never_piles_up_in_the_conversation():
    """**관측이 매 턴 user 로 쌓이고 있었다.** 한 요청 안에 예산이 네 개(100·177·196·215),
    비용표가 네 번 있었다 — 낭비이면서 모순이고, 그 부피가 context_limit 을 밀어
    **대화 이력을 방출시켰다.** 상태를 쌓느라 대화를 버린 것이다.

    이제 경계가 하나다: **지금 그러한 것은 system, 일어난 일은 대화.**
    """
    cfg = _cfg(3)
    end = assistant_msg(tool_call("end_turn", "e", reasoning="r"))
    ids = [f"{c}{i}" for c in ("Asla", "Ranoa", "Miris") for i in (1, 2, 3)]
    clients = {a: StubClient([end] * 4) for a in ids}
    res = run_agentic(cfg, random.Random(1), lambda a: clients[a],
                      StubClient([{"role": "assistant", "content": "t",
                                   "tool_calls": []}] * 30),
                      48.0, prompts.render_turn_open, prompts.system_for,
                      parallel=False)
    convo = res.world.agents["Asla1"].convo
    users = [m["content"] for m in convo if m["role"] == "user"]
    assert len(users) == 3                       # 턴마다 한 마디
    for u in users:
        assert "予算" not in u                    # 상태가 대화에 없다
        assert "行動の費用" not in u               # 비용표도 없다
        assert "になりました" in u                 # 턴을 여는 한 마디는 있다


def test_arrived_messages_stay_in_the_conversation():
    """도착한 메시지는 **사건**이라 쌓여야 한다. 그것만이 에이전트 컨텍스트 안의 유일한
    대화 기록이다 — 상태처럼 갈아치우면 누가 무슨 말을 했는지 잊는다."""
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    world.turn = 2
    a = world.agents["Asla1"]
    box = [{"msg_id": 7, "from": "Ranoa1", "label": "[AI translation]",
            "text": "MARK_ARRIVED", "original": None}]
    txt = prompts.render_turn_open(world, a, cfg, 48.0, box)
    assert "MARK_ARRIVED" in txt and "Ranoa1" in txt
    # 그리고 관측(system)에는 없다 — 두 군데 있으면 어긋날 수 있다
    assert "MARK_ARRIVED" not in prompts.system_for(a, world, cfg, 48.0)


def test_system_without_a_world_is_just_the_rules():
    """문구 검사용. 규칙만 돌려준다."""
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    a = world.agents["Asla1"]
    assert "予算" not in prompts.system_for(a)
    assert "予算" in prompts.system_for(a, world, cfg, 48.0)


def test_an_empty_inbox_says_nothing_at_all():
    """**「도착한 메시지: 없음」 을 붙이면 아무 일도 없었다는 사실이 매 턴 쌓인다.**

    없는 것을 굳이 적지 않는다 (`prompt_audit` 0절). 안 적혀 있으면 안 온 것이다.
    """
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    world.turn = 3
    a = world.agents["Asla1"]
    empty = prompts.render_turn_open(world, a, cfg, 48.0, [])
    assert empty.count("\n") == 0                     # 한 줄뿐이다
    for none_word in ("なし", "没有", "aucun", "Aucun"):
        assert none_word not in empty
    got = prompts.render_turn_open(world, a, cfg, 48.0,
                                   [{"msg_id": 1, "from": "Ranoa1", "label": None,
                                     "text": "MARK", "original": None}])
    assert "MARK" in got and got.startswith(empty)    # 머리말은 같다
