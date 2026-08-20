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
    log = run_agent_turn(world, a, cfg, cl, Sink(), 48.0, prompts.system_for(a, None, cfg),
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


def test_only_the_testament_survives_death(cfg, world):
    """개인에 속한 것은 전부 소실. **유언 한 문장만** 아이의 기억으로 넘어간다.

    유언을 별도 블록이 아니라 `memory` 초기값으로 두는 이유 — 다른 모든 것과 같은
    컨텍스트에서 관리되고, 아이가 `memory_write` 로 덮어쓰면 사라집니다.
    **그 덮어쓰기가 곧 구전의 감쇠**이고, 무엇을 남길 가치가 있다고 봤는지가 관측됩니다.
    """
    a = world.agents["Asla1"]
    a.memory = "내가 평생 알아낸 것"
    a.convo.append({"role": "user", "content": "옛 기억"})
    loop._procreate_child(world, "Asla1", "유언만 넘어간다", cfg,
                          itertools.count(9000), loop.RunResult(world=world))
    assert "Asla1" not in world.agents        # id 는 재사용하지 않는다
    child = world.agents["Asla4"]             # 3명 다음이니 4번
    assert child.convo == []                       # 대화 이력은 소실
    assert "내가 평생 알아낸 것" not in child.memory  # 부모의 메모도 소실
    assert child.memory == "유언만 넘어간다"          # 유언만 기억으로
    assert world.testaments["Asla4"][0] == "유언만 넘어간다"   # 유언은 아이에게
    assert "Asla1" not in world.testaments                    # 죽은 자리는 비운다


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


def test_free_actions_keep_can_act_true(cfg, world):
    """**종료 조건 ② 는 자유 행동이 생기면서 사실상 죽었다** (8/17).

    `memory_write` 와 `procreate` 가 돈도 AP 도 안 쓰므로, 자원이 완전히 바닥나도
    고를 것이 남아 있다. 여기서 거짓으로 False 를 돌려주면 **합법적인 행동을 잘라내게**
    된다 — 빈털터리가 유언을 남기고 죽는 것이 이 세계에서 가장 흔한 결말이다.
    정상 종료는 `end_turn` 이고, 폭주는 `RUNAWAY_CAP`(64) 이 막는다.
    """
    a = world.agents["Asla1"]
    a.budget, a.ap = 0.0, 0.0
    assert cfg.ap.memory_write == 0.0 and cfg.ap.procreate == 0.0
    assert can_act(a, cfg, 48.0) is True


def test_no_max_steps_constant():
    """임의 상한은 없다. 폭주 보험만 남는다."""
    import core.agent_loop as al
    assert not hasattr(al, "MAX_STEPS")
    assert al.RUNAWAY_CAP >= 40      # 정상 턴은 도구 5~15회. 닿지 않을 높이면 된다


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
                           prompts.render_turn_open, prompts.system_for, parallel=False)
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
                           48.0, prompts.render_turn_open, prompts.system_for, parallel=False)
    # **신원은 이제 system 에 있다.** 관측(지금 그러한 것)이 system 으로 옮겨갔고,
    # 대화에는 턴을 여는 한 마디와 내 행동·결과만 쌓인다.
    for aid in ids:
        agent = res.world.agents[aid]
        sysp = prompts.system_for(agent, res.world, cfg, 48.0)
        assert (f"あなたは {aid}" in sysp or f"你是 {aid}" in sysp
                or f"Vous êtes {aid}" in sysp), aid
        for other in ids:
            if other != aid:
                assert f"あなたは {other}" not in sysp
                assert f"你是 {other}" not in sysp
                assert f"Vous êtes {other}" not in sysp
        # 대화에 남의 신원이 섞이지 않는다
        blob = json.dumps(agent.convo, ensure_ascii=False)
        for other in ids:
            if other != aid:
                assert f"あなたは {other}" not in blob


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
    run_agent_turn(world, a, cfg, cl, Sink(), 48.0, prompts.system_for(a, None, cfg),
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
    run_agent_turn(world, a, cfg, cl, Sink(), 48.0, prompts.system_for(a, None, cfg),
                   prompts.render_observation(world, a, cfg, 48.0))
    echoed = [m for m in a.convo if m["role"] == "assistant" and m.get("tool_calls")]
    for m in echoed:
        for tc in m["tool_calls"]:
            json.loads(tc["function"]["arguments"])      # 파싱되어야 한다 (안 되면 400)


def test_repeat_guard_ignores_successful_repeats(cfg, world):
    """성공한 호출은 자원을 쓰므로 ②가 막는다 — ③이 정상 행동을 끊으면 안 된다."""
    a = world.agents["Asla1"]
    a.budget = 10000.0
    same = tool_call("speak", "1", to="Asla2", text="같은 말", reasoning="r")
    _, _, log = _turn(world, cfg, "Asla1",
                      [assistant_msg(same, same, same),
                       assistant_msg(tool_call("end_turn", "9", reasoning="r"))])
    assert log["ended_by"] == "ended", "동일한 성공 호출 3건에 끊기면 안 된다"
    assert sum(1 for r in log["reasonings"] if r["ok"] and r["tool"] == "speak") == 3


def test_tool_schema_counted_in_eviction(cfg):
    """도구 스키마 909 토큰을 빼면 실질 한계가 8192 가 아니라 9100 쯤으로 느슨해진다."""
    from core.agent_loop import _TOOL_TOKENS, estimate_tokens
    assert _TOOL_TOKENS > 500, "도구 스키마가 계산에 잡혀야 한다"
    msgs = [{"role": "user", "content": "x" * 300}]
    assert estimate_tokens(msgs, _TOOL_TOKENS) > estimate_tokens(msgs)


def test_the_memo_header_says_it_overwrites(cfg, world):
    """**덧붙이는 것으로 읽히고 있었다.**

    `memory_write` 는 통째로 덮어쓴다 — 그것이 spec 3.3 의 「구전의 감쇠」 다. 무엇을
    버릴지 고르는 것이 관측 대상인데, 덧붙는 줄로 알면 그 선택 자체가 일어나지 않는다.

    안내를 **머리말**에 둔다. 비어 있을 때만 적으면 정작 오해가 생기는 자리 — 이미 뭔가
    적혀 있는 상태 — 에서 안 보인다. 그리고 「어떻게 쓰는가」 와 「덮어쓴다」 를 한 줄이
    같이 말하므로 빈 칸의 안내는 없어도 된다.
    """
    from domains.meteor import prompts
    marks = {"ja": "書き足すのではなく", "zh": "不是追加", "fr": "n'ajoute rien"}
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        ag = world.agents[aid]
        hdr = prompts.T[ag.native_lang]["mem_hdr"]
        assert "memory_write" in hdr and marks[ag.native_lang] in hdr, aid
        for memo in ("", "요격기에 몰아줘라"):
            ag.memory = memo
            obs = prompts.system_for(ag, world, cfg, 48.0)
            assert hdr in obs                     # 채워져 있어도 보인다
            if memo:
                assert memo in obs
