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


# **인구가 늘어난다** (8/21). `bear_child` 는 부모를 죽이지 않으므로 초기 9명 말고도
# 사람이 생긴다 — 초기 id 로만 만든 클라이언트 사전은 새 사람에게서 KeyError 를 낸다.
# 없는 id 는 즉시 끝내는 스텁으로 채운다.
def _client_for(clients, script_end):
    def get(aid):
        if aid not in clients:
            clients[aid] = StubClient([script_end] * 4)
        return clients[aid]
    return get


@pytest.fixture()
def cfg():
    return config.load("configs/base.yaml")


@pytest.fixture()
def world(cfg):
    w = loop.init_world(cfg, itertools.count(1))
    # **개체 차이를 1.0 으로 눕힌다** (8/22) — 다른 기제를 재는 테스트가 사람마다 다른
    # 액수에 흔들리지 않게. 차이 자체는 test_world_rules_v2 의 전용 테스트가 본다.
    for a in w.agents.values():
        a.income_mult = a.invest_mult = 1.0
    return w


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
    """memory_write 로 적은 것이 다음 턴 관측에 보인다.

    **압박선 위에서만 쓸 수 있다** (8/21) — 그래서 먼저 그 상태로 만든다. 기억은 잃을
    것이 생긴 뒤에 뜻이 있는 도구다.
    """
    world.agents["Asla1"].last_prompt_tokens = cfg.llm.context_limit
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


def test_nothing_of_a_life_survives_a_natural_death(cfg, world):
    """**유언이 없어졌다** (8/21). 자연사에는 넘길 통로가 아예 없다.

    전에는 `procreate` 가 유언·예산·학습 진척 절반을 아이에게 넘겼다. 이제 재생산은
    `bear_child` 이고 **부모가 죽지 않으므로**, 넘기고 싶은 것은 살아서 넘긴다 — 돈은
    주고, 말은 하고, 언어는 가르친다 (부모가 국내 구사자이므로 아이의 학습이 싸다).

    그래서 죽음은 **순수한 소실**이 됐다. 개인에 속한 것은 전부 사라진다.
    """
    a = world.agents["Asla1"]
    a.memory = "내가 평생 알아낸 것"
    a.convo.append({"role": "user", "content": "옛 기억"})
    a.lang_progress = {"fr": 400.0}
    a.budget = 500.0

    child = loop._newborn("Asla9", "Asla", "ja", 0.0, set(), world.turn,
                          "natural", cfg, itertools.count(9000))
    loop._replace(world, "Asla1", child, [])

    assert "Asla1" not in world.agents        # id 는 재사용하지 않는다
    assert child.convo == [] and child.memory == ""
    assert child.lang_progress == {} and child.budget == 0.0
    assert child.parent_langs == set()       # 자연사에는 부모가 없다 (3.2)
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

    `memory_write` 가 돈도 AP 도 안 쓰므로, 자원이 바닥나도 고를 것이 남아 있다. 여기서
    거짓으로 False 를 돌려주면 **합법적인 행동을 잘라내게** 된다. 정상 종료는 `end_turn`
    이고, 폭주는 `RUNAWAY_CAP`(64) 이 막는다.

    **`procreate` 는 없어졌다** (8/21). 그것도 AP 0 이었는데 `bear_child` 는 1.0 을 문다 —
    그래서 「공짜 행동」 은 이제 `memory_write` 하나이고, 그것도 압박선 위에서만 열린다.
    """
    a = world.agents["Asla1"]
    a.budget, a.ap = 0.0, 0.0
    a.memory_open = True                       # 압박선 위 — 기억이 열려 있다
    assert cfg.ap.memory_write == 0.0
    assert can_act(a, cfg, 48.0) is True


def test_can_act_is_false_when_nothing_is_affordable(cfg, world):
    """**`can_act` 가 항상 참이었다** (#47).

    마지막 줄이 `_afford(ap, min(ap.memory_write, ap.vote))` 였고 `ap.memory_write = 0.0`
    이라 값이 늘 0 이 되어 **AP 가 0 이어도 참**이었다. 게다가 `memory_write` 는 압박선
    아래에서 도구 목록에 아예 없는데 그 값을 「공짜 행동」 으로 세고 있었다.

    종료 조건 ②(자원 소진)가 그렇게 죽어 있었고, 라운드로빈은 할 수 있는 것이 없는
    사람을 한 번 더 깨워 `end_turn` 만 시켰다 — API 호출 하나가 그대로 나간다.

    위의 `test_free_actions_keep_can_act_true` 와 짝이다: **열려 있으면 참, 닫혀 있고
    자원이 없으면 거짓.**
    """
    a = world.agents["Asla1"]
    a.memory_open = False                      # 압박선 아래 — 기억은 목록에 없다
    a.budget, a.ap = 0.0, 0.0
    assert can_act(a, cfg, 48.0) is False
    a.budget = 10_000.0                        # 돈이 넘쳐도 AP 가 0 이면 못 한다
    assert can_act(a, cfg, 48.0) is False
    a.ap = cfg.ap.vote                         # 採決일이면 표는 던질 수 있다
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
    res = loop.run_agentic(cfg, random.Random(1), _client_for(clients, assistant_msg(tool_call("end_turn", "z", reasoning="r"))), tr, 48.0,
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
    clients = {a: StubClient([end, end]) for a in ids}   # 새로 태어난 사람은 _client_for 가 채운다
    res = loop.run_agentic(cfg, random.Random(1), _client_for(clients, assistant_msg(tool_call("end_turn", "z", reasoning="r"))),
                           StubClient([{"role": "assistant", "content": "t", "tool_calls": []}] * 30),
                           48.0, prompts.render_turn_open, prompts.system_for, parallel=False)
    # **신원은 이제 system 에 있다.** 관측(지금 그러한 것)이 system 으로 옮겨갔고,
    # 대화에는 턴을 여는 한 마디와 내 행동·결과만 쌓인다.
    # **9명이 그대로 있지 않다** (8/21). 자연사는 자리를 갈고, 아이 낳기는 사람을 늘린다.
    # 그래서 「초기 id」 가 아니라 **지금 살아 있는 사람들**을 본다.
    live = sorted(res.world.agents)
    for aid in live:
        agent = res.world.agents[aid]
        sysp = prompts.system_for(agent, res.world, cfg, 48.0)
        assert (f"あなたは {aid}" in sysp or f"你是 {aid}" in sysp
                or f"Vous êtes {aid}" in sysp), aid
        for other in live:
            if other != aid:
                assert f"あなたは {other}" not in sysp
                assert f"你是 {other}" not in sysp
                assert f"Vous êtes {other}" not in sysp
        # 대화에 남의 신원이 섞이지 않는다
        blob = json.dumps(agent.convo, ensure_ascii=False)
        for other in live:
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
        # **도구가 열려 있을 때의 머리말이다** (8/21). 기억은 압박선 위에서만 목록에
        # 오르므로, 그 아래에서는 도구 이름을 말하지 않는 머리말(`mem_hdr_ro`)이 나간다.
        ag.memory_open = True
        for memo in ("", "요격기에 몰아줘라"):
            ag.memory = memo
            obs = prompts.system_for(ag, world, cfg, 48.0)
            assert hdr in obs                     # 채워져 있어도 보인다
            if memo:
                assert memo in obs


def test_the_memo_is_visible_even_when_the_tool_is_closed(cfg, world):
    """**메모 자체는 늘 보인다.** 물려받은 유언이 여기 들어 있고, 그건 쓸 수 없을 때도
    자기가 들고 다니는 것이다 (spec 3.3).

    다만 머리말이 도구 이름을 말하는 것은 그 도구가 있을 때뿐이다 — 없는 도구를 설명하면
    「부를 수 있다」 는 거짓이 되고, 실제로 부르면 거절당한다.
    """
    from domains.meteor import prompts
    ag = world.agents["Asla1"]
    ag.memory = "부모의 유언: 요격기에 몰아줘라"
    ag.memory_open = False                        # 도구가 닫혀 있다
    obs = prompts.system_for(ag, world, cfg, 48.0)
    assert ag.memory in obs                       # 유언은 보인다
    assert prompts.T["ja"]["mem_hdr_ro"] in obs
    assert "書き足すのではなく" not in obs          # 쓰는 방법은 말하지 않는다
    assert "  memory_write" not in obs            # 비용표에도 없다


# ── 기억을 쓰면 자리를 산다 ────────────────────────────────────────────────────

def _fat_convo(n_blocks: int, chars: int = 2200) -> list[dict]:
    """user 로 시작하는 블록 n 개. `_turn_blocks` 가 user 를 경계로 쪼갠다."""
    out = []
    for i in range(n_blocks):
        out.append({"role": "user", "content": f"[{i}] " + "x" * chars})
        out.append({"role": "assistant", "content": None, "tool_calls": []})
        out.append({"role": "tool", "tool_call_id": "t", "content": '{"ok": true}'})
    return out


def test_writing_memory_under_pressure_buys_room(cfg, world):
    """**경고를 받고 기억을 적어도 아무것도 줄지 않았다.**

    압박 경고는 사실 통지인데, memory_write 를 해도 대화는 그대로 쌓이고
    `last_prompt_tokens` 도 그대로여서 경고가 다음 호출에도 계속 붇었다. 20턴 런에서:

        압박 판정      135 에이전트-해 중 **94건(70%)**. 한 번 걸리면 죽을 때까지 유지
        Miris6 턴14    한 해에 memory_write **10회** — 안 꺼지는 경고를 계속 껐다
        Asla3 턴7~15   거꾸로 **한 번도 안 적었다**. 안 꺼지는 경고는 잡음이 된다

    그래서 압박 아래의 memory_write 를 **거래**로 만든다 — 적으면 자리가 생긴다.
    """
    from core import agent_loop
    a = world.agents["Asla1"]
    a.ap, a.convo = 1.0, _fat_convo(8)
    a.last_prompt_tokens = cfg.llm.context_limit          # 압박 안
    assert agent_loop.under_pressure(a, cfg)

    dropped = agent_loop.compact_after_memory(a, cfg, agent_loop._TOOL_TOKENS)
    assert dropped > 0
    # **경고선 아래로** 내려간다 — 한계선까지만 내리면 다음 호출에 다시 켜진다
    warn = cfg.llm.context_limit * cfg.llm.warn_ratio
    assert agent_loop.estimate_tokens(a.convo, agent_loop._TOOL_TOKENS) <= warn
    # 그리고 그 자리에서 경고가 꺼진다 — 다음 호출을 기다리지 않는다
    assert not agent_loop.under_pressure(a, cfg)


def test_compaction_keeps_the_exchange_in_progress(cfg, world):
    """**지금 진행 중인 주고받기는 남는다.** 압축은 한계선보다 낮은 값까지 내려가므로
    블록 하나만 남기면 방금 부른 도구의 결과가 사라질 수 있다."""
    from core import agent_loop
    a = world.agents["Asla1"]
    a.convo = _fat_convo(6)
    a.convo[-1] = {"role": "tool", "tool_call_id": "t", "content": '{"ok": true, "mine": 1}'}
    a.last_prompt_tokens = cfg.llm.context_limit
    agent_loop.compact_after_memory(a, cfg, agent_loop._TOOL_TOKENS)
    assert len(agent_loop._turn_blocks(a.convo)) >= 2
    assert a.convo[-1]["content"] == '{"ok": true, "mine": 1}'


def test_no_compaction_when_there_is_room(cfg, world):
    """여유가 있으면 버리지 않는다 — 아무 이득 없이 대화만 잃는다."""
    from core import agent_loop
    a = world.agents["Asla1"]
    a.convo = _fat_convo(2, chars=50)
    a.last_prompt_tokens = 10
    assert not agent_loop.under_pressure(a, cfg)
    before = list(a.convo)
    assert agent_loop.compact_after_memory(a, cfg, agent_loop._TOOL_TOKENS) == 0
    assert a.convo == before


def test_memory_write_compacts_on_both_paths(cfg, world):
    """**두 경로가 같이 움직여야 한다.** 순차 라운드로빈에만 넣으면 병렬 경로에서는
    기억을 적어도 압박이 안 풀리고, 그 차이는 테스트가 한쪽만 보면 안 보인다.

    소스를 읽어 확인한다 — 배선은 두 함수 안에 있고, 한쪽을 지워도 다른 쪽 테스트는
    통과하기 때문이다.
    """
    import inspect

    from core import agent_loop
    for fn in (agent_loop._agent_one_call, agent_loop.run_agent_turn):
        src = inspect.getsource(fn)
        assert "compact_after_memory" in src, fn.__name__
        assert "under_pressure(agent, cfg)" in src, fn.__name__


def test_memory_write_is_refused_below_the_threshold(cfg, world):
    """**목록에 없을 때 불러도 거절한다.** 대화에 남은 옛 스키마를 보고 부를 수 있고,
    그때 조용히 통과시키면 「압박 뒤에만」 이 절반만 지켜진다.

    30해 실측에서 `memory_write` 가 **206번** 불렸다 — 압박이 걸리기 한참 전부터다. 그
    값이 공짜(돈 0 · AP 0)라 무엇도 막지 않지만, 순차 라운드로빈은 스텝 단위로 도므로
    한 번 부르면 그만큼 남들이 먼저 움직인다. **공짜가 아니라 차례를 쓴다.**
    """
    from core import tools as T
    from core.agent_loop import Sink, execute_tool
    a = world.agents["Asla1"]
    a.ap = 1.0

    a.memory_open = False                         # 압박 아래
    assert not any(t["function"]["name"] == "memory_write"
                   for t in T.tools_for(cfg, memory=False))
    r, _ = execute_tool("memory_write", {"text": "x"}, world, a, cfg, Sink(), 48.0)
    assert not r["ok"] and "not available yet" in r["error"]
    assert a.memory == "" and a.ap == 1.0         # 아무것도 쓰지 않고 AP 도 안 쓴다

    a.memory_open = True                          # 압박 위
    assert any(t["function"]["name"] == "memory_write"
               for t in T.tools_for(cfg, memory=True))
    r, _ = execute_tool("memory_write", {"text": "x"}, world, a, cfg, Sink(), 48.0)
    assert r["ok"] and a.memory == "x"
