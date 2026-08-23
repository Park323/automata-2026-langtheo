"""에이전트 루프 통합. 과제 2 Part B-3. StubClient 로 검증 (API 안 씀)."""
from __future__ import annotations

import itertools
import json
import random
import re
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


# **인구가 늘어난다** (8/21). `bear_child` 는 부모를 죽이지 않으므로 초기 9명 말고도
# 사람이 생긴다 — 초기 id 로만 만든 클라이언트 사전은 새 사람에게서 KeyError 를 낸다.
# 없는 id 는 즉시 끝내는 스텁으로 채운다.
def _client_for(clients, script_end):
    def get(aid):
        if aid not in clients:
            clients[aid] = StubClient([script_end] * 4)
        return clients[aid]
    return get


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
                       sequential=sequential, render_events=prompts.render_events,
                       render_arrivals=prompts.render_arrivals)


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
                 48, itertools.count(1000), RunResult(world=world), {}, [])
    got = _dequeue_inbox_pop(world, "Asla2")
    assert "HELLO_SAME_TURN" in str(got)                    # 같은 턴에 도착
    assert _dequeue_inbox_pop(world, "Asla2") == []         # 두 번 안 옴 (제거됨)

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
    res = run_agentic(cfg, random.Random(1), _client_for(clients, assistant_msg(tool_call("end_turn", "z", reasoning="r"))),
                      StubClient([{"role": "assistant", "content": "t",
                                   "tool_calls": []}] * 30),
                      48.0, prompts.render_turn_open, prompts.system_for,
                      parallel=False)
    convo = res.world.agents["Asla1"].convo
    users = [m["content"] for m in convo if m["role"] == "user"]
    assert len(users) == 3                       # 해마다 한 마디
    for u in users:
        assert "行動の費用" not in u               # 비용표는 대화에 없다
        assert "自国の進捗" not in u                # 진척·국토도 없다
        assert "になりました" in u                 # 해를 여는 한 마디는 있다
        # **소득과 그때의 예산은 있다** — 그 해에 일어난 일이고, 관측에 두면 매 콜 다시
        # 계산돼 값이 흔들린다 (실측: 한 해 안에 +100 → +104 → +105).
        assert "今年の収入は" in u


def test_arrived_messages_stay_in_the_conversation():
    """도착한 메시지는 **사건**이라 쌓여야 한다. 그것만이 에이전트 컨텍스트 안의 유일한
    대화 기록이다 — 상태처럼 갈아치우면 누가 무슨 말을 했는지 잊는다."""
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    world.turn = 2
    a = world.agents["Asla1"]
    box = [{"msg_id": 7, "from": "Ranoa1", "label": "[AI translation]",
            "text": "MARK_ARRIVED", "original": None}]
    txt = prompts.render_arrivals(a, box)
    assert "MARK_ARRIVED" in txt and "Ranoa1" in txt
    # 그리고 관측(system)에는 없다 — 두 군데 있으면 어긋날 수 있다
    assert "MARK_ARRIVED" not in prompts.system_for(a, world, cfg, 48.0)


def test_system_without_a_world_is_just_the_rules():
    """문구 검사용. 규칙만 돌려준다."""
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    a = world.agents["Asla1"]
    assert "予算" not in prompts.system_for(a, None, cfg)
    assert "予算" in prompts.system_for(a, world, cfg, 48.0)


def test_an_empty_inbox_says_nothing_at_all():
    """**「도착한 메시지: 없음」 을 붙이면 아무 일도 없었다는 사실이 매 턴 쌓인다.**

    없는 것을 굳이 적지 않는다 (`prompt_audit` 0절). 안 적혀 있으면 안 온 것이다.
    """
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    world.turn = 3
    a = world.agents["Asla1"]
    # **오프닝은 이제 도착분을 담지 않는다** — 각 채널이 자기 자리를 갖는다
    empty = prompts.render_turn_open(world, a, cfg, 48.0, [])
    assert empty.count("\n") == 1                     # 소득·예산 한 줄 + 집행 한 줄
    assert prompts.render_arrivals(a, []) == ""        # 온 것이 없으면 빈 문자열
    for none_word in ("なし", "没有", "aucun", "Aucun"):
        assert none_word not in empty
    got = prompts.render_arrivals(a, [{"msg_id": 1, "from": "Ranoa1", "label": None,
                                       "text": "MARK", "original": None}])
    assert "MARK" in got


def test_income_is_stated_once_a_year_not_recomputed_each_call():
    """**소득이 관측에 있던 동안 턴 안에서 값이 흔들렸다.**

    실측(`ovh15` · Asla1 · 44년):

        step 2   予算: 100 · 今年の収入: +100     ← 맞다
        step 3   予算:   0 · 今年の収入: +104
        step 6   予算:   0 · 今年の収入: +105

    관측은 매 콜 새로 렌더되므로, 남들이 `national` 에 넣어 배수가 커지면 **이미 받은
    소득**이 올라간다. 그리고 예산 0 옆에 붙어서 「104 를 받았는데 0」 으로 읽힌다.

    소득은 **그 해에 일어난 일**이다. 해가 열릴 때 한 번 적으면 사실로 굳는다.
    """
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    world.turn = 3
    a = world.agents["Asla1"]
    a.budget = 137.0
    open_ = prompts.render_turn_open(world, a, cfg, 48.0, [])
    assert "+100" in open_ and "137" in open_

    # 그 해 안에 남들이 국가에 투자해 배수가 커져도 이미 적힌 말은 안 변한다
    world.countries["Asla"].national_capital = 12_000.0
    obs = prompts.render_observation(world, a, cfg, 48.0)
    assert "収入:" not in obs and "収入は" not in obs


def test_growing_older_accumulates_in_the_conversation():
    """**나이가 관측에 있으면 매 콜 덮여서 나이 드는 것이 느껴지지 않는다.**

    해가 열릴 때 적으면 대화에 6살 · 7살 · 8살이 차례로 남는다. 그것이 `bear_child` 를
    고를 시점을 가늠하는 유일한 재료다 — 수명 곡선은 비공개이고(4.1), 부고에 찍힌 나이와
    자기 나이의 흐름만이 단서다. 세 런 21명이 전부 자연사한 뒤에 붙인 것이 부고의
    나이였고, 이건 그 짝이다.
    """
    cfg = _cfg(3)
    world = init_world(cfg, itertools.count(1))
    a = world.agents["Asla1"]
    seen = []
    for t, age in ((1, 6), (2, 7), (3, 8)):
        world.turn, a.age = t, age
        seen.append(prompts.render_turn_open(world, a, cfg, 48.0, []))
    assert ["6 歳" in seen[0], "7 歳" in seen[1], "8 歳" in seen[2]] == [True] * 3
    # 관측에는 자기 나이가 없다 — **다만 비용표의 「10 歳から」 는 규칙 상수다** (8/21).
    # 그것까지 막으면 아이 낳기의 조건을 적을 수 없다.
    #
    # 관측에는 없다 — 그리고 죽은 문구도 남기지 않는다
    obs = prompts.render_observation(world, a, cfg, 48.0)
    assert f"{a.age} 歳" not in obs                 # 내 나이는 없다
    for line in obs.splitlines():
        if "歳" in line:                            # 남는 것은 규칙 상수 한 줄뿐이다
            assert f"{cfg.world.adult_age} 歳" in line and "bear_child" in line, line
    assert "age" not in prompts.T["ja"]


def test_a_mid_year_arrival_does_not_reopen_the_year():
    """**같은 해가 여러 번 열린 것처럼 보이고 있었다.**

    순차 라운드로빈은 한 해에 여러 번 차례가 오고 그 사이에 메시지가 도착한다. 그때마다
    해 오프닝을 다시 붙이면 실측처럼 된다:

        到了 42 年。你 5 岁。今年的收入是 +100，手上的预算是 100。
        到了 42 年。你 5 岁。今年的收入是 +100，手上的预算是 97。   ← 같은 해다

    안의 예산이 흔들리고(100 → 97) 이미 받은 소득을 다시 말한다. `ovh15` 에서 135
    에이전트-해 중 **49건(36%)** 이 오프닝을 두 번 이상 받았다.

    재방문에는 **도착분만** 적는다.
    """
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    world.turn = 1
    a = world.agents["Asla1"]
    a.budget = 100.0
    box = [{"msg_id": 3, "from": "Asla2", "label": None, "text": "MARK", "original": None}]

    first = prompts.render_turn_open(world, a, cfg, 48.0, box, opening=True)
    later = prompts.render_turn_open(world, a, cfg, 48.0, box, opening=False)
    arrived = prompts.render_arrivals(a, box)

    assert "になりました" in first and "MARK" not in first     # 오프닝은 오프닝만
    assert later == ""                                        # 재방문엔 해를 열지 않는다
    assert "MARK" in arrived and "になりました" not in arrived
    assert "100" not in arrived                               # 소득·예산도 없다


def test_an_ended_agent_wakes_when_mail_arrives():
    """**끝냈다고 했는데 그 뒤에 말이 오면 다시 깨운다.**

    `end_turn` 은 「지금 더 할 일이 없다」 는 판단이다. 그 뒤에 도착한 메시지는 **그
    판단의 근거를 무너뜨리는 새 정보**다 — 누가 협력을 청했는데 이미 끝냈다고 그 해가
    통째로 지나가면, 같은 해 왕복 대화라는 순차 라운드로빈의 취지가 절반만 산다.

    필요한 것은 **수신자가 발신자보다 먼저 오는 순서**다. Asla2 가 먼저 끝내고, 그 뒤
    Asla1 이 말을 보내고, 그래서 Asla2 가 다시 불린다.

    **시드를 박아 두지 않는다.** 전에는 「시드 3」 이라고 적어 두었는데, 초기화가 rng 를
    쓰는 방식이 바뀌자(나이 범위·개체 배수 추첨) 그 시드의 순서가 달라져 테스트가 조용히
    다른 것을 재게 됐다. 필요한 순서가 나오는 시드를 **찾는다.**
    """
    end = assistant_msg(tool_call("end_turn", "e", reasoning="r"))
    speak = assistant_msg(tool_call("speak", "s", to="Asla2", text="WAKE_UP", reasoning="r"))
    for seed in range(40):
        cfg = _cfg(1)
        clients = _clients({"Asla1": [speak, end], "Asla2": [end, end]})
        res = _run(cfg, clients, seed=seed, parallel=False, sequential=True)
        if len(clients["Asla2"].calls) == 2:
            break
    else:
        raise AssertionError("수신자가 먼저 오는 시드를 40개 안에서 못 찾았다")

    # 깨어나 두 번 불렸다 (한 번은 처음, 한 번은 도착 뒤)
    assert len(clients["Asla2"].calls) == 2
    convo = json.dumps(res.world.agents["Asla2"].convo, ensure_ascii=False)
    assert "WAKE_UP" in convo                         # 같은 해에 봤다
    # 깨어난 뒤 붙은 것은 도착분만 — 해를 다시 열지 않는다
    users = [m["content"] for m in res.world.agents["Asla2"].convo if m["role"] == "user"]
    assert len(users) == 2 and "になりました" not in users[1]
    assert all(u for u in users)                      # 빈 항목이 없다


def test_a_stopped_agent_is_not_woken_when_it_cannot_act():
    """깨우는 조건은 셋이다 — **스스로 끝냈고 · 행동력이 남았고 · 받을 것이 있다.**

    `exhausted`·`error`·`repeat_guard`·`runaway` 로 멈춘 것을 깨우면 같은 실패를
    되풀이한다. 행동력이 0 이면 깨워도 할 수 있는 것이 `memory_write` 뿐이다.
    """
    from core.loop import _has_inbox
    cfg = _cfg(1)
    world = init_world(cfg, itertools.count(1))
    world.turn = 1
    assert _has_inbox(world, "Asla2") is False
    world.inbox_queue.append({"deliver_turn": 1, "to": "Asla2",
                              "to_uid": world.agents["Asla2"].uid,
                              "msg": {"msg_id": 1, "from": "Asla1", "text": "x"}})
    assert _has_inbox(world, "Asla2") is True
    # **꺼내지 않고 본다** — 깨울지 판단만 하고, 실제 수령은 그 차례에 한다
    assert _has_inbox(world, "Asla2") is True
    assert len(world.inbox_queue) == 1

    # 수신 슬롯이 바뀌었으면(사망·교체) 받을 것이 아니다
    world.inbox_queue[0]["to_uid"] = 9999
    assert _has_inbox(world, "Asla2") is False


def test_world_events_get_their_own_context_entry():
    """**세계의 사건은 해 오프닝과 섞이지 않는다.**

    죽음·출자 결과·전달 실패는 「올해가 시작됐다」 와 성질이 다르다. 오프닝에 묶으면
    **새해 인사에 부고가 딸려 오고**, 무엇이 언제 일어났는지가 한 덩어리로 뭉개진다.

    대화에서 사건이 **앞**에 온다 — 해 끝에 죽은 사람 소식이 다음 해가 열리기 전에
    놓인다. 모델은 대화를 볼 때만 무엇을 알게 되므로, 그 순서가 곧 「그 시점」 이다.

    머리말도 다르다. 「도착한 메시지」 는 사람이 나에게 한 말이고, 사건은 세계가 나에게
    알리는 것이다 — 부고를 「메시지」 라고 부르면 누가 보낸 것처럼 읽힌다.
    """
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    world.turn = 2
    a = world.agents["Asla2"]
    a.budget = 137.0
    box = [{"died": "Asla1", "born": "Asla4", "age": 14},
           {"fac_gain": 61, "amount": 200.0, "to": "Asla"},
           {"from": "Ranoa1", "label": None, "text": "SAID", "original": None}]

    ev = prompts.render_events(a, box)
    arrived = prompts.render_arrivals(a, box)

    assert "Asla1" in ev and "61" in ev and "SAID" not in ev
    assert "起きたこと" in ev and "になりました" not in ev      # 해를 열지 않는다
    assert "SAID" in arrived and "Asla1" not in arrived       # 사건은 여기 없다
    assert prompts.T["ja"]["in_hdr"] not in ev                # 머리말이 다르다
    # 사건만 왔으면 도착분은 빈 문자열 — 루프가 붙이지 않는다
    assert prompts.render_arrivals(a, box[:2]) == ""


def test_the_event_entry_lands_before_the_year_opens():
    """루프가 **사건을 먼저** 붙인다. 부고가 「43년이 되었습니다」 뒤에 오면 순서가
    거꾸로다 — 그 죽음은 42년 끝에 일어난 일이다."""
    from core.loop import _push_events
    cfg = _cfg(2)
    world = init_world(cfg, itertools.count(1))
    a = world.agents["Asla2"]
    a.convo = []
    _push_events(a, [{"died": "Asla1", "born": "Asla4", "age": 14}], prompts.render_events)
    assert len(a.convo) == 1 and a.convo[0]["role"] == "user"
    assert "Asla1" in a.convo[0]["content"]

    # 사건이 없으면 아무것도 붙이지 않는다
    _push_events(a, [{"from": "X", "text": "y"}], prompts.render_events)
    assert len(a.convo) == 1
    # 렌더러를 안 주면 아무 일도 하지 않는다 (옛 경로 호환)
    _push_events(a, [{"died": "Z"}], None)
    assert len(a.convo) == 1


def test_the_year_opens_before_anything_that_happened_in_it():
    """**나중에 차례가 온 사람이 이런 대화를 받고 있었다.**

        user: 起きたこと: 自国の技術力が上がりました。
        user: 42 年になりました。…

    그 기술력 상승은 42년에 일어난 일이다. **해는 모두에게 같은 때 밝는다** — 소득도 AP 도
    턴 시작에 한꺼번에 주어진다. 먼저 행동한 사람의 결과가 남의 새해보다 앞에 놓이면
    시간이 거꾸로 읽힌다.
    """
    cfg = _cfg(1)
    inv = assistant_msg(tool_call("invest", "i", target="national", reasoning="r"))
    end = assistant_msg(tool_call("end_turn", "e", reasoning="r"))
    clients = _clients({aid: [inv, end] for aid in IDS})
    res = _run(cfg, clients, seed=3, parallel=False, sequential=True)
    opens = {"ja": "になりました", "zh": "到了", "fr": "est arrivé"}
    for aid in IDS:
        agent = res.world.agents[aid]
        users = [m["content"] for m in agent.convo if m["role"] == "user"]
        assert users and opens[agent.native_lang] in users[0], aid   # 해가 먼저 밝는다
        assert all(u for u in users), aid                            # 빈 항목이 없다
        # **받은 소득 그대로.** 렌더 때 다시 계산하면 나중에 차례가 온 사람은 남들이
        # national 에 넣은 뒤의 값(+102)을 보게 된다 — 실제로 받은 것은 100 이다.
        assert f"+{agent.income_this_year:.0f}" in users[0], aid


def test_capital_notice_carries_the_gain_as_a_percentage():
    """「기술력이 올랐다」 에 **이번 상승분**을 싣는다 (8/23).

    배수(「1.174 배」)도 누적(「당초보다 17%」)도 아니다 — 사건 줄은 「방금 무슨 일이
    있었나」 이고, 한 차례 상승분은 0.05~0.6% 라 소수 두 자리여야 값이 남는다.

    전에는 값이 없는 사실이라 해마다 한 번으로 접었다. 이제 값이 있으므로 그 제한을
    뗐고, 접는 일은 `render_inbox._add` 가 값으로 판단한다 — 진척과 같은 취급이다.

    값이 없으면 「national 에 더 부을까 facility 에 부을까」 를 수치로 비교할 수 없다.
    """
    cfg = _cfg(1)
    inv = assistant_msg(tool_call("invest", "i", target="national", reasoning="r"))
    end = assistant_msg(tool_call("end_turn", "e", reasoning="r"))
    clients = _clients({aid: [inv, inv, end] for aid in IDS})
    res = _run(cfg, clients, seed=3, parallel=False, sequential=True)
    blob = "\n".join(m["content"] for m in res.world.agents["Asla2"].convo
                     if m["role"] == "user")
    assert "技術力が" in blob
    # 상승분은 0 보다 크고, 합치면 누적 배수와 맞아야 한다 (곱으로 쌓인다)
    got = [float(x) for x in re.findall(r"技術力が ([\d.]+)% 上がりました", blob)]
    assert got and all(v > 0 for v in got), got
    prod = 1.0
    for v in got:
        prod *= 1 + v / 100
    assert prod == pytest.approx(res.world.countries["Asla"].multiplier(cfg), rel=1e-3)


def test_identical_rows_inside_one_batch_collapse():
    """한 묶음 안에서도 같은 줄은 한 번만. 실측에서 세 줄이 나란히 붙은 적이 있다."""
    cfg = _cfg(1)
    world = init_world(cfg, itertools.count(1))
    a = world.agents["Asla1"]
    ev = prompts.render_events(a, [{"cap_up": True, "cap_gain": 0.23}] * 3)
    assert ev.count("上がりました") == 1
    # **소수 두 자리에서 갈리면 다른 줄이다** — 낸 액수가 다르면 오른 폭도 다르고,
    # 그건 접어서 없앨 정보가 아니다 (진척 `prog_up` 과 같은 취급).
    ev3 = prompts.render_events(a, [{"cap_up": True, "cap_gain": 0.23},
                                    {"cap_up": True, "cap_gain": 0.07}])
    assert ev3.count("上がりました") == 2
    # 값이 다르면 접히지 않는다
    ev2 = prompts.render_events(a, [{"prog_up": 18, "now": 18},
                                    {"prog_up": 34, "now": 52}])
    assert ev2.count("進捗が") == 2
