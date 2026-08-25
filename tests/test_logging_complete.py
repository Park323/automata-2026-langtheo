"""로깅 전수 — **빠진 것을 사람이 아니라 기계가 잡게 한다.**

로깅 누락은 늘 같은 방식으로 발견됐다. 분석하려고 파일을 열어 보니 없었고, 그때는
이미 그 런을 다시 돌려야 했다. 세 런 60건의 `memory_write` 가 전부
`{"type": "memory_write"}` 로만 남은 것이 그랬다 — *"본문은 messages 에 있으므로 뺀다"*
라는 주석이 `speak` 에만 맞는 말이었는데 종류를 안 가리고 잘랐다.

그래서 여기서 지키는 것은 값이 아니라 **덮개**다. 세계 상태의 필드가 늘면 로그에
안 들어간 채로 통과하지 못하게 하고, API 왕복은 요청·응답 전문이 그대로 남는지 본다.

    ① 상태     Agent·Country 의 필드가 전부 로그에 닿는가
    ② 호출     도구 호출의 인자·결과·실패 사유가 남는가
    ③ API      요청 본문과 응답 전문이 가공 없이 남는가
    ④ 이음매   raw_calls 를 events·messages 와 이어붙일 키가 있는가
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import random

import pytest

from core import config, loop
from core.agent_loop import Sink
from core.run_io import RunWriter
from core.state import Agent, Country


# **노브는 이제 AP 다** (8/25). 돈 값 48 을 넘기면 「48 AP」 가 되어
# 한 해(1.0)를 넘고 발신이 불가능해진다 — 타입이 같아 아무도 안 잡았다.
KNOB = 0.5          # comm_intl_ai_ap 의 최고값

BASE = "configs/base.yaml"


@pytest.fixture()
def cfg():
    return config.load(BASE)


@pytest.fixture()
def world(cfg):
    w = loop.init_world(cfg, itertools.count(1), random.Random(1))
    w.turn = 1
    # **개체 차이를 1.0 으로 눕힌다** (8/22). 소득 배수·처리량 배수는 태어날 때 뽑히므로,
    # 그것을 그대로 두면 **다른 기제를 재는 테스트가 사람마다 다른 액수에 흔들린다.**
    # 차이 자체는 `test_world_rules_v2.py` 의 전용 테스트가 본다.
    for _a in w.agents.values():
        _a.invest_mult = 1.0
    return w


def _rows(w, name):
    """RunWriter 는 파일을 열어둔 채 버퍼링한다 — 읽기 전에 닫아야 한다."""
    w.close() if hasattr(w, "close") else None
    for f in getattr(w, "_files", {}).values():
        try:
            f.flush()
        except ValueError:
            pass
    path = w.dir / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_one_turn(tmp_path, cfg, world, logs=None):
    w = RunWriter("t", cfg_raw={"x": 1}, root=tmp_path)
    r = loop.RunResult(world=world)
    r.agent_logs = [logs or {}]
    w.on_turn_end(1, r)
    return w, r


# ── ① 상태 ─────────────────────────────────────────────────────────────────────

# 로그로 안 내보내도 되는 필드와 **그 이유**. 이유를 못 적는 필드는 내보내야 한다.
AGENT_EXEMPT = {
    "id": "state 행의 `agent` 가 곧 id",
    "lam": "`lambda` 로 나간다 (본인에게도 비공개인 값이지만 로그에는 남는다)",
    "ap": "`ap_left` 로 나간다",
    "convo": "raw_calls 의 messages 가 매 호출의 전문을 담는다 — 중복이고 훨씬 크다",
    "last_prompt_tokens": "events.agent_turn.prompt_tokens 로 나간다",
    "memory": "state 의 `memory` 로 나간다",
    "lang_progress": "state 의 `lang_progress` 로 나간다",
    "invested_turn": "턴 안에서만 쓰는 누적치 — calls 의 결과에 실제 과금이 남는다",
}


# 턴 로그(`_turn_log`)에서 로그로 안 내보내도 되는 키와 **그 이유**.
TURN_LOG_EXEMPT: dict[str, str] = {}


def test_every_turn_log_key_reaches_the_log(tmp_path, cfg, world):
    """**`_turn_log` 가 내는 키가 전부 events.agent_turn 에 닿아야 한다.**

    `run_io` 는 필드를 **손으로 열거**한다. `_turn_log` 에 키를 더하고 그쪽을 잊으면
    조용히 사라지는데, 기존 가드는 `Agent`·`Country` **dataclass** 만 봐서 이걸 못 봤다.

    실제로 `compacted_blocks`(기억으로 산 자리)와 `truncated_calls` 가 그렇게 빠져
    있었다 — `inh30` 30해에서 압박 아래 `memory_write` 가 127건이었는데 **거래가
    일어났는지 사후에 확인할 수 없었다.** 그것이 spec 4.5 의 관측 대상이다.
    """
    from core.agent_loop import _StepAcc, _turn_log
    keys = set(_turn_log(world.agents["Asla1"], _StepAcc(), "exhausted", 0.0))
    w, _ = _write_one_turn(
        tmp_path, cfg, world,
        logs={"Asla1": _turn_log(world.agents["Asla1"], _StepAcc(), "exhausted", 0.0)})
    got = set()
    for row in _rows(w, "events"):
        if row.get("type") == "agent_turn":
            got |= set(row)
    missing = {k for k in keys - got if k not in TURN_LOG_EXEMPT}
    assert not missing, f"events.agent_turn 에 안 나가는 턴 로그 키: {sorted(missing)}"


def test_every_agent_field_reaches_the_log(tmp_path, cfg, world):
    """**필드를 늘리면 로그도 같이 늘어야 한다.**

    면제하려면 `AGENT_EXEMPT` 에 이유를 적어야 한다. 이유를 적을 수 없으면 그건
    빠뜨린 것이다.
    """
    w, _ = _write_one_turn(tmp_path, cfg, world)
    keys = set()
    for row in _rows(w, "state"):
        keys |= set(row)
    fields = {f.name for f in dataclasses.fields(Agent)}
    missing = {f for f in fields - keys if f not in AGENT_EXEMPT}
    assert not missing, f"state.jsonl 에 안 나가는 Agent 필드: {sorted(missing)}"


def test_country_state_including_the_open_proposal(tmp_path, cfg, world):
    """열린 제안이 매 턴 남아야 한다 — vote 이벤트만으로는 *구간* 을 복원해야 한다."""
    world.countries["Asla"].proposal = {"target": "bunker", "by": "Asla1",
                                        "opened_turn": 1, "vote_turn": 5}
    w, _ = _write_one_turn(tmp_path, cfg, world)
    (m,) = _rows(w, "metrics")
    fields = {f.name for f in dataclasses.fields(Country)} - {"id", "lang"}
    assert fields <= set(m), f"metrics 에 안 나가는 Country 필드: {sorted(fields - set(m))}"
    assert m["proposal"]["Asla"]["target"] == "bunker"


def test_memory_is_logged_every_turn_not_only_when_written(tmp_path, cfg, world):
    """쓴 턴에만 남기면 **들고 다니는 것** 을 볼 수 없다 — 유언을 물려받고 한 번도
    고치지 않은 아이가 특히 그렇다."""
    world.agents["Asla1"].memory = "요격기에 몰아줘라"
    w, _ = _write_one_turn(tmp_path, cfg, world)
    row = next(r for r in _rows(w, "state") if r["agent"] == "Asla1")
    assert row["memory"] == "요격기에 몰아줘라"


# ── ② 도구 호출 ────────────────────────────────────────────────────────────────

def test_failed_calls_keep_their_reason(tmp_path, cfg, world):
    """실패가 이름과 ok=False 로만 남으면 **왜 실패했는지가 사라진다** — AP 부족인지
    국가 이름을 틀렸는지 이미 아는 언어였는지 구분이 안 된다."""
    logs = {"Asla1": {"calls": [
        {"step": 1, "tool": "learn", "args": {"country": "NOPE", "amount": 10},
         "ok": False, "error": "unknown nation: NOPE", "result": {"ok": False}}]}}
    w, _ = _write_one_turn(tmp_path, cfg, world, logs)
    (e,) = [r for r in _rows(w, "events") if r["type"] == "agent_turn"]
    (c,) = e["calls"]
    assert c["error"] == "unknown nation: NOPE" and c["args"]["country"] == "NOPE"


def test_calls_record_the_real_cost_not_the_request(tmp_path, cfg, world):
    """`actions` 는 **요청한 인자**다. 실제로 얼마가 나갔는지는 `calls` 의 결과에만 있다.

    금액을 인자로 받던 때는 요청과 과금이 갈릴 수 있어서 응답이 그 차이를 알렸다.
    지금은 금액이 고정이라 **요청에 금액이 아예 없다** — 그래서 sink·결과 말고는 얼마가
    나갔는지 알 길이 없고, 로그가 그것을 담아야 한다.
    """
    from core.agent_loop import Sink, execute_tool
    world.countries["Asla"].land = "interceptor"
    a = world.agents["Asla1"]; a.ap = 1.0
    sink = Sink()
    res, _ = execute_tool("invest", {"target": "facility", "reasoning": "r"},
                          world, a, cfg, sink, KNOB)
    assert res["ok"] and "amount" not in res
    assert sink.facility == [("Asla", cfg.costs.unit, "Asla1")]
    assert res["ap_left"] is not None


def test_memory_and_testament_are_not_stripped_from_calls():
    from core.agent_loop import _redact_args
    assert _redact_args("memory_write", {"text": "x", "reasoning": "r"}) == {"text": "x"}
    assert _redact_args("procreate", {"testament": "y", "reasoning": "r"}) == {"testament": "y"}
    # speak 본문만 뺀다 — messages.jsonl 에 원문·도착문이 함께 있다
    assert _redact_args("speak", {"to": "R2", "text": "z"}) == {"to": "R2"}


# ── ③ API 왕복 ─────────────────────────────────────────────────────────────────

def test_request_and_response_are_stored_whole(tmp_path, cfg, world):
    """**가공하지 않는다.** usage·finish_reason·reasoning·tools 스키마까지 그대로 남아야
    파생 로그를 raw 에서 재생성할 수 있다 (spec 9장)."""
    from core.llm import StubClient, tool_call
    w = RunWriter("t", cfg_raw={"x": 1}, root=tmp_path)
    client = StubClient([{"role": "assistant", "content": None,
                          "tool_calls": [tool_call("end_turn", "1")]}],
                        recorder=w.recorder(kind="agent"))
    client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}],
                tool_choice="required", log_tag={"turn": 3, "agent": "Asla1", "step": 1})
    (r,) = _rows(w, "raw_calls")
    assert r["request"]["messages"] == [{"role": "user", "content": "hi"}]
    assert r["request"]["tools"] and r["request"]["tool_choice"] == "required"
    assert r["response"]["choices"][0]["message"]["tool_calls"]


# ── ④ 이음매 ───────────────────────────────────────────────────────────────────

def test_raw_calls_can_be_joined_to_who_and_when(tmp_path, cfg, world):
    """`kind` 는 클라이언트를 만들 때 붙는 고정 태그라 turn·agent 를 담을 수 없었다.
    그래서 raw_calls 를 events 와 이어붙일 방법이 없었고 **호출 단위 분석이 막혀
    있었다.**"""
    from core.llm import StubClient
    w = RunWriter("t", cfg_raw={"x": 1}, root=tmp_path)
    client = StubClient([], recorder=w.recorder(kind="agent"))
    client.chat([], log_tag={"turn": 7, "agent": "Ranoa2", "step": 3})
    (r,) = _rows(w, "raw_calls")
    assert (r["turn"], r["agent"], r["step"], r["kind"]) == (7, "Ranoa2", 3, "agent")


def test_translation_calls_carry_the_message_id(cfg):
    """번역 호출이 kind 로만 구분되면 원문·도착문과 실제 API 왕복을 이어붙일 수 없다 —
    지표 6a·9 가 재는 것이 바로 그 이음매다."""
    from core import translate as tr
    seen = {}

    class Rec:
        def chat(self, messages, tools=None, temperature=None, tool_choice=None,
                 log_tag=None):
            seen.update(log_tag or {})
            return {"choices": [{"message": {"content": "訳"}}]}

    tr.translate(Rec(), "fr", "ja", "bonjour", None, log_tag={"turn": 4, "msg_id": 12})
    assert seen == {"turn": 4, "msg_id": 12, "src_lang": "fr", "dst_lang": "ja"}


def test_message_id_is_assigned_before_the_translation_runs(cfg, world):
    """나중에 뽑으면 번역 raw 기록의 msg_id 가 null 로 남는다."""
    seen = []

    class Rec:
        def chat(self, messages, tools=None, temperature=None, tool_choice=None,
                 log_tag=None):
            seen.append(log_tag)
            return {"choices": [{"message": {"content": "訳"}}]}

    sink = Sink()
    sink.messages = [{"from": "Miris1", "to": "Asla1", "from_lang": "fr", "to_lang": "ja",
                      "from_country": "Miris", "to_country": "Asla",
                      "text": "bonjour", "route": "ai"}]
    r = loop.RunResult(world=world)
    loop._settle_agentic(world, cfg, random.Random(0), sink, Rec(), KNOB,
                         itertools.count(500), r, itertools.count(900))
    assert seen and seen[0]["msg_id"] == 900
    assert r.messages_log[0]["msg_id"] == 900


# ── 호출마다 이름 (8/20) ────────────────────────────────────────────────────

def test_every_raw_call_gets_a_unique_id(tmp_path, cfg):
    """**한 호출을 가리키려면 (turn, agent, step, attempt) 네 개를 나열해야 했다.**

    `call_id` 하나면 된다. 재시도까지 갈리므로 같은 스텝의 1차·2차 시도도 서로 다른
    이름을 갖는다.
    """
    from core.llm import StubClient
    w = RunWriter("ids", cfg_raw={"x": 1}, root=tmp_path)
    client = StubClient([], recorder=w.recorder(kind="agent"))
    for step in (1, 2, 3):
        client.chat([], log_tag={"turn": 4, "agent": "Asla1", "step": step})
    rows = _rows(w, "raw_calls")
    ids = [r["call_id"] for r in rows]
    assert ids == ["c00001", "c00002", "c00003"]
    assert len(set(ids)) == len(ids)
    # 맨 앞 키다 — 한 줄을 눈으로 훑을 때 먼저 보인다
    assert next(iter(rows[0])) == "call_id"


def test_ids_do_not_collide_after_a_resume(tmp_path, cfg):
    """이어할 때 0 에서 다시 시작하면 앞 구간과 **같은 이름이 두 번** 생긴다.
    raw_calls_total 도 이어붙인 구간만 센 값이 된다."""
    from core.llm import StubClient
    a = RunWriter("rid", cfg_raw={"x": 1}, root=tmp_path)
    c1 = StubClient([], recorder=a.recorder(kind="agent"))
    c1.chat([]); c1.chat([])
    a.close()

    b = RunWriter("rid", cfg_raw={"x": 1}, root=tmp_path, overwrite=True, append=True)
    c2 = StubClient([], recorder=b.recorder(kind="agent"))
    c2.chat([])
    rows = _rows(b, "raw_calls")
    ids = [r["call_id"] for r in rows]
    assert ids == ["c00001", "c00002", "c00003"], ids
    assert b.counts["raw"] == 3          # 이어붙인 구간만 세지 않는다


def test_the_counter_does_not_leak_under_parallel_writes(tmp_path, cfg):
    """카운터가 락 **밖**이라 병렬 경로에서 집계가 새고 있었다."""
    import threading

    from core.llm import StubClient
    w = RunWriter("par", cfg_raw={"x": 1}, root=tmp_path)
    N, T = 40, 8

    def work(k):
        c = StubClient([], recorder=w.recorder(kind="agent"))
        for i in range(N):
            c.chat([], log_tag={"turn": 1, "agent": f"A{k}", "step": i})

    ts = [threading.Thread(target=work, args=(k,)) for k in range(T)]
    [t.start() for t in ts]; [t.join() for t in ts]
    rows = _rows(w, "raw_calls")
    assert w.counts["raw"] == N * T == len(rows)
    assert len({r["call_id"] for r in rows}) == N * T      # 겹치지 않는다
