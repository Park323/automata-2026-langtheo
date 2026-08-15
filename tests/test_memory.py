"""개체 기억 — 컨텍스트 누적·축출·종료 조건·memory_write·누수 불변식. spec 4.5.

StubClient 로 검증 (API 안 씀). MAX_STEPS 는 폐지됐다 — 종료는 세계 규칙으로만.
반환 계약은 작성자의 #7·#8 것을 따른다: reasonings·ended_by·reasoning_missing.
"""
from __future__ import annotations

import itertools
import json
import random
import re
from pathlib import Path

import pytest
import yaml

from core import config, memory
from core.agent_loop import Sink, run_agent_turn
from core.llm import StubClient, assistant_msg, tool_call
from core.loop import init_world, run_agentic
from domains.meteor import prompts

BASE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
IDS = [f"{n}{i}" for n in ("Asla", "Ranoa", "Miris") for i in (1, 2, 3)]


@pytest.fixture()
def cfg():
    return config.load(BASE)


@pytest.fixture()
def world(cfg):
    return init_world(cfg, itertools.count(1))


def _turn(world, cfg, aid, script, knob_ai=48, budget=None):
    """실제 루프의 1단계(AP 리셋)를 대신하고 한 에이전트의 한 턴을 돌린다."""
    agent = world.agents[aid]
    agent.ap = cfg.turn.action_points
    if budget is not None:
        agent.budget = budget
    sink = Sink()
    client = StubClient(script)
    sp = prompts.system_for(agent)
    up = prompts.render_observation(world, agent, cfg, knob_ai)
    log = run_agent_turn(world, agent, cfg, client, sink, knob_ai, sp, up)
    return agent, sink, client, log


def _cfg_turns(turns, ctx=None):
    d = yaml.safe_load(open(BASE, encoding="utf-8"))
    d["world"]["total_turns"] = turns
    if ctx is not None:
        d["llm"]["context_limit"] = ctx
    return config.from_dict(d)


def _clients(overrides=None):
    d = {aid: StubClient([]) for aid in IDS}
    for aid, script in (overrides or {}).items():
        d[aid] = StubClient(script)
    return d


# ── 컨텍스트 누적 ────────────────────────────────────────────────────────────

def test_context_accumulates_across_turns(cfg, world):
    """messages 는 태어나서 죽을 때까지 이어진다. system 은 정확히 하나, 맨 앞."""
    agent = world.agents["Asla1"]
    _turn(world, cfg, "Asla1", [assistant_msg(tool_call("end_turn", "1", reasoning="r1"))])
    n1 = len(agent.messages)
    _turn(world, cfg, "Asla1", [assistant_msg(tool_call("end_turn", "1", reasoning="r2"))])
    assert agent.messages[0]["role"] == "system"
    assert sum(1 for m in agent.messages if m["role"] == "system") == 1   # system 재삽입 안 됨
    assert len(agent.messages) > n1                                       # 뒤에 계속 붙는다


def test_ended_by_and_reasoning(cfg, world):
    """end_turn 인자의 reasoning 을 reasonings 에 담는다. 있으면 reasoning_missing=False."""
    _, _, _, log = _turn(world, cfg, "Asla1",
                         [assistant_msg(tool_call("end_turn", "1", reasoning="because"))])
    assert log["ended_by"] == "ended"
    assert log["reasonings"][-1]["reasoning"] == "because"
    assert log["reasoning_missing"] is False


# ── 종료 조건 (MAX_STEPS 폐지) ───────────────────────────────────────────────

def test_no_arbitrary_step_cap(cfg, world):
    """12회 연속 성공 행동이 전부 실행된다 — 옛 MAX_STEPS=8 이면 잘렸을 것."""
    script = [assistant_msg(tool_call("invest", str(i), target="facility", amount=1, reasoning="r"))
              for i in range(12)]
    _, sink, _, log = _turn(world, cfg, "Asla1", script, budget=10000)
    assert len(sink.facility) == 12                    # 8 에서 안 잘린다
    assert log["ended_by"] == "ended"                  # 스크립트 소진(도구 없음) → 종료


def test_repeat_guard_stops_repeated_failure(cfg, world):
    """동일 (도구, 인자) 실패가 repeat_guard 회면 종료 — 실패는 자원을 안 쓰므로 ②로 못 막는다."""
    guard = cfg.turn.repeat_guard
    script = [assistant_msg(tool_call("speak", "c", to="NOBODY", text="x", reasoning="r"))] * (guard + 2)
    _, _, _, log = _turn(world, cfg, "Asla1", script, budget=10000)
    assert log["ended_by"] == "repeat_guard"


def test_exhausted_when_nothing_affordable(cfg, world):
    """예산 0 · AP 가 memory_write 미만이면 실행 가능한 도구가 없어 자연 종료(exhausted)."""
    agent = world.agents["Asla1"]
    agent.ap = 0.0
    agent.budget = 0.0
    sink = Sink()
    client = StubClient([assistant_msg(tool_call("invest", "1", target="wellness", amount=5, reasoning="r"))])
    sp = prompts.system_for(agent)
    up = prompts.render_observation(world, agent, cfg, 48)
    log = run_agent_turn(world, agent, cfg, client, sink, 48, sp, up)
    assert log["ended_by"] == "exhausted"


# ── memory_write ─────────────────────────────────────────────────────────────

def test_memory_write_costs_ap_not_budget(cfg, world):
    """기억은 AP 0.1 을 쓰고 예산은 건드리지 않는다. 관측에 [내 메모] 로 뜬다."""
    agent, sink, client, log = _turn(
        world, cfg, "Asla1",
        [assistant_msg(tool_call("memory_write", "1", text="RanoaのAsla2はjaを読める", reasoning="메모")),
         assistant_msg(tool_call("end_turn", "2", reasoning="r"))],
        budget=100)
    assert agent.memory == "RanoaのAsla2はjaを読める"
    assert agent.budget == 100                         # 예산 불변
    assert abs(agent.ap - (cfg.turn.action_points - cfg.ap.memory_write)) < 1e-9
    obs = prompts.render_observation(world, agent, cfg, 48)
    assert "RanoaのAsla2はjaを読める" in obs             # 다음 관측에 실린다


def test_memory_tool_requires_reasoning():
    """memory_write 도 예외 없이 reasoning 필수 (작성자 규칙, _fn 자동 주입)."""
    from core import tools
    mw = next(t for t in tools.TOOLS if t["function"]["name"] == "memory_write")
    assert "reasoning" in mw["function"]["parameters"]["required"]
    assert "reasoning" in mw["function"]["parameters"]["properties"]


# ── 🔴 발신자 컨텍스트 누수 불변식 (spec 4.5) ────────────────────────────────

def test_sender_context_no_translation_leak():
    """ai 경로 발신 후, 발신자 컨텍스트에 번역 결과가 절대 없다. speak 결과는 접수·과금만."""
    cfg = _cfg_turns(2)
    translator = StubClient([{"role": "assistant", "content": "XLATION_LEAK", "tool_calls": []}] * 20)
    clients = _clients({"Asla1": [assistant_msg(
        tool_call("speak", "1", to="Ranoa2", route="ai", text="MY_ORIGINAL", reasoning="r"))]})
    run_agentic(cfg, random.Random(1), clients.__getitem__, translator, 48,
                prompts.render_observation, prompts.system_for, parallel=False)

    a1 = clients["Asla1"]
    tool_results = [json.loads(m["content"]) for call in a1.calls
                    for m in call["messages"] if m.get("role") == "tool"]
    receipts = [r for r in tool_results if r.get("queued")]
    assert receipts, "speak 접수 결과가 있어야 한다"
    allowed = {"ok", "queued", "charged", "budget_left", "ap_left"}
    for r in receipts:
        assert set(r) <= allowed, f"speak 결과에 누수 채널: {set(r) - allowed}"
    blob = "".join(m.get("content") or "" for call in a1.calls for m in call["messages"])
    assert "XLATION_LEAK" not in blob


# ── 축출 (오래된 블록부터, 짝 보존) ──────────────────────────────────────────

def test_evict_keeps_system_and_pairs():
    """한계 초과 시 오래된 user 블록부터 통째로 축출. system·최근 블록은 남고 짝이 안 깨진다."""
    msgs = [{"role": "system", "content": "S"}]
    for _ in range(30):
        msgs.append({"role": "user", "content": "U" * 400})
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": "c", "type": "function",
                                     "function": {"name": "end_turn", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": "c", "content": "{}"})
    before = len(msgs)
    dropped = memory.evict(msgs, limit=800, tool_tokens=0)
    assert dropped > 0 and len(msgs) < before
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            assert msgs[i - 1]["role"] == "assistant" and msgs[i - 1].get("tool_calls")


def test_no_leak_after_eviction():
    """축출을 강제한 뒤에도 발신자 누적 컨텍스트에 번역 결과가 새지 않는다."""
    cfg = _cfg_turns(4, ctx=300)
    translator = StubClient([{"role": "assistant", "content": "XLATION_LEAK", "tool_calls": []}] * 60)
    a1 = [assistant_msg(tool_call("speak", f"s{t}", to="Ranoa2", route="ai", text=f"ORIG{t}", reasoning="r"),
                        tool_call("end_turn", f"e{t}", reasoning="r")) for t in range(4)]
    clients = _clients({"Asla1": a1})
    run_agentic(cfg, random.Random(1), clients.__getitem__, translator, 48,
                prompts.render_observation, prompts.system_for, parallel=False)
    last = clients["Asla1"].calls[-1]["messages"]
    assert last[0]["role"] == "system"
    assert len(last) <= 6, f"축출이 안 일어난 듯: {len(last)}"
    blob = "".join(m.get("content") or "" for call in clients["Asla1"].calls
                   for m in call["messages"])
    assert "XLATION_LEAK" not in blob


def _api_valid(messages):
    """OpenAI 대화 유효성: 모든 tool 메시지는 tool_calls 를 든 assistant 의 열린 id 와 짝이
    맞아야 하고, 그 사이에 user/system 이 끼면 안 된다 (끼면 실제 API 가 400 을 돌려준다)."""
    pending = set()
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            pending = {tc["id"] for tc in m["tool_calls"]}
        elif role == "tool":
            if m.get("tool_call_id") not in pending:
                return False
            pending.discard(m["tool_call_id"])
        elif role in ("user", "system") and pending:
            return False           # tool 응답이 안 온 채 다음 블록 시작 → 짝 깨짐
    return True


def test_conversation_stays_api_valid_across_accumulation_and_eviction(cfg):
    """한 assistant 에 여러 tool_call 이 붙는 실제 패턴에서도, 누적·축출 내내 대화가
    API-유효해야 한다 (assistant(tool_calls)↔tool 응답 짝이 절대 안 깨진다)."""
    d = yaml.safe_load(open(BASE, encoding="utf-8"))
    d["llm"]["context_limit"] = 1300          # 축출 강제
    cfg = config.from_dict(d)
    world = init_world(cfg, itertools.count(1))
    agent = world.agents["Asla1"]
    patterns = [
        # 한 assistant 에 tool_call 3개 → tool 응답 3개
        [assistant_msg(tool_call("memory_write", "1", text="m", reasoning="r"),
                       tool_call("speak", "2", to="Asla3", text="x", reasoning="r"),
                       tool_call("end_turn", "3", reasoning="r"))],
        # 여러 스텝 (assistant/tool 라운드 2회)
        [assistant_msg(tool_call("speak", "1", to="Asla3", text="y", reasoning="r")),
         assistant_msg(tool_call("end_turn", "2", reasoning="r"))],
    ]
    for t in range(9):
        agent.ap = cfg.turn.action_points
        agent.budget = 500
        up = prompts.render_observation(world, agent, cfg, 48)
        run_agent_turn(world, agent, cfg, StubClient(patterns[t % 2]), Sink(),
                       48, prompts.system_for(agent), up)
        assert _api_valid(agent.messages), f"턴 {t} 후 대화가 API-무효 (짝 깨짐)"
    assert agent.messages[0]["role"] == "system"


def test_agent_messages_isolated():
    """에이전트별 messages 는 별개 객체이고, 축출 로직이 남의 조각을 섞지 않는다."""
    cfg = _cfg_turns(3, ctx=400)
    clients = _clients({"Asla1": [assistant_msg(
        tool_call("memory_write", "1", text="ASLA1_PRIVATE_MEMO", reasoning="r"),
        tool_call("end_turn", "2", reasoning="r"))]})
    res = run_agentic(cfg, random.Random(1), clients.__getitem__,
                      StubClient([{"role": "assistant", "content": "x", "tool_calls": []}] * 60),
                      48, prompts.render_observation, prompts.system_for, parallel=True)
    agents = res.world.agents
    obj_ids = [id(a.messages) for a in agents.values()]
    assert len(set(obj_ids)) == len(obj_ids)
    for a in agents.values():
        assert a.messages[0]["content"] == prompts.SYSTEM[a.native_lang]
    for aid, c in clients.items():
        if aid == "Asla1":
            continue
        blob = "".join(m.get("content") or "" for call in c.calls for m in call["messages"])
        assert "ASLA1_PRIVATE_MEMO" not in blob


# ── 압박 통지 (MemGPT) — 언어 위생 ──────────────────────────────────────────

def test_pressure_notice_language_hygiene():
    """통지는 모국어 하나로, 한글 없이 (프롬프트 언어 위생 규칙과 동일)."""
    HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
    KANA = re.compile(r"[぀-ヿ]")
    CJK = re.compile(r"[一-鿿]")
    for lang, txt in prompts.PRESSURE_NOTICE.items():
        assert not HANGUL.search(txt), f"{lang} 통지에 한글"
        if lang == "ja":
            assert KANA.search(txt)
        elif lang == "zh":
            assert not KANA.search(txt) and CJK.search(txt)
        elif lang == "fr":
            assert not KANA.search(txt) and not CJK.search(txt)


def test_pressure_notice_prepended_when_flagged(cfg, world):
    """mem_pressure 가 서면 관측 맨 앞에 통지가 붙는다 (사실 통지, 지시 아님)."""
    agent = world.agents["Asla1"]
    agent.mem_pressure = True
    obs = prompts.render_observation(world, agent, cfg, 48)
    assert obs.startswith(prompts.PRESSURE_NOTICE["ja"])
