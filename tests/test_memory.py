"""개체 기억 — 컨텍스트 누적·축출·종료 조건·memory_write·누수 불변식. spec 4.5.

StubClient 로 검증 (API 안 씀). MAX_STEPS 는 폐지됐다 — 종료는 세계 규칙으로만.
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


def _turn(world, cfg, aid, script, turn=1, knob_ai=48, budget=None):
    """실제 루프의 1단계(AP 리셋)를 대신하고 한 에이전트의 한 턴을 돌린다."""
    agent = world.agents[aid]
    agent.ap = cfg.turn.action_points
    if budget is not None:
        agent.budget = budget
    sink = Sink()
    client = StubClient(script)
    sp = prompts.system_for(agent)
    up = prompts.render_observation(world, agent, cfg, knob_ai)
    log = run_agent_turn(world, agent, cfg, client, sink, knob_ai, sp, up, turn=turn)
    return agent, sink, client, log


def _cfg_turns(turns):
    d = yaml.safe_load(open(BASE, encoding="utf-8"))
    d["world"]["total_turns"] = turns
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
    _turn(world, cfg, "Asla1", [assistant_msg(tool_call("end_turn", "1", reasoning="r1"))], turn=1)
    n1 = len(agent.messages)
    _turn(world, cfg, "Asla1", [assistant_msg(tool_call("end_turn", "1", reasoning="r2"))], turn=2)
    assert agent.messages[0]["role"] == "system"
    assert sum(1 for m in agent.messages if m["role"] == "system") == 1   # system 재삽입 안 됨
    assert len(agent.messages) > n1                                       # 뒤에 계속 붙는다


def test_end_reason_and_reasoning(cfg, world):
    """end_turn 인자의 reasoning 을 취한다. 있으면 reasoning_missing=False."""
    _, _, _, log = _turn(world, cfg, "Asla1",
                         [assistant_msg(tool_call("end_turn", "1", reasoning="because"))])
    assert log["end_reason"] == "ended"
    assert log["reasoning"] == "because" and log["reasoning_missing"] is False


def test_reasoning_missing_when_absent(cfg, world):
    """end_turn 에 reasoning 이 비면 누락 플래그가 선다 (분모 제외용, spec 9)."""
    _, _, _, log = _turn(world, cfg, "Asla1", [assistant_msg(tool_call("end_turn", "1"))])
    assert log["reasoning"] == "" and log["reasoning_missing"] is True


# ── 종료 조건 (MAX_STEPS 폐지) ───────────────────────────────────────────────

def test_no_arbitrary_step_cap(cfg, world):
    """12회 연속 성공 행동이 전부 실행된다 — 옛 MAX_STEPS=8 이면 잘렸을 것."""
    script = [assistant_msg(tool_call("invest", str(i), target="facility", amount=1))
              for i in range(12)]
    _, sink, _, log = _turn(world, cfg, "Asla1", script, budget=10000)
    assert len(sink.facility) == 12                    # 8 에서 안 잘린다
    assert log["end_reason"] == "ended"                # 스크립트 소진(도구 없음) → 종료


def test_repeat_guard_stops_repeated_failure(cfg, world):
    """동일 (도구, 인자) 실패가 repeat_guard 회면 종료 — 실패는 자원을 안 쓰므로 ②로 못 막는다."""
    guard = cfg.turn.repeat_guard
    script = [assistant_msg(tool_call("speak", "c", to="NOBODY", text="x"))] * (guard + 2)
    _, _, _, log = _turn(world, cfg, "Asla1", script, budget=10000)
    assert log["end_reason"] == "repeat_guard"


def test_exhausted_when_nothing_affordable(cfg, world):
    """예산 0 · AP 가 memory_write 미만이면 실행 가능한 도구가 없어 자연 종료(exhausted)."""
    agent = world.agents["Asla1"]
    agent.ap = 0.0                                     # 어떤 AP 도구도 불가
    agent.budget = 0.0                                 # invest 도 불가
    sink = Sink()
    # 첫 스텝에서 실패 행동을 하나 주면, 그 뒤 affordable 검사가 exhausted 를 잡는다
    client = StubClient([assistant_msg(tool_call("invest", "1", target="wellness", amount=5))])
    sp = prompts.system_for(agent)
    up = prompts.render_observation(world, agent, cfg, 48)
    log = run_agent_turn(world, agent, cfg, client, sink, 48, sp, up, turn=1)
    assert log["end_reason"] == "exhausted"


# ── memory_write ─────────────────────────────────────────────────────────────

def test_memory_write_costs_ap_not_budget(cfg, world):
    """기억은 AP 0.1 을 쓰고 예산은 건드리지 않는다. 관측에 [내 메모] 로 뜬다."""
    agent, sink, client, log = _turn(
        world, cfg, "Asla1",
        [assistant_msg(tool_call("memory_write", "1", text="RanoaのAsla2はjaを読める")),
         assistant_msg(tool_call("end_turn", "2", reasoning="r"))],
        budget=100)
    assert agent.memory == "RanoaのAsla2はjaを読める"
    assert agent.budget == 100                         # 예산 불변
    assert abs(agent.ap - (cfg.turn.action_points - cfg.ap.memory_write)) < 1e-9
    obs = prompts.render_observation(world, agent, cfg, 48)
    assert "RanoaのAsla2はjaを読める" in obs             # 다음 관측에 실린다


# ── 🔴 발신자 컨텍스트 누수 불변식 (spec 4.5) ────────────────────────────────

def test_sender_context_no_translation_leak():
    """ai 경로 발신 후, 발신자 컨텍스트에 번역 결과가 절대 없다. speak 결과는 접수·과금만."""
    cfg = _cfg_turns(2)
    translator = StubClient([{"role": "assistant", "content": "XLATION_LEAK", "tool_calls": []}] * 20)
    clients = _clients({"Asla1": [assistant_msg(
        tool_call("speak", "1", to="Ranoa2", route="ai", text="MY_ORIGINAL"))]})
    run_agentic(cfg, random.Random(1), clients.__getitem__, translator, 48,
                prompts.render_observation, prompts.system_for, parallel=False)

    a1 = clients["Asla1"]
    tool_results = [json.loads(m["content"]) for call in a1.calls
                    for m in call["messages"] if m.get("role") == "tool"]
    receipts = [r for r in tool_results if r.get("queued")]
    assert receipts, "speak 접수 결과가 있어야 한다"
    # 접수 결과 키는 접수·과금뿐 — text_sent·translation·delivered·understood 채널이 없다
    allowed = {"ok", "queued", "charged", "budget_left", "ap_left"}
    for r in receipts:
        assert set(r) <= allowed, f"speak 결과에 누수 채널: {set(r) - allowed}"
    # 누적 컨텍스트 전체에도 번역 결과가 없다 (수신자 Ranoa2 만 봐야 한다)
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
    assert msgs[0]["role"] == "system"                 # system 보존
    assert msgs[1]["role"] == "user"                   # 남은 첫 비-system 은 user (짝 안 깨짐)
    # tool 메시지 앞에는 반드시 tool_calls 를 든 assistant 가 있다
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            assert msgs[i - 1]["role"] == "assistant" and msgs[i - 1].get("tool_calls")


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


# ── 🔴 누수 불변식: 축출 후에도 · 에이전트별 분리 (spec 4.5) ──────────────────

def _cfg_small_ctx(turns, limit):
    d = yaml.safe_load(open(BASE, encoding="utf-8"))
    d["world"]["total_turns"] = turns
    d["llm"]["context_limit"] = limit          # 도구 스키마만으로도 넘겨 매 턴 축출 유발
    return config.from_dict(d)


def test_no_leak_after_eviction():
    """축출을 강제한 뒤에도 발신자 누적 컨텍스트에 번역 결과가 새지 않는다.
    (한 번 새면 생애 내내 남으므로 축출 경로까지 불변식을 못 박는다.)"""
    cfg = _cfg_small_ctx(turns=4, limit=300)
    translator = StubClient([{"role": "assistant", "content": "XLATION_LEAK", "tool_calls": []}] * 60)
    a1 = [assistant_msg(tool_call("speak", f"s{t}", to="Ranoa2", route="ai", text=f"ORIG{t}"),
                        tool_call("end_turn", f"e{t}", reasoning="r")) for t in range(4)]
    clients = _clients({"Asla1": a1})
    res = run_agentic(cfg, random.Random(1), clients.__getitem__, translator, 48,
                      prompts.render_observation, prompts.system_for, parallel=False)

    # 축출이 실제로 일어났다 — 4턴 누적인데 마지막 호출 messages 가 짧다 (system + 최근 블록)
    last = clients["Asla1"].calls[-1]["messages"]
    assert last[0]["role"] == "system"
    assert len(last) <= 6, f"축출이 안 일어난 듯: {len(last)}"
    # 발신자 컨텍스트(축출 후 누적 messages 전체 + 매 호출 스냅샷)에 번역 결과 없음
    blob = "".join(m.get("content") or "" for call in clients["Asla1"].calls
                   for m in call["messages"])
    persisted = "".join(m.get("content") or "" for m in res.world.agents["Asla1"].messages)
    assert "XLATION_LEAK" not in blob and "XLATION_LEAK" not in persisted


def test_agent_messages_isolated():
    """에이전트별 messages 는 별개 객체이고, 축출 로직이 남의 조각을 섞지 않는다.
    한 에이전트가 memory_write 로 쓴 메모는 다른 에이전트 컨텍스트에 절대 안 나타난다."""
    cfg = _cfg_small_ctx(turns=3, limit=400)   # 축출을 켠 채로 분리 검사
    clients = _clients({"Asla1": [assistant_msg(
        tool_call("memory_write", "1", text="ASLA1_PRIVATE_MEMO"),
        tool_call("end_turn", "2", reasoning="r"))]})
    res = run_agentic(cfg, random.Random(1), clients.__getitem__,
                      StubClient([{"role": "assistant", "content": "x", "tool_calls": []}] * 60),
                      48, prompts.render_observation, prompts.system_for, parallel=True)

    agents = res.world.agents
    # 전부 서로 다른 리스트 객체 (공유·혼합 없음)
    obj_ids = [id(a.messages) for a in agents.values()]
    assert len(set(obj_ids)) == len(obj_ids)
    # 각 에이전트의 system(0번)은 자기 모국어 프롬프트
    for a in agents.values():
        assert a.messages[0]["content"] == prompts.SYSTEM[a.native_lang]
    # Asla1 의 사적 메모가 다른 누구의 프롬프트에도 없다
    for aid, c in clients.items():
        if aid == "Asla1":
            continue
        blob = "".join(m.get("content") or "" for call in c.calls for m in call["messages"])
        assert "ASLA1_PRIVATE_MEMO" not in blob
