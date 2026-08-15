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


def _run(cfg, clients, translator=None, knob_ai=48, seed=1, parallel=True):
    translator = translator or StubClient([{"role": "assistant", "content": "译", "tool_calls": []}] * 50)
    return run_agentic(cfg, random.Random(seed), clients.__getitem__, translator, knob_ai,
                       prompts.render_observation, prompts.system_for, parallel=parallel)


# ── #5 도착 지연 ─────────────────────────────────────────────────────────────

def test_message_delivery_delayed():
    """이번 턴 발신은 다음 턴 관측에 나타난다 (같은 턴엔 없음)."""
    cfg = _cfg(2)
    clients = _clients({"Asla1": [assistant_msg(
        tool_call("speak", "1", to="Asla2", text="HELLO_MARK"))]})
    _run(cfg, clients, parallel=False)
    a2 = clients["Asla2"]                       # 빈 스크립트 → 턴당 chat 1회
    # 기억이 누적되므로(spec 4.5) 각 턴의 관측은 그 시점 messages 의 **마지막** 원소다
    turn1 = a2.calls[0]["messages"][-1]["content"]
    turn2 = a2.calls[1]["messages"][-1]["content"]
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
                        itertools.count(1000), RunResult(world=world))
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
