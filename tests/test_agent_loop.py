"""에이전트 루프. 과제 2 Part A. StubClient 로 검증 (API 안 씀)."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from core import config
from core.agent_loop import RUNAWAY_CAP, Sink, learn_cost, run_agent_turn
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
    sys_p = prompts.system_for(agent)
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
        tool_call("speak", "1", to="Asla2", text="a"),
        tool_call("speak", "2", to="Asla2", text="b"),
        tool_call("speak", "3", to="Asla2", text="c"),
        tool_call("speak", "4", to="Asla2", text="d"),
    ), assistant_msg(tool_call("end_turn", "5"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    results = _results(client)
    oks = [r for r in results if "ok" in r]
    assert sum(1 for r in oks if r["ok"]) == 3          # 앞 3건 성공
    assert any((not r["ok"]) and "AP" in r.get("error", "") for r in oks)


# ── #3 예산 고갈 ─────────────────────────────────────────────────────────────

def test_budget_never_negative(cfg, world):
    world.countries["Asla"].land = "interceptor"   # 투표 전에는 애초에 투자가 막힌다
    script = [assistant_msg(tool_call("invest", "1", target="facility", amount=999999)),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=100)
    results = _results(client)
    assert any((not r["ok"]) and "budget" in r.get("error", "") for r in results)
    assert agent.budget >= 0                            # 음수 안 됨
    assert sink.facility == []                          # 실패했으니 sink 에 안 들어감


# ── #4 procreate 즉시 종료 ───────────────────────────────────────────────────

def test_invest_facility_invalid_country(cfg, world):
    """LLM 이 국가 대신 에이전트 id(B2)를 주면 ok:False, 예산 미차감, sink 미반영."""
    world.countries["Ranoa"].land = "interceptor"
    script = [assistant_msg(tool_call("invest", "1", target="facility", amount=50, to="Ranoa2")),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    results = _results(client)
    assert any((not r["ok"]) and "nation" in r.get("error", "") for r in results)
    assert agent.budget == 10000                # 검증이 차감보다 먼저
    assert sink.facility == []


def test_invest_amount_not_number(cfg, world):
    """amount 가 숫자 아닌 문자열이어도 크래시하지 않고 ok:False."""
    world.countries["Asla"].land = "interceptor"
    script = [assistant_msg(tool_call("invest", "1", target="facility", amount="많이")),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    assert any(not r["ok"] for r in _results(client))
    assert sink.facility == []


def test_speak_to_self_rejected(cfg, world):
    """자기 자신에게는 메시지를 못 보낸다 (무의미한 낭비)."""
    script = [assistant_msg(tool_call("speak", "1", to="Asla1", text="x")),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    assert any((not r["ok"]) and "yourself" in r.get("error", "") for r in _results(client))
    assert sink.messages == []


def test_malformed_tool_call_no_crash(cfg, world):
    """모델이 function/name 없는 malformed tool_call 을 줘도 run 이 죽지 않는다."""
    bad = {"role": "assistant", "content": "",
           "tool_calls": [{"id": "x", "type": "function"}]}   # function 키 없음
    script = [bad, assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    assert log["error"] is None                 # 예외 없이 정상 종료


def test_speak_text_coerced_to_str(cfg, world):
    """text 가 문자열이 아니어도(숫자) 크래시하지 않고 문자열로 저장된다."""
    script = [assistant_msg(tool_call("speak", "1", to="Asla2", text=123)),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    assert len(sink.messages) == 1
    assert sink.messages[0]["text"] == "123"    # str 강제


def test_procreate_ends_turn(cfg, world):
    """procreate 뒤의 tool_call 은 실행되지 않는다."""
    script = [assistant_msg(
        tool_call("procreate", "1", testament="믿지 마라"),
        tool_call("invest", "2", target="facility", amount=10),   # 버려져야 함
    )]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    assert len(sink.procreations) == 1
    assert sink.facility == []                          # procreate 뒤 invest 무시


# ── #10 학습 할인 ────────────────────────────────────────────────────────────

def test_learn_discount_levels(cfg, world):
    """국내 구사자 없음/있음/부모까지 → 600 / 300 / 150 (L · L/2 · L/4)."""
    a1 = world.agents["Asla1"]           # 국가 A, ja
    # 아무 할인 없음
    cost, _ = learn_cost(a1, "Ranoa", world, cfg)       # B = zh
    assert cost == 600
    # 국내 구사자: A2 가 zh 를 앎
    world.agents["Asla2"].known_langs.add("zh")
    cost, reason = learn_cost(a1, "Ranoa", world, cfg)
    assert cost == 300 and "nation" in reason
    # 부모까지: a1 의 부모가 zh
    a1.parent_langs.add("zh")
    cost, reason = learn_cost(a1, "Ranoa", world, cfg)
    assert cost == 150


def test_learn_self_not_counted(cfg, world):
    """자기 자신은 국내 구사자로 세지 않는다."""
    a1 = world.agents["Asla1"]
    a1.known_langs.add("zh")          # 자기가 zh 를 알아도
    cost, _ = learn_cost(a1, "Ranoa", world, cfg)
    assert cost == 600                # 할인 안 됨


def test_learn_is_paid_in_instalments(cfg, world):
    """**한 번에 다 낼 필요가 없다.** 낸 만큼 쌓이고 다 차야 읽을 수 있다.

    Asla2 가 Miris(fr) 를 배운다 — Asla 에는 fr 구사자가 없어 정가 600 이다.
    """
    script = [assistant_msg(tool_call("learn", "1", country="Miris", amount=250)),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla2", script, budget=10000)
    assert agent.budget == 10000 - 250
    (rec,) = sink.learns
    assert rec["charged"] == 250 and rec["required"] == 600 and rec["rung"] == 1.0
    assert rec["progress_before"] == 0.0
    res = next(r for r in _results(client) if r.get("toward"))
    assert res["progress"] == 250 and res["remaining"] == 350
    assert "not enough yet" in res["effect"]


def test_learn_never_takes_more_than_needed(cfg, world):
    """넘치게 내면 필요한 만큼만 받는다 — 남는 돈이 조용히 사라지면 안 된다."""
    script = [assistant_msg(tool_call("learn", "1", country="Miris", amount=9999)),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla2", script, budget=10000)
    assert agent.budget == 10000 - 600
    assert sink.learns[0]["charged"] == 600


def test_learn_rejects_a_language_already_read(cfg, world):
    """Asla1 은 초기화로 zh 를 안다. 또 낼 수 없다."""
    script = [assistant_msg(tool_call("learn", "1", country="Ranoa", amount=100)),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    assert any((not r["ok"]) and "already read" in r.get("error", "")
               for r in _results(client))
    assert agent.budget == 10000 and sink.learns == []


def test_learn_uses_less_than_a_whole_turn(cfg, world):
    """분할 납부라 한 턴을 통째로 쓸 이유가 없다 (전에는 ap.learn = 1.0)."""
    assert cfg.ap.learn < cfg.turn.action_points
    script = [assistant_msg(tool_call("learn", "1", country="Miris", amount=100)),
              assistant_msg(tool_call("speak", "2", to="Asla3", text="x")),
              assistant_msg(tool_call("end_turn", "3"))]
    agent, sink, client, log = _run(world, cfg, "Asla2", script, budget=10000)
    assert len(sink.learns) == 1 and len(sink.messages) == 1   # 같은 턴에 둘 다


# ── #11 정보 은닉 (가장 중요) ────────────────────────────────────────────────

# 영어 프롬프트에서 새면 안 되는 것: 내부 파라미터 + 왜곡 언급 + 재앙 카운트다운
FORBIDDEN = ["success_prob", "lambda", "hazard", "distort", "inaccurate",
             "turns until", "death prob"]


def test_inbox_renders_delivery_failure():
    """발신자 실패 통지가 'None로부터'가 아니라 명확한 알림으로 렌더된다."""
    notice = {"from": None, "text": None, "label": None, "original": None,
              "delivery_failed_to": "Ranoa2", "msg_id": 1}
    for lang in ("ja", "zh", "fr"):
        s = prompts.render_inbox([notice], lang)
        assert "None" not in s
        assert "Ranoa2" in s
        # 언어별 통지 문구가 실제로 그 언어로 렌더되는지 (영어 잔재가 아닌지)
        assert s.splitlines()[-1] == prompts.T[lang]["in_fail"].format(id=1, to="Ranoa2")


def test_prompt_hides_secrets(cfg, world):
    """프롬프트(system·관측)에 success_prob·λ·하자드·재앙까지 남은 턴이 없다."""
    a0 = world.agents["Asla1"]
    p = prompts.system_for(a0) + "\n" + prompts.render_observation(world, a0, cfg, knob_ai=48)
    for bad in FORBIDDEN:
        assert bad not in p, f"프롬프트에 금지어 '{bad}' 노출"


def test_tool_results_hide_progress(cfg, world):
    """invest(facility) 결과에 진척 증가분이 없다 (success_prob 역산 방지)."""
    script = [assistant_msg(tool_call("invest", "1", target="facility", amount=50)),
              assistant_msg(tool_call("invest", "2", target="wellness", amount=30)),
              assistant_msg(tool_call("end_turn", "3"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    for r in _results(client):
        blob = json.dumps(r, ensure_ascii=False)
        for bad in ["success_prob", "lambda", "λ", "gained", "증가분", "progress"]:
            assert bad not in blob, f"도구 결과에 금지어 '{bad}' 노출: {blob}"
