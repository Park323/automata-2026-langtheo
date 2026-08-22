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
    w = init_world(cfg, itertools.count(1))
    # **개체 차이를 1.0 으로 눕힌다** (8/22) — 다른 기제를 재는 테스트가 사람마다 다른
    # 액수에 흔들리지 않게. 차이 자체는 test_world_rules_v2 의 전용 테스트가 본다.
    for a in w.agents.values():
        a.income_mult = a.invest_mult = 1.0
    return w


def _run(world, cfg, agent_id, script, knob_ai=48, budget=None):
    agent = world.agents[agent_id]
    agent.ap = cfg.turn.action_points        # 실제 루프의 1단계(소득·AP 리셋)를 대신
    if budget is not None:
        agent.budget = budget
    sink = Sink()
    client = StubClient(script)
    sys_p = prompts.system_for(agent, None, cfg)
    usr_p = prompts.render_observation(world, agent, cfg, knob_ai)
    log = run_agent_turn(world, agent, cfg, client, sink, knob_ai, sys_p, usr_p)
    return agent, sink, client, log


def _results(client):
    """마지막 chat 호출 시점의 messages 스냅샷에서 tool 응답을 훑는다.
    (스크립트 끝에 end_turn 을 두면 그 직전까지의 모든 tool 결과가 담긴다.)"""
    last = client.calls[-1]["messages"]
    return [json.loads(m["content"]) for m in last if m.get("role") == "tool"]


# ── #2 AP 상한 ───────────────────────────────────────────────────────────────

def test_speaking_stops_when_the_year_runs_out(cfg, world):
    """**말할 수 있는 횟수는 `ap.speak` 이 정한다.** 8/22 에 0.3 → 0.2 로 내려 한 해에
    다섯 번이 됐다 — 실측에서 AP 가 병목이고 돈이 남았다 (턴 끝 예산 중앙 74 → 435,
    남은 AP 중앙 0.0). 세 번이면 대화가 거기서 끊긴다.

    **횟수를 여기 적지 않는다** — 상수에서 유도한다.
    """
    n = int(cfg.turn.action_points / cfg.ap.speak)
    calls = [tool_call("speak", str(i), to="Asla2", text="x") for i in range(n + 1)]
    script = [assistant_msg(*calls), assistant_msg(tool_call("end_turn", "z"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    oks = [r for r in _results(client) if "ok" in r]
    assert sum(1 for r in oks if r["ok"]) == n
    # 「not enough AP」 → 「not enough action」. **에이전트에게 AP 는 없는 말이다** —
    # 관측·비용표가 「行動力 / 行动力 / action」 이라고 부른다. 그리고 남은 값을 알려준다.
    fail = next(r for r in oks if not r["ok"])
    assert "not enough action" in fail["error"] and "have 0.00" in fail["error"]


# ── #3 예산 고갈 ─────────────────────────────────────────────────────────────

def test_budget_never_negative(cfg, world):
    world.countries["Asla"].land = "interceptor"   # 투표 전에는 애초에 투자가 막힌다
    inv = assistant_msg(tool_call("invest", "1", target="facility"))
    script = [inv, inv, assistant_msg(tool_call("end_turn", "2"))]
    # **한 번의 값에서 예산을 잡는다** — 40 으로 올렸을 때 30 이 낡았다 (8/22).
    # 한 번은 되고 두 번은 안 되는 액수: unit × 1.5
    agent, sink, client, log = _run(world, cfg, "Asla1", script,
                                    budget=cfg.costs.unit * 1.5)
    results = _results(client)
    # 한 번의 값 1.5배로 두 번 — 두 번째가 거절된다
    assert any((not r["ok"]) and "budget" in r.get("error", "") for r in results)
    assert agent.budget >= 0                            # 음수 안 됨
    assert len(sink.facility) == 1                      # 실패한 것은 sink 에 안 들어감


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


def test_invest_ignores_a_stray_amount(cfg, world):
    """**`amount` 인자는 없어졌다** (8/20). 한 번에 정해진 액수만 낸다.

    모델이 옛 스키마를 기억해 `amount` 를 실어 보내도 **그것 때문에 실패하지는 않는다** —
    남는 인자는 무시하고 단위만 나간다. 거절하면 프롬프트를 못 따라온 대가를 세계가
    치르게 된다.
    """
    world.countries["Asla"].land = "interceptor"
    script = [assistant_msg(tool_call("invest", "1", target="facility", amount="많이")),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    assert sink.facility == [("Asla", cfg.costs.unit, "Asla1")]
    assert agent.budget == 10000 - cfg.costs.unit


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


def test_bearing_a_child_does_not_end_the_turn(cfg, world):
    """**부모가 죽지 않으므로 뒤 호출을 버릴 이유가 없다** (8/21).

    전에는 `procreate` 가 그 자리에서 죽여서 뒤따르는 tool_call 을 전부 버렸다. 이제는
    행동력이 0 이 된 것뿐이고, 그 판정은 각 도구가 스스로 한다.
    """
    world.countries["Asla"].land = "interceptor"
    script = [assistant_msg(
        tool_call("bear_child", "1"),
        tool_call("invest", "2", target="facility"),      # AP 0 이라 실패해야 한다
    ), assistant_msg(tool_call("end_turn", "3"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    assert sink.births == ["Asla1"] and agent.has_borne
    assert agent.ap == 0.0
    # **버려진 것이 아니라 실행되고 거절됐다** — 그 차이가 로그에 남는다
    res = _results(client)
    assert any((not r["ok"]) and "not enough action" in r.get("error", "") for r in res)


def test_a_child_can_be_had_once_and_only_by_an_adult(cfg, world):
    """생애 1회 · 성인부터. **나이와 횟수를 자원보다 먼저 말한다** — 되돌릴 수 없는
    조건이라, 「행동력이 없다」 로 거절하면 다음 해에 다시 시도하게 된다."""
    from core.agent_loop import Sink, execute_tool
    a = world.agents["Asla1"]

    a.age, a.ap = cfg.world.adult_age - 1, 1.0
    r, _ = execute_tool("bear_child", {}, world, a, cfg, Sink(), 48.0)
    assert not r["ok"] and "not old enough" in r["error"]
    assert not a.has_borne and a.ap == 1.0

    a.age = cfg.world.adult_age
    r, ctl = execute_tool("bear_child", {}, world, a, cfg, Sink(), 48.0)
    assert r["ok"] and a.has_borne and ctl is None       # 턴을 끝내지 않는다
    assert a.ap == 0.0                                  # 그 해를 통째로 쓴다

    a.ap = 1.0
    r, _ = execute_tool("bear_child", {}, world, a, cfg, Sink(), 48.0)
    assert not r["ok"] and "already had a child" in r["error"]


# ── #10 학습 할인 ────────────────────────────────────────────────────────────

def test_help_makes_learning_faster_not_cheaper(cfg, world):
    """**필요액은 고정이고 속도가 오른다** (8/22).

    전에는 필요액을 깎았다 (200 → 150 → 100). 그러면 **목표가 움직인다** — 반쯤 낸 학습이
    구사자가 생기는 순간 갑자기 완성되는 경로가 생기고, 그 주변에서 이미 버그를 잡았다.

    이제 회당 수확이 오른다. 사유 하나마다 `+learn_speedup`, **곱이 아니라 합**이다 —
    ×1.5 를 두 번 곱하면 2.25 배라 정가와 너무 벌어진다.

        사유 없음   회당 40   →  5회 · 200원 · AP 1.0
        하나        회당 60   →  4회 · 160원 · AP 0.8
        둘          회당 80   →  3회 · 120원 · AP 0.6
    """
    from core.agent_loop import learn_speed
    up = cfg.costs.learn_speedup
    base = cfg.costs.learn_base
    a1 = world.agents["Asla1"]

    m, why = learn_speed(a1, "Ranoa", world, cfg)
    assert m == 1.0 and "no help" in why
    assert learn_cost(a1, "Ranoa", world, cfg)[0] == base

    world.agents["Asla2"].known_langs.add("zh")          # 국내 구사자
    m, why = learn_speed(a1, "Ranoa", world, cfg)
    assert m == 1.0 + up and "nation" in why
    assert learn_cost(a1, "Ranoa", world, cfg)[0] == base

    a1.parent_langs.add("zh")                            # 부모까지
    m, why = learn_speed(a1, "Ranoa", world, cfg)
    assert m == 1.0 + 2 * up and "parent" in why
    assert learn_cost(a1, "Ranoa", world, cfg)[0] == base


def test_learn_self_not_counted(cfg, world):
    """자기 자신은 국내 구사자로 세지 않는다."""
    a1 = world.agents["Asla1"]
    a1.known_langs.add("zh")          # 자기가 zh 를 알아도
    cost, _ = learn_cost(a1, "Ranoa", world, cfg)
    assert cost == cfg.costs.learn_base       # 필요액은 언제나 고정이다 (8/22)


def test_learn_is_paid_in_instalments(cfg, world):
    """**한 번에 다 낼 필요가 없다.** 낸 만큼 쌓이고 다 차야 읽을 수 있다.

    Asla2 가 Miris(fr) 를 배운다 — Asla 에는 fr 구사자가 없어 정가다.
    """
    L = cfg.costs.learn_base
    learn = assistant_msg(tool_call("learn", "1", country="Miris"))
    script = [learn, learn, assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla2", script, budget=10000)
    # 한 번에 20. 정가는 L/20 번이고, 그 횟수 × 0.1 이 드는 AP 다.
    assert agent.budget == 10000 - 2 * cfg.costs.unit
    assert [r["charged"] for r in sink.learns] == [cfg.costs.unit] * 2
    assert [r["progress_before"] for r in sink.learns] == [0.0, cfg.costs.unit]
    (rec,) = sink.learns[:1]
    assert rec["required"] == L
    # **응답은 내가 몰랐던 것만 담는다** — 요청한 국가·액수는 되돌려주지 않는다.
    res = [r for r in _results(client) if "progress" in r]
    # **회당 수확은 `costs.unit × 배율`** — Asla2 → Miris 는 도움이 없어 정가다
    u = cfg.costs.unit
    assert [r["progress"] for r in res] == [u, 2 * u]   # 같은 해에도 쌓인다
    assert res[-1]["remaining"] == L - 2 * u
    assert res[-1]["complete"] is False   # 일정이 아니라 사실만
    assert "toward" not in res[0]         # 요청한 국가를 되돌려주지 않는다


def test_learn_never_takes_more_than_needed(cfg, world):
    """**마지막 한 번은 남은 만큼만 받는다** — 남는 돈이 조용히 사라지면 안 된다."""
    from core.agent_loop import Sink, execute_tool
    a = world.agents["Asla2"]; a.ap, a.budget = 1.0, 10_000.0
    L = cfg.costs.learn_base
    a.lang_progress = {"fr": L - 5.0}                # 정가에 5 만 남았다
    sink = Sink()
    r, _ = execute_tool("learn", {"country": "Miris", "reasoning": "r"},
                        world, a, cfg, sink, 48.0)
    assert r["ok"] and r["progress"] == L and r["complete"] is True
    assert a.budget == 10_000.0 - 5                  # 20 이 아니라 5 만 나간다
    assert sink.learns[0]["charged"] == 5


def test_learn_rejects_a_language_already_read(cfg, world):
    """Asla1 은 초기화로 zh 를 안다. 또 낼 수 없다."""
    script = [assistant_msg(tool_call("learn", "1", country="Ranoa", amount=100)),
              assistant_msg(tool_call("end_turn", "2"))]
    agent, sink, client, log = _run(world, cfg, "Asla1", script, budget=10000)
    assert any((not r["ok"]) and "already read" in r.get("error", "")
               for r in _results(client))
    assert agent.budget == 10000 and sink.learns == []


def test_learn_uses_less_than_a_whole_turn(cfg, world):
    """한 번의 납부는 한 해의 십분의 일이다 — 열 번이면 AP 를 다 쓴다."""
    # 정가 학습이 딱 한 해분의 행동력이다
    assert (cfg.costs.learn_base / cfg.costs.unit) * cfg.ap.unit == cfg.turn.action_points
    script = [assistant_msg(tool_call("learn", "1", country="Miris", amount=100)),
              assistant_msg(tool_call("speak", "2", to="Asla3", text="x")),
              assistant_msg(tool_call("end_turn", "3"))]
    agent, sink, client, log = _run(world, cfg, "Asla2", script, budget=10000)
    assert len(sink.learns) == 1 and len(sink.messages) == 1   # 같은 턴에 둘 다


def test_learn_action_points_scale_with_the_amount(cfg, world):
    """**분할이 손해면 안 된다.** 정액 0.3 이었을 때 정가를 여섯 번에 나눠 내면 AP 1.8,
    한 번에 내면 0.3 이었다 — 분할을 넣어놓고 분할에 벌을 주고 있었다.

    비례로 두면 나눠 내든 몰아 내든 합계가 같고, **정가 전액이 딱 한 해의 행동력**이 된다
    (L 200 ÷ 20 = 10회 × 0.1 = 1.0). 8/20 에 L 을 600 에서 내리면서 그렇게 맞췄다 —
    600 일 때는 세 해였고 아무도 끝내지 못했다.
    """
    from core.agent_loop import Sink, execute_tool
    base = cfg.costs.learn_base
    n = int(base / cfg.costs.unit)
    # 정가는 딱 한 해분의 행동력이다. 여기가 학습이 몇 해 걸리는지를 정하는 자리다.
    assert n * cfg.ap.unit == cfg.turn.action_points

    lump = world.agents["Asla2"]; lump.ap, lump.budget = 1.0, 10_000.0
    sink = Sink()
    for i in range(n):
        r, _ = execute_tool("learn", {"country": "Miris", "reasoning": "r"},
                            world, lump, cfg, sink, 48.0)
        assert r["ok"], (i, r)
    assert lump.ap == 0.0                       # 격자에 붙어 있다 (부동소수 아님)
    assert lump.budget == 10_000.0 - base       # 나눠 냈어도 합계는 정가 그대로
    assert r["complete"] is True                # 그리고 마지막 한 번에 끝난다

    # **행동력과 완성이 같은 지점에서 만난다.** 한 해를 학습에 다 쓰면 정가가 채워지므로,
    # 여기서 「AP 부족」 을 볼 수 없다 — 그건 test_learn_stops_when_action_runs_out 이 본다.
    r, _ = execute_tool("learn", {"country": "Miris", "reasoning": "r"},
                        world, lump, cfg, sink, 48.0)
    assert not r["ok"] and "already" in r["error"]


def test_learn_stops_when_action_runs_out(cfg, world):
    """AP 가 한 번 값보다 적으면 그 해에는 더 못 낸다 — 금액이 고정이라 절삭할 것이 없다."""
    from core.agent_loop import Sink, execute_tool
    a = world.agents["Asla2"]; a.ap, a.budget = cfg.ap.unit / 2, 10_000.0
    r, _ = execute_tool("learn", {"country": "Miris", "reasoning": "r"},
                        world, a, cfg, Sink(), 48.0)
    assert not r["ok"] and "not enough action" in r["error"]
    assert a.budget == 10_000.0                    # 한 푼도 안 나갔다


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
    p = prompts.system_for(a0, None, cfg) + "\n" + prompts.render_observation(world, a0, cfg, knob_ai=48)
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
