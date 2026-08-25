"""행동마다의 reasoning — 지표 4 를 여기서 역추적한다. spec 4.2.

understood 를 받는 별도 도구는 폐기했다. 세계를 바꾸지 않는 도구는 모델이 부르지
않는다는 것이 실측으로 확인됐기 때문이다 (MAX_STEPS 8·20 양쪽에서 0건, 행동 분포
invest 16 · speak 8 · propose_vote 7 · procreate 2 · report_understanding 0).

대신 모든 도구 호출에 reasoning 이 필수로 붙는다. 결측이 거의 사라지고, "설명해보라"
고 물어서 받는 사후 합리화 대신 실제 결정의 근거가 남는다.
"""
from __future__ import annotations

import random

import pytest

from core import config, loop, tools
from core.llm import StubClient, assistant_msg, tool_call
from domains.meteor import prompts



# **노브는 이제 AP 다** (8/25). 돈 값 48 을 넘기면 「48 AP」 가 되어
# 한 해(1.0)를 넘고 발신이 불가능해진다 — 타입이 같아 아무도 안 잡았다.
KNOB = 0.5          # comm_intl_ai_ap 의 최고값

# **인구가 늘어난다** (8/21). `bear_child` 는 부모를 죽이지 않으므로 초기 9명 말고도
# 사람이 생긴다 — 초기 id 로만 만든 클라이언트 사전은 새 사람에게서 KeyError 를 낸다.
# 없는 id 는 즉시 끝내는 스텁으로 채운다.
def _client_for(clients, script_end):
    def get(aid):
        if aid not in clients:
            clients[aid] = StubClient([script_end] * 4)
        return clients[aid]
    return get


@pytest.fixture
def cfg_think(cfg):
    """사고형 모드 (도구 reasoning 없음). base.yaml 은 지금 켠 상태다 — DeepInfra 에서
    effort 가 안 먹어 사고를 아예 끄고 도구 인자로 되돌렸다."""
    import dataclasses
    return dataclasses.replace(cfg, llm=dataclasses.replace(cfg.llm, tool_reasoning=False))


@pytest.fixture
def cfg_tool(cfg):
    """도구마다 reasoning 을 받는 모드 (비사고형 모델용). base.yaml 은 지금 끈 상태다."""
    import dataclasses
    return dataclasses.replace(cfg, llm=dataclasses.replace(cfg.llm, tool_reasoning=True))


@pytest.fixture
def cfg():
    return config.load("configs/base.yaml")


def _run(cfg, scripts, turns=1):
    object.__setattr__(cfg.world, "total_turns", turns)
    ids = [f"{c}{i}" for c in ("Asla", "Ranoa", "Miris") for i in (1, 2, 3)]
    clients = {a: StubClient(list(scripts.get(a, []))) for a in ids}
    return loop.run_agentic(cfg, random.Random(1), _client_for(clients, assistant_msg(tool_call("end_turn", "z", reasoning="r"))),
                            StubClient([{"role": "assistant", "content": "译文",
                                         "tool_calls": []}] * 60),
                            48.0, prompts.render_turn_open, prompts.system_for,
                            parallel=False)


def test_every_acting_tool_requires_reasoning():
    """**행동하는** 도구는 전부. 하나라도 빠지면 그 행동의 근거가 사라진다.

    `end_turn` 만 예외다 — 행동이 아니라 행동을 그만두는 신호라 "행동 하나에 근거
    하나" 에 대응하지 않는다. 실측에서 근거가 있는 에이전트턴 407 중 end_turn 근거
    **뿐**인 것은 14건(3%)이라, 빼도 지표 4 의 표본이 3% 준다.
    """
    for t in tools.TOOLS:
        f = t["function"]
        if f["name"] == "end_turn":
            assert "reasoning" not in f["parameters"]["properties"]
            assert f["parameters"]["required"] == []
            continue
        assert "reasoning" in f["parameters"]["required"], f"{f['name']} 에 없다"
        assert "reasoning" in f["parameters"]["properties"]


def test_report_understanding_is_gone():
    """폐기됨 — 세계를 안 바꾸는 도구는 모델이 부르지 않는다."""
    assert "report_understanding" not in tools.TOOL_NAMES


def test_reasoning_recorded_per_action(cfg_tool):
    """행동마다 하나씩. 실패한 호출의 근거도 남는다 (왜 그걸 시도했는지)."""
    scripts = {"Asla1": [
        assistant_msg(tool_call("speak", "1", to="Asla2", text="x", reasoning="같은 나라라 싸다")),
        assistant_msg(tool_call("speak", "2", to="NOBODY", text="x", reasoning="타국에 알리려 했다")),
        assistant_msg(tool_call("end_turn", "3", reasoning="예산이 없다")),
    ]}
    res = _run(cfg_tool, scripts)
    rs = res.agent_logs[0]["Asla1"]["reasonings"]
    assert [r["tool"] for r in rs] == ["speak", "speak", "end_turn"]
    assert [r["ok"] for r in rs] == [True, False, True]
    assert rs[1]["reasoning"] == "타국에 알리려 했다"      # 실패해도 근거는 남는다


def test_missing_flag_only_when_all_blank(cfg_tool):
    """하나라도 근거가 있으면 누락이 아니다."""
    scripts = {"Asla1": [assistant_msg(tool_call("end_turn", "1", reasoning="이래서"))]}
    assert _run(cfg_tool, scripts).agent_logs[0]["Asla1"]["reasoning_missing"] is False
    scripts = {"Asla1": [assistant_msg(tool_call("end_turn", "1", reasoning=""))]}
    assert _run(cfg_tool, scripts).agent_logs[0]["Asla1"]["reasoning_missing"] is True


def test_api_reasoning_is_separate(cfg_tool):
    """API 의 message.reasoning(추론 모델의 사고 과정)과 섞이지 않는다."""
    msg = assistant_msg(tool_call("end_turn", "e", reasoning="스펙쪽"))
    msg["reasoning"] = "모델의 사고 과정"
    log = _run(cfg_tool, {"Asla1": [msg]}).agent_logs[0]["Asla1"]
    assert log["reasonings"][0]["reasoning"] == "스펙쪽"
    assert log["api_reasoning"] == "모델의 사고 과정"


def test_msg_id_is_global(cfg):
    """msg_id 는 전역 — 수신자별 지역 번호면 로그를 조인할 수 없다."""
    end = assistant_msg(tool_call("end_turn", "e", reasoning="r"))
    scripts = {
        "Asla1": [assistant_msg(tool_call("speak", "1", to="Asla2", text="a", reasoning="r")), end],
        "Asla3": [assistant_msg(tool_call("speak", "1", to="Asla2", text="b", reasoning="r")), end],
    }
    ids = [m["msg_id"] for m in _run(cfg, scripts).messages_log]
    assert len(ids) == len(set(ids)) == 2


def test_intent_is_gone(cfg):
    """intent 는 도구 스키마에도 로그에도 없다."""
    for t in tools.TOOLS:
        assert "intent" not in t["function"]["parameters"]["properties"]
    end = assistant_msg(tool_call("end_turn", "e", reasoning="r"))
    res = _run(cfg, {"Asla1": [
        assistant_msg(tool_call("speak", "1", to="Asla2", text="x", reasoning="r")), end]})
    assert "intent" not in res.messages_log[0]


# ── 새어나온 도구 호출 회수 (8/16) ────────────────────────────────────────────

@pytest.fixture
def world(cfg):
    import itertools
    w = loop.init_world(cfg, itertools.count(1))
    # **개체 차이를 1.0 으로 눕힌다** (8/22) — 다른 기제를 재는 테스트가 사람마다 다른
    # 액수에 흔들리지 않게. 차이 자체는 test_world_rules_v2 의 전용 테스트가 본다.
    for a in w.agents.values():
        a.invest_mult = 1.0
    return w


def test_recovers_a_tool_call_that_leaked_into_content():
    """모델이 `tool_calls` 대신 `content` 에 도구 호출을 넣는다. **전송 장애다.**

    8턴 실측에서 도구를 안 부른 응답 19건이 **전부** content 안의 도구 호출이었고
    (learn 6 · vote 6 · invest 3 …) 통째로 버려지고 있었다. 43턴 런에서는 learn 만
    19건이 날아갔다 — "학습 0건" 의 원인이 여기 있을 수 있다.
    """
    from core.agent_loop import recover_tool_calls as R
    import json as _json

    cases = [
        ('{\n"name": "learn",\n"arguments": {"country": "Ranoa", "reasoning": "r"}\n}',
         "learn", {"country": "Ranoa", "reasoning": "r"}),
        ('```json\n{"name":"vote","arguments":{"approve":true,"reasoning":"r"}}\n```',
         "vote", {"approve": True, "reasoning": "r"}),
        ('설명입니다. {"name":"invest","arguments":{"target":"facility","amount":50,'
         '"reasoning":"r"}} 끝.',
         "invest", {"target": "facility", "amount": 50, "reasoning": "r"}),
        ('{"tool_calls":[{"function":{"name":"speak","arguments":'
         '"{\\"to\\":\\"Ranoa1\\",\\"text\\":\\"x\\",\\"reasoning\\":\\"r\\"}"}}]}',
         "speak", {"to": "Ranoa1", "text": "x", "reasoning": "r"}),
    ]
    for content, name, args in cases:
        (c,) = R(content)
        assert c["function"]["name"] == name
        assert _json.loads(c["function"]["arguments"]) == args


def test_recovery_ignores_prose_and_unknown_tools():
    """잡담을 도구 호출로 오인하면 세계가 하지도 않은 일을 한다."""
    from core.agent_loop import recover_tool_calls as R
    assert R("그냥 잡담입니다") == []
    assert R('{"name":"unknown_tool","arguments":{}}') == []
    assert R(None) == [] and R("") == []
    assert R('{"country": "Ranoa"}') == []          # name 이 없다


def test_recovered_call_actually_runs(cfg, world):
    """회수한 호출이 실제로 세계를 바꾼다 — 줍기만 하고 안 쓰면 의미가 없다."""
    from core.agent_loop import Sink, run_agent_turn
    a = world.agents["Asla2"]
    a.ap = 1.0
    leaked = {"role": "assistant",
              "content": '{"name":"learn","arguments":{"country":"Miris",'
                         '"reasoning":"r"}}'}
    lg = run_agent_turn(world, a, cfg, StubClient([leaked]), Sink(), KNOB,
                        prompts.system_for(a, None, cfg), prompts.render_observation(world, a, cfg, KNOB))
    assert lg["recovered_calls"] == 1
    assert [x["type"] for x in lg["actions"]] == ["learn"]


def test_no_tool_call_is_not_reported_as_exhausted(cfg, world):
    """**로그가 거짓말을 하고 있었다.** tool_calls 가 비어 끝난 것을 `exhausted`
    (자원 고갈)로 기록했다 — 실측에서 행동 0건 23건 중 21건이 이 경우였고,
    Phase 1 을 "에이전트가 가난해서 못 움직인다" 로 읽을 뻔했다.
    """
    from core.agent_loop import Sink, run_agent_turn
    a = world.agents["Asla1"]
    a.ap = 1.0                      # 자원은 충분하다
    lg = run_agent_turn(world, a, cfg,
                        StubClient([{"role": "assistant", "content": "생각 중입니다"}]),
                        Sink(), KNOB, prompts.system_for(a, None, cfg),
                        prompts.render_observation(world, a, cfg, KNOB))
    assert lg["ended_by"] == "no_tool_call"
    assert lg["recovered_calls"] == 0
    assert "생각 중입니다" in lg["no_tool_content"]    # 무엇을 답했는지 남는다


def test_exhausted_still_means_exhausted(cfg, world):
    """진짜 자원 고갈은 그대로 `exhausted` 여야 한다.

    자유 행동(`memory_write`·`procreate`)이 생겨 `can_act` 만으로는 더 이상 고갈을
    못 만든다. 그것들까지 막아야 진짜 바닥이다 — 그때도 라벨이 `no_tool_call` 로
    새지 않는지가 이 테스트가 지키는 것이다.
    """
    from core.agent_loop import Sink, run_agent_turn
    from dataclasses import replace
    a = world.agents["Asla1"]
    a.ap = 0.0
    # `procreate` 는 없어졌고 `bear_child` 가 이미 1.0 이다 (8/21)
    broke = replace(cfg, ap=replace(cfg.ap, memory_write=0.1))
    lg = run_agent_turn(world, a, broke, StubClient([]), Sink(), KNOB,
                        prompts.system_for(a, None, cfg),
                        prompts.render_observation(world, a, broke, KNOB))
    assert lg["ended_by"] == "exhausted" and lg["steps"] == 0


def test_recovers_a_call_closed_with_a_paren():
    """모델이 마지막을 `}` 대신 `)` 로 닫는다 — 실측 2건, **둘 다 learn** 이었다."""
    from core.agent_loop import recover_tool_calls as R
    (c,) = R('{"name": "learn", "arguments": {"country": "Ranoa", "reasoning": "r"})')
    assert c["function"]["name"] == "learn"


def test_prose_after_acting_is_left_alone():
    """이미 행동한 뒤의 마무리 말은 **줍지 않는다.**

    회수 실패 36건 중 20건이 이것이고, 전부 그 턴에 도구를 부른 뒤였다. 사실상
    `end_turn` 이라 정상이다. 이걸 호출로 오인하면 세계가 하지도 않은 일을 한다.
    """
    from core.agent_loop import recover_tool_calls as R
    assert R("我已经成功提议将本国设施的目标改为拦截器。接下来，我将等待 Ranoa2 的回复。") == []
    assert R("Ce tour, je me suis concentré sur ma santé en investissant dans la wellness.") == []


def test_max_tokens_is_sent(monkeypatch):
    """상한이 없으면 반복 붕괴가 안 잘린다 — 실측 최악 40,935자."""
    import io
    import json as _json
    from core import llm as _llm

    seen = {}

    def _fake(req, timeout=None):
        seen["body"] = _json.loads(req.data.decode())
        return io.StringIO(_json.dumps({"choices": [{"message": {"content": "ok"}}]}))

    monkeypatch.setattr(_llm.urllib.request, "urlopen", _fake)
    _llm.OpenRouterClient("m", api_key="k", max_tokens=1024).chat([{"role": "user", "content": "x"}])
    assert seen["body"]["max_tokens"] == 1024

    seen.clear()
    _llm.OpenRouterClient("m", api_key="k").chat([{"role": "user", "content": "x"}])
    assert "max_tokens" not in seen["body"]          # 안 주면 안 보낸다


# ── 사고형 모델 모드 (tool_reasoning: false) ─────────────────────────────────

def test_thinking_replaces_the_tool_reasoning_argument(cfg_think):
    """사고형 모델에서는 도구마다 reasoning 을 또 받지 않는다 (spec 12.1).

    **그냥 끄면 지표 4 가 죽는다** — 2단계 판정의 ①이 읽을 근거가 통째로 사라진다.
    그래서 모델 자신의 사고를 `reasonings` 스트림에 넣어 이어준다.
    """
    assert cfg_think.llm.tool_reasoning is False          # base.yaml 이 사고형 모델을 쓴다
    msg = assistant_msg(tool_call("speak", "1", to="Asla2", text="x"))
    msg["reasoning"] = "같은 나라라 싸니 먼저 말을 걸어본다"
    log = _run(cfg_think, {"Asla1": [msg]}).agent_logs[0]["Asla1"]
    rs = log["reasonings"]
    assert rs[0]["source"] == "thinking"
    assert rs[0]["reasoning"] == "같은 나라라 싸니 먼저 말을 걸어본다"
    assert rs[0]["tool"] is None and rs[0]["step"] == 1
    assert log["reasoning_missing"] is False        # 판정이 읽을 것이 있다


def test_tool_schema_drops_reasoning_in_thinking_mode(cfg_think):
    """스키마에서 빠져야 모델이 그 자리를 안 채운다 — 안 그러면 사고를 두 번 시킨다."""
    from core import tools as _t
    picked = _t.tools_for(cfg_think)
    assert picked is _t.TOOLS_NO_REASONING
    for t in picked:
        assert "reasoning" not in t["function"]["parameters"]["properties"]


def test_judge_can_still_read_the_stream(cfg_think):
    """판정 1단계는 `reasonings[*].reasoning` 을 읽는다. 모드가 바뀌어도 그대로여야 한다."""
    from tools.score import judge
    msg = assistant_msg(tool_call("speak", "1", to="Asla2", text="x"))
    msg["reasoning"] = "Ranoa1 이 보낸 요격기 얘기에 답한다"
    log = _run(cfg_think, {"Asla1": [msg]}).agent_logs[0]["Asla1"]
    ev = [{"turn": 2, "type": "agent_turn", "agent": "Asla1",
           "reasonings": log["reasonings"]}]
    ms = [{"turn": 1, "msg_id": 1, "from": "Ranoa1", "to": "Asla1", "route": "ai",
           "delivered": True,
           "meta": {"src_lang": "zh", "dst_lang": "ja", "text_sent": "A",
                    "text_delivered": "B", "reader": False}}]
    (r,) = judge.link(ms, ev)
    assert r["skip"] is None
    assert "요격기" in r["reasonings"][0]


def test_tool_choice_comes_from_the_config_on_every_step(cfg, world):
    """**강제는 config 손잡이다** (8/25 · Eddie).

    `"required"` 가 코드에 박혀 있었다. 그런데 그것이 **프로바이더 선택을 덮어쓴다** —
    업체 전부가 `supported_parameters` 에 `tool_choice` 를 신고하지만 실제로 `required`
    를 받는 곳은 적고, OpenRouter 가 실제 지원으로 걸러내면서 우리 `order` 를 통째로
    건너뛴다. 같은 요청 6콜씩 실측:

        required   지연중앙 26,675ms   {Sail Research 5, CoreWeave 1}   ← order 무시
        auto       지연중앙  9,208ms   {GMICloud 6}                     ← 2.9배

    강제는 원래 **사고를 껐을 때** 넣은 우회책이었다 (8/16) — 「모델이 content 에 숙고를
    쏟고 그대로 끝낸다 · 턴의 2~7% 가 날아갔다」. 사고를 되켰으므로 숙고는 reasoning
    채널로 가고, 실측 6콜에서 도구 없는 응답은 **0건**이었다.

    값을 박지 않고 **손잡이가 모든 스텝에 전해지는지**만 본다 — 그래야 되돌릴 때
    코드를 안 고친다 (오늘 배운 것: 결정이 코드나 CLI 에만 있으면 조용히 낡는다).
    """
    from core.agent_loop import Sink, run_agent_turn
    from core import config as cfgmod
    import dataclasses

    for want in ("auto", "required"):
        llm = dataclasses.replace(cfg.llm, tool_choice=want)
        c = dataclasses.replace(cfg, llm=llm)
        stub = StubClient([assistant_msg(tool_call("speak", "1", to="Asla3", text="x",
                                                   reasoning="r")),
                           assistant_msg(tool_call("end_turn", "2"))])
        a = world.agents["Asla2"]; a.ap = 1.0
        run_agent_turn(world, a, c, stub, Sink(), KNOB,
                       prompts.system_for(a, None, c),
                       prompts.render_observation(world, a, c, KNOB))
        assert stub.calls, want
        # **모든 스텝**에 같은 값이 간다 — 첫 스텝만 강제하는 식의 어긋남이 없어야 한다
        assert [x["tool_choice"] for x in stub.calls] == [want] * len(stub.calls), want
