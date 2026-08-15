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


@pytest.fixture
def cfg():
    return config.load("configs/base.yaml")


def _run(cfg, scripts, turns=1):
    object.__setattr__(cfg.world, "total_turns", turns)
    ids = [f"{c}{i}" for c in ("Asla", "Ranoa", "Miris") for i in (1, 2, 3)]
    clients = {a: StubClient(list(scripts.get(a, []))) for a in ids}
    return loop.run_agentic(cfg, random.Random(1), lambda a: clients[a],
                            StubClient([{"role": "assistant", "content": "译文",
                                         "tool_calls": []}] * 60),
                            48.0, prompts.render_observation, prompts.system_for,
                            parallel=False)


def test_every_tool_requires_reasoning():
    """예외 없이 전부. 하나라도 빠지면 그 행동의 근거가 사라진다."""
    for t in tools.TOOLS:
        f = t["function"]
        assert "reasoning" in f["parameters"]["required"], f"{f['name']} 에 없다"
        assert "reasoning" in f["parameters"]["properties"]


def test_report_understanding_is_gone():
    """폐기됨 — 세계를 안 바꾸는 도구는 모델이 부르지 않는다."""
    assert "report_understanding" not in tools.TOOL_NAMES


def test_reasoning_recorded_per_action(cfg):
    """행동마다 하나씩. 실패한 호출의 근거도 남는다 (왜 그걸 시도했는지)."""
    scripts = {"Asla1": [
        assistant_msg(tool_call("speak", "1", to="Asla2", text="x", reasoning="같은 나라라 싸다")),
        assistant_msg(tool_call("speak", "2", to="NOBODY", text="x", reasoning="타국에 알리려 했다")),
        assistant_msg(tool_call("end_turn", "3", reasoning="예산이 없다")),
    ]}
    res = _run(cfg, scripts)
    rs = res.agent_logs[0]["Asla1"]["reasonings"]
    assert [r["tool"] for r in rs] == ["speak", "speak", "end_turn"]
    assert [r["ok"] for r in rs] == [True, False, True]
    assert rs[1]["reasoning"] == "타국에 알리려 했다"      # 실패해도 근거는 남는다


def test_missing_flag_only_when_all_blank(cfg):
    """하나라도 근거가 있으면 누락이 아니다."""
    scripts = {"Asla1": [assistant_msg(tool_call("end_turn", "1", reasoning="이래서"))]}
    assert _run(cfg, scripts).agent_logs[0]["Asla1"]["reasoning_missing"] is False
    scripts = {"Asla1": [assistant_msg(tool_call("end_turn", "1", reasoning=""))]}
    assert _run(cfg, scripts).agent_logs[0]["Asla1"]["reasoning_missing"] is True


def test_api_reasoning_is_separate(cfg):
    """API 의 message.reasoning(추론 모델의 사고 과정)과 섞이지 않는다."""
    msg = assistant_msg(tool_call("end_turn", "e", reasoning="스펙쪽"))
    msg["reasoning"] = "모델의 사고 과정"
    log = _run(cfg, {"Asla1": [msg]}).agent_logs[0]["Asla1"]
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
