"""8/16 규칙 개정 — 국토 배타성 · 부고 · 진척 공개 · 유언.

`propose_vote` 는 **아무 일도 하지 않고 있었습니다.** 국토는 첫 시설 투자로
`DEFAULT_FACILITY_TYPE` 이 되는 게 전부였고, 43턴 실측에서 세 나라가 모두
`interceptor` 였던 것은 고른 게 아니라 **기본값**이었습니다. 그 54건은 6원씩 내고
효과가 없었고, 지표 3(정책 전환 유발율)이 재던 것도 실은 무효 행동이었습니다.
"""
from __future__ import annotations

import itertools
import random

import pytest

from core import config, loop
from core.agent_loop import Sink

BASE = "configs/base.yaml"


@pytest.fixture()
def cfg():
    return config.load(BASE)


@pytest.fixture()
def world(cfg):
    w = loop.init_world(cfg, itertools.count(1))
    w.turn = 5
    return w


def _settle(world, cfg, sink, result=None):
    result = result or loop.RunResult(world=world)
    loop._settle_agentic(world, cfg, random.Random(0), sink, None, 48.0,
                         itertools.count(500), result, itertools.count(900))
    return result


# ── 국토 배타성 ─────────────────────────────────────────────────────────────────

def _propose(world, cfg, who, target):
    sink = Sink()
    sink.votes = [(who, world.agents[who].country, target)]
    return _settle(world, cfg, sink)


def _ballot(world, cfg, *votes):
    sink = Sink()
    sink.ballots = [(w, world.agents[w].country, a) for w, a in votes]
    return _settle(world, cfg, sink)


def test_proposal_does_not_change_anything_for_three_turns(cfg, world):
    """제안한 순간에는 아무 일도 일어나지 않는다. **유예가 상의할 시간이다.**"""
    c = world.countries["Ranoa"]
    c.land, c.progress = "interceptor", 295.0
    world.turn = 10
    _propose(world, cfg, "Ranoa1", "bunker")
    assert (c.land, c.progress) == ("interceptor", 295.0)
    assert c.proposal["target"] == "bunker" and c.proposal["by"] == "Ranoa1"
    assert c.proposal["vote_turn"] == 10 + loop.VOTE_DELAY


def test_ballot_only_counts_on_the_ballot_turn(cfg, world):
    """유예 중에 던진 표는 무효다 — 그때 통과되면 상의할 시간이 사라진다."""
    from core.agent_loop import execute_tool
    c = world.countries["Ranoa"]
    world.turn = 10
    _propose(world, cfg, "Ranoa1", "bunker")
    a = world.agents["Ranoa2"]; a.ap = 1.0
    world.turn = 12                                    # 아직 유예 중
    res, _ = execute_tool("vote", {"approve": True, "reasoning": "r"},
                          world, a, cfg, Sink(), 48.0)
    assert not res["ok"] and "14" in res["error"]


def test_silence_counts_as_consent(cfg, world):
    """찬성이 반대보다 많으면 통과 — **1찬 0반도 통과한다.**

    반대하려면 그 턴에 표를 내야 한다. 한 사람이 나라를 망칠 수 있다는 것이 의도이고,
    막는 것은 규칙이 아니라 사람들이다. 유예 3턴은 그러라고 준 시간이다.
    """
    c = world.countries["Ranoa"]
    c.land, c.progress = "interceptor", 295.0
    world.turn = 10
    _propose(world, cfg, "Ranoa1", "bunker")
    world.turn = 10 + loop.VOTE_DELAY
    r = _ballot(world, cfg, ("Ranoa1", True))
    assert (c.land, c.progress) == ("bunker", 0.0)
    (ch,) = r.land_changes
    assert (ch["yes"], ch["no"], ch["passed"]) == (1, 0, True)
    assert ch["progress_lost"] == 295.0
    assert c.proposal is None


def test_two_against_one_blocks_it(cfg, world):
    """설득에 성공하면 막힌다. 진척은 그대로 남는다."""
    c = world.countries["Ranoa"]
    c.land, c.progress = "interceptor", 295.0
    world.turn = 10
    _propose(world, cfg, "Ranoa1", "bunker")
    world.turn = 10 + loop.VOTE_DELAY
    r = _ballot(world, cfg, ("Ranoa1", True), ("Ranoa2", False), ("Ranoa3", False))
    assert (c.land, c.progress) == ("interceptor", 295.0)
    assert r.land_changes[0]["passed"] is False
    assert c.proposal is None                     # 부결돼도 제안은 닫힌다


def test_nobody_votes_nothing_passes(cfg, world):
    """아무도 표를 안 내면 통과도 부결도 없다 — 0 > 0 이 아니다."""
    c = world.countries["Ranoa"]
    c.land, c.progress = "interceptor", 295.0
    world.turn = 10
    _propose(world, cfg, "Ranoa1", "bunker")
    world.turn = 10 + loop.VOTE_DELAY
    r = _ballot(world, cfg)
    assert (c.land, c.progress) == ("interceptor", 295.0)
    assert r.land_changes[0]["passed"] is False


def test_only_one_proposal_at_a_time(cfg, world):
    """제안이 열려 있으면 새 제안을 못 연다 — 안 그러면 유예가 무의미해진다."""
    from core.agent_loop import execute_tool
    world.turn = 10
    _propose(world, cfg, "Ranoa1", "bunker")
    a = world.agents["Ranoa2"]; a.ap, a.budget = 1.0, 100.0
    res, _ = execute_tool("propose_vote", {"target": "interceptor", "reasoning": "r"},
                          world, a, cfg, Sink(), 48.0)
    assert not res["ok"] and "already has an open proposal" in res["error"]


def test_cannot_invest_before_the_facility_is_decided(cfg, world):
    """투표로 정해지기 전에는 지을 것이 없다. **예산은 차감되지 않는다.**"""
    from core.agent_loop import execute_tool
    a = world.agents["Asla1"]; a.ap, a.budget = 1.0, 500.0
    sink = Sink()
    res, _ = execute_tool("invest", {"target": "facility", "amount": 100,
                                     "to": "Ranoa", "reasoning": "r"},
                          world, a, cfg, sink, 48.0)
    assert not res["ok"] and "has not decided" in res["error"]
    assert a.budget == 500.0 and sink.facility == []


def test_can_invest_once_decided(cfg, world):
    from core.agent_loop import execute_tool
    world.countries["Ranoa"].land = "interceptor"
    a = world.agents["Asla1"]; a.ap, a.budget = 1.0, 500.0
    sink = Sink()
    res, _ = execute_tool("invest", {"target": "facility", "amount": 100,
                                     "to": "Ranoa", "reasoning": "r"},
                          world, a, cfg, sink, 48.0)
    assert res["ok"] and sink.facility == [("Ranoa", 100.0, "Asla1")]


def test_only_nationals_may_vote(cfg, world):
    """투표는 그 나라 주민에 한한다. **도구 설명에 명시돼 있어야 한다.**

    8턴 실측에서 Asla2 가 외국인 Ranoa2 에게 자국 제안의 투표를 부탁했다 —
    「Asla2はinterceptorの建設に賛成し、Ranoa2の投票を依頼します」. 자연스러운
    오해지만 의도 밖이라 규칙을 말로 적었다.
    """
    from core import tools
    d = {t["function"]["name"]: t["function"]["description"] for t in tools.TOOLS}
    assert "own nation" in d["propose_vote"] and "foreigner cannot" in d["propose_vote"]
    assert "your own nation" in d["vote"] and "another nation" in d["vote"]

    # 말뿐이 아니라 코드도 막는다 — vote 는 언제나 제안자 자신의 나라에만 들어간다
    from core.agent_loop import execute_tool
    world.turn = 10
    _propose(world, cfg, "Ranoa1", "bunker")           # Ranoa 에 제안이 열림
    a = world.agents["Asla1"]                          # 외국인
    a.ap = 1.0
    sink = Sink()
    res, _ = execute_tool("vote", {"approve": True, "reasoning": "r"},
                          world, a, cfg, sink, 48.0)
    assert not res["ok"] and "no open proposal" in res["error"]
    assert sink.ballots == []


# ── 진척 공개 ───────────────────────────────────────────────────────────────────

def test_progress_gain_is_reported_after_the_fact(cfg, world):
    """행위 **전**에는 안 알려주지만 **후**에는 알려준다.

    확률적이라 한 건으로는 success_prob 을 읽을 수 없고, 모르면 "얼마를 더 내야
    하는가" 를 판단할 근거가 아예 없다.
    """
    sink = Sink()
    sink.facility = [("Ranoa", 90.0, "Asla1"), ("Ranoa", 30.0, "Ranoa2")]
    r = _settle(world, cfg, sink)
    assert len(r.facility_gains) == 2
    by = {g["agent"]: g for g in r.facility_gains}
    assert by["Asla1"]["amount"] == 90.0 and by["Asla1"]["to"] == "Ranoa"
    assert sum(g["gain"] for g in r.facility_gains) == world.countries["Ranoa"].progress
    # 출자자에게 다음 턴 인박스로 간다
    gains = [e for e in world.inbox_queue if "fac_gain" in e["msg"]]
    assert {e["to"] for e in gains} == {"Asla1", "Ranoa2"}
    assert all(e["deliver_turn"] == world.turn + 1 for e in gains)


def test_gain_notice_reaches_a_foreign_contributor(cfg, world):
    """타국 요격기에 낸 사람도 자기 몫의 결과를 받는다 — 안 그러면 남의 땅에 내는
    선택이 영영 깜깜이가 된다."""
    sink = Sink()
    sink.facility = [("Ranoa", 120.0, "Asla1")]
    _settle(world, cfg, sink)
    (e,) = [x for x in world.inbox_queue if "fac_gain" in x["msg"]]
    assert e["to"] == "Asla1" and e["msg"]["to"] == "Ranoa"


# ── 부고 ────────────────────────────────────────────────────────────────────────

def test_death_is_announced_to_the_same_nation_only(cfg, world):
    """같은 나라 사람은 안다. 타국의 인구 구성은 여전히 메시지로만 안다 (spec 4.1).

    국내 구사자 할인이 갑자기 2배가 된 이유를 알 수 있게 하는 정보이기도 하다.
    """
    for a in world.agents.values():
        a.age = 40                                  # 확실히 죽게
    result = loop.RunResult(world=world)
    loop._death_birth(world, cfg, random.Random(1), sorted(world.agents), set(),
                      itertools.count(700), result)
    assert result.deaths_log
    dead_ranoa = [d["who"] for d in result.deaths_log if d["country"] == "Ranoa"]
    assert dead_ranoa
    # 배선은 run_turn_agentic 에 있으므로 여기서는 기록만 확인한다
    assert all(d["by"] == "natural" for d in result.deaths_log)
    assert {d["country"] for d in result.deaths_log} <= {"Asla", "Ranoa", "Miris"}


# ── 유언 ────────────────────────────────────────────────────────────────────────

def test_testament_becomes_the_child_s_memory(cfg, world):
    """유언은 별도 블록이 아니라 아이의 기억 초기값이다.

    아이가 `memory_write` 로 덮어쓰면 사라진다 — **그 덮어쓰기가 구전의 감쇠다.**
    """
    world.agents["Asla1"].memory = "부모의 메모"
    loop._procreate_child(world, "Asla1", "요격기에만 내라", cfg,
                          itertools.count(800), loop.RunResult(world=world))
    child = world.agents["Asla1"]
    assert child.memory == "요격기에만 내라"
    assert "부모의 메모" not in child.memory


def test_observation_has_no_separate_testament_block(cfg, world):
    """유언 블록·'알아낸 것' 블록은 폐지됐다. 전부 memory 하나로 관리된다."""
    from domains.meteor import prompts
    world.agents["Asla1"].memory = "요격기에만 내라"
    obs = prompts.render_observation(world, world.agents["Asla1"], cfg, 48.0)
    assert obs.count("要撃機にだけ") == 0                    # 별도 블록 없음
    assert "要撃機にだけ出せ" not in obs
    assert "要撃機" not in obs or "覚え書き" in obs
    assert obs.count("요격기에만 내라") == 1                  # 메모 안에 딱 한 번


# ── 관측의 새 항목 ──────────────────────────────────────────────────────────────

def test_year_starts_at_42(cfg, world):
    """1 로 시작하면 '첫 해라서 아직 괜찮다' 같은 편향이 붙는다."""
    from domains.meteor import prompts
    world.turn = 1
    assert "42" in prompts.render_observation(world, world.agents["Asla1"], cfg, 48.0)
    world.turn = 10
    assert "51" in prompts.render_observation(world, world.agents["Asla1"], cfg, 48.0)


def test_threshold_is_shown_but_success_prob_is_not(cfg, world):
    """목표를 모르면 '자국의 진척: 728' 은 해석할 수 없는 숫자다.

    임계는 spec 4.1 은닉 목록에 없다 — 거기 있는 것은 `success_prob` 이다.
    """
    from domains.meteor import prompts
    obs = prompts.render_observation(world, world.agents["Asla1"], cfg, 48.0)
    assert str(int(cfg.thresholds.interceptor)) in obs
    for hidden in ("0.3", "success_prob", str(int(cfg.thresholds.bunker_scale))):
        assert hidden not in obs


def test_production_multiplier_is_gone(cfg, world):
    """배수는 안 알려준다 — 수입에서 추론 가능하다."""
    from domains.meteor import prompts
    world.countries["Asla"].national_capital = 3000.0
    obs = prompts.render_observation(world, world.agents["Asla1"], cfg, 48.0)
    assert "1.28" not in obs and "倍率: 1" not in obs
    assert "+128" in obs or "+1" in obs          # 수입에는 반영돼 보인다


def test_ask_is_gone(cfg):
    """되묻기는 speak 로 충분하다. 도구를 하나 줄이면 실패 모드도 하나 준다."""
    from core import tools
    assert "ask" not in tools.TOOL_NAMES
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert "ask" not in names and "speak" in names
