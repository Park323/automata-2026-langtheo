"""8/16 규칙 개정 — 국토 배타성 · 부고 · 진척 공개 · 유언.

`propose_vote` 는 **아무 일도 하지 않고 있었습니다.** 국토는 첫 시설 투자로
`DEFAULT_FACILITY_TYPE` 이 되는 게 전부였고, 43턴 실측에서 세 나라가 모두
`interceptor` 였던 것은 고른 게 아니라 **기본값**이었습니다. 그 54건은 6원씩 내고
효과가 없었고, 지표 3(정책 전환 유발율)이 재던 것도 실은 무효 행동이었습니다.
"""
from __future__ import annotations

import itertools
import json
import random
import re

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
    # **개체 차이를 1.0 으로 눕힌다** (8/22). 소득 배수·처리량 배수는 태어날 때 뽑히므로,
    # 그것을 그대로 두면 **다른 기제를 재는 테스트가 사람마다 다른 액수에 흔들린다.**
    # 차이 자체는 `test_world_rules_v2.py` 의 전용 테스트가 본다.
    for _a in w.agents.values():
        _a.income_mult = _a.invest_mult = 1.0
    return w


def _settle(world, cfg, sink, result=None):
    result = result or loop.RunResult(world=world)
    loop._settle_agentic(world, cfg, random.Random(0), sink, None, 48.0,
                         itertools.count(500), result, itertools.count(900))
    return result



def _do(world, cfg, agent, name, args):
    """도구 하나를 직접 실행한다 (정산 없이). 행동력 산술만 볼 때 쓴다."""
    from core import agent_loop
    r, _ = agent_loop.execute_tool(name, args, world, agent, cfg, Sink(), 48.0)
    return r


# ── 국토 배타성 ─────────────────────────────────────────────────────────────────

def _call(world, cfg, who):
    """採決을 소집한다. **무엇을 지을지는 여기서 정하지 않는다.**"""
    sink = Sink()
    sink.votes = [(who, world.agents[who].country)]
    return _settle(world, cfg, sink)


def _ballot(world, cfg, *votes):
    """(who, choice) 들. choice 는 interceptor / bunker / abstain."""
    sink = Sink()
    sink.ballots = [(w, world.agents[w].country, ch) for w, ch in votes]
    return _settle(world, cfg, sink)


def test_calling_a_ballot_changes_nothing_for_three_turns(cfg, world):
    """소집한 순간에는 아무 일도 일어나지 않는다. **유예가 상의할 시간이다.**"""
    c = world.countries["Ranoa"]
    c.land, c.progress = "interceptor", 295.0
    world.turn = 10
    _call(world, cfg, "Ranoa1")
    assert (c.land, c.progress) == ("interceptor", 295.0)
    assert c.proposal["by"] == "Ranoa1" and c.proposal["vote_turn"] == 10 + loop.VOTE_DELAY
    assert "target" not in c.proposal          # 소집에는 내용이 없다


def test_calling_a_ballot_carries_no_choice(cfg, world):
    """**전에는 `target` 을 들고 「이것으로 하자」 를 열었다.** 같은 턴에 둘이 제안하면
    둘 다 도구를 통과하는데 하나만 열렸고, 밀린 쪽은 AP 0.6 을 내고 아무 일도 안 일어난
    것을 알 방법이 없었다. 소집에 내용이 없으면 겹칠 것이 없다."""
    from core.agent_loop import execute_tool
    world.turn = 10
    a = world.agents["Ranoa1"]; a.ap, a.budget = 1.0, 100.0
    res, _ = execute_tool("propose_vote", {"reasoning": "r"}, world, a, cfg, Sink(), 48.0)
    assert res["ok"] and res["ballot_turn"] == 10 + loop.VOTE_DELAY


def test_two_people_calling_the_same_ballot_is_harmless(cfg, world):
    """둘이 소집해도 **같은 採決**이다. 유령 제안이 생기지 않는다."""
    world.turn = 10
    sink = Sink()
    sink.votes = [("Ranoa1", "Ranoa"), ("Ranoa3", "Ranoa")]
    r = _settle(world, cfg, sink)
    calls = [v for v in r.votes_log if v["kind"] == "propose"]
    assert [c["opened"] for c in calls] == [True, False]     # id 순으로 앞선 것만 연다
    assert all(c["vote_turn"] == 10 + loop.VOTE_DELAY for c in calls)  # 같은 날짜다
    assert world.countries["Ranoa"].proposal["by"] == "Ranoa1"


def test_ballot_only_counts_on_the_ballot_turn(cfg, world):
    """유예 중에 던진 표는 무효다 — 그때 정해지면 상의할 시간이 사라진다."""
    from core.agent_loop import execute_tool
    world.turn = 10
    _call(world, cfg, "Ranoa1")
    a = world.agents["Ranoa2"]; a.ap = 1.0
    # 유예 안쪽. **VOTE_DELAY 를 4 에서 2 로 줄이면서 12 가 採決일 자체가 됐다** —
    # 상수를 고쳤을 때 이 숫자가 함께 움직이지 않으면 테스트가 조용히 다른 것을 잰다.
    world.turn = 10 + loop.VOTE_DELAY - 1
    res, _ = execute_tool("vote", {"choice": "bunker", "reasoning": "r"},
                          world, a, cfg, Sink(), 48.0)
    # **연도로 말한다.** 「turn 14」 라고 말하고 있었다 — 세계는 55년인데 내부 인덱스다.
    from domains.meteor.prompts import FIRST_YEAR
    assert not res["ok"] and str(FIRST_YEAR + 10 + loop.VOTE_DELAY - 1) in res["error"]
    assert "turn" not in res["error"]


def test_one_vote_decides_when_nobody_else_shows_up(cfg, world):
    """**한 표만 나오면 그 한 표가 나라를 정한다.**

    한 사람이 나라를 망칠 수 있다는 것이 의도이고, 막는 것은 규칙이 아니라 사람들이다.
    유예 3턴은 그러라고 준 시간이다.
    """
    c = world.countries["Ranoa"]
    c.land, c.progress = "interceptor", 295.0
    world.turn = 10
    _call(world, cfg, "Ranoa1")
    world.turn = 10 + loop.VOTE_DELAY
    r = _ballot(world, cfg, ("Ranoa1", "bunker"))
    assert (c.land, c.progress) == ("bunker", 0.0)
    (ch,) = r.land_changes
    assert ch["chosen"] == "bunker" and ch["changed"] is True
    assert ch["counts"] == {"interceptor": 0, "bunker": 1, "abstain": 0}
    assert ch["progress_lost"] == 295.0 and c.proposal is None


def test_the_majority_choice_wins(cfg, world):
    """설득에 성공하면 막힌다. 진척은 그대로 남는다."""
    c = world.countries["Ranoa"]
    c.land, c.progress = "interceptor", 295.0
    world.turn = 10
    _call(world, cfg, "Ranoa1")
    world.turn = 10 + loop.VOTE_DELAY
    r = _ballot(world, cfg, ("Ranoa1", "bunker"),
                ("Ranoa2", "interceptor"), ("Ranoa3", "interceptor"))
    assert (c.land, c.progress) == ("interceptor", 295.0)
    (ch,) = r.land_changes
    assert ch["chosen"] == "interceptor" and ch["changed"] is False     # 이미 그것이다
    assert ch["progress_lost"] == 0.0


def test_a_tie_keeps_what_the_nation_has(cfg, world):
    """**동수면 현 상태 그대로고 진척도 살아 있다.**

    합의 실패의 대가를 진척 파괴로 물리면, 소집 한 번이 남의 나라 진척을 지우는
    무기가 된다.
    """
    c = world.countries["Ranoa"]
    c.land, c.progress = "interceptor", 295.0
    world.turn = 10
    _call(world, cfg, "Ranoa1")
    world.turn = 10 + loop.VOTE_DELAY
    r = _ballot(world, cfg, ("Ranoa1", "bunker"), ("Ranoa2", "interceptor"))
    assert (c.land, c.progress) == ("interceptor", 295.0)
    (ch,) = r.land_changes
    assert ch["chosen"] is None and ch["changed"] is False
    assert c.proposal is None                     # 정해지지 않아도 採決은 닫힌다


def test_nobody_voting_keeps_what_the_nation_has(cfg, world):
    """아무도 표를 안 내면 정해지지 않는다 — 0 > 0 이 아니다."""
    c = world.countries["Ranoa"]
    c.land, c.progress = "interceptor", 295.0
    world.turn = 10
    _call(world, cfg, "Ranoa1")
    world.turn = 10 + loop.VOTE_DELAY
    r = _ballot(world, cfg)
    assert (c.land, c.progress) == ("interceptor", 295.0)
    assert r.land_changes[0]["chosen"] is None


def test_abstain_counts_for_neither_but_is_recorded(cfg, world):
    """**기권은 개표상 표를 안 낸 것과 같다.** 다만 「생각해봤지만 정하지 않았다」 가
    근거와 함께 로그에 남아 지표가 읽는다."""
    c = world.countries["Ranoa"]
    c.land = None
    world.turn = 10
    _call(world, cfg, "Ranoa1")
    world.turn = 10 + loop.VOTE_DELAY
    r = _ballot(world, cfg, ("Ranoa1", "abstain"), ("Ranoa2", "abstain"),
                ("Ranoa3", "bunker"))
    (ch,) = r.land_changes
    assert ch["counts"] == {"interceptor": 0, "bunker": 1, "abstain": 2}
    assert ch["chosen"] == "bunker"                # 기권은 어느 쪽으로도 안 센다
    votes = [v for v in r.votes_log if v["kind"] == "ballot"]
    assert sorted(v["choice"] for v in votes) == ["abstain", "abstain", "bunker"]


def test_only_one_ballot_at_a_time(cfg, world):
    """採決이 열려 있으면 새로 소집할 수 없다 — 안 그러면 유예가 무의미해진다."""
    from core.agent_loop import execute_tool
    world.turn = 10
    _call(world, cfg, "Ranoa1")
    a = world.agents["Ranoa2"]; a.ap, a.budget = 1.0, 100.0
    res, _ = execute_tool("propose_vote", {"reasoning": "r"}, world, a, cfg, Sink(), 48.0)
    assert not res["ok"] and "already called" in res["error"]


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

    # 말뿐이 아니라 코드도 막는다 — vote 는 언제나 부른 사람 자신의 나라에만 들어간다
    from core.agent_loop import execute_tool
    world.turn = 10
    _call(world, cfg, "Ranoa1")                        # Ranoa 에 採決이 열림
    a = world.agents["Asla1"]; a.ap = 1.0
    res, _ = execute_tool("vote", {"choice": "bunker", "reasoning": "r"},
                          world, a, cfg, Sink(), 48.0)
    assert not res["ok"] and "no open proposal" in res["error"]


def test_investing_never_reveals_whether_a_nation_decided(cfg, world):
    """접수와 과금만 답한다. **타국이 시설을 정했는지 알려주지 않는다.**

    알려주면 10원짜리 조회로 타국 국토를 읽을 수 있다 — 국제 메시지가 24~48원인데
    그보다 싸서, *"타국 사정은 소통해야만 안다"* 는 전제가 통째로 무너진다
    (spec 4.1 은닉 목록: 타국의 진척·예산·국토·언어 능력).
    """
    from core.agent_loop import execute_tool
    world.countries["Ranoa"].land = "interceptor"      # 정한 나라
    world.countries["Miris"].land = None               # 안 정한 나라
    a = world.agents["Asla1"]; a.ap, a.budget = 1.0, 500.0
    outs = []
    for to in ("Ranoa", "Miris"):
        res, _ = execute_tool("invest", {"target": "facility", "to": to, "reasoning": "r"},
                              world, a, cfg, Sink(), 48.0)
        outs.append(res)
    # 나라 이름·잔액·AP 말고는 한 글자도 달라선 안 된다 — 다르면 그것이 곧 조회다.
    # (잔액과 AP 는 두 번 연달아 내서 줄어든 것이지 나라 차이가 아니다)
    # 잔액·AP 말고는 한 글자도 달라선 안 된다 — 다르면 그것이 곧 조회다.
    # (둘을 두 번 연달아 내서 줄어든 것이지 나라 차이가 아니다)
    seq = ("budget_left", "ap_left")
    shape = [{k: (v if k not in seq else None) for k, v in o.items()} for o in outs]
    assert shape[0] == shape[1]
    assert all(o["ok"] for o in outs)
    # 나라 이름조차 안 나온다 — 응답이 입력을 되돌려주지 않게 된 부수 효과다
    assert not any("Ranoa" in json.dumps(o) or "Miris" in json.dumps(o) for o in outs)
    assert a.budget == 500.0 - 2 * cfg.costs.unit       # 둘 다 과금됐다


def test_money_into_an_undecided_nation_just_vanishes(cfg, world):
    """정해지지 않았으면 돈은 나가고 아무 일도 일어나지 않는다 — route=original 과 같은 도박."""
    sink = Sink()
    sink.facility = [("Miris", 200.0, "Asla1")]        # Miris 는 land None
    r = _settle(world, cfg, sink)
    assert world.countries["Miris"].progress == 0.0
    (g,) = r.facility_gains
    assert g["gain"] == 0 and g["amount"] == 200.0


def test_foreign_money_digs_whatever_they_are_building(cfg, world):
    """**벙커라도 들어간다.** 요격기를 짓는 척하며 벙커를 파고 타국 출자를 받는
    전략이 성립합니다 (spec 4.4 개정).

    이전 판은 "벙커는 자국에만" 이었는데, 그대로 두면 배신이 *"남 손해"* 에 그칩니다.
    이렇게 두면 **배신이 실제로 이득**이 되고, 동시에 *"너희 무엇을 짓고 있는가"* 가
    이 세계에서 가장 값비싼 정보가 됩니다 — 그 답은 번역을 거쳐야만 오니까요.
    """
    world.countries["Ranoa"].land = "bunker"
    sink = Sink()
    sink.facility = [("Ranoa", 300.0, "Asla1")]        # 외국인이 낸 돈
    r = _settle(world, cfg, sink)
    assert world.countries["Ranoa"].progress > 0       # 남의 돈으로 벙커가 깊어진다
    (g,) = r.facility_gains
    assert g["agent"] == "Asla1" and g["gain"] > 0
    # 출자자는 **늘었다는 것만** 알 뿐, 무엇이 깊어졌는지도 얼마나인지도 모른다
    (e,) = [x for x in world.inbox_queue if "fac_moved" in x["msg"]]
    assert e["msg"]["fac_moved"] is True and "fac_gain" not in e["msg"]
    assert "bunker" not in json.dumps(e["msg"]) and "interceptor" not in json.dumps(e["msg"])


def test_invest_tool_states_the_rule(cfg):
    """규칙을 모르면 그 도박이 선택이 아니라 우연이 된다."""
    from core import tools
    d = {t["function"]["name"]: t["function"]["description"] for t in tools.TOOLS}
    assert "whatever that nation is currently building" in d["invest"]
    assert "Only that nation knows" in d["invest"]


def test_the_gain_notice_arrives_either_way(cfg, world):
    """통지가 없으면 **그 부재가 곧 '아직 안 정했다'** 가 된다. 똑같이 보낸다."""
    sink = Sink()
    sink.facility = [("Miris", 50.0, "Asla1")]         # 미정
    _settle(world, cfg, sink)
    (e,) = [x for x in world.inbox_queue if "fac_moved" in x["msg"]]
    assert e["to"] == "Asla1" and e["msg"]["fac_moved"] is False


def test_can_invest_once_decided(cfg, world):
    from core.agent_loop import execute_tool
    world.countries["Ranoa"].land = "interceptor"
    a = world.agents["Asla1"]; a.ap, a.budget = 1.0, 500.0
    sink = Sink()
    res, _ = execute_tool("invest", {"target": "facility", "to": "Ranoa", "reasoning": "r"},
                          world, a, cfg, sink, 48.0)
    assert res["ok"] and sink.facility == [("Ranoa", cfg.costs.unit, "Asla1")]


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
    # 출자자에게 다음 턴 인박스로 간다. **자국민은 액수까지, 외국인은 여부만.**
    gains = [e for e in world.inbox_queue
             if "fac_gain" in e["msg"] or "fac_moved" in e["msg"]]
    assert {e["to"] for e in gains} == {"Asla1", "Ranoa2"}
    assert all(e["deliver_turn"] == world.turn + 1 for e in gains)
    by = {e["to"]: e["msg"] for e in gains}
    assert "fac_gain" in by["Ranoa2"] and "fac_moved" not in by["Ranoa2"]
    assert "fac_moved" in by["Asla1"] and "fac_gain" not in by["Asla1"]


def test_gain_notice_reaches_a_foreign_contributor(cfg, world):
    """타국 요격기에 낸 사람도 자기 몫의 결과를 받는다 — 안 그러면 남의 땅에 내는
    선택이 영영 깜깜이가 된다."""
    sink = Sink()
    sink.facility = [("Ranoa", 120.0, "Asla1")]
    _settle(world, cfg, sink)
    (e,) = [x for x in world.inbox_queue if "fac_moved" in x["msg"]]
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
    """연도는 **해 시작 문구**가 말한다. 관측에는 없다 — 같은 사실이 두 군데면 어긋난다."""
    from domains.meteor import prompts
    world.turn = 1
    a = world.agents["Asla1"]
    assert "42" in prompts.render_turn_open(world, a, cfg, 48.0, [])
    obs = prompts.render_observation(world, a, cfg, 48.0)
    assert "42" not in obs and prompts.T["ja"]["year"].split(":")[0] not in obs


def test_threshold_is_no_longer_free(cfg, world):
    """임계는 **관측으로만** 알 수 있다 — `observe_risk` 를 사서 재야 한다.

    공짜로 보이면 진척과 임계가 둘 다 정확해져 산수로 풀리는 문제가 된다.
    이제 둘 다 기술력에 따라 흐릿하고, 알아낸 값은 **개인의 것**이라 남에게 알리려면
    말해야 한다 — 국제로 보내면 번역을 타고, 그게 지표 6a 가 재는 경로다.
    """
    from dataclasses import replace

    from domains.meteor import prompts
    # **success_prob 를 0.3 에서 떼어놓고 잽니다.** AP 가 관측에 나오면서 0.3 이 정당하게
    # 찍히고(speak·learn), 맨 숫자 검사가 오탐을 냈습니다. 0.37 로 두면 그 숫자가 보이는
    # 경우는 누출뿐입니다 — 그물은 넓게 두되 우연한 일치만 없앱니다.
    probe = replace(cfg, world=replace(cfg.world, success_prob=0.37))
    obs = prompts.render_observation(world, world.agents["Asla1"], probe, 48.0)
    for hidden in (str(int(probe.thresholds.interceptor)), "0.37", "success_prob",
                   str(int(probe.thresholds.bunker_scale))):
        assert hidden not in obs, hidden
    assert "observe_risk" in obs          # 살 수 있다는 것은 안다


def test_risk_reading_sharpens_with_national_capital(cfg, world):
    """정확도는 **국가 자본(기술력)**이 좌우한다. `national` 투자에 두 번째 쓸모다."""
    from core.agent_loop import Sink, execute_tool, risk_error
    world.turn = 10
    errs = []
    for nc in (0.0, 3000.0, 12000.0):
        world.countries["Asla"].national_capital = nc
        a = world.agents["Asla1"]; a.ap, a.budget = 1.0, 1000.0
        sink = Sink()
        r, _ = execute_tool("observe_risk", {"reasoning": "r"}, world, a, cfg, sink, 48.0)
        assert r["ok"]
        errs.append(r["typical_error"])
    assert errs[0] > errs[1] > errs[2], errs
    assert risk_error(world.countries["Asla"], cfg) == pytest.approx(errs[-1], abs=0.05)


def test_readings_are_normal_around_the_truth(cfg, world):
    """**정규분포다.** 꼬리에서 크게 빗나가도 된다 — 대체로 맞되 가끔 크게 틀리는 쪽이
    늘 일정 폭 안에서 틀리는 것보다 실제 계측에 가깝다.

    평균은 진실에 붙고 표준편차는 보고된 값과 맞아야 한다.
    """
    import statistics
    from core.agent_loop import Sink, execute_tool
    world.turn = 10
    world.countries["Asla"].national_capital = 3000.0
    a = world.agents["Asla1"]; a.budget = 1e9
    sink = Sink()
    seen = []
    for _ in range(400):
        a.ap = 1.0
        r, _ = execute_tool("observe_risk", {"reasoning": "r"}, world, a, cfg, sink, 48.0)
        seen.append(r["years_until_impact"])
    truth = cfg.world.total_turns - world.turn
    assert statistics.mean(seen) == pytest.approx(truth, abs=0.15 * r["typical_error"])
    assert statistics.pstdev(seen) == pytest.approx(r["typical_error"], rel=0.25)
    assert max(seen) - min(seen) > 3 * r["typical_error"], "꼬리가 없다"


def test_each_reading_is_fresh_but_costs(cfg, world):
    """매번 새로 잰다. 여러 번 재면 좁혀지지만 **공짜가 아니다** — 그 값이 곧
    국가 자본과 겨루는 가격이다."""
    from core.agent_loop import Sink, execute_tool
    world.turn = 10
    a = world.agents["Asla1"]; a.budget = 1000.0
    sink = Sink()
    seen = []
    for _ in range(5):
        a.ap = 1.0
        r, _ = execute_tool("observe_risk", {"reasoning": "r"}, world, a, cfg, sink, 48.0)
        seen.append(r["years_until_impact"])
    assert len(set(seen)) > 1, "매번 같으면 새 관측이 아니다"
    assert a.budget == 1000.0 - 5 * cfg.costs.observe_risk
    assert [o["nth"] for o in sink.observations] == [0, 1, 2, 3, 4]


def test_reading_is_private(cfg, world):
    """알아낸 값은 개인의 것이다. 남에게 알리려면 말해야 하고, 국제로 보내면 번역을 탄다."""
    from core.agent_loop import Sink, execute_tool
    world.turn = 10
    a = world.agents["Asla1"]; a.ap, a.budget = 1.0, 1000.0
    r, _ = execute_tool("observe_risk", {"reasoning": "r"}, world, a, cfg, Sink(), 48.0)
    assert set(r) == {"ok", "years_until_impact", "typical_error", "interceptor_needs",
                      "interceptor_typical_error_pct", "budget_left", "ap_left"}
    # "당신만의 것" 은 **도구 설명**에 있다 — 응답마다 되풀이할 규칙이 아니다
    from core.tools import TOOLS
    (t,) = [f for f in TOOLS if f["function"]["name"] == "observe_risk"]
    assert "yours alone" in t["function"]["description"]

def test_production_multiplier_is_gone(cfg, world):
    """배수는 안 알려준다 — 수입에서 추론 가능하다.

    수입은 **해가 열릴 때** 한 번 말한다 (관측에 두면 매 콜 다시 계산돼 값이 흔들렸다).
    """
    from domains.meteor import prompts
    world.countries["Asla"].national_capital = 3000.0
    a = world.agents["Asla1"]
    obs = prompts.render_observation(world, a, cfg, 48.0)
    assert "1.28" not in obs and "倍率: 1" not in obs
    # 관측에는 수입 **값**이 없다. `invest 효과` 의 「収入も…良くなる」 은 값이 아니라
    # national 이 무엇을 올리는지의 설명이라 남는다.
    assert "収入:" not in obs and "収入は" not in obs
    assert "income" not in prompts.T["ja"]        # 죽은 문구도 남기지 않는다
    # 배수가 반영된 값으로 **해 시작 문구에** 보인다. 옛 테스트는 `"+128" or "+1"` 이라
    # "+100" 에도 통과했다 — growth_coef 가 0.3 에서 0.2 로 바뀐 뒤에도 안 걸렸다.
    open_ = prompts.render_turn_open(world, a, cfg, 48.0, [])
    mult = world.countries["Asla"].multiplier(cfg)
    expect = f"+{cfg.income.per_turn * mult:.0f}"
    assert expect in open_ and expect != "+100"
    assert f"{mult:.2f}" not in open_             # 배수 자체는 없다


def test_ask_is_gone(cfg):
    """되묻기는 speak 로 충분하다. 도구를 하나 줄이면 실패 모드도 하나 준다."""
    from core import tools
    assert "ask" not in tools.TOOL_NAMES
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert "ask" not in names and "speak" in names


# ── 부지 독립 · invest 대상 (8/16 저녁) ───────────────────────────────────────

def test_sites_are_independent_is_stated(cfg, world):
    """**진척이 부지별로 따로 쌓인다는 것이 어디에도 없었다.**

    `interceptor_best = max(...)` 인데 에이전트는 알 길이 없었다. 20턴 실측에서 세
    나라가 각자 자기 땅을 팠고(860 · 1,159 · 1,267), 합치면 3,286 이라 임계 16,038 을
    향해 쌓이는 것처럼 보인다. **그 믿음 아래서는 자국에 붓는 것이 합리적이다.**

    이기심이 아니었다 — 국제 메시지 41건 중 24건이 요격기 협력 얘기였고 13건은 부지까지
    지목했다. 「너도 짓고 나도 짓고 같이 진척을 쌓자」 를 협력이라고 이해하고 있었다.
    """
    from domains.meteor import prompts
    marks = {"ja": "国ごとに別々", "zh": "按国家分别累积", "fr": "séparément pour chaque nation"}
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        a = world.agents[aid]
        assert marks[a.native_lang] in prompts.system_for(a, None, cfg), a.native_lang


def test_invest_names_a_nation_and_says_the_default(cfg, world):
    """`to` 가 159건 **전부** 생략됐다. 기본값이 있으니 생략이 안전하고, 생략하면 자국이다.

    생략하면 어디로 가는지 말해야 생략이 사고가 아니라 선택이 된다.
    """
    from core import tools
    from domains.meteor import prompts
    d = next(t["function"] for t in tools.TOOLS if t["function"]["name"] == "invest")
    assert "your own or another" in d["description"]
    assert "Defaults to your own nation" in d["parameters"]["properties"]["to"]["description"]

    # 기본값만이 아니라 **타국도 된다는 것**을 관측에 적어야 한다. 도구 설명(영어)에만
    # 있고 관측(모국어)에는 없었는데, 매 턴 읽는 것은 관측이다 — 실측에서 `to` 를 84번
    # 쓰면서 **전부 자국**을 넣었다.
    marks = {"ja": ("省くと自国", "他国でもよい"),
             "zh": ("不写则本国", "本国或别国都可以"),
             "fr": ("sans `to`, la vôtre", "la vôtre ou une autre")}
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        a = world.agents[aid]
        obs = prompts.render_observation(world, a, cfg, 48.0)
        for m in marks[a.native_lang]:
            assert m in obs, f"{a.native_lang}: {m}"


def test_the_rule_does_not_tell_them_what_to_do(cfg, world):
    """규칙만 말하고 전략은 말하지 않는다 (spec 4.1 ① 평가어 금지 · ② 목적함수 금지).

    「합치는 게 낫다」·「한 곳에 모아라」 같은 말이 들어가면 조율이 발견이 아니라
    지시 이행이 된다.
    """
    from domains.meteor import prompts
    banned = ["집중", "모아", "한 곳", "concentr", "should", "better", "集中", "最好", "推荐"]
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        t = prompts.system_for(world.agents[aid], None, cfg)
        for b in banned:
            assert b not in t, f"{aid}: {b}"


def test_investing_before_a_territory_is_settled_is_stated(cfg, world):
    """**국토가 정해지기 전에 부은 돈은 그냥 사라진다.** 20턴 실측에서 16% 가 증발했다
    (2,509원 / 15,278원). 투표가 t5~8 에 통과하는데 t1 부터 붓기 때문이다.

    일반 규칙만 말한다 — 어느 나라가 정했는지는 여전히 안 알려준다. 그건 10원짜리
    조회가 되고 "타국 사정은 소통해야만 안다" 는 전제가 무너진다.
    """
    from core import tools
    from domains.meteor import prompts
    marks = {"ja": "積むものがありません", "zh": "没有可积累的东西",
             "fr": "n'a rien où accumuler"}
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        a = world.agents[aid]
        assert marks[a.native_lang] in prompts.system_for(a, None, cfg), a.native_lang
    d = next(t["function"]["description"] for t in tools.TOOLS
             if t["function"]["name"] == "invest")
    # **「국토」 라는 말을 걷어냈다** (8/20). `land` 는 무엇을 짓는가인데 세 언어 모두
    # 国土·領土·territoire 로 옮겨 두어, 에이전트들이 지리로 읽고 없는 절차를 발명했다
    # (「先定国土，再推动表决」 · 「你们的领土是固定的吗？」).
    assert "yet decided what to build" in d and "buys no progress" in d
    for stale in ("territor", "国土", "领土"):
        assert stale not in d, stale


def test_the_rule_still_hides_which_nation_decided(cfg, world):
    """규칙은 말하되 **어느 나라가 정했는지는 여전히 감춘다.**"""
    from domains.meteor import prompts
    world.countries["Ranoa"].land = "interceptor"
    world.countries["Miris"].land = None
    obs = prompts.render_observation(world, world.agents["Asla1"], cfg, 48.0)
    assert "Ranoa" in obs and "Miris" in obs        # 명단에는 있다
    for tok in ("interceptor 未定", "Ranoa: interceptor", "Miris: 未定"):
        assert tok not in obs


# ── id 를 재사용하지 않는다 (8/16 밤) ────────────────────────────────────────

def test_ids_are_never_reused(cfg, world):
    """**「Asla1 이 죽었다」 는 부고 직후 명단에 Asla1 이 그대로 있으면 말이 안 된다.**

    죽은 자리에는 Asla4, Asla5 … 로 새 번호가 온다. 덤으로 id 가 곧 개체 식별자가 되어
    로그 조인이 깔끔해진다 (전에는 id 가 슬롯이라 uid 를 따로 봐야 했다).
    """
    import random
    for a in world.agents.values():
        a.age = 40                                   # 확실히 죽게
    r = loop.RunResult(world=world)
    loop._death_birth(world, cfg, random.Random(1), sorted(world.agents), set(),
                      itertools.count(700), r)
    dead = {d["who"] for d in r.deaths_log}
    assert dead and not (dead & set(world.agents)), "죽은 id 가 명단에 남아 있다"
    born = {b["id"] for b in r.births}
    assert born <= set(world.agents)
    assert all(b["replaces"] in dead for b in r.births)   # 누구 자리인지 남는다
    assert len(world.agents) == 9                          # 인구는 그대로


def test_roster_sorts_by_number_not_alphabetically(cfg, world):
    """사전순이면 Asla10 이 Asla2 앞에 온다. 번호가 두 자리로 넘어가므로 숫자순."""
    from domains.meteor import prompts
    from core.state import Agent
    for n in (10, 2):
        world.agents[f"Asla{n}"] = Agent(id=f"Asla{n}", country="Asla", native_lang="ja",
                                         known_langs={"ja"}, parent_langs=set(), budget=0)
    line = prompts.render_observation(world, world.agents["Asla1"], cfg, 48.0)
    row = next(l for l in line.splitlines() if "Asla1" in l and "Ranoa1" in l)
    assert row.index("Asla2") < row.index("Asla10")


def test_the_obituary_names_the_successor(cfg, world):
    """**누가 죽고 누가 그 자리에 왔는지를 한 쌍으로 알린다.**

    id 를 재사용하지 않으므로 이름만으로는 짝지을 수 없다 — 「Asla1 이 죽었다」 만
    보면 명단에 새로 생긴 Asla4 가 그 자리인지 알 수 없다.
    """
    import random
    from domains.meteor import prompts
    for a in world.agents.values():
        a.age = 40
    r = loop.RunResult(world=world)
    loop._death_birth(world, cfg, random.Random(1), sorted(world.agents), set(),
                      itertools.count(700), r)
    for d in r.deaths_log:
        assert d["born"] in world.agents and d["born"] != d["who"]

    d = next(x for x in r.deaths_log if x["country"] == "Asla")
    line = prompts.render_inbox([{"msg_id": 1, "died": d["who"], "born": d["born"]}], "ja")
    assert d["who"] in line and d["born"] in line


def test_the_delivery_rule_matches_the_loop_that_is_running(cfg, world):
    """**문구가 거짓이었다.** 순차 라운드로빈은 메시지를 **같은 해**에 배달하는데
    (`deliver_turn = world.turn`) 관측은 「翌年に届きます」 라고 적고 도구 설명은
    「a round trip takes two years」 라고 했다.

    에이전트가 그 거짓을 믿고 계획했다 — 실측 근거: 「メッセージ送付は翌年43年に届く」.
    같은 해에 답이 올 수 있다는 것은 **큰 차이**라, 모르면 한 해 안의 대화를 시도하지
    않는다. 이번 주에 「규칙을 고치고 말을 두었다」 를 다섯 번째로 겪은 자리다.

    도구 설명은 **두 경로에서 다 참인** 말로 바꿨다 — 「상대가 다음에 행동할 때 도착한다」.
    """
    from core import tools
    from domains.meteor import prompts
    d = next(t["function"]["description"] for t in tools.TOOLS
             if t["function"]["name"] == "speak")
    assert "next year" not in d and "two years" not in d
    assert "when that person next acts" in d
    assert "does not make it arrive sooner" in d

    a = world.agents["Asla1"]
    par = prompts.render_observation(world, a, cfg, 48.0)
    seq = prompts.render_observation(world, a, cfg, 48.0, same_year=True)
    assert "翌年に届きます" in par                   # 병렬은 다음 해가 맞다
    assert "翌年に届きます" not in seq
    assert "同じ年のうちに返事" in seq               # 순차는 같은 해에 올 수 있다

    marks = {"ja": "同じ年のうちに", "zh": "同一年内", "fr": "la même année"}
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        ag = world.agents[aid]
        txt = prompts.render_observation(world, ag, cfg, 48.0, same_year=True)
        assert marks[ag.native_lang] in txt, ag.native_lang


def test_one_speaker_per_nation_at_the_start(cfg):
    """나라마다 **한 명**이 이웃 나라 말을 이미 안다 (순환).

    그전에는 국내에 구사자가 아무도 없어 학습이 **늘 정가** 였고, 20턴 동안 학습 시도가
    **0건**이었다. `x̂` 는 할인 눈금이 존재해야 구간으로 좁혀진다 (spec 7장).
    """
    import random
    from core.agent_loop import learn_cost
    w = loop.init_world(cfg, itertools.count(1), random.Random(1))
    langs = {c.id: c.lang for c in w.countries.values()}

    for cid in w.countries:
        mine = [a for a in w.agents.values() if a.country == cid]
        bi = [a for a in mine if len(a.known_langs) > 1]
        assert len(bi) == 1, f"{cid}: {len(bi)}명"
        (extra,) = bi[0].known_langs - {langs[cid]}
        assert extra != langs[cid]

    # 어느 나라도 고립되지 않고, 어느 나라도 두 개를 갖지 않는다
    seeded = {next(iter(a.known_langs - {langs[a.country]}))
              for a in w.agents.values() if len(a.known_langs) > 1}
    assert seeded == set(langs.values()) - set()  # 세 언어가 각각 한 번씩
    assert len(seeded) == 3

    # 그 나라 사람들에게 그 언어 학습이 절반이 된다
    other = next(a for a in w.agents.values()
                 if a.country == "Asla" and len(a.known_langs) == 1)
    speaker = next(a for a in w.agents.values()
                   if a.country == "Asla" and len(a.known_langs) > 1)
    tgt_lang = next(iter(speaker.known_langs - {"ja"}))
    tgt = next(c.id for c in w.countries.values() if c.lang == tgt_lang)
    # **할인이 아니라 가속이다** (8/22). 필요액은 고정이고 회당 수확이 오른다.
    from core.agent_loop import learn_speed
    cost, why = learn_cost(other, tgt, w, cfg)
    mult, _ = learn_speed(other, tgt, w, cfg)
    assert cost == cfg.costs.learn_base
    assert mult == 1.0 + cfg.costs.learn_speedup and "nation" in why


def test_initial_ages_are_spread(cfg):
    """전원 0살이면 **한꺼번에 죽는다.** 20턴 실측에서 t14~19 에 9명이 전부 교체됐고
    그 6턴 사이에 쌓아둔 기억·관계·예산이 통째로 사라졌다."""
    import random
    w = loop.init_world(cfg, itertools.count(1), random.Random(1))
    # **처음 사람들은 성인으로 시작한다** (8/21). 소득을 성인부터로 바꾼 순간, 1~10 에서
    # 뽑으면 대부분이 빈손인데 **줄 부모도 없다** — 아이의 무소득은 「부모가 준다」 가
    # 있어서 성립하는 규칙이고, 세계 첫 해에는 그 부모가 없다.
    ages = [a.age for a in w.agents.values()]
    lo, hi = cfg.world.adult_age, cfg.world.adult_age + cfg.world.init_age_spread
    assert all(lo <= x <= hi for x in ages), ages
    # **둘 이상이면 된다.** 요점은 「전원이 같은 해에 죽지 않는다」 이고, 뽑는 폭이 4 이며
    # 개체 배수 추첨이 같은 rng 를 쓰므로 시드에 따라 2~4 가지가 나온다.
    assert len(set(ages)) >= 2, ages


def test_initialisation_is_reproducible(cfg):
    """같은 시드면 나이도 같아야 한다 — 안 그러면 재현성 검사가 통째로 깨진다."""
    import random
    a = loop.init_world(cfg, itertools.count(1), random.Random(7))
    b = loop.init_world(cfg, itertools.count(1), random.Random(7))
    assert [(k, v.age, sorted(v.known_langs)) for k, v in sorted(a.agents.items())] == \
           [(k, v.age, sorted(v.known_langs)) for k, v in sorted(b.agents.items())]


def test_learning_progress_is_visible_without_paying_to_look(cfg, world):
    """언어별 진척은 **별도 관측 없이** 그대로 보인다. 얼마 냈고 얼마 남았는지."""
    from domains.meteor import prompts
    # **필요액은 어느 말이든 같다** (8/22) — 다른 것은 회당 수확이다
    L = cfg.costs.learn_base
    a = world.agents["Asla2"]
    a.lang_progress = {"fr": 140.0, "zh": 120.0}
    obs = prompts.render_observation(world, a, cfg, 48.0)
    assert f"140 / {L:.0f}" in obs          # Miris(fr) — Asla 에 fr 구사자 없음
    assert f"120 / {L:.0f}" in obs          # Ranoa(zh) — Asla1 이 zh 를 안다 (더 빠르다)


def test_the_target_never_moves(cfg, world):
    """**목표가 움직이면 반쯤 낸 학습이 갑자기 완성된다** — 그 경로를 없앴다 (8/22).

    전에는 국내 구사자가 생기는 순간 필요액이 200 → 150 으로 내려가, 이미 150 을 낸 사람이
    **그 자리에서** 말을 하게 됐다. 완료 판정을 「그 순간의 학습가」 로 한 결과였고, 그
    주변에서 버그를 한 번 잡았다.

    이제 필요액은 고정이고 **회당 수확**이 오른다. 구사자가 생기면 앞으로가 빨라질 뿐,
    이미 낸 것이 갑자기 충분해지지는 않는다.
    """
    import random

    from core.agent_loop import learn_cost, learn_speed
    L, up = cfg.costs.learn_base, cfg.costs.learn_speedup
    a = world.agents["Asla2"]
    a.lang_progress = {"fr": L - cfg.costs.unit}      # 한 번 남았다
    r = loop.RunResult(world=world)
    loop._settle_agentic(world, cfg, random.Random(0), Sink(), None, 48.0,
                         itertools.count(500), r, itertools.count(900))
    assert "fr" not in a.known_langs                  # 아직 모자라다

    world.agents["Asla3"].known_langs.add("fr")       # 국내 구사자 등장
    assert learn_cost(a, "Miris", world, cfg)[0] == L     # **목표는 그대로**
    assert learn_speed(a, "Miris", world, cfg)[0] == 1.0 + up   # 속도만 오른다
    loop._settle_agentic(world, cfg, random.Random(0), Sink(), None, 48.0,
                         itertools.count(500), r, itertools.count(900))
    assert "fr" not in a.known_langs                  # 그래도 완성되지 않는다


def test_the_obituary_says_how_old_they_were(cfg, world):
    """**몇 살에 죽었는지 알린다.**

    수명 곡선은 은닉 목록이지만(4.1) 부고에 찍힌 나이는 사실이고, 그것이 쌓이면
    인구가 경험으로 배웁니다. 자기 수명을 모르면 `procreate`(죽고 물려주기)를 고를
    시점을 알 수 없습니다 — 세 런 21명이 전부 자연사했고 procreate 는 **0건**이었습니다.
    """
    import random
    from domains.meteor import prompts
    for a in world.agents.values():
        a.age = 22
    r = loop.RunResult(world=world)
    loop._death_birth(world, cfg, random.Random(1), sorted(world.agents), set(),
                      itertools.count(700), r)
    assert r.deaths_log and all(d["age"] == 22 for d in r.deaths_log)

    d = r.deaths_log[0]
    for lang, mark in (("ja", "22 歳"), ("zh", "22 岁"), ("fr", "22 ans")):
        line = prompts.render_inbox(
            [{"msg_id": 1, "died": d["who"], "born": d["born"], "age": d["age"]}], lang)
        assert mark in line, lang


def test_obituary_survives_a_missing_age(cfg):
    """구버전 런에는 나이가 없다. 렌더가 터지면 그 런을 아예 못 본다."""
    from domains.meteor import prompts
    line = prompts.render_inbox([{"msg_id": 1, "died": "Asla1", "born": "Asla4"}], "ja")
    assert "Asla1" in line and "Asla4" in line


def test_national_investment_states_all_three_effects(cfg, world):
    """`national` 은 **셋을 동시에** 올린다. 하나만 적으면 나머지 둘이 안 보인다.

        수입          income × multiplier           (loop.py 소득 지급)
        시설 전환율    eff × multiplier              (그 나라 시설에 낸 돈의 효율)
        관측 정확도    σ ∝ 1/√(national_capital)     (observe_risk)

    실측에서 근거 12건이 전부 *수입* 쪽만 말했고 관측 정확도는 1건뿐이었다 —
    두 번째 쓸모를 붙여 놓고 알려주지 않았다.
    """
    from core import tools
    from domains.meteor import prompts
    d = next(t["function"]["description"] for t in tools.TOOLS
             if t["function"]["name"] == "invest")
    assert "technical level" in d and "income" in d and "observe_risk" in d

    marks = {"ja": ("技術力", "収入", "observe_risk"),
             "zh": ("技术水平", "收入", "observe_risk"),
             "fr": ("niveau technique", "revenu", "observe_risk")}
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        a = world.agents[aid]
        obs = prompts.render_observation(world, a, cfg, 48.0)
        for m in marks[a.native_lang]:
            assert m in obs, f"{a.native_lang}: {m}"


def test_cost_table_says_which_nations_are_guaranteed(cfg, world):
    """**나라별로 보장 여부를 적는다.** 규칙만 적었을 때 연결을 못 했다.

    20턴 실측에서 자기가 아는 말의 나라에 24원짜리 `ai` 를 6번 썼다 — `original` 5원이면
    **확실히** 닿는데도. Ranoa1(fr·zh)이 Miris 에 그랬다.

    자기 언어 능력에서 나오는 사실이라 타국 사정을 흘리지 않는다.
    """
    from domains.meteor import prompts
    a = world.agents["Ranoa1"]                 # 초기화로 fr 을 안다
    assert "fr" in a.known_langs
    obs = prompts.render_observation(world, a, cfg, 48.0)
    sure = prompts.T[a.native_lang]["c_orig_sure"].format(nation="Miris")
    risk = prompts.T[a.native_lang]["c_orig_risk"].format(nation="Asla")
    assert sure in obs and risk in obs

    mono = world.agents["Ranoa2"]              # 자기 말만 안다
    obs2 = prompts.render_observation(world, mono, cfg, 48.0)
    assert prompts.T[mono.native_lang]["c_orig_sure"].format(nation="Miris") not in obs2
    for cid in ("Asla", "Miris"):
        assert prompts.T[mono.native_lang]["c_orig_risk"].format(nation=cid) in obs2


def test_the_guarantee_line_leaks_nothing_about_others(cfg, world):
    """보장 여부는 **내 언어 능력**만으로 정해진다 — 상대가 무엇을 읽는지와 무관하다."""
    from domains.meteor import prompts
    a = world.agents["Ranoa1"]
    before = prompts.render_observation(world, a, cfg, 48.0)
    for other in world.agents.values():        # 타국 사람들이 언어를 더 배워도
        if other.country != a.country:
            other.known_langs.add("zh")
    assert prompts.render_observation(world, a, cfg, 48.0) == before


# ── 투자의 AP 비용 · 학습 진척 상속 (8/17) ───────────────────────────────────

def test_one_investment_costs_one_fixed_unit(cfg, world):
    """**금액을 인자로 받지 않는다** (8/20). 한 번에 정해진 돈과 정해진 AP 만 나간다.

    금액이 자유였을 때는 요청·절삭·과금이 서로 달라서, 응답이 그 차이를 알려야 했고
    (`_clamped`) 표에는 「額÷300」 이라는 비율이 필요했다. 그리고 표의 학습 줄은
    `600 · 額÷300` 이라 **총액과 비율이 한 줄에 섞여** 읽혔다. 고정하면 `20 · 0.1` 뿐이다.
    """
    from core.agent_loop import Sink, execute_tool
    world.countries["Asla"].land = "interceptor"
    a = world.agents["Asla1"]; a.ap, a.budget = 1.0, 10_000.0
    sink = Sink()
    r, _ = execute_tool("invest", {"target": "facility", "reasoning": "r"},
                        world, a, cfg, sink, 48.0)
    assert r["ok"] and a.ap == 1.0 - cfg.ap.unit
    assert a.budget == 10_000.0 - cfg.costs.unit
    assert sink.facility == [("Asla", cfg.costs.unit, "Asla1")]
    assert "charged" not in r          # 절삭이 없으니 알릴 차이가 없다


def test_the_year_holds_a_fixed_number_of_investments(cfg, world):
    """한 해에 몇 번 낼 수 있나 — `turn.action_points / ap.unit`. **횟수를 여기 적지
    않는다** (8/22 에 0.1 → 0.2 로 바뀌며 열 번이 다섯 번이 됐다)."""
    a = world.agents["Ranoa1"]
    a.ap, a.budget = cfg.turn.action_points, 10_000.0
    n = int(cfg.turn.action_points / cfg.ap.unit)
    for i in range(n):
        assert _do(world, cfg, a, "invest", {"target": "wellness"})["ok"], i
    assert a.ap == 0.0
    assert not _do(world, cfg, a, "invest", {"target": "wellness"})["ok"]


def test_every_target_costs_the_same_unit(cfg, world):
    """세 대상이 같은 값이다. **규칙이 하나면 문구가 갈리지 않는다** — wellness 만
    정액이던 때 같은 화면에 「無料」 와 「0.1 定額」 이 함께 있었다."""
    from core.agent_loop import Sink, execute_tool
    for target in ("wellness", "national", "facility"):
        w = loop.init_world(cfg, itertools.count(1)); w.turn = 5
        w.countries["Asla"].land = "interceptor"
        a = w.agents["Asla1"]; a.ap, a.budget = 1.0, 10_000.0
        # **액수는 사람마다 다르다** (8/22) — 대상 셋이 같은 값이라는 것이 요점이므로
        # 그 사람의 배수로 잰다.
        unit = cfg.costs.unit * a.invest_mult
        r, _ = execute_tool("invest", {"target": target, "reasoning": "r"},
                            w, a, cfg, Sink(), 48.0)
        assert r["ok"] and a.ap == round(1.0 - cfg.ap.unit, 3)
        assert a.budget == 10_000.0 - unit


def test_the_action_rate_does_not_grow_with_wealth(cfg, world):
    """**생산배수를 곱하지 않는다** (8/20). 곱했을 때 두 가지가 나빴다.

    ① **표에 숫자가 둘 나왔다.** invest 는 `額÷318`, learn 은 `額÷300` — 배수를 학습에는
       걸 수 없어서(걸면 국가 투자가 학습률을 올려 지표 1 이 노브 아닌 이유로 움직인다)
       「하나로 통일」 이 말뿐이었다. 100해 자본이면 530 대 300 으로 77% 벌어진다.

    ② **후반에 상한이 영영 안 걸렸다.** 수입도 배수로 커지므로 `수입 ÷ 상한` 이 0.33 에
       고정된다 — 1해든 100해든 같다. 부유해질수록 쏟아붓기 속도도 같이 커지면
       「모아둔 걸 쏟아붓기를 막는다」 는 역할이 흐려진다.

    초반은 그대로고(자본 0 이면 배수 1.0) 후반에만 압박이 생긴다 — 수입 177 대 상한 300.
    """
    from core.agent_loop import Sink, execute_tool, risk_sigma
    world.countries["Asla"].land = "interceptor"
    for cap in (0.0, 27_000.0):
        world.countries["Asla"].national_capital = cap
        a = world.agents["Asla1"]; a.ap, a.budget = 1.0, 10_000.0
        r, _ = execute_tool("invest", {"target": "facility", "reasoning": "r"},
                            world, a, cfg, Sink(), 48.0)
        assert r["ok"] and a.ap == 1.0 - cfg.ap.unit          # 부유해도 같은 값
        assert a.budget == 10_000.0 - cfg.costs.unit
    # 국가 투자에는 여전히 세 가지 쓸모가 남는다
    assert world.countries["Asla"].multiplier(cfg) > 1.0      # 수입·시설 전환율
    assert risk_sigma(world.countries["Asla"], cfg) < cfg.risk.sigma_ratio   # 관측 정확도


def test_natural_death_passes_nothing(cfg, world):
    """자연사는 계보와 무관한 뒷세대다 (3.2). 진척도 안 넘어간다."""
    import random
    a = world.agents["Asla1"]
    a.lang_progress = {"fr": 400.0}
    a.age = 40
    r = loop.RunResult(world=world)
    loop._death_birth(world, cfg, random.Random(1), ["Asla1"], set(),
                      itertools.count(700), r)
    child = next(x for x in world.agents.values() if x.country == "Asla" and x.age == 0)
    assert child.lang_progress == {} and child.parent_langs == set()


# ── 로그에 본문이 남는가 (8/18) ─────────────────────────────────────────────

def test_memory_and_testament_survive_the_log(cfg, world, tmp_path):
    """**무엇을 적어뒀고 무엇을 남기고 죽었는가는 로그에서 잘리면 안 된다.**

    `text`·`testament` 를 종류와 무관하게 잘라내고 있었다. 근거는 *"본문은 messages 에
    있으므로"* 였는데 그건 `speak` 에만 맞는 말이었다 — `memory_write` 의 본문과
    `procreate` 의 유언은 messages 에도 events 에도 없어서 **통째로 사라졌다.**
    세 런 60건의 memory_write 가 전부 `{"type": "memory_write"}` 로만 남아 있었다.
    """
    from core.run_io import _redact
    assert _redact({"type": "memory_write", "text": "요격기에 몰아줘라",
                    "reasoning": "r"}) == {"type": "memory_write", "text": "요격기에 몰아줘라"}
    assert _redact({"type": "procreate", "testament": "AI 를 믿지 마라",
                    "reasoning": "r"}) == {"type": "procreate", "testament": "AI 를 믿지 마라"}
    # speak 의 본문은 messages.jsonl 에 원문·도착문이 함께 있으므로 계속 뺀다
    assert _redact({"type": "speak", "to": "Ranoa2", "text": "x"}) == {"type": "speak",
                                                                      "to": "Ranoa2"}


def test_foreign_gain_amount_is_hidden(cfg, world):
    """**액수를 주면 상대국 생산배수가 새어 나온다.**

        E[gain] / amount = facility.eff × success_prob × multiplier(받는 나라)

    상수가 모든 나라에 같으므로 **두 나라를 비교하면 상수가 지워지고 배수 비율만
    남는다.** 실측 런에서 통지를 쌓아 배수 1.13 을 복원했다 (실제 1.13~1.15).

    자국 배수보다 나쁜 누출이다 — 자국은 수입에서 추론하는 정당한 경로가 있어서
    관측에서 배수 자체를 뺐지만(prompt_audit), 타국은 4.1 이 *"소통으로만"* 이라고
    못 박았다. 10원짜리 조회로 읽히면 안 된다.
    """
    world.countries["Ranoa"].land = "interceptor"
    world.countries["Ranoa"].national_capital = 27_000.0     # 배수가 확 벌어진 나라
    sink = Sink()
    sink.facility = [("Ranoa", 200.0, "Asla1"), ("Ranoa", 200.0, "Ranoa2")]
    _settle(world, cfg, sink)
    msgs = {e["to"]: e["msg"] for e in world.inbox_queue
            if "fac_gain" in e["msg"] or "fac_moved" in e["msg"]}
    # 외국인: 숫자가 한 글자도 없다
    assert msgs["Asla1"] == {"msg_id": msgs["Asla1"]["msg_id"], "amount": 200.0,
                             "to": "Ranoa", "fac_moved": True}
    # 자국민: 그대로 — 자국 진척 델타로 어차피 보이는 값이다
    assert isinstance(msgs["Ranoa2"].get("fac_gain"), int)


def test_the_notice_still_tells_whether_anything_moved(cfg, world):
    """여부는 살린다. 두 가지가 그것에 걸려 있다 —
    ① 없으면 남의 땅에 내는 선택이 영영 깜깜이가 된다.
    ② 늘지 않았다는 것이 곧 *"그 나라가 아직 국토를 안 정했다"* 다. 통지 자체가
       없으면 그 부재가 같은 말을 하므로, 통지는 어느 쪽이든 똑같이 가야 한다."""
    for land, moved in (("interceptor", True), (None, False)):
        w = loop.init_world(cfg, itertools.count(1)); w.turn = 5
        w.countries["Ranoa"].land = land
        sink = Sink(); sink.facility = [("Ranoa", 300.0, "Asla1")]
        _settle(w, cfg, sink)
        (e,) = [x for x in w.inbox_queue if "fac_moved" in x["msg"]]
        assert e["msg"]["fac_moved"] is moved


def test_the_prompt_never_prints_a_foreign_gain_number(cfg, world):
    """문구까지 확인한다 — 규칙을 고쳐도 렌더러가 숫자를 찍으면 그대로 새어 나간다."""
    from domains.meteor import prompts
    for lang in ("ja", "zh", "fr"):
        for moved in (True, False):
            out = prompts.render_inbox(
                [{"msg_id": 3, "from": None, "text": None, "label": None,
                  "original": None, "amount": 200.0, "to": "Ranoa", "fac_moved": moved}], lang)
            assert "200" in out                      # 내가 낸 액수는 내가 안다
            assert "Ranoa" in out
            # 진척 숫자가 될 수 있는 다른 수가 없다
            nums = [n for n in re.findall(r"\d+", out) if n not in ("3", "200")]
            assert not nums, (lang, moved, out)


# ── 내가 그 나라에 낸 누적 (8/18) ───────────────────────────────────────────

def test_facility_investment_returns_my_running_total(cfg, world):
    """**`learn` 은 누적을 돌려주는데 `facility` 는 안 돌려주고 있었다.**

    실측에서 13턴에 885원을 한 나라에 나눠 낸 에이전트가 자기가 얼마 냈는지를
    **메모로만** 알았다 — `memory_write` 로 덮이면 그마저 사라진다.
    """
    from core.agent_loop import Sink, execute_tool
    world.countries["Ranoa"].land = "interceptor"
    a = world.agents["Asla1"]; a.budget = 10_000.0
    sink = Sink()
    for n in (1, 2, 3):
        a.ap = 1.0
        r, _ = execute_tool("invest", {"target": "facility",
                                       "to": "Ranoa", "reasoning": "r"},
                            world, a, cfg, sink, 48.0)
        assert r["your_total_into"] == n * cfg.costs.unit
    assert a.facility_invested == {"Ranoa": 3 * cfg.costs.unit}


def test_the_running_total_is_per_nation_and_leaks_nothing_about_them(cfg, world):
    """나라별로 따로 센다. 그리고 이것은 **내 행동의 합**이라 상대 국가 정보가 아니다 —
    그 나라의 총 진척이나 이번 턴 그 나라에 모인 총액은 여전히 안 알려준다."""
    from core.agent_loop import Sink, execute_tool
    for c in ("Ranoa", "Miris"):
        world.countries[c].land = "interceptor"
    world.countries["Ranoa"].progress = 5555.0        # 알려주면 안 되는 값
    a = world.agents["Asla1"]; a.budget = 10_000.0
    sink = Sink()
    outs = []
    for c in ("Ranoa", "Miris", "Ranoa"):
        a.ap = 1.0
        r, _ = execute_tool("invest", {"target": "facility",
                                       "to": c, "reasoning": "r"}, world, a, cfg, sink, 48.0)
        outs.append(r)
    u = cfg.costs.unit
    assert a.facility_invested == {"Ranoa": 2 * u, "Miris": u}
    assert "5555" not in json.dumps(outs)             # 타국 진척은 안 새어 나온다
    assert not any("progress" in o for o in outs)


def test_the_observation_shows_only_my_own_contributions(cfg, world):
    """관측에도 적는다 — `learn` 진척이 그러는 것과 같다. 낸 적 없는 나라는 안 나온다."""
    from domains.meteor import prompts
    a = world.agents["Asla1"]
    a.facility_invested = {"Ranoa": 210.0}
    world.countries["Miris"].progress = 7777.0
    obs = prompts.render_observation(world, a, cfg, 48.0)
    assert "210" in obs and "Ranoa" in obs
    assert "7777" not in obs                          # 타국 진척은 여전히 없다


def test_the_child_inherits_only_the_discount(cfg, world):
    """**물려주는 것은 부모 할인 자격뿐이다** (8/21 개정).

    전에는 예산·유언·학습 진척 절반이 함께 넘어갔다. 부모가 죽었으니 넘길 수밖에 없었다.
    이제 부모가 살아 있으므로 —

      · 돈은 **살아서 주면 된다** (아이는 빈손으로 시작하고 성인까지 소득이 없다)
      · 유언은 **말로 하면 된다** (`speak` — 여러 해에 걸쳐, 기존 채널로 관측된다)
      · 학습 진척 절반은 **과하다** — 부모가 살아 있는 것이 이미 국내 구사자이므로
        아이의 학습이 두 겹으로 싸다 (부모 −50 × 국내 −50)

    내 생애의 기록(어디에 얼마를 냈나)은 물려주지 않는다.
    """
    import itertools

    from core.loop import RunResult, _bear_child
    a = world.agents["Asla1"]
    a.facility_invested = {"Ranoa": 900.0}
    a.lang_progress = {"fr": 400.0}
    a.budget = 777.0
    r = RunResult(world=world)
    cid = _bear_child(world, "Asla1", cfg, itertools.count(99), r)

    child = world.agents[cid]
    assert child.facility_invested == {}
    assert child.lang_progress == {}                   # 진척은 안 넘어간다
    assert child.budget == 0.0                         # 빈손
    assert child.parent_langs == a.known_langs         # 할인 자격만
    assert child.memory == ""

    # **부모는 살아 있다.** 그것이 이 개정의 전부다.
    assert "Asla1" in world.agents and world.agents["Asla1"] is a
    assert a.budget == 777.0                           # 돈도 안 뺏긴다
    assert r.deaths == 0 and not r.deaths_log

    # 그리고 부모가 국내 구사자이므로 아이의 학습이 두 겹으로 싸다
    from core.agent_loop import learn_cost, learn_speed
    tgt = next(c.id for c in world.countries.values()
               if c.lang in a.known_langs and c.id != a.country)
    cost, why = learn_cost(child, tgt, world, cfg)
    mult, _ = learn_speed(child, tgt, world, cfg)
    assert cost == cfg.costs.learn_base                       # 목표는 고정
    assert mult == 1.0 + 2 * cfg.costs.learn_speedup          # 두 배속
    assert "parent" in why and "nation" in why


def test_a_birth_is_announced_to_the_whole_world(cfg, world):
    """명단은 GLOBAL 이므로 새 사람이 나타난 것은 어차피 보인다. **누구의 아이인가**만
    새로 새는 것이고, 그건 세대 간 전달을 관측하려면 필요하다 (3.3)."""
    import itertools

    from core.loop import RunResult, _bear_child
    cid = _bear_child(world, "Asla1", cfg, itertools.count(900),
                      RunResult(world=world))
    got = {e["to"] for e in world.inbox_queue}
    alive = {a.id for a in world.agents.values() if a.alive and a.id != cid}
    assert got == alive, "전 세계가 안다 (갓 태어난 본인만 빼고)"
    msg = world.inbox_queue[0]["msg"]
    assert msg == {"born": cid, "parent": "Asla1"}

    from domains.meteor import prompts
    txt = prompts.render_events(world.agents["Ranoa1"], [msg])
    assert cid in txt and "Asla1" in txt


def test_the_observation_never_describes_the_old_ballot(cfg, world):
    """**규칙을 고치고 문구를 두면 그대로 남는다.** 소집이 내용을 갖지 않게 된 뒤에도
    관측이 「제안」·「찬반」 을 말하고 있으면 에이전트는 옛 세계를 산다.

    그리고 `propose_vote` 행에는 오래도록 설명이 **비어 있었다** — 다른 행은 다 있는데
    성격이 바뀐 바로 그 도구만 없었다.
    """
    from domains.meteor import prompts
    STALE = ("賛否", "赞成", "反対", "提案なし", "没有提案", "Aucune proposition",
             "prononcer")
    world.turn = 10
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        obs = prompts.render_observation(world, world.agents[aid], cfg, 48.0)
        for w in STALE:
            assert w not in obs, (aid, w)
        # propose_vote 행에 설명이 있고, 그 설명이 「고르지 않는다」 를 말한다
        (line,) = [l for l in obs.splitlines() if l.strip().startswith("propose_vote")]
        assert len(line.split()) > 3, line
    # 採決 당일에는 세 선택지가 이름 그대로 보인다
    world.countries["Asla"].proposal = {"by": "Asla2", "opened_turn": 6, "vote_turn": 10}
    obs = prompts.render_observation(world, world.agents["Asla1"], cfg, 48.0)
    for w in ("interceptor", "bunker", "abstain"):
        assert w in obs


def test_learn_reports_completion_not_a_schedule(cfg, world):
    """**`can_read_next_turn` 은 순차 라운드로빈에서 거짓이 됐다.**

    `_settle_step` 이 학습을 차례마다 반영하므로, 다 낸 순간부터 그 턴에 바로 쓸 수
    있다. "다음 턴부터" 라고 말하면 막 배운 말을 그 턴에 안 쓰게 만든다 — 하필 학습이
    살아나기를 바라는 지점이다. 병렬 경로는 여전히 턴 끝이므로, **언제부터인지는
    응답이 말하지 않고** 관측의 「읽을 수 있는 언어」 가 답한다.
    """
    from core.agent_loop import Sink, execute_tool
    a = world.agents["Asla2"]; a.ap, a.budget = 1.0, 10_000.0
    sink = Sink()
    # 국내 구사자가 있어 할인가이고, 5 만 남았다
    # 남은 것이 회당 수확보다 작으면 마지막 한 번이 그만큼만 걷는다
    a.lang_progress = {"zh": cfg.costs.learn_base - 5.0}
    a.parent_langs = set()                # 배율을 정가로 고정해 계산을 단순하게
    r, _ = execute_tool("learn", {"country": "Ranoa", "reasoning": "r"},
                        world, a, cfg, sink, 48.0)
    assert r["complete"] is True and r["remaining"] == 0.0
    assert not any("turn" in k for k in r)        # 일정을 말하는 필드가 없다

    # 그리고 순차 정산은 실제로 같은 턴에 반영한다
    res = loop.RunResult(world=world)
    loop._settle_step(world, cfg, random.Random(0), sink, None, 48.0,
                      itertools.count(900), res, {}, [], [])
    assert "zh" in world.agents["Asla2"].known_langs


def test_my_resources_are_not_in_the_observation(cfg, world):
    """**예산·남은 행동력은 관측에 없다.**

    그 둘은 「세계가 어떤가」 가 아니라 **내 행동의 결과**다. 결과는 도구 채널이 말한다 —
    성공 응답마다 `budget_left`·`ap_left` 가 오고, 실패 응답도 얼마가 필요하고 얼마가
    있는지 말한다. 해가 열릴 때의 값은 시작 문구가 적는다.

    관측에 두면 **관측이 매 콜 흔들리는 숫자를 담게 된다.** 오늘 그 부류로 세 번 물렸다 —
    소득 드리프트(+100→+104→+105) · wellness 정액 모순 · 해 중간 재렌더.
    """
    from core.agent_loop import Sink, execute_tool
    from domains.meteor import prompts
    a = world.agents["Asla1"]
    a.ap, a.budget = 0.9, 80.0
    obs = prompts.render_observation(world, a, cfg, 48.0)
    assert "80" not in obs and "0.90" not in obs
    for gone in ("budget", "ap_now"):
        assert gone not in prompts.T["ja"]        # 죽은 문구도 남기지 않는다

    # 해가 열릴 때는 적는다
    assert "80" in prompts.render_turn_open(world, a, cfg, 48.0, [])

    # 그리고 도구가 매번 돌려준다 — 성공도, 실패도
    r, _ = execute_tool("speak", {"to": "Asla2", "text": "x", "reasoning": "r"},
                        world, a, cfg, Sink(), 48.0)
    assert r["ok"] and "budget_left" in r and "ap_left" in r
    a.ap = 0.01
    r, _ = execute_tool("speak", {"to": "Asla2", "text": "x", "reasoning": "r"},
                        world, a, cfg, Sink(), 48.0)
    assert not r["ok"] and "have 0.01" in r["error"]      # 남은 값을 알려준다


def test_the_note_names_the_right_reason_and_the_yield(cfg, world):
    """사유가 둘로 갈린다 — 국내 구사자냐 부모냐. 그리고 **회당 수확**을 적는다 (8/22).

    문구가 「割引あり」 처럼 뭉개져 있던 동안에는 어느 쪽인지 알 수 없었다. 그리고 「반값」
    은 무엇의 반인지 총액을 되짚어야 알았다 — 이제 「한 번에 얼마가 쌓이나」 를 적는다.

    「先輩」 같은 말은 쓸 수 없다 — **실측에서 국내 구사자가 배우는 사람보다 어린 경우가
    13%**(805짝 중 108건)다. 나이 관계는 이 세계에 없다.
    """
    from core.agent_loop import learn_speed
    from domains.meteor import prompts
    t = prompts.T["ja"]
    u, up = cfg.costs.unit, cfg.costs.learn_speedup
    a = world.agents["Asla2"]                       # Asla1 이 zh 를 안다 (씨앗)

    a.parent_langs = {"fr"}
    obs = prompts.system_for(a, world, cfg, 48.0)
    zh = next(l for l in obs.splitlines() if "Ranoa の言語を学ぶ" in l)
    fr = next(l for l in obs.splitlines() if "Miris の言語を学ぶ" in l)
    assert t["c_cheap"].format(gain=u * (1 + up)).strip() in zh     # 국내 구사자
    assert t["c_disc"].format(gain=u * (1 + up)).strip() in fr      # 부모
    assert "自国に話せる人" not in fr                                # 섞이지 않는다

    a.parent_langs = {"zh"}                          # 둘 다 걸리면 두 배속
    lines = prompts.system_for(a, world, cfg, 48.0).splitlines()
    i = next(n for n, l in enumerate(lines) if "Ranoa の言語を学ぶ" in l)
    assert learn_speed(a, "Ranoa", world, cfg)[0] == 1 + 2 * up
    assert t["c_both"].format(gain=u * (1 + 2 * up)).strip() in lines[i]
    # **목표는 진척 줄에 있고 움직이지 않는다**
    assert f"0 / {cfg.costs.learn_base:.0f}" in lines[i + 1]


def test_no_prose_hardcodes_the_grace_period(cfg, world):
    """**숫자를 두 군데 적으면 하나가 낡는다.**

    `VOTE_DELAY` 를 4 에서 2 로 줄였을 때 도구 설명은 *"three years pass … in the fourth
    year"* 로 남아 있었다. 이번 주에 그 부류를 다섯 번 겪었다 — `can_read_next_turn` ·
    採決 문구 · wellness 무료 · 「기술력이 비율을 올린다」 · 「메시지는 다음 해에 도착」.

    유예 길이는 **관측만** 말한다 (「採決은 44년」). 도구 설명은 모양만 말한다.
    """
    from core import tools
    d = next(t["function"]["description"] for t in tools.TOOLS
             if t["function"]["name"] == "propose_vote")
    for stale in ("three years", "fourth year", "two years", "third"):
        assert stale not in d, stale
    assert "which year that is" in d          # 대신 관측이 알려준다고 적는다

    # 관측이 실제 날짜를 말한다
    from domains.meteor import prompts
    world.turn = 10
    _call(world, cfg, "Ranoa1")
    obs = prompts.system_for(world.agents["Ranoa2"], world, cfg, 48.0)
    from domains.meteor.prompts import FIRST_YEAR
    assert str(FIRST_YEAR + 10 + loop.VOTE_DELAY - 1) in obs


def test_the_grace_is_one_year_now(cfg, world):
    """**3해는 메시지 왕복에 두 해가 들던 때 정해진 길이다.** 순차 라운드로빈은 같은 해에
    왕복이 되므로, 소집한 해의 대화 + 한 해면 충분하다.

    기대수명이 16해인데 다섯 해를 절차에 쓰면 한 사람이 겪는 採決이 세 번뿐이다.
    """
    assert loop.VOTE_DELAY == 2
    world.turn = 10
    _call(world, cfg, "Ranoa1")
    p = world.countries["Ranoa"].proposal
    assert p["opened_turn"] == 10 and p["vote_turn"] == 12   # 소집 · 유예 11 · 採決 12


def test_exactly_affordable_is_affordable(cfg, world):
    """**딱 낼 수 있는 사람이 거절당하고 있었다.**

    행동력에서 0.3·0.3·0.1 을 빼면 2진 부동소수에서 0.3 이 아니라 0.29999999999999993 이
    나온다. `ap < 0.3` 이 참이 되어 발화가 막혔고, 오류 문구는 `.2f` 로 반올림해
    「0.3 이 필요한데 0.30 을 갖고 있다」 는 말을 했다.

    3해 실측에서 **25건** — 투자 20 · 발화 5. 에이전트는 그 뒤 대개 end_turn 을 불렀다.
    """
    a = world.agents["Ranoa1"]
    a.ap, a.budget = cfg.turn.action_points, 1000.0
    # **상수에서 짠다** (8/22 에 speak 0.3→0.2, unit 0.1→0.2 로 바뀌었다). AP 를 딱
    # `speak` 한 번 남기고, 그 한 번이 통하는지 본다.
    left = cfg.ap.speak
    while round(a.ap - cfg.ap.unit, 3) >= left:
        assert _do(world, cfg, a, "invest", {"target": "wellness"})["ok"]
    assert a.ap == left                      # 0.19999… 이 아니라 정확히 그 값

    r = _do(world, cfg, a, "speak", {"to": "Ranoa2", "text": "y"})
    assert r["ok"], r                        # 딱 맞으면 낼 수 있다
    assert a.ap == 0.0


def test_ap_stays_on_the_grid_over_a_full_year(cfg, world):
    """단위가 0.05 이므로 소수 세 자리 격자에 계속 붙어 있어야 한다. 비교와 차감이 같은
    격자를 쓰는 한 오차가 누적되지 않는다."""
    a = world.agents["Ranoa1"]
    a.ap, a.budget = cfg.turn.action_points, 10_000.0
    n = int(cfg.turn.action_points / cfg.ap.unit)
    for _ in range(n):
        assert _do(world, cfg, a, "invest", {"target": "wellness"})["ok"]
        assert a.ap == round(a.ap, 3)
    assert a.ap == 0.0                       # n 번 빼면 정확히 0
    r = _do(world, cfg, a, "invest", {"target": "wellness"})
    assert not r["ok"] and "have 0.00" in r["error"]


def test_a_language_you_already_read_is_not_on_the_learning_table(cfg, world):
    """**같은 화면에서 두 줄이 서로를 부정했다.**

    맨 위는 「掌握している言語: Miris の言葉」 라고 적고, 비용표는 「Miris の言葉を学ぶ
    … 0 / 600」 이라고 적었다. 3해 실측에서 `learn` 이 14번 거절당했고
    (`you already read Ranoa's language`), 한 에이전트는 메모에

        学习Miris语已投入 0/600（但已掌握？笔记需更新：已掌握Miris语）

    라고 적어 **스스로 모순을 기록했다.** 아는 말은 배울 표에서 뺀다.
    """
    from domains.meteor import prompts
    a = world.agents["Ranoa1"]
    a.known_langs = {world.countries["Ranoa"].lang, world.countries["Miris"].lang}

    sysmsg = prompts.system_for(a, world, cfg, 48.0)
    lines = sysmsg.splitlines()
    # **총액을 여기 적지 않는다** — 이 테스트를 쓴 다음 날 L 이 600 에서 200 으로 갔다.
    prog = [i for i, ln in enumerate(lines) if " / " in ln and "0" in ln
            and any(c.isdigit() for c in ln.split(" / ")[0])]
    assert prog, sysmsg                          # 모르는 말(Asla)은 여전히 표에 있다
    for i in prog:
        near = " ".join(lines[max(0, i - 1):i + 1])
        assert "Miris" not in near, near
        assert "Asla" in near, near

    # **말을 걸 때는 여전히 나온다** — 아는 말이라 반드시 닿는다는 사실은 남아야 한다
    assert "Miris" in sysmsg


def test_one_person_gets_one_vote(cfg, world):
    """**한 사람이 두 번 던졌고 두 표가 다 집계됐다.**

    3해 실측에서 Ranoa1 이 같은 採決에 두 번 `vote` 를 불렀고, 세 사람 나라에서
    interceptor **3표**가 나왔다 — 실제로 던진 사람은 둘이고 Ranoa2 는 던지지 않았다.
    국토를 정하는 자리에서 한 사람이 나라의 용도를 두 번 밀 수 있었던 것이다.

    두 겹으로 막는다. 도구가 두 번째를 거절하고, 집계가 사람마다 한 표만 센다 — 순차
    라운드로빈은 한 해에 같은 사람을 두 번 방문할 수 있다 (메일로 깨우는 경로).
    """
    a = world.agents["Ranoa1"]
    _call(world, cfg, "Ranoa1")
    # **採決일을 제안에서 읽는다** — 숫자를 여기 적으면 VOTE_DELAY 를 고칠 때 낡는다
    world.turn = world.countries["Ranoa"].proposal["vote_turn"]
    a.ap = 1.0

    r = _do(world, cfg, a, "vote", {"choice": "interceptor"})
    assert r["ok"] and a.voted_turn == world.turn
    r = _do(world, cfg, a, "vote", {"choice": "bunker"})
    assert not r["ok"] and "already voted" in r["error"]

    # 그리고 집계는 새는 경로가 있어도 한 표만 센다
    assert loop._one_vote_each(
        [("Ranoa1", "interceptor"), ("Ranoa1", "interceptor"),
         ("Ranoa3", "bunker")]) == [("Ranoa1", "interceptor"), ("Ranoa3", "bunker")]


def test_the_ballot_day_says_only_that_it_is_today(cfg, world):
    """**예정과 「오늘이다」 를 겹쳐 내보내고 있었다.**

        表决将在 44 年举行（由 Ranoa3 召集）。建什么在那时决定
        ★ 今年就是表决之年。可以用 vote 选 …

    유예를 한 해로 줄이면서 이 겹침이 제안 수명의 3분의 1 이 됐다. 그날은 「오늘이다」
    한 줄만 내보내고, 소집자는 그 줄이 데려간다.
    """
    from domains.meteor import prompts
    _call(world, cfg, "Ranoa1")
    p = world.countries["Ranoa"].proposal

    def prop_lines(w):
        obs = prompts.system_for(world.agents["Ranoa2"], world, cfg, 48.0)
        # 비용표에도 「表决」 이 있다 — 제안 블록만 본다
        return [l for l in obs.splitlines() if "表决将在" in l or "★" in l]

    year = str(prompts.FIRST_YEAR + p["vote_turn"] - 1)
    world.turn = p["opened_turn"]                    # 소집한 해 — 예정만
    (line,) = prop_lines(world)
    assert year in line and "★" not in line

    world.turn = p["vote_turn"]                      # 採決일 — 「오늘」 만
    (line,) = prop_lines(world)
    assert "★" in line and "Ranoa1" in line
    assert year not in line                          # 예정 줄이 겹치지 않는다


# ── 주기 (8/22) ───────────────────────────────────────────────────────────────

def _exec(name, args, world, agent, cfg, sink, knob=48.0):
    from core import agent_loop
    return agent_loop.execute_tool(name, args, world, agent, cfg, sink, knob)


def test_giving_moves_money_and_the_receiver_is_told(cfg, world):
    """**잉여의 용처가 없었다.** 10해 실측에서 성인의 턴 끝 예산이 74 → 198 → 435 로
    쌓이는데 남은 AP 중앙은 0.0 이었다 — 돈은 남고 쓸 행동력이 없다. 그리고 사람에게 돈을
    주는 행동이 아예 없었다.

    그래서 `bear_child` 가 순수 비용이었다. 1.0 AP 를 내고, 열 해 동안 아무것도 못 하고,
    **도울 수도 없는** 사람을 얻는 거래다. 줄 수 있게 되면 「내 잉여를 쓸 손」 이 된다.

    받는 이에게 알린다 — 예산은 PRIVATE 이고, 갑자기 늘어난 이유를 본인이 모르면 그 돈을
    쓸 판단을 못 한다.
    """
    import random

    giver, taker = world.agents["Ranoa1"], world.agents["Ranoa2"]
    giver.ap, giver.budget = cfg.turn.action_points, 500.0
    taker.budget = 0.0

    sink = Sink()
    r, _ = _exec("give", {"to": "Ranoa2", "amount": 300},
                                   world, giver, cfg, sink, 48.0)
    assert r["ok"] and giver.budget == 200.0
    assert giver.ap == cfg.turn.action_points - cfg.ap.give
    # **받는 쪽은 아직 안 늘었다** — 남의 상태라 정산에서 넣는다 (병렬 안전)
    assert taker.budget == 0.0 and sink.gifts == [("Ranoa1", "Ranoa2", 300.0)]

    loop._settle_agentic(world, cfg, random.Random(0), sink, None, 48.0,
                         itertools.count(500), loop.RunResult(world=world),
                         itertools.count(900))
    assert taker.budget == 300.0
    got = [e for e in world.inbox_queue if e["to"] == "Ranoa2"]
    assert len(got) == 1 and got[0]["msg"] == {"gift_from": "Ranoa1", "gift": 300.0}
    assert not [e for e in world.inbox_queue if e["to"] != "Ranoa2"]   # 받는 이만

    from domains.meteor import prompts
    txt = prompts.render_events(taker, [got[0]["msg"]])
    assert "Ranoa1" in txt and "300" in txt


def test_giving_is_refused_when_it_cannot_be_honoured(cfg, world):
    """**넘치게 주지 않는다.** 잘라서 주면 받는 쪽이 얼마를 받았는지 되짚어야 한다."""
    a = world.agents["Ranoa1"]
    a.ap, a.budget = cfg.turn.action_points, 100.0
    for args, mark in ((  {"amount": 10},                    "`to`"),
                       ({"to": "Nobody1", "amount": 10},     "unknown recipient"),
                       ({"to": "Ranoa1", "amount": 10},      "yourself"),
                       ({"to": "Ranoa2", "amount": 0},       "more than 0"),
                       ({"to": "Ranoa2", "amount": "많이"},   "must be a number"),
                       ({"to": "Ranoa2", "amount": 101},     "not enough budget")):
        r, _ = _exec("give", args, world, a, cfg, Sink(), 48.0)
        assert not r["ok"] and mark in r["error"], (args, r)
    assert a.budget == 100.0 and a.ap == cfg.turn.action_points   # 아무것도 안 나갔다


def test_the_size_of_a_gift_does_not_change_the_effort(cfg, world):
    """**금액을 인자로 받는 유일한 도구다.**

    `invest`·`learn` 에서 금액을 뺀 이유는 비용표가 `600 · 額÷300` 처럼 두 숫자를 읽게
    만들었기 때문이다. 주는 것은 다르다 — 크기가 드는 수고를 바꾸지 않는다. 한 번에 40 씩만
    옮길 수 있으면 435 를 넘기는 데 열한 해가 걸리고, 그러면 **잉여의 용처**라는 이 도구의
    존재 이유가 사라진다.
    """
    a = world.agents["Ranoa1"]
    a.ap, a.budget = cfg.turn.action_points, 1000.0
    for amt in (1, 999):
        before = a.ap
        r, _ = _exec("give", {"to": "Ranoa2", "amount": amt},
                     world, a, cfg, Sink(), 48.0)
        # **격자에서 비교한다.** `before - a.ap` 를 날로 빼면 1.0 − 0.8 이
        # 0.19999999999999996 이다 — 코드가 막고 있는 그 부동소수를 테스트가 다시 만든다.
        assert r["ok"] and a.ap == round(before - cfg.ap.give, 3)
        a.budget = 1000.0


def test_income_grows_with_age_so_the_end_of_life_cannot_spend_it(cfg, world):
    """**말년에 소비가 못 따라간다** (8/22).

    10해 실측에서 아이를 낳은 사람이 **0명**이었다. 면담이 이유를 말했다 —
    *"생애 한 번뿐이니 좀 더 나이 들어 상황이 안정된 뒤에도 늦지 않다"*. 「한 번뿐」 이
    미루기를 최적으로 만든 것이다.

    미루기를 벌하는 대신 **낳을 이유**를 만든다. 소득이 나이와 함께 오르고, 한 해에 쓸 수
    있는 돈은 행동력이 묶으므로(invest 40원·0.2AP → 상한 200) 말년에는 잉여가 **강제로**
    쌓인다. 그 잉여의 용처가 `give` 이고, 줄 사람을 만드는 것이 `bear_child` 다.
    """
    a = world.agents["Ranoa1"]
    g, adult = cfg.income.age_growth, cfg.world.adult_age
    assert g > 0
    base = cfg.income.per_turn * world.countries["Ranoa"].multiplier(cfg)

    a.age = adult
    assert loop.income_for(a, world, cfg) == pytest.approx(base)
    a.age = adult + 6
    assert loop.income_for(a, world, cfg) == pytest.approx(base * (1 + 6 * g))

    # **한 해에 쓸 수 있는 돈보다 많아지는 나이가 있다** — 거기부터 잉여가 강제된다
    ceiling = (cfg.turn.action_points / cfg.ap.unit) * cfg.costs.unit
    over = next(x for x in range(adult, 60)
                if base * (1 + g * (x - adult)) > ceiling)
    assert adult < over < 25, over          # 기대수명(16) 근처여야 뜻이 있다

    # 미성년은 어리다고 더 받지 않는다 (배수는 성인 나이부터)
    a.age = 0
    assert loop.income_for(a, world, cfg) == pytest.approx(base)


def test_the_rule_that_income_grows_is_stated(cfg, world):
    """**적지 않으면 계획할 수 없다.** 나이가 들면 더 번다는 것은 세계의 사실이고,
    그것을 알아야 「지금 쓸까 나중에 쓸까」 를 저울질할 수 있다."""
    from domains.meteor import prompts
    marks = {"ja": "収入は増えます", "zh": "收入越多", "fr": "revenu augmente"}
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        ag = world.agents[aid]
        assert marks[ag.native_lang] in prompts.system_for(ag, world, cfg, 48.0), aid


# ── 개체 차이 (8/22) ─────────────────────────────────────────────────────────

def test_people_differ_in_what_they_earn_and_what_they_can_move(cfg):
    """**전원이 동일해서 조율할 것이 없었다.**

    지금까지는 「무엇을 지을까」 하나뿐이었다. 소득과 처리량이 사람마다 다르면 **비교우위**
    가 생긴다 — 「고소득·저처리」 는 줘야 하고 「저소득·고처리」 는 받아야 한다. 그리고 그
    조합을 알아내려면 **말을 해야 한다.**

    두 축은 **독립**이다. 묶으면 두 부류만 나오는데, 독립이면 스물다섯 조합이 생기고 세계가
    그만큼 덜 깔끔하다.
    """
    import random
    w = loop.init_world(cfg, itertools.count(1), random.Random(1))
    inc = {a.id: a.income_mult for a in w.agents.values()}
    thr = {a.id: a.invest_mult for a in w.agents.values()}
    assert len(set(inc.values())) > 1, inc      # 실제로 갈린다
    assert len(set(thr.values())) > 1, thr
    assert set(inc.values()) <= set(cfg.income.spread)
    assert set(thr.values()) <= set(cfg.facility.throughput_spread)
    # **독립이다** — 같은 값끼리 붙어 다니지 않는다
    assert any(inc[k] != thr[k] for k in inc)


def test_the_spreads_average_to_one_or_the_window_breaks(cfg):
    """**평균이 1 이어야 한다.** 임계값 창이 `per_turn × n × total` 에서 나오므로, 평균이
    1 이 아니면 창이 어긋나고 방금 나이 배수로 한 재계산을 또 해야 한다."""
    for sp in (cfg.income.spread, cfg.facility.throughput_spread):
        assert sum(sp) / len(sp) == pytest.approx(1.0), sp
        assert len(sp) >= 3                    # 눈금이 있어야 말로 전할 수 있다


def test_my_multipliers_show_but_nobody_elses_do(cfg):
    """**남의 값이 보이면 소통이 필요 없어진다.**

    내 액수는 비용표에 적힌다 (내 자원이다). 남의 값은 어디에도 없다 — 같은 나라 사람의
    것도 마찬가지다. 그래야 국내 조율도 대화를 요구한다.
    """
    import random

    from domains.meteor import prompts
    w = loop.init_world(cfg, itertools.count(1), random.Random(1))
    w.turn = 1
    me = w.agents["Asla1"]
    obs = prompts.system_for(me, w, cfg, 48.0)
    mine = cfg.costs.unit * me.invest_mult
    inv_row = next(l for l in obs.splitlines() if l.startswith("  invest "))
    assert f"{mine:g}" in inv_row

    # 남의 액수·소득이 다른 값이면 관측 어디에도 없다
    for other in w.agents.values():
        if other.id == me.id:
            continue
        theirs = cfg.costs.unit * other.invest_mult
        if theirs != mine:
            assert f"{theirs:g}" not in inv_row, other.id
        assert str(round(loop.income_for(other, w, cfg))) not in obs or \
            other.income_mult == me.income_mult


def test_a_child_does_not_inherit_the_multipliers(cfg):
    """부모와 아이의 배수는 **독립**이다. 물려받으면 한 계보가 누적 우위를 갖고, spec 3.3
    의 「능력은 상속되지 않는다」 와도 어긋난다."""
    import random
    rng = random.Random(7)
    w = loop.init_world(cfg, itertools.count(1), rng)
    parent = w.agents["Asla1"]
    parent.age = cfg.world.adult_age
    parent.income_mult, parent.invest_mult = 1.4, 1.4

    kids = []
    for _ in range(12):
        parent.has_borne = False
        cid = loop._bear_child(w, "Asla1", cfg, itertools.count(900),
                               loop.RunResult(world=w), rng)
        kids.append(w.agents[cid])
    # 열두 명이 전부 부모와 같을 확률은 사실상 0 이다
    assert any(k.income_mult != 1.4 or k.invest_mult != 1.4 for k in kids)
