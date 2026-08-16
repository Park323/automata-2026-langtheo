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


def test_same_turn_duplicate_proposals_are_marked(cfg, world):
    """같은 턴에 둘이 제안하면 **둘 다 도구를 통과한다** — 행동 시점엔 둘 다
    proposal=None 을 본다. 실제로 열리는 것은 하나뿐이므로 어느 쪽이 열렸는지
    기록한다. 안 하면 로그만 보고 유령 제안을 진짜로 읽게 된다.
    """
    world.turn = 10
    sink = Sink()
    sink.votes = [("Ranoa1", "Ranoa", "bunker"), ("Ranoa3", "Ranoa", "interceptor")]
    r = _settle(world, cfg, sink)
    props = [v for v in r.votes_log if v["kind"] == "propose"]
    assert len(props) == 2
    assert [p["opened"] for p in props] == [True, False]        # id 순으로 앞선 것만
    assert world.countries["Ranoa"].proposal["target"] == "bunker"


def test_only_one_proposal_at_a_time(cfg, world):
    """제안이 열려 있으면 새 제안을 못 연다 — 안 그러면 유예가 무의미해진다."""
    from core.agent_loop import execute_tool
    world.turn = 10
    _propose(world, cfg, "Ranoa1", "bunker")
    a = world.agents["Ranoa2"]; a.ap, a.budget = 1.0, 100.0
    res, _ = execute_tool("propose_vote", {"target": "interceptor", "reasoning": "r"},
                          world, a, cfg, Sink(), 48.0)
    assert not res["ok"] and "already has an open proposal" in res["error"]


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
        res, _ = execute_tool("invest", {"target": "facility", "amount": 10,
                                         "to": to, "reasoning": "r"},
                              world, a, cfg, Sink(), 48.0)
        outs.append(res)
    # 나라 이름과 잔액 말고는 한 글자도 달라선 안 된다 — 다르면 그것이 곧 조회다
    shape = [{k: (v if k not in ("accepted", "budget_left") else None)
              for k, v in o.items()} for o in outs]
    assert shape[0] == shape[1]
    assert outs[0]["accepted"].replace("Ranoa", "") == outs[1]["accepted"].replace("Miris", "")
    assert all(o["ok"] for o in outs)
    assert a.budget == 480.0                           # 둘 다 과금됐다


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
    # 출자자는 진척이 얼마나 늘었는지만 알 뿐, **무엇이 깊어졌는지는 모른다**
    (e,) = [x for x in world.inbox_queue if "fac_gain" in x["msg"]]
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
    (e,) = [x for x in world.inbox_queue if "fac_gain" in x["msg"]]
    assert e["to"] == "Asla1" and e["msg"]["fac_gain"] == 0


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
    child = world.agents["Asla4"]
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


def test_threshold_is_no_longer_free(cfg, world):
    """임계는 **관측으로만** 알 수 있다 — `observe_risk` 를 사서 재야 한다.

    공짜로 보이면 진척과 임계가 둘 다 정확해져 산수로 풀리는 문제가 된다.
    이제 둘 다 기술력에 따라 흐릿하고, 알아낸 값은 **개인의 것**이라 남에게 알리려면
    말해야 한다 — 국제로 보내면 번역을 타고, 그게 지표 6a 가 재는 경로다.
    """
    from domains.meteor import prompts
    obs = prompts.render_observation(world, world.agents["Asla1"], cfg, 48.0)
    for hidden in (str(int(cfg.thresholds.interceptor)), "0.3", "success_prob",
                   str(int(cfg.thresholds.bunker_scale))):
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
        seen.append(r["turns_until_impact"])
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
        seen.append(r["turns_until_impact"])
    assert len(set(seen)) > 1, "매번 같으면 새 관측이 아니다"
    assert a.budget == 1000.0 - 5 * cfg.costs.observe_risk
    assert [o["nth"] for o in sink.observations] == [0, 1, 2, 3, 4]


def test_reading_is_private(cfg, world):
    """알아낸 값은 개인의 것이다. 남에게 알리려면 말해야 하고, 국제로 보내면 번역을 탄다."""
    from core.agent_loop import Sink, execute_tool
    world.turn = 10
    a = world.agents["Asla1"]; a.ap, a.budget = 1.0, 1000.0
    r, _ = execute_tool("observe_risk", {"reasoning": "r"}, world, a, cfg, Sink(), 48.0)
    assert "nobody else" in r["note"]

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
        assert marks[a.native_lang] in prompts.system_for(a), a.native_lang


def test_invest_names_a_nation_and_says_the_default(cfg, world):
    """`to` 가 159건 **전부** 생략됐다. 기본값이 있으니 생략이 안전하고, 생략하면 자국이다.

    생략하면 어디로 가는지 말해야 생략이 사고가 아니라 선택이 된다.
    """
    from core import tools
    from domains.meteor import prompts
    d = next(t["function"] for t in tools.TOOLS if t["function"]["name"] == "invest")
    assert "your own or another" in d["description"]
    assert "Defaults to your own nation" in d["parameters"]["properties"]["to"]["description"]

    marks = {"ja": "省くと自国", "zh": "不写则本国", "fr": "sans `to`, la vôtre"}
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        a = world.agents[aid]
        obs = prompts.render_observation(world, a, cfg, 48.0)
        assert marks[a.native_lang] in obs, a.native_lang


def test_the_rule_does_not_tell_them_what_to_do(cfg, world):
    """규칙만 말하고 전략은 말하지 않는다 (spec 4.1 ① 평가어 금지 · ② 목적함수 금지).

    「합치는 게 낫다」·「한 곳에 모아라」 같은 말이 들어가면 조율이 발견이 아니라
    지시 이행이 된다.
    """
    from domains.meteor import prompts
    banned = ["집중", "모아", "한 곳", "concentr", "should", "better", "集中", "最好", "推荐"]
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        t = prompts.system_for(world.agents[aid])
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
        assert marks[a.native_lang] in prompts.system_for(a), a.native_lang
    d = next(t["function"]["description"] for t in tools.TOOLS
             if t["function"]["name"] == "invest")
    assert "has not settled its territory" in d and "buys no progress" in d


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


def test_procreate_also_announces_the_pair(cfg, world):
    """스스로 죽는 것도 같은 나라 사람에게는 같은 사건이다."""
    r = loop.RunResult(world=world)
    loop._procreate_child(world, "Asla1", "유언", cfg, itertools.count(900), r)
    (d,) = r.deaths_log
    assert (d["who"], d["by"]) == ("Asla1", "procreate")
    assert d["born"] == "Asla4" and d["born"] in world.agents


def test_round_trip_takes_two_turns_is_stated(cfg, world):
    """**도착만 알려주고 답신까지 한 턴 더라는 건 안 알려줬다.**

    그래서 같은 말을 반복해서 보내는 일이 잦았다 — 답이 안 오니 안 갔다고 여긴 것이다.
    """
    from core import tools
    from domains.meteor import prompts
    d = next(t["function"]["description"] for t in tools.TOOLS
             if t["function"]["name"] == "speak")
    assert "round trip takes two turns" in d
    assert "does not make it arrive sooner" in d

    marks = {"ja": "返事が来るのはさらに次のターン", "zh": "回信要再下一回合",
             "fr": "une réponse n'arrive qu'au tour d'après"}
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        a = world.agents[aid]
        assert marks[a.native_lang] in prompts.render_observation(world, a, cfg, 48.0)


# ── 초기화 (8/17) ────────────────────────────────────────────────────────────

def test_one_speaker_per_nation_at_the_start(cfg):
    """나라마다 **한 명**이 이웃 나라 말을 이미 안다 (순환).

    그전에는 국내에 구사자가 아무도 없어 학습이 **늘 정가 600** 이었고, 20턴 동안
    학습 시도가 **0건**이었다. `x̂` 는 L/2 눈금이 존재해야 구간으로 좁혀진다 (spec 7장).
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
    cost, why = learn_cost(other, tgt, w, cfg)
    assert cost == cfg.costs.learn_base / 2 and "nation" in why


def test_initial_ages_are_spread(cfg):
    """전원 0살이면 **한꺼번에 죽는다.** 20턴 실측에서 t14~19 에 9명이 전부 교체됐고
    그 6턴 사이에 쌓아둔 기억·관계·예산이 통째로 사라졌다."""
    import random
    w = loop.init_world(cfg, itertools.count(1), random.Random(1))
    ages = [a.age for a in w.agents.values()]
    assert all(1 <= x <= cfg.world.init_age_max for x in ages)
    assert len(set(ages)) >= 4, ages


def test_initialisation_is_reproducible(cfg):
    """같은 시드면 나이도 같아야 한다 — 안 그러면 재현성 검사가 통째로 깨진다."""
    import random
    a = loop.init_world(cfg, itertools.count(1), random.Random(7))
    b = loop.init_world(cfg, itertools.count(1), random.Random(7))
    assert [(k, v.age, sorted(v.known_langs)) for k, v in sorted(a.agents.items())] == \
           [(k, v.age, sorted(v.known_langs)) for k, v in sorted(b.agents.items())]
