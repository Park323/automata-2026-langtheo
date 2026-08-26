"""산출물 기록. spec 9장.

파생 로그는 전부 raw_calls.jsonl 에서 재생성할 수 있어야 한다 — 정의는 나중에 바뀌고,
원본이 없으면 그때 다시 돌려야 한다.
"""
from __future__ import annotations

import json
import random

import pytest

from core import config, loop, run_io
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
def cfg():
    return config.load("configs/base.yaml")


def _run_with_writer(cfg, tmp_path, turns=2):
    object.__setattr__(cfg.world, "total_turns", turns)
    w = run_io.RunWriter("test_run", cfg_raw={"world": {"total_turns": turns}},
                         root=tmp_path)
    ids = [f"{c}{i}" for c in ("Asla", "Ranoa", "Miris") for i in (1, 2, 3)]
    end = assistant_msg(tool_call("end_turn", "e", reasoning="이래서"))
    script = {"Asla1": [assistant_msg(tool_call("speak", "1", to="Ranoa1",
                                                route="ai", text="隕石")), end]}
    clients = {a: StubClient(list(script.get(a, [])), recorder=w.recorder(kind="agent", agent=a))
               for a in ids}
    tr = StubClient([{"role": "assistant", "content": "陨石", "tool_calls": []}] * 40,
                    recorder=w.recorder(kind="translate"))
    res = loop.run_agentic(cfg, random.Random(1), _client_for(clients, assistant_msg(tool_call("end_turn", "z", reasoning="r"))), tr, KNOB,
                           prompts.render_turn_open, prompts.system_for,
                           parallel=False, on_turn_end=w.on_turn_end)
    w.close({"final": res.final, "deaths": res.deaths})
    return w, res


def _lines(w, name):
    p = w.dir / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def test_all_artifacts_written(cfg, tmp_path):
    w, _ = _run_with_writer(cfg, tmp_path)
    for f in ("raw_calls.jsonl", "state.jsonl", "messages.jsonl",
              "events.jsonl", "metrics.jsonl", "summary.json", "config_snapshot.yaml"):
        assert (w.dir / f).exists(), f"{f} 가 없다"


def test_raw_keeps_request_and_response(cfg, tmp_path):
    """요청·응답을 가공 없이. 이게 있어야 파생 로그를 재생성할 수 있다."""
    w, _ = _run_with_writer(cfg, tmp_path)
    raw = _lines(w, "raw_calls")
    assert raw, "raw_calls 가 비었다"
    r = raw[0]
    for k in ("run_id", "kind", "attempt", "latency_ms", "request", "response"):
        assert k in r, f"raw 에 {k} 가 없다"
    assert r["request"]["messages"], "요청 messages 가 비었다"
    assert {x["kind"] for x in raw} >= {"agent"}


def test_config_snapshot_has_commit(cfg, tmp_path):
    """재현에는 설정만으로 부족하다 — 코드 커밋도 남긴다."""
    import yaml
    w, _ = _run_with_writer(cfg, tmp_path)
    snap = yaml.safe_load((w.dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert "code_commit" in snap and "config" in snap


def test_state_has_fields_for_x_hat(cfg, tmp_path):
    """x̂ 를 나이로 층화하려면 age 가 있어야 한다 (spec 8.2)."""
    w, _ = _run_with_writer(cfg, tmp_path)
    st = _lines(w, "state")
    assert st
    for k in ("turn", "agent", "age", "lambda", "known_langs", "parent_langs", "born_by"):
        assert k in st[0], f"state 에 {k} 가 없다"


def test_metrics_have_failure_rate(cfg, tmp_path):
    """실패율이 조건 간에 다르면 교란 신호 — 집계 가능해야 한다 (지표 16)."""
    w, _ = _run_with_writer(cfg, tmp_path)
    m = _lines(w, "metrics")
    assert m and "llm_failure_rate" in m[0] and "ended_by" in m[0]


def test_appended_per_turn(cfg, tmp_path):
    """턴마다 append — 45턴에서 죽어도 거기까지는 남아야 한다."""
    w, _ = _run_with_writer(cfg, tmp_path, turns=3)
    turns = [m["turn"] for m in _lines(w, "metrics")]
    assert turns == [1, 2, 3]


def test_messages_written_once(cfg, tmp_path):
    """메시지가 턴마다 중복 기록되지 않는다."""
    w, _ = _run_with_writer(cfg, tmp_path, turns=3)
    ids = [m["msg_id"] for m in _lines(w, "messages")]
    assert len(ids) == len(set(ids))


def test_refuses_to_append_to_existing_run(cfg, tmp_path):
    """같은 run_id 로 다시 돌리면 두 런이 한 파일에 섞여 지표가 조용히 오염된다."""
    import pytest as _pytest
    w = run_io.RunWriter("dup", root=tmp_path)
    w.raw({"kind": "agent", "request": {}, "response": {}})
    w.close()
    with _pytest.raises(FileExistsError):
        run_io.RunWriter("dup", root=tmp_path)
    run_io.RunWriter("dup", root=tmp_path, overwrite=True)      # 명시하면 허용


def test_every_result_log_reaches_disk(tmp_path):
    """**RunResult 에만 있고 파일에 안 남는 로그가 있으면 안 된다.**

    투표·국토 전환·부고·진척 기여가 전부 그랬다 — 오늘 만든 규칙이 통째로 관측
    불가였다. metrics 의 land 로 전환은 역산되지만 찬반 수·소실 진척은 복구되지 않는다.
    """
    from core import loop, run_io

    class _W:
        agents = {}
        countries = {}

    r = loop.RunResult(world=_W())
    r.votes_log = [{"turn": 1, "kind": "propose", "by": "Ranoa1",
                    "country": "Ranoa", "target": "bunker", "vote_turn": 5}]
    r.land_changes = [{"turn": 1, "country": "Ranoa", "target": "bunker",
                       "yes": 2, "no": 1, "passed": True, "progress_lost": 295.0}]
    r.deaths_log = [{"turn": 1, "who": "Asla2", "country": "Asla", "by": "natural"}]
    r.facility_gains = [{"turn": 1, "agent": "Asla1", "to": "Ranoa",
                         "amount": 90.0, "gain": 27}]
    r.agent_logs = [{}]

    w = run_io.RunWriter("t", cfg_raw={"a": 1}, root=tmp_path)
    w.on_turn_end(1, r)
    w.close()

    kinds = [json.loads(l)["type"]
             for l in (tmp_path / "t" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert set(kinds) == {"vote", "land_change", "death", "facility_gain"}
    # 행 안의 키가 이벤트 type 을 덮어쓰면 안 된다 (votes_log 의 kind=propose/ballot)
    rows = [json.loads(l) for l in
            (tmp_path / "t" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    vote = next(r for r in rows if r["type"] == "vote")
    assert vote["kind"] == "propose" and vote["vote_turn"] == 5


def test_logs_are_not_written_twice(tmp_path):
    """턴마다 append 하므로 같은 행이 두 번 나가면 집계가 배로 뛴다."""
    from core import loop, run_io

    class _W:
        agents = {}
        countries = {}

    r = loop.RunResult(world=_W())
    r.deaths_log = [{"turn": 1, "who": "Asla2", "country": "Asla", "by": "natural"}]
    r.agent_logs = [{}, {}]
    w = run_io.RunWriter("t", cfg_raw={"a": 1}, root=tmp_path)
    w.on_turn_end(1, r)
    w.on_turn_end(2, r)          # 같은 기록이 남아 있어도 다시 안 쓴다
    w.close()
    lines = (tmp_path / "t" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum(1 for l in lines if json.loads(l)["type"] == "death") == 1


def test_agent_turn_records_its_own_wall_time():
    """**한 사람이 한 턴을 사는 데 걸린 시간**을 남긴다.

    llm_ms 를 따로 재는 이유 — 벽시계의 거의 전부가 API 대기여야 정상이고, 둘이
    갈리면 우리 코드가 병목이라는 뜻이다. 3턴 실측에서 32,523 / 32,526 ms 였다.
    """
    import itertools
    from core import config, loop
    from core.agent_loop import Sink, run_agent_turn
    from core.llm import StubClient, assistant_msg, tool_call
    from domains.meteor import prompts

    cfg = config.load("configs/base.yaml")
    w = loop.init_world(cfg, itertools.count(1))
    a = w.agents["Asla1"]; a.ap = 1.0
    lg = run_agent_turn(
        w, a, cfg, StubClient([assistant_msg(tool_call("end_turn", "1"))]),
        Sink(), KNOB, prompts.system_for(a, None, cfg), prompts.render_observation(w, a, cfg, KNOB))
    for k in ("elapsed_ms", "llm_ms", "ms_per_step"):
        assert lg[k] is not None and lg[k] >= 0, k
    assert lg["llm_ms"] <= lg["elapsed_ms"] + 1


def test_wellness_spend_accumulates_over_a_life():
    """개인 누적 wellness 출자. **본인에게는 여전히 비공개**이고 로그에만 남는다."""
    import itertools
    import random
    from core import config, loop
    from core.agent_loop import Sink

    cfg = config.load("configs/base.yaml")
    w = loop.init_world(cfg, itertools.count(1))
    r = loop.RunResult(world=w)
    for _ in range(3):
        sink = Sink(); sink.wellness = [("Asla1", 40.0)]
        loop._settle_agentic(w, cfg, random.Random(0), sink, None, KNOB,
                             itertools.count(500), r, itertools.count(900))
    assert w.agents["Asla1"].wellness_spent == 120.0

    obs = prompts.render_observation(w, w.agents["Asla1"], cfg, KNOB)
    assert "120" not in obs                      # 관측에는 안 나온다


def test_mid_turn_snapshots_are_marked_and_skipped_by_metrics(tmp_path, cfg):
    """**해 도중에도 상태를 남긴다** (8/26 · Eddie).

    순차 라운드로빈은 차례마다 `_settle_step` 이 즉시 정산하므로, 한 차례가 끝난 시점의
    세계는 **완전히 일관된다** — 「아직 안 움직인 사람이 있는 해」 이고 그것이 그 순간의
    진실이다. (병렬은 턴 끝에 한꺼번에 정산하므로 중간이 거짓이고, 그래서 훅이 없다.)

    **그 줄이 지표에 섞이면 한 해가 여러 번 세어진다.** 그래서 `step` 으로 갈라 두고,
    소비자 넷(`interview`·`score/metrics`·`score/xhat`·뷰어)이 `step is None` 만 읽는다.
    이 테스트는 그 규약을 양쪽에서 못 박는다 — 표시가 붙는지, 그리고 읽는 쪽이 거르는지.
    """
    import itertools
    import json
    import random

    from core import loop, run_io

    w = run_io.RunWriter("t_step", cfg_raw={}, knob_ai=None, seed=1, root=tmp_path)
    try:
        world = loop.init_world(cfg, itertools.count(1), random.Random(0))
        res = loop.RunResult(world=world)
        w.on_step_end(3, 10, res)          # 해 도중
        w.on_step_end(3, 20, res)
        w.on_turn_end(3, res)              # 해 끝
    finally:
        w.close()

    rows = [json.loads(l) for l in
            (tmp_path / "t_step" / "state.jsonl").read_text(encoding="utf-8").splitlines()]
    n = len(world.agents)
    assert len(rows) == 3 * n
    mid = [r for r in rows if r.get("step") is not None]
    end = [r for r in rows if r.get("step") is None]
    assert len(mid) == 2 * n and len(end) == n
    assert sorted({r["step"] for r in mid}) == [10, 20]
    # 중간 줄도 **완전한 상태**다 — 턴 끝 줄과 같은 필드를 갖는다
    assert set(mid[0]) == set(end[0])

    # **읽는 쪽이 실제로 거른다** — 소비자 넷의 코드를 구조로 본다
    for path, needle in (
            ("tools/interview.py", 'r.get("step") is None'),
            ("tools/score/metrics.py", 'r.get("step") is None'),
            ("tools/score/xhat.py", 'r.get("step") is None'),
            ("viewer/index.html", "r.step == null")):
        src = (run_io.ROOT / path).read_text(encoding="utf-8")
        assert needle in src, path


def test_the_viewer_never_draws_a_turn_twice():
    """**중간 스냅샷을 섞어 그리면 사람이 증식한다** (8/26 · Eddie 발견).

    실측에서 1해가 9명이 아니라 **54줄**로 그려졌다 (스냅샷 5회 + 턴 끝 1회 = 6배).

    갈라 두는 것만으로는 부족했다 — 처음엔 중간 줄을 **걸러내기만** 했는데, 그러면
    진행 중인 해가 통째로 사라져 뷰어가 한 해 내내(2~4분) 옛 화면에 멈춘다.
    스냅샷을 넣은 이유가 바로 그것이었으므로 정반대가 된다.

    규칙은 둘이다:
      · 끝난 해는 **턴 끝 줄**로 그린다 (스냅샷은 그 앞의 스냅사진일 뿐이다)
      · 끝나지 않은 해는 **마지막 스냅샷**으로 그리고 「진행 중」 이라고 밝힌다
    """
    from core import run_io

    src = (run_io.ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
    # 갈라 둔다
    assert "g(\"state.jsonl\").filter(r => r.step == null)" in src
    assert "stepRows" in src
    # **진행 중인 해도 센다** — `turns` 계산에 stepRows 가 들어가야 한다
    assert "...d.stepRows.map(s => s.turn)" in src
    # **이미 끝난 해는 덮어쓰지 않는다** — 이 가드가 빠지면 다시 증식한다
    assert "if (!b || b.state.length) return;" in src
    # 진행 중임을 화면에 밝힌다
    assert "진행 중" in src
