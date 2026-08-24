"""런 이어하기. 하루치를 통째로 날려 봐야 필요성을 안다.

노트북이 자는 사이 20턴 런이 9시간을 흘려보냈고, 다시 돌리려면 1턴부터였다.
크래시·레이트리밋·강제 종료도 같다.

여기서 지키는 것은 하나다 — **이어붙인 세계가 끊지 않은 것과 같아야 한다.**
아니면 이어하기는 없는 편이 낫다. 조용히 다른 세계를 만들어 놓고 같은 런인 척한다.
"""
from __future__ import annotations

import itertools
import random

import pytest

from core import checkpoint, config, loop
from core.llm import StubClient, assistant_msg, tool_call
from domains.meteor import prompts


@pytest.fixture(scope="module")
def cfg():
    return config.load("configs/base.yaml")


def _script(n=200):
    """매 턴 한 마디 하고 끝낸다. 에이전트마다 별개 Stub 이어야 한다 (병렬)."""
    return [assistant_msg(tool_call("speak", "1", to="Ranoa1", text="x", reasoning="r")),
            assistant_msg(tool_call("end_turn", "2"))] * n


def _run(cfg, turns, seed=3, resume_from=None, checkpoint_to=None):
    clients = {}

    def client_for(aid):
        clients.setdefault(aid, StubClient(_script()))
        return clients[aid]

    return loop.run_agentic(
        cfg, random.Random(seed), client_for, StubClient(_script()), 24.0,
        prompts.render_turn_open, prompts.system_for, parallel=False,
        sim_turns=turns, resume_from=resume_from, checkpoint_to=checkpoint_to)


def test_resumed_world_matches_an_uninterrupted_one(tmp_path, cfg):
    """**끊고 이어붙인 6턴 = 통으로 돈 6턴.** 바이트 단위로 같아야 한다.

    난수 상태나 카운터가 하나라도 빠지면 여기서 갈린다.
    """
    whole = _run(cfg, 6)

    ck = tmp_path / "checkpoint.json"
    _run(cfg, 3, checkpoint_to=ck)                     # 3턴 돌고 끊는다
    tail = _run(cfg, 6, resume_from=ck, checkpoint_to=ck)   # 이어서 6턴까지

    assert tail.world.turn == 6
    assert loop._state_line(tail.world) == loop._state_line(whole.world)


def test_checkpoint_carries_what_state_jsonl_cannot(tmp_path, cfg):
    """`state.jsonl` 로는 못 이어붙인다 — 대화 이력·기억·언어 진척·열린 제안·
    인박스 큐가 거기 없다. 하나라도 빠지면 이어붙인 뒤가 다른 세계가 된다."""
    ck = tmp_path / "checkpoint.json"
    r = _run(cfg, 2, checkpoint_to=ck)
    a = r.world.agents["Asla1"]
    a.memory = "기억"
    a.lang_progress = {"zh": 120.0}
    r.world.countries["Asla"].proposal = {"target": "bunker", "by": "Asla2",
                                          "opened_turn": 1, "vote_turn": 5}
    checkpoint.save(ck, r.world, random.Random(1), 99, 77)

    w, rng, uid, mid, done = checkpoint.load(ck)
    b = w.agents["Asla1"]
    assert b.memory == "기억" and b.lang_progress == {"zh": 120.0}
    assert b.convo == a.convo and b.convo != []          # 대화 이력이 통째로
    assert isinstance(b.known_langs, set)                # set 으로 복원
    assert w.countries["Asla"].proposal["vote_turn"] == 5
    assert w.inbox_queue == r.world.inbox_queue
    assert (next(uid), next(mid), done) == (99, 77, 2)


def test_ids_do_not_leak_across_checkpoints(tmp_path, cfg):
    """`itertools.count` 는 현재 값을 못 읽는다. 훔쳐본 값을 버리면 턴마다 id 가
    하나씩 새어 나가 uid 가 듬성듬성해진다."""
    ck = tmp_path / "checkpoint.json"
    r = _run(cfg, 4, checkpoint_to=ck)
    uids = sorted(a.uid for a in r.world.agents.values())
    # **인구가 늘어난다** (8/21) — 아이 낳기가 부모를 죽이지 않으므로 9명이 아니다.
    # 그래도 uid 는 **틈 없이** 이어져야 한다. 훔쳐본 값을 버리면 턴마다 하나씩 새어
    # 듬성듬성해지고, 그게 이 검사가 잡으려는 것이다.
    # **살아 있는 uid 는 이제 띄엄띄엄하다** — 자연사가 자리를 갈면 옛 uid 가 빠진다.
    # 검사할 것은 「빠짐」 이 아니라 **「건너뜀」** 이다: 지금까지 만든 사람 수와 uid 의
    # 최대값이 같아야 한다. 훔쳐본 값을 버리면 턴마다 하나씩 새어 최대값이 앞서 나간다.
    n0 = cfg.world.agents_per_country * len(cfg.world.countries)
    made = n0 + len(r.births)
    assert max(uids) == made, f"uid 최대 {max(uids)} ≠ 만든 사람 {made}"
    assert len(set(uids)) == len(uids)


def test_a_stale_checkpoint_is_refused(tmp_path, cfg):
    """세계 구조가 바뀐 뒤의 체크포인트를 조용히 이어붙이면 다른 세계가 된다."""
    ck = tmp_path / "checkpoint.json"
    _run(cfg, 1, checkpoint_to=ck)
    import json
    blob = json.loads(ck.read_text(encoding="utf-8"))
    blob["version"] = 999
    ck.write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(ValueError, match="버전"):
        checkpoint.load(ck)


def test_writing_a_checkpoint_is_atomic(tmp_path, cfg):
    """쓰다 죽으면 **이전 것이 남아야** 한다. 반쯤 쓰인 파일은 이어할 수도 버릴 수도 없다."""
    ck = tmp_path / "checkpoint.json"
    _run(cfg, 1, checkpoint_to=ck)
    first = ck.read_text(encoding="utf-8")
    assert not (tmp_path / "checkpoint.tmp").exists()    # 임시 파일이 안 남는다
    _run(cfg, 2, resume_from=ck, checkpoint_to=ck)
    assert ck.read_text(encoding="utf-8") != first       # 갱신됐다
