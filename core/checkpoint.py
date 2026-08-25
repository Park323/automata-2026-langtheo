"""런 이어하기. 100턴 × 12런 야간 배치를 위한 것.

**하루치를 통째로 날려 봐야 필요성을 안다.** 노트북이 자는 사이 20턴 런이 9시간을
흘려보냈고, 다시 돌리려면 1턴부터였다. 크래시·레이트리밋·강제 종료도 같다.

이어하려면 **세계 전부**가 필요하다. `state.jsonl` 로는 안 된다 — 거기엔 분석용
요약만 있고 대화 이력·기억·열린 제안·인박스 큐·난수 상태가 없다. 그중 하나라도 빠지면
이어붙인 뒤가 원래 런과 다른 세계가 된다.

    세계        agents(대화 이력·기억·언어 진척 포함) · countries(열린 제안 포함)
                testaments · inbox_queue · next_idx · turn
    난수        rng.getstate()  ← 빠지면 이어붙인 뒤가 재현되지 않는다
    카운터      uid · msg_id 의 **다음 값**

`itertools.count` 는 현재 값을 읽을 수 없으므로 다음 값을 따로 넘겨받아 다시 만든다.
"""
from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict
from pathlib import Path

from core.state import Agent, Country, World

VERSION = 3   # 8/25: 돈 삭제 · Country.build_mult · Agent.income_mult 흡수
#
# **버전을 올려야 조용히 틀리지 않는다.** `Country(**v)` 는 없는 키를 기본값으로
# 떨어뜨리므로, 8/23 이전 체크포인트를 이어받으면 `build_mult` 가 전부 1.0 이 되어
# **국가 효율 순열이 사라진다** — 세계가 달라진 것을 아무도 모른다. `Agent` 쪽은
# 지운 키(budget 등)가 남아 있어 TypeError 로 시끄럽게 죽지만, Country 는 아니었다.


def _agent_to_json(a: Agent) -> dict:
    d = asdict(a)
    d["known_langs"] = sorted(a.known_langs)      # set 은 JSON 이 못 담는다
    d["parent_langs"] = sorted(a.parent_langs)
    return d


def _agent_from_json(d: dict) -> Agent:
    d = dict(d)
    d["known_langs"] = set(d.get("known_langs") or [])
    d["parent_langs"] = set(d.get("parent_langs") or [])
    return Agent(**d)


def save(path: Path, world: World, rng: random.Random,
         next_uid: int, next_msg_id: int) -> None:
    """턴 끝 상태를 통째로 적는다. **원자적으로** — 쓰다 죽으면 이전 것이 남아야 한다."""
    blob = {
        "version": VERSION,
        "turn": world.turn,
        "agents": {k: _agent_to_json(v) for k, v in world.agents.items()},
        "countries": {k: asdict(v) for k, v in world.countries.items()},
        "testaments": world.testaments,
        "inbox_queue": world.inbox_queue,
        "next_idx": world.next_idx,
        "rng_state": rng.getstate(),
        "next_uid": next_uid,
        "next_msg_id": next_msg_id,
    }
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(blob, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def load(path: Path):
    """(world, rng, uid_counter, msg_ids, turn_done) 를 돌려준다."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if blob.get("version") != VERSION:
        raise ValueError(f"체크포인트 버전이 다릅니다 ({blob.get('version')} != {VERSION}). "
                         "세계 구조가 바뀐 뒤라 이어붙이면 다른 세계가 됩니다.")
    world = World(
        turn=blob["turn"],
        countries={k: Country(**v) for k, v in blob["countries"].items()},
        agents={k: _agent_from_json(v) for k, v in blob["agents"].items()},
        testaments=blob.get("testaments") or {},
        inbox_queue=blob.get("inbox_queue") or [],
        next_idx=blob.get("next_idx") or {},
    )
    rng = random.Random()
    st = blob["rng_state"]
    # JSON 은 튜플을 배열로 만든다. setstate 는 튜플을 요구한다.
    rng.setstate((st[0], tuple(st[1]), st[2]))
    return (world, rng,
            itertools.count(blob["next_uid"]),
            itertools.count(blob["next_msg_id"]),
            blob["turn"])
