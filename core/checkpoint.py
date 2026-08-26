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
    # **매해를 따로 남긴다** (8/25 · Eddie). 전에는 한 파일을 덮어써서 복원점이 늘
    # 「마지막 턴 끝」 하나였다 — 규칙을 고친 뒤 「n해부터 다시」 가 **원리적으로**
    # 불가능했고, 그것을 알았을 때는 이미 되돌릴 곳이 없었다.
    #
    # 한 해가 12~40초인데 이 파일은 수십~수백 KB 다. 50해면 몇 MB — 런 하나의
    # `raw_calls.jsonl` 이 30MB 인 것에 비하면 무료다. 안 남길 이유가 없었다.
    d = Path(path).parent / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"t{world.turn:03d}.json").write_bytes(Path(path).read_bytes())


def at_turn(run_dir: Path, turn: int | None) -> tuple[Path, int]:
    """되돌릴 복원점을 고른다 — `(경로, 그 해)`.

    `turn=None` 이면 가장 마지막. `turn=N` 이면 **N해가 끝난 뒤**, 즉 다음이 N+1해다.
    「N해부터 다시」 를 원하면 `N-1` 을 준다.
    """
    d = Path(run_dir) / "checkpoints"
    have = sorted(int(f.stem[1:]) for f in d.glob("t*.json")) if d.is_dir() else []
    if turn is None:
        latest = Path(run_dir) / "checkpoint.json"
        if latest.exists():
            return latest, json.loads(latest.read_text(encoding="utf-8"))["turn"]
        if not have:
            raise FileNotFoundError(f"{run_dir} 에 복원점이 없습니다.")
        return d / f"t{have[-1]:03d}.json", have[-1]
    if turn == 0:
        raise ValueError("0해로 되돌리는 것은 새 런입니다 — --run-id 를 바꾸세요.")
    if turn not in have:
        raise FileNotFoundError(
            f"{turn}해 복원점이 없습니다. 있는 해: {have or '없음'}\n"
            "  (매해 저장은 8/25 부터입니다 — 그 전 런은 마지막 하나뿐입니다.)")
    return d / f"t{turn:03d}.json", turn


# 되돌릴 때 잘라낼 로그. **여기 빠진 파일은 조용히 두 번 들어간다.**
_TURN_LOGS = ("events", "messages", "metrics", "state", "raw_calls")


def rewind_logs(run_dir: Path, turn: int) -> dict[str, int]:
    """`turn` 보다 뒤의 행을 지운다. 지운 행 수를 파일별로 돌려준다.

    **되돌리기의 절반은 로그다.** 세계만 되돌리고 로그를 그대로 두면 다시 돌린 해가
    **두 번** 들어가고, 그 뒤 모든 지표가 조용히 오염된다 (`run_io` 가 겪었다).

    `turn` 필드가 없는 행은 남긴다 — 크래시 행처럼 턴에 속하지 않는 기록이다.
    """
    cut: dict[str, int] = {}
    for name in _TURN_LOGS:
        f = Path(run_dir) / f"{name}.jsonl"
        if not f.exists():
            continue
        keep, dropped = [], 0
        for line in f.read_text(encoding="utf-8").splitlines(keepends=True):
            if not line.strip():
                continue
            try:
                t = json.loads(line).get("turn")
            except json.JSONDecodeError:
                keep.append(line)          # 못 읽는 줄은 건드리지 않는다
                continue
            if isinstance(t, int) and t > turn:
                dropped += 1
            else:
                keep.append(line)
        if dropped:
            tmp = f.with_suffix(".tmp")
            tmp.write_text("".join(keep), encoding="utf-8")
            tmp.replace(f)
        cut[name] = dropped
    return cut


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
