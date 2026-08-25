"""1회 실행 요약. 과제 1 제출물 (결과 한 문단). 길이는 `world.total_turns` 가 정한다.

    python -m scripts.run_once            # seed 는 config 의 run.seed
    python -m scripts.run_once --seed 7
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from core import config
from core.loop import run

BASE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(BASE))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = config.load(args.config)
    seed = args.seed if args.seed is not None else cfg.run.seed
    r = run(cfg, random.Random(seed))

    lands = {cid: c.land for cid, c in r.world.countries.items()}
    progress = {cid: round(c.progress, 1) for cid, c in r.world.countries.items()}
    mean_age = sum(r.death_ages) / len(r.death_ages) if r.death_ages else 0.0
    by_kind = Counter(b["born_by"] for b in r.births)

    print(f"[seed {seed}] {cfg.world.total_turns}턴 완료")
    print(f"  사망(=출생) {r.deaths}회  (자연사 {by_kind['natural']} / procreate {by_kind['procreate']})")
    print(f"  사망 나이 평균 {mean_age:.2f}, 최대 {max(r.death_ages) if r.death_ages else 0}")
    print(f"  국토 {lands}")
    print(f"  진척 {progress},  요격기 최고부지 {r.interceptor_best:.1f} / 임계 {cfg.thresholds.interceptor}")
    print(f"  생존 판정: {r.final['outcome']}  생존국 {r.final['survivors']}")


if __name__ == "__main__":
    main()
