#!/usr/bin/env python3
"""언어 학습의 암묵 효용 x 를 읽어내기 위한 자(ruler)를 맞춘다.

우리는 학습의 효용을 설계에 넣지 않는다. 넣으면 학습 여부가 설정의 귀결이 되어
관측이 불가능해진다 (왜곡을 프롬프트로 주입하지 않는 것과 같은 이유).

  known_langs 는 '읽을 수 있는 언어 집합'이고 route=original 의 실패 조건은
  '수신자가 못 읽으면' 이다. 메시지는 발신자 언어로 쓰이므로 학습자 본인의
  예산 회수는 0 이다. 그래서 학습 결정은 순수하게 암묵 효용 x 로만 결정된다.

      학습이 일어남  ⟺  x ≥ 실제 학습비

**할인이 아니라 가속이다** (8/22). `learn_base` 는 고정이고, 사유 하나당 한 번의
호출이 낳는 진척이 `learn_speedup` 만큼 빨라진다:

      배수 = 1 + speedup × 사유 수
      한 호출의 진척 = unit × 배수,  그 호출의 지출 = 진척 / 배수 = unit

  진척 목표가 고정이므로 **총 지출도 총 AP 도 1/배수** 로 줄어든다. 눈금이 셋이라는
  구조는 그대로 살아 있고 (사유 0 / 1 / 2), 비율만 1 : 1/1.5 : 1/2 로 바뀌었다.

이 파일은 몬테카를로를 쓰지 않는다 — ③만 예외다.

    ./.venv/bin/python tools/balance/learning.py
    ./.venv/bin/python tools/balance/learning.py --mult 1.2

**값을 손으로 적지 않는다** — `configs/base.yaml` 에서 읽는다. 이 파일이 실제로
`total_turns 50 · λ 8.26 · L 300` 을 박아둔 채로 낡아 있었다.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from core import config, survival as surv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def survival(a, lam, k):
    return math.exp(-((a / lam) ** k))


def expected_remaining(age, lam, k, tmax=60):
    """나이 age 까지 살아 있을 때 앞으로 더 살 것으로 기대되는 턴."""
    s0 = survival(age, lam, k)
    if s0 <= 0:
        return 0.0
    return sum(survival(age + j, lam, k) for j in range(0, tmax + 1)) / s0


def remaining_income(age, cfg, lam, k, mult, tmax=60):
    """남은 생애 기대 소득. **나이와 함께 오르는 소득**을 반영한다 (8/22).

    `expected_remaining × income` 으로 계산하면 젊은 에이전트를 과대평가한다 —
    남은 해가 많다는 것과 그 해들이 비싸다는 것은 다른 얘기다.
    """
    s0 = survival(age, lam, k)
    if s0 <= 0:
        return 0.0
    tot = 0.0
    for j in range(0, tmax + 1):
        a = age + j
        grown = 1.0 + cfg.income.age_growth * max(0, a - cfg.world.adult_age)
        tot += survival(a, lam, k) * cfg.income.per_turn * mult * grown
    return tot / s0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--mult", type=float, default=1.0,
                    help="자국 생산 배수 × 개체 소득 배수")
    a = ap.parse_args()
    cfg = config.load(a.config)

    L = cfg.costs.learn_base
    k, lam = cfg.survival.k, cfg.survival.lambda_base
    EL = surv.expected_life(lam, k)
    T = cfg.world.total_turns
    N = cfg.world.agents_per_country
    AP = cfg.turn.action_points

    # 눈금 — 사유 수별 배수. 사유는 spec 3.4: 국내 구사자 · 부모 구사자.
    marks = []
    for r, label in ((0, "사유 0 (연고 없음)"),
                     (1, "사유 1 (국내 구사자 또는 부모)"),
                     (2, "사유 2 (국내 구사자 + 부모)")):
        m = 1.0 + cfg.costs.learn_speedup * r
        # **총지출과 총 AP 는 다른 자로 나온다** (#49). 지출은 `L / 배수` 로 연속이지만,
        # AP 는 **호출 수**에 걸리고 호출 수는 올림이다 — 마지막 한 번은 남은 만큼만 내도
        # AP 는 그대로 한 번을 먹는다. 전에는 둘 다 `1/배수` 로 적어 AP 를 0.67·0.50 으로
        # 찍었는데 실제로는 0.8·0.6 이다.
        calls = math.ceil(L / (cfg.costs.unit * m) - 1e-9)
        marks.append((label, m, L / m, calls * cfg.ap.unit))

    print("=" * 84)
    print("학습 암묵효용 x 의 자 맞추기 — 산수. ③만 몬테카를로")
    print("=" * 84)
    print(f"  확정   {T}턴 · 기대수명 {EL:.2f}턴 · adult_age {cfg.world.adult_age}")
    print(f"         learn_base L = {L:.0f} (고정) · speedup +{cfg.costs.learn_speedup}/사유")
    print(f"         한 호출 진척 {cfg.costs.unit:.0f}×배수 · AP {cfg.ap.unit}/호출 · "
          f"한 해 AP {AP}")

    print("\n" + "=" * 84)
    print("① 자가 닿는 범위")
    print("=" * 84)
    # 성인 진입 시점 소득으로 상한을 잡는다 — 학습은 어린 쪽의 투자다.
    income0 = cfg.income.per_turn * a.mult
    ceil_ = remaining_income(cfg.world.adult_age, cfg, lam, k, a.mult)
    print(f"  x 측정 상한 ≈ 성인 진입({cfg.world.adult_age}세) 시점 잔여 기대소득 "
          f"= {ceil_:.0f}")
    print("  이보다 비싼 눈금은 저축이 생애 안에 끝나지 않아 x 를 놓칩니다.\n")
    print(f"{'눈금':<34}{'배수':>6}{'총지출':>8}{'저축해':>8}"
          f"{'총AP':>7}{'AP해':>7}   판정")
    for name, m, cost, ap_tot in marks:
        save = cost / income0
        ap_years = ap_tot / AP
        v = ("측정 불가 — 저축이 안 끝남" if cost > ceil_ else
             "바닥 — 예산이 제약을 멈춤" if save < 1.0 and ap_years < 1.0 else
             "AP 가 제약" if ap_years >= save else "쓸 수 있는 눈금")
        print(f"{name:<34}{m:>6.1f}{cost:>8.0f}{save:>8.1f}"
              f"{ap_tot:>7.2f}{ap_years:>7.2f}   {v}")
    costs = sorted(m[2] for m in marks)
    edges = " ".join(f"[{lo:.0f}, {hi:.0f})" for lo, hi in
                     zip([0] + costs, costs)) + f" [{costs[-1]:.0f}, ∞)"
    print(f"\n  → x 를 네 구간으로 나눕니다: {edges}")
    print("  ⚠ 지출과 AP 는 **같은 비율로 줄지 않습니다** (#49). 지출은 L/배수 로 연속이고")
    print("    (1 : 0.667 : 0.500), AP 는 호출 수에 걸려 올림입니다 (1 : 0.8 : 0.6). 그래서")
    print("    이 자는 「돈이 없어서」 와 「AP 가 없어서」 를 **조금은** 가릅니다 — 다만 그 차이가")
    print("    작으므로 판정은 여전히 로그의 거부 사유에 기댑니다")
    print("    (`insufficient_budget` / `insufficient_ap`).")

    print("\n" + "=" * 84)
    print("② 나이가 자를 흔듭니다")
    print("=" * 84)
    print("  '남은 생애'가 개인마다 다릅니다. 같은 눈금이라도 늙은 에이전트에게는")
    print("  회수 기간이 없어 사실상 더 비쌉니다. 게다가 소득이 나이와 함께 오르므로")
    print(f"  (age_growth {cfg.income.age_growth}) 젊을수록 남은 해가 싸기도 합니다.\n")
    # **「감당 가능한가」 로는 이제 아무것도 안 갈립니다** — L 이 300 → 200 으로 내려가
    # 20세에도 셋 다 감당됩니다. 갈리는 건 **남은 소득에서 차지하는 비중**입니다.
    dear, cheap = max(m[2] for m in marks), min(m[2] for m in marks)
    print(f"{'나이':>5}{'기대 잔여':>11}{'남은 소득':>11}"
          f"{'사유0 비중':>11}{'사유2 비중':>11}   판정")
    for age in (0, cfg.world.adult_age, 8, 12, 16, 20):
        rem = expected_remaining(age, lam, k)
        earn = remaining_income(age, cfg, lam, k, a.mult)
        hi, lo = dear / earn, cheap / earn
        v = ("연고가 생사를 가른다" if hi > 0.5 >= lo else
             "연고 없이도 가볍다" if hi <= 0.25 else
             "연고가 있어도 무겁다" if lo > 0.5 else "중간")
        print(f"{age:>5}{rem:>11.2f}{earn:>11.0f}{hi:>10.0%}{lo:>11.0%}   {v}")
    print("\n  ⚠ 이것이 x 추정의 노이즈원입니다. 학습 여부만 보면 'x 가 작아서'인지")
    print("    '늙어서'인지 구분되지 않습니다. **나이를 함께 기록해 층화해야 합니다.**")

    print("\n" + "=" * 84)
    print("③ 마지막 구사자가 죽으면 되돌릴 수 없습니다 — 임계 위험")
    print("=" * 84)
    print("  국내 구사자가 0 이 되면 배수가 내려가고(사유 하나가 사라짐), 그래서 더")
    print("  안 배우게 됩니다. 학습률 p(턴당 국가별 신규 학습자 기대수)별로 봅니다.\n")
    print(f"{'학습률 p':>10}{'단절 발생':>11}{'첫 단절 시점':>13}{'단절 후 회복':>13}")
    slow = 1.0 / (1.0 + cfg.costs.learn_speedup)   # 사유 하나를 잃으면 이만큼 느려진다
    for p in (0.02, 0.05, 0.10, 0.20, 0.40):
        broke, when, recovered = 0, [], 0
        for s in range(400):
            rng = random.Random(f"{p}|{s}")
            speakers = [[0, lam]]                  # 초기 구사자 1명
            ever, first = False, None
            for t in range(T):
                rate = p if speakers else p * slow
                if rng.random() < rate and len(speakers) < N:
                    speakers.append([0, lam])
                alive = []
                for sp in speakers:
                    sp[0] += 1
                    s0, s1 = survival(sp[0] - 1, lam, k), survival(sp[0], lam, k)
                    if rng.random() >= (1 - s1 / s0 if s0 > 0 else 1):
                        alive.append(sp)
                speakers = alive
                if not speakers and not ever:
                    ever, first = True, t
            if ever:
                broke += 1
                when.append(first)
                if speakers:
                    recovered += 1
        w = f"{sum(when)/len(when):.0f}턴" if when else "—"
        r = f"{recovered/broke:.0%}" if broke else "—"
        print(f"{p:>10.2f}{broke/400:>11.0%}{w:>13}{r:>13}")
    print("\n  학습률이 0.1 아래면 단절이 거의 확실하고, 단절 뒤 회복은 드뭅니다.")
    print("  **이것이 연쇄 2칸(학습자 감소)의 자기강화 메커니즘입니다.**")
    print("  ⚠ 상속이 들어온 뒤로는 '부모 구사자' 사유가 계보를 타고 이어집니다 —")
    print("    이 모형에는 그 경로가 없으므로 실제 단절 확률은 여기보다 낮습니다.")

    print("\n" + "=" * 84)
    print("④ 부등식으로만 묶인 것들")
    print("=" * 84)
    print("  comm_domestic < comm_intl_learner < comm_intl_ai   (전 구간, assert)")
    print("  comm_intl_learner   학습자 본인은 회수하지 못하므로 x 도출식에 들어가지 않는다")
    print("  propose_vote        국내/국제 비대칭의 크기")


if __name__ == "__main__":
    main()
