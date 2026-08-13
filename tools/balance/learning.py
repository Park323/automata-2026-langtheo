#!/usr/bin/env python3
"""언어 학습의 암묵 효용 x 를 읽어내기 위한 자(ruler)를 맞춘다.

우리는 학습의 효용을 설계에 넣지 않는다. 넣으면 학습 여부가 설정의 귀결이 되어
관측이 불가능해진다 (왜곡을 프롬프트로 주입하지 않는 것과 같은 이유).

  known_langs 는 '읽을 수 있는 언어 집합'이고 route=original 의 실패 조건은
  '수신자가 못 읽으면' 이다. 메시지는 발신자 언어로 쓰이므로 학습자 본인의
  예산 회수는 0 이다. 그래서 학습 결정은 순수하게 암묵 효용 x 로만 결정된다.

      학습이 일어남  ⟺  x ≥ 실제 학습비 ∈ {L, L/2, L/4}

  할인은 spec 3.4 — 국내 구사자 ×0.5, 부모 구사자 ×0.5, 중복 ×0.25.
  눈금이 셋이므로 x 를 네 구간으로 좁힐 수 있다.

이 파일은 몬테카를로를 쓰지 않는다 — ③만 예외다.

  python3 tools/balance/learning.py
  python3 tools/balance/learning.py --L 300
"""
import argparse, math, random

FIXED = dict(
    total_turns=50, income=100, initial_budget=0,
    countries=3, agents_per_country=3,
    action_points=1.0, ap_learn=1.0,
    surv_k=8, surv_lambda=8.26,
)


def survival(a, lam, k):
    return math.exp(-((a / lam) ** k))


def expected_life(lam, k, tmax=40):
    return sum(survival(t, lam, k) for t in range(1, tmax + 1))


def expected_remaining(age, lam, k, tmax=40):
    """나이 age 에서 앞으로 더 살 것으로 기대되는 턴."""
    s0 = survival(age, lam, k)
    if s0 <= 0:
        return 0.0
    return sum(survival(t, lam, k) for t in range(age + 1, tmax + 1)) / s0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=300)
    ap.add_argument("--mult", type=float, default=1.0, help="자국 생산 배수")
    a = ap.parse_args()
    f = FIXED
    L, k, lam = a.L, f["surv_k"], f["surv_lambda"]
    EL = expected_life(lam, k)
    income = f["income"] * a.mult
    marks = [("L      (연고 없음)", L),
             ("L/2    (국내 구사자 또는 부모)", L / 2),
             ("L/4    (국내 구사자 + 부모)", L / 4)]

    print("=" * 84)
    print("학습 암묵효용 x 의 자 맞추기 — 산수. ③만 몬테카를로")
    print("=" * 84)
    print(f"  확정   {f['total_turns']}턴 · 세대 경계 없음 · 기대수명 {EL:.2f}턴")
    print(f"  잠정   income {income:.0f} (배수 {a.mult})  learn_base L = {L:.0f}")

    print("\n" + "=" * 84)
    print("① 자가 닿는 범위")
    print("=" * 84)
    ceil_ = income * (EL - f["ap_learn"])
    print(f"  x 측정 상한 ≈ income × (기대수명 − ap.learn) = {ceil_:.0f}")
    print(f"  이보다 비싼 눈금은 저축이 생애 안에 끝나지 않아 x 를 놓칩니다.\n")
    print(f"{'눈금':<32}{'비용':>8}{'저축턴':>8}{'점유턴':>8}{'생애비중':>10}   판정")
    for name, cost in marks:
        save = cost / income
        occ = math.ceil(save) + f["ap_learn"]
        share = occ / EL
        v = ("측정 불가 — 저축이 안 끝남" if cost > ceil_ else
             "바닥 — 예산이 제약을 멈춤" if save < 1.0 else "쓸 수 있는 눈금")
        print(f"{name:<32}{cost:>8.0f}{save:>8.1f}{occ:>8.0f}{share:>9.0%}   {v}")
    print(f"\n  → x 를 [0, {L/4:.0f}) [{L/4:.0f}, {L/2:.0f}) "
          f"[{L/2:.0f}, {L:.0f}) [{L:.0f}, ∞) 네 구간으로 나눕니다.")

    print("\n" + "=" * 84)
    print("② 나이가 자를 흔듭니다")
    print("=" * 84)
    print("  세대가 없어져 '남은 생애'가 개인마다 다릅니다. 같은 눈금이라도")
    print("  늙은 에이전트에게는 회수 기간이 없어 사실상 더 비쌉니다.\n")
    print(f"{'나이':>5}{'기대 잔여':>11}{'남은 소득':>11}   L 을 감당할 수 있나")
    for age in (0, 2, 4, 6, 8):
        rem = expected_remaining(age, lam, k)
        earn = rem * income
        can = [n.split()[0] for n, c in marks if c <= earn]
        print(f"{age:>5}{rem:>11.2f}{earn:>11.0f}   "
              f"{'· '.join(can) if can else '없음'}")
    print("\n  ⚠ 이것이 x 추정의 노이즈원입니다. 학습 여부만 보면 'x 가 작아서'인지")
    print("    '늙어서'인지 구분되지 않습니다. **나이를 함께 기록해 층화해야 합니다.**")

    print("\n" + "=" * 84)
    print("③ 마지막 구사자가 죽으면 되돌릴 수 없습니다 — 임계 위험")
    print("=" * 84)
    print("  국내 구사자가 0 이 되면 비용이 2배가 되고, 그래서 더 안 배우게 됩니다.")
    print("  학습률 p(턴당 국가별 신규 학습자 기대수)별로 단절 확률을 봅니다.\n")
    print(f"{'학습률 p':>10}{'단절 발생':>11}{'첫 단절 시점':>13}{'단절 후 회복':>13}")
    N = f["agents_per_country"]
    for p in (0.02, 0.05, 0.10, 0.20, 0.40):
        broke, when, recovered = 0, [], 0
        for s in range(400):
            rng = random.Random(f"{p}|{s}")
            speakers = [[0, lam] for _ in range(1)]   # 초기 구사자 1명
            ever, first = False, None
            for t in range(f["total_turns"]):
                rate = p if speakers else p / 2      # 단절되면 학습률 절반
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
                broke += 1; when.append(first)
                if speakers:
                    recovered += 1
        w = f"{sum(when)/len(when):.0f}턴" if when else "—"
        r = f"{recovered/broke:.0%}" if broke else "—"
        print(f"{p:>10.2f}{broke/400:>11.0%}{w:>13}{r:>13}")
    print("\n  학습률이 0.1 아래면 단절이 거의 확실하고, 단절 뒤 회복은 드뭅니다.")
    print("  **이것이 연쇄 2칸(학습자 감소)의 자기강화 메커니즘입니다.**")

    print("\n" + "=" * 84)
    print("④ 부등식으로만 묶인 것들")
    print("=" * 84)
    print("  comm_domestic < comm_intl_learner < comm_intl_ai   (전 구간, assert)")
    print("  comm_intl_learner   학습자 본인은 회수하지 못하므로 x 도출식에 들어가지 않는다")
    print("  ask_clarification   검증이 먼저 잘리는 강도")
    print("  propose_vote        국내/국제 비대칭의 크기")


if __name__ == "__main__":
    main()
