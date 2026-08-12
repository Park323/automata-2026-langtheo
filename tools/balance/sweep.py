#!/usr/bin/env python3
"""밸런스 사전 조정 — LLM 없이 파라미터 공간을 훑는다.

세계를 만든 뒤 8개 파라미터를 맞추는 건 늦고 비싸다. 대부분은 산수와
몬테카를로로 미리 좁힐 수 있다. LLM 에이전트 대신 규칙 기반 정책 몇 개를
쓰고, 실제 에이전트는 그 공간 어딘가에 떨어진다고 가정한다.

  ★ 1차 기준 : 조율 성공률을 낮췄을 때 회피율이 충분히 갈리는가 (민감도)
                노브가 결과를 못 바꾸면 회피율이 50%여도 실험은 죽는다
    2차 기준 : 중간값이 바닥(0)·천장(1)에 붙지 않는가
    3차 기준 : 정책 간 편차가 작은가 (강건성)

이것은 세계의 '경제 구조'를 맞추는 도구이지 가설을 검증하는 도구가 아니다.
번역 왜곡의 효과는 coord 파라미터로 추상화되어 있을 뿐 모형화되지 않았다.

  python3 tools/balance/sweep.py
  python3 tools/balance/sweep.py --top 20 --seeds 60
"""
import argparse, itertools, json, math, random
from dataclasses import dataclass, asdict

COUNTRIES = 3


@dataclass
class Cfg:
    income: float
    learn_new: float
    lifespan: int
    generations: int
    success_prob: float
    growth_coef: float
    growth_scale: float
    interceptor: float
    bunker: float
    facility_eff: float
    warm_glow: float
    agents: int = 3


# ── 산수로 먼저 거른다 (assert. spec 7장) ──────────────────────────
def caps(c: Cfg):
    """세대별 (multiplier, 국가 총소득). 반복문 — 상호 재귀는 2^G 로 폭발한다."""
    out, mults, total = [], [], 0.0
    for g in range(c.generations):
        m = 1.0 if g == 0 else 1 + c.growth_coef * math.sqrt(total / c.growth_scale)
        cap = c.income * m * c.agents * c.lifespan
        mults.append(m); out.append(cap); total += cap
    return out, mults


def to_progress(income_amount: float, c: Cfg) -> float:
    """소득을 기대 진척도로 환산. 임계값은 진척 단위이므로 반드시 거쳐야 한다."""
    return income_amount * c.facility_eff * c.success_prob


def bounds(c: Cfg):
    """요격기 임계가 놓여야 할 창(window). 전부 **진척 단위**로 환산한다.

    A 미루기 방지 : 마지막 세대 3국이 전력을 다해도 불가
    B 조율 강제   : 한 나라가 전 세대를 다 써도 불가   ← 스펙에 없던 부등식
    C 도달 가능   : 세 나라가 모으면 가능              ← 스펙에 없던 부등식
    """
    cp, _ = caps(c)
    k = c.facility_eff * c.success_prob
    A = 3 * cp[-1] * k
    B = sum(cp) * k                      # 1국 × 전 세대
    C = 3 * sum(cp) * k
    return A, B, C


def passes_asserts(c: Cfg):
    cp, mu = caps(c)
    A, B, C = bounds(c)
    if not c.interceptor > A:
        return False, "A 미루기 방지 위반"
    if not c.interceptor > B:
        return False, "B 조율 강제 위반 (한 나라가 혼자 해낼 수 있다)"
    if not c.interceptor < C * 0.6:      # 정책계수 — 전액 투입은 비현실적
        return False, "C 도달 불가"
    if not c.bunker <= cp[0] * c.facility_eff * c.success_prob:
        return False, "bunker 가 첫 세대 총소득으로 도달 불가"
    b1 = c.bunker / (c.agents * c.lifespan)
    i1 = c.interceptor / (3 * c.agents * c.lifespan * c.generations)
    if not b1 > i1:
        return False, "벙커 1인부담 <= 요격기 1인부담"
    return True, ""


# ── 정책 ───────────────────────────────────────────────────────────
# 실제 LLM 에이전트가 어디에 떨어질지 모르므로, 공간의 꼭짓점들을 훑는다.
POLICIES = {
    "selfish":    lambda ctx: (1.0, 0.0, 0.0),   # 전액 행복
    "bunker":     lambda ctx: (0.2, 0.0, 0.8),   # 자국 벙커 우선
    "altruist":   lambda ctx: (0.1, 0.0, 0.9),   # 전액 요격기
    "freerider":  lambda ctx: (0.6, 0.0, 0.4) if ctx["others_investing"] else (0.2, 0.0, 0.8),
    "mixed":      lambda ctx: (0.34, 0.33, 0.33),
    "farsighted": lambda ctx: (0.2, 0.6, 0.2) if ctx["gen"] < ctx["G"] - 1 else (0.2, 0.0, 0.8),
}


def simulate(c: Cfg, policy, coord: float, rng: random.Random) -> float:
    """한 판. 반환 = 생존 인구 비율 (1.0 이면 전 인류 생존).

    coord = 3국이 요격기 부지를 하나로 모을 확률(조율 성공률).
    조율에 성공하면 비유치국도 유치국 요격기에 낼 수 있다 (타국 투자).
    실패하면 각자 자국에 착수해 진척이 쪼개진다.
    번역 왜곡의 효과는 이 한 숫자로 추상화되어 있을 뿐 모형화되지 않았다.
    """
    coordinated = rng.random() < coord
    host = rng.randrange(COUNTRIES)

    land = [None] * COUNTRIES
    prog = [0.0] * COUNTRIES
    natl = [0.0] * COUNTRIES
    mult = [1.0] * COUNTRIES

    if coordinated:
        land[host] = "interceptor"
        for i in range(COUNTRIES):
            if i != host:
                land[i] = "bunker"          # 유치를 안 맡은 나라는 자국에 벙커
    else:
        for i in range(COUNTRIES):          # 조율 실패 — 각자 알아서
            land[i] = "interceptor" if rng.random() < 0.5 else "bunker"

    for g in range(c.generations):
        for i in range(COUNTRIES):
            mult[i] = 1 + c.growth_coef * math.sqrt(natl[i] / c.growth_scale) if g else 1.0
        others = any(prog[j] > 0 for j in range(COUNTRIES) if land[j] == "interceptor")
        for _turn in range(c.lifespan):
            for i in range(COUNTRIES):
                budget = c.income * mult[i] * c.agents
                w_h, w_n, w_f = policy({"gen": g, "G": c.generations,
                                        "others_investing": others})
                natl[i] += budget * w_n
                spend = budget * w_f
                if spend <= 0:
                    continue
                # 조율 성공 시 비유치국은 자국 벙커와 유치국 요격기에 반씩 낸다.
                # 조율 실패 시 자국에만 낼 수 있다 (어디에 낼지 모르므로).
                targets = [(i, spend)]
                if coordinated and i != host:
                    targets = [(i, spend * 0.5), (host, spend * 0.5)]
                for tgt, amt in targets:
                    n = int(amt * c.facility_eff)
                    prog[tgt] += sum(1 for _ in range(n) if rng.random() < c.success_prob)

    itot = sum(prog[i] for i in range(COUNTRIES) if land[i] == "interceptor")
    if not coordinated:                     # 쪼개져 합산되지 않는다
        cand = [prog[i] for i in range(COUNTRIES) if land[i] == "interceptor"]
        itot = max(cand) if cand else 0.0
    if itot >= c.interceptor:
        return 1.0                          # 전 인류 생존
    saved = sum(1 for i in range(COUNTRIES)
                if land[i] == "bunker" and prog[i] >= c.bunker)
    return saved / COUNTRIES


def evaluate(c: Cfg, seeds: int, coords=(0.2, 0.9)):
    """정책 × 조율률로 회피율을 재고 민감도·강건성을 낸다."""
    res = {}
    for cd in coords:
        per_pol = []
        for name, pol in POLICIES.items():
            rng = random.Random(hash((name, cd)) & 0xFFFF)
            runs = [simulate(c, pol, cd, rng) for _ in range(seeds)]
            per_pol.append(sum(1 for x in runs if x >= 1.0) / seeds)   # 전 인류 생존률
        res[cd] = per_pol
    lo, hi = res[min(coords)], res[max(coords)]
    mean_lo, mean_hi = sum(lo) / len(lo), sum(hi) / len(hi)
    mid = (mean_lo + mean_hi) / 2
    spread = sum((x - sum(hi) / len(hi)) ** 2 for x in hi) / len(hi)
    return {
        "sensitivity": mean_hi - mean_lo,       # ★ 1차 기준
        "mid": mid,                              # 2차
        "policy_var": spread ** 0.5,             # 3차
        "lo": mean_lo, "hi": mean_hi,
        "by_policy_hi": dict(zip(POLICIES, hi)),
    }


# ── 스윕 ───────────────────────────────────────────────────────────
GRID = {
    "income":        [8, 12],
    "learn_new":     [30, 50],
    "lifespan":      [10, 15, 20],
    "generations":   [3, 4, 5],
    "success_prob":  [0.3, 0.5, 0.7],
    "growth_coef":   [0.0, 0.3],
    "growth_scale":  [2000],
    "facility_eff":  [1.0],
    "warm_glow":     [0.05],
}
# 임계값은 유도하되 여유 계수를 스윕한다.
# bunker_ratio 가 낮으면 각자 벙커로 다 살아남아 조율이 무의미해진다 —
# 스윕이 실제로 그것을 잡아냈다.
BUNKER_RATIO = [0.85, 0.95, 1.00]
INTERCEPT_POS = [0.15, 0.35, 0.55]   # 창 안에서의 위치


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"격자 {len(combos)}개 조합\n")

    ok, rejected = [], {}
    for vals in combos:
        d = dict(zip(keys, vals))
        probe = Cfg(interceptor=1, bunker=1, **d)
        cp, _ = caps(probe)
        k = probe.facility_eff * probe.success_prob
        A, B, C = bounds(Cfg(**dict(d, bunker=1, interceptor=1)))
        for br, pos in itertools.product(BUNKER_RATIO, INTERCEPT_POS):
            lo = max(A, B)
            c = Cfg(**dict(d, bunker=cp[0] * k * br,
                           interceptor=lo + (C * 0.6 - lo) * pos))
            good, why = passes_asserts(c)
            if not good:
                k = why.split("(")[0]
                rejected[k] = rejected.get(k, 0) + 1
                continue
            ok.append(c)

    print(f"assert 통과 {len(ok)}개 / 탈락 {len(combos)-len(ok)}개")
    for k, v in sorted(rejected.items(), key=lambda x: -x[1]):
        print(f"   {v:>4}  {k}")
    if not ok:
        return

    scored = []
    for c in ok:
        m = evaluate(c, a.seeds)
        scored.append((c, m))
    # 민감도 우선, 바닥·천장 회피, 정책 편차 작은 순
    scored.sort(key=lambda x: (-x[1]["sensitivity"],
                               abs(x[1]["mid"] - 0.5),
                               x[1]["policy_var"]))

    print(f"\n{'='*100}")
    print("민감도 상위 — 조율률 0.2 → 0.9 일 때 회피율이 얼마나 갈리는가")
    print(f"{'='*100}")
    hdr = f"{'inc':>4}{'life':>5}{'gen':>4}{'p':>5}{'grow':>5}{'벙커비':>7}  "
    print(hdr + f"{'민감도':>7}{'저조율':>7}{'고조율':>7}{'정책편차':>8}")
    for c, m in scored[:a.top]:
        cp, _ = caps(c)
        print(f"{c.income:>4.0f}{c.lifespan:>5}{c.generations:>4}"
              f"{c.success_prob:>5.1f}{c.growth_coef:>5.1f}{c.bunker/(cp[0]*c.facility_eff*c.success_prob):>7.2f}  "
              f"{m['sensitivity']:>7.2f}{m['lo']:>7.2f}{m['hi']:>7.2f}{m['policy_var']:>8.2f}")

    best, bm = scored[0]
    print(f"\n{'='*100}\n추천 조합 — 정책별 회피율 (조율률 0.9)\n{'='*100}")
    for k, v in bm["by_policy_hi"].items():
        print(f"  {k:<12} {v:>5.2f}")
    cp, mu = caps(best)
    print(f"\n  세대별 국가 총소득 : {[round(x) for x in cp]}")
    print(f"  세대별 multiplier  : {[round(x,2) for x in mu]}")
    print(f"  요격기 임계 {best.interceptor:.0f} / 벙커 임계 {best.bunker:.0f}")
    print(f"  학습 저축 필요턴   : {best.learn_new/best.income:.1f} / 수명 {best.lifespan}")
    print("\n" + json.dumps(asdict(best), indent=2))


if __name__ == "__main__":
    main()
