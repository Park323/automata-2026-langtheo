#!/usr/bin/env python3
"""밸런스 사전 조정 (Phase 0) — LLM 없이 파라미터 공간을 훑는다.

세계를 만든 뒤 파라미터를 맞추는 건 늦고 비싸다. 대부분은 산수와 몬테카를로로 미리
좁힐 수 있다. LLM 에이전트 대신 규칙 기반 정책을 쓰고, 실제 에이전트는 그 공간
어딘가에 떨어진다고 가정한다.

  ★ 1차 기준 : 필요 기여율 w* 가 목표(0.5)에 오는가
                재앙 회피율은 목표로 삼을 수 없다. 시행 횟수가 많아 진척이 사실상
                결정적이 되고 '회피율 ≈ 조율 성공률' 이 되어 파라미터로 안 움직인다
    2차 기준 : 조율률을 낮췄을 때 회피율이 갈리는가 (민감도)
    3차 기준 : 정책 간 편차가 작은가 (강건성)

세대 경계가 없다 (spec 2.2). 50턴을 통으로 돌면서 각자 다른 시점에 죽는다.

이것은 세계의 '경제 구조'를 맞추는 도구이지 가설을 검증하는 도구가 아니다.
번역 왜곡의 효과는 coord 파라미터로 추상화되어 있을 뿐 모형화되지 않았다.

  python3 tools/balance/sweep.py
  python3 tools/balance/sweep.py --top 20 --seeds 30
"""
import argparse, itertools, json, math, random
from dataclasses import dataclass, asdict

COUNTRIES = 3
POLICY_COEF = 0.6      # 전액 투입은 비현실적. 현실적 최대 투자율
W_TARGET = 0.5         # Phase 1 캘리브레이션 목표 (spec 12.4)


@dataclass
class Cfg:
    income: float            # **한 해 용량** (AP/ap_unit × unit). 옛 소득 자리다
    total_turns: int
    epoch_turns: int
    success_prob: float
    growth_coef: float
    growth_scale: float
    initial_budget: float   # 남겨 둔다 — 0 이 아닌 격자를 훑던 이력이 있다
                            # **기본값을 주지 말 것** — 뒤에 기본값 없는 필드가 있다
                            # (이 프로젝트에서 다섯 번째로 밟은 자리다)
    surv_k: float
    surv_lambda: float
    wellness_gain: float
    interceptor: float
    bunker: float
    facility_eff: float
    agents: int = 3
    # **나이가 들면 더 번다** (8/22). 창은 「한 나라의 전 기간 총소득」 에서 나오므로,
    # 나이 배수를 안 넣으면 창이 실제보다 좁아지고 임계가 「도달 가능」 쪽에 붙는다.
    # **가장 잘 짓는 나라의 효율** (8/23). 나라마다 요격기 진척 속도가 다르므로
    # (`facility.build_spread`) 네 조건이 「최선의 나라에 몰아줬을 때」 로 걸려야 한다.
    # ★B 가 결정적이다 — 혼자 해내면 안 되는 것은 **가장 잘 짓는 나라**다.
    # 벙커에는 안 걸린다 (요격기 전용).
    build_best: float = 1.0


# ── 수명 ───────────────────────────────────────────────────────────
def hazard(age: int, lam: float, k: float) -> float:
    """나이 age 를 마칠 때 죽을 확률. S(a) = exp(-(a/lam)^k)."""
    s0 = math.exp(-((age / lam) ** k))
    s1 = math.exp(-(((age + 1) / lam) ** k))
    return 1.0 if s0 <= 0 else 1.0 - s1 / s0


def expected_life(lam: float, k: float, tmax: int = 60) -> float:
    """E[사망 나이] = Σ_(a≥0) S(a).  a=0 항을 빼면 1턴이 모자란다."""
    return sum(math.exp(-((a / lam) ** k)) for a in range(0, tmax + 1))


# ── 창 (spec 7장) ──────────────────────────────────────────────────
def bounds(c: Cfg):
    """요격기 임계가 놓여야 할 창. 전부 **진척 단위**.

    A 미루기 방지 : 마지막 한 주기에 3국이 전력을 다해도 불가
    B 조율 강제   : 한 나라가 전 기간을 다 써도 불가
    C 도달 가능   : 세 나라가 모으면 가능
    E 지속 참여   : 한 주기가 통째로 쉬면 불가 — 양변에 같은 정책계수를 쓴다

    성장은 유도에서 뺀다. 성장은 국가 투자의 결과이고 그 돈은 요격기와 경쟁하므로,
    성장을 전제로 임계를 잡으면 "국가 투자를 안 하면 구조적으로 도달 불가" 가 된다.
    """
    k = c.facility_eff * c.success_prob * c.build_best
    # **나이 배수를 안 곱한다** (8/25). `age_growth` 를 없앴다 — AP 는 매년 리셋돼
    # 나이와 무관하게 같은 용량이다.
    per_turn_country = c.income * c.agents
    whole = per_turn_country * c.total_turns
    epoch = per_turn_country * c.epoch_turns
    A = 3 * epoch * k
    B = whole * k
    C = 3 * whole * k
    E = 3 * (whole - epoch) * k * POLICY_COEF
    return A, B, C, E


def passes_asserts(c: Cfg):
    A, B, C, E = bounds(c)
    # **벙커에는 `build_best` 를 안 곱한다** — 국가 효율은 요격기 전용이다 (8/23).
    k = c.facility_eff * c.success_prob
    # **실효 소득으로 잰다** (8/22) — `bounds()` 와 같은 식이다. 여기만 안 곱하면 벙커 창이
    # 창보다 좁아져서 「전 기간을 부어도 의미 없다」 가 거짓으로 걸린다.
    eff = c.income
    epoch_progress = eff * c.agents * c.epoch_turns * k
    whole_progress = eff * c.agents * c.total_turns * k
    if not c.interceptor > A:
        return False, "A 미루기 방지 위반"
    if not c.interceptor > B:
        return False, "B 조율 강제 위반 (한 나라가 혼자 해낼 수 있다)"
    if not c.interceptor < C * POLICY_COEF:
        return False, "C 도달 불가"
    if not c.interceptor > E:
        return False, "E 지속 참여 위반 (한 주기가 쉬어도 지어진다)"
    if not c.bunker >= epoch_progress:
        return False, "벙커가 한 주기로 너무 깊어진다"
    if not c.bunker <= whole_progress:
        return False, "벙커가 전 기간을 부어도 의미 있는 깊이에 못 간다"
    # **같은 기간으로 나눈다** (#51). 벙커만 한 주기로 나누고 있었다 — `core.asserts` 와
    # 같은 자를 써야 두 도구가 같은 말을 한다.
    b1 = c.bunker / (c.agents * c.total_turns)
    i1 = c.interceptor / (3 * c.agents * c.total_turns)
    # **벙커가 더 비싸야 한다.** 같으면 협력의 추가 비용(말하기·학습·배신 위험)을
    # 감수할 이유가 없어진다 — `core.asserts` 와 같은 규칙이다 (8/25 되돌림).
    if not b1 > i1:
        return False, "벙커 1인부담 <= 요격기 1인부담"
    return True, ""


# ── 정책 ───────────────────────────────────────────────────────────
# 배분 = (wellness, national, 자국시설, 타국요격기).  합이 1 이 아니어도 된다 —
# 남는 몫은 쓰지 않고 이월된다. 실제 갈림은 뒤 두 항에 있다.
POLICIES = {
    "longevist":  lambda ctx: (0.9, 0.0, 0.0, 0.0),   # 내 수명에만 — happiness 를 대신한다
    "bunker":     lambda ctx: (0.1, 0.0, 0.8, 0.0),
    "altruist":   lambda ctx: (0.1, 0.0, 0.0, 0.8),
    "hedger":     lambda ctx: (0.1, 0.0, 0.4, 0.4),
    "freerider":  lambda ctx: (0.3, 0.0, 0.6, 0.0) if ctx["others_investing"]
                              else (0.1, 0.0, 0.0, 0.8),
    "farsighted": lambda ctx: (0.1, 0.5, 0.2, 0.2) if ctx["t"] < ctx["T"] * 0.6
                              else (0.1, 0.0, 0.0, 0.8),
}


def contrib_policy(w: float):
    """소득의 90% 를 시설에 쓰되 그중 w 를 타국 요격기에. w* 를 재는 자."""
    return lambda ctx: (0.0, 0.0, 0.9 * (1 - w), 0.9 * w)


def simulate(c: Cfg, policy, coord: float, rng: random.Random):
    """한 판. 반환 = (생존 인구 비율, 사망 횟수).

    coord = 3국이 요격기 부지를 하나로 모을 확률. 조율에 성공하면 비유치국도
    유치국 요격기에 낼 수 있다. 실패하면 각자 자국에 착수해 진척이 쪼개진다.
    """
    coordinated = rng.random() < coord
    host = rng.randrange(COUNTRIES)

    land = [None] * COUNTRIES
    if coordinated:
        land[host] = "interceptor"
        for i in range(COUNTRIES):
            if i != host:
                land[i] = "bunker"
    else:
        for i in range(COUNTRIES):
            land[i] = "interceptor" if rng.random() < 0.5 else "bunker"

    prog = [0.0] * COUNTRIES
    natl = [0.0] * COUNTRIES
    # 에이전트: (나이, lambda, 예산)
    people = [[[0, c.surv_lambda, c.initial_budget] for _ in range(c.agents)]
              for _ in range(COUNTRIES)]
    deaths = 0

    for t in range(c.total_turns):
        others = any(prog[j] > 0 for j in range(COUNTRIES) if land[j] == "interceptor")
        for i in range(COUNTRIES):
            mult = 1 + c.growth_coef * math.sqrt(natl[i] / c.growth_scale)
            for a in people[i]:
                # **이월이 없다** (8/25). 전에는 `+=` 였고 남은 예산이 쌓였다.
                # 국가 배수는 용량이 아니라 **진척 전환**에만 걸린다 (아래 eff).
                a[2] = c.income
                w_well, w_nat, w_own, w_int = policy(
                    {"t": t, "T": c.total_turns, "age": a[0],
                     "others_investing": others})
                spend = a[2]
                well, nat = spend * w_well, spend * w_nat
                own, to_host = spend * w_own, spend * w_int
                if not coordinated or i == host:
                    own, to_host = own + to_host, 0.0
                a[2] -= well + nat + own + to_host
                a[1] += well * c.wellness_gain      # 수명 척도 증가
                natl[i] += nat
                eff = c.facility_eff * mult
                for tgt, amt in ((i, own), (host, to_host)):
                    if amt <= 0:
                        continue
                    # **받는 나라의 요격기 효율** (8/23). 조율이 완벽하다는 것은 부지를
                    # 하나로 모은다는 뜻이고, 합리적인 조율자는 **가장 잘 짓는 나라**를
                    # 고른다 — 그래서 host 는 `build_best` 다. 자국 투자는 나라 평균(1.0).
                    n = int(amt * eff * (c.build_best if tgt == host else 1.0))
                    prog[tgt] += sum(1 for _ in range(n)
                                     if rng.random() < c.success_prob)
        # 턴 끝 — 개인별 생존 판정. 죽으면 그 자리에 0 상태 신규
        for i in range(COUNTRIES):
            for j, a in enumerate(people[i]):
                a[0] += 1
                if rng.random() < hazard(a[0] - 1, a[1], c.surv_k):
                    people[i][j] = [0, c.surv_lambda, c.initial_budget]
                    deaths += 1

    itot = sum(prog[i] for i in range(COUNTRIES) if land[i] == "interceptor")
    if not coordinated:
        cand = [prog[i] for i in range(COUNTRIES) if land[i] == "interceptor"]
        itot = max(cand) if cand else 0.0
    if itot >= c.interceptor:
        return 1.0, deaths
    saved = 0
    for i in range(COUNTRIES):
        if land[i] != "bunker":
            continue
        if rng.random() < 1 - math.exp(-prog[i] / c.bunker):
            saved += 1
    return saved / COUNTRIES, deaths


def growth_then_build(w: float, split: int):
    """앞 split 턴은 전액 국가 투자, 이후는 소득의 w 를 타국 요격기에.

    국가 투자도 시간 축 투자다 — 초기 세대가 반드시 무언가를 해야 하고 수익은
    뒷사람이 받는다. 그래서 이 경로는 '미루기' 가 아니다 (spec 3.2).
    성장이 소득과 facility_eff 에 모두 곱해지므로 진척이 m^2 로 늘어난다.
    """
    def pol(ctx):
        if ctx["t"] < split:
            return (0.0, 0.9, 0.0, 0.0)
        return (0.0, 0.0, 0.9 * (1 - w), 0.9 * w)
    return pol


def _min_w(c: Cfg, make_policy, seeds: int, step: float) -> float:
    w = 0.0
    while w <= 1.0 + 1e-9:
        rng = random.Random(f"w{w:.2f}")
        runs = [simulate(c, make_policy(w), 1.0, rng)[0] for _ in range(seeds)]
        if sum(1 for x in runs if x >= 1.0) / seeds >= 0.5:
            return w
        w += step
    return 1.5


def required_w(c: Cfg, seeds: int = 8, step: float = 0.05) -> float:
    """조율이 완벽할 때 살아남는 데 필요한 최소 기여율. 1.0 초과 = 도달 불가.

    coord=1.0 이므로 **요격기 부지가 하나로 모인** 전제다. 요격기는 부지별로 독립이고
    판정이 max 이므로(spec 2.4), 조율이 깨지면 w* 를 넘겨도 못 짓는다.

    ⚠ 이것은 **성장을 쓰지 않는 경로**의 값이다. 국가 투자를 먼저 하면 더 적게 내도
      되므로(required_w_growth), 세계 난이도의 보수적 상한으로 읽어야 한다.
    """
    return _min_w(c, contrib_policy, seeds, step)


def required_w_growth(c: Cfg, seeds: int = 6, step: float = 0.05,
                      splits=(0, 5, 10, 15, 20, 25)) -> tuple[float, int]:
    """성장을 최적으로 태울 때의 필요 기여율과 그 배분점. (w*, split)

    ★C 상한이 성장을 빼고 계산되므로 실제 용량은 C 보다 크다. 이 함수가 그 차이를
    수치로 드러낸다.
    """
    best = (1.5, -1)
    for sp in splits:
        w = _min_w(c, lambda ww, sp=sp: growth_then_build(ww, sp), seeds, step)
        if w < best[0]:
            best = (w, sp)
    return best


def evaluate(c: Cfg, seeds: int, coords=(0.2, 0.9)):
    res, dth = {}, []
    for cd in coords:
        per_pol = []
        for name, pol in POLICIES.items():
            rng = random.Random(f"{name}|{cd}")
            runs = [simulate(c, pol, cd, rng) for _ in range(seeds)]
            per_pol.append(sum(1 for x, _ in runs if x >= 1.0) / seeds)
            dth += [d for _, d in runs]
        res[cd] = per_pol
    lo, hi = res[min(coords)], res[max(coords)]
    mean_lo, mean_hi = sum(lo) / len(lo), sum(hi) / len(hi)
    spread = sum((x - mean_hi) ** 2 for x in hi) / len(hi)
    return {
        "sensitivity": mean_hi - mean_lo,
        "mid": (mean_lo + mean_hi) / 2,
        "policy_var": spread ** 0.5,
        "lo": mean_lo, "hi": mean_hi,
        "deaths": sum(dth) / len(dth),
        "by_policy_hi": dict(zip(POLICIES, hi)),
    }


# ── 스윕 ───────────────────────────────────────────────────────────
GRID = {
    "income":         [200],          # 한 해 용량 = (1.0/0.2)회 × unit 40
    # **config 를 따라간다** (#52). 50 · 10 · 8.26 이 남아 있었다 — 세계는 8/21 에 60해로,
    # 수명은 8/19 에 두 배(16.52)로 갔다. 낡은 격자로 훑으면 **지금 없는 세계**에서 후보를
    # 고르게 된다. 확정 뒤의 재검증은 `verify_config.py` 가 맡는다.
    "total_turns":    [60],           # 확정 (configs/base.yaml · world.total_turns)
    "epoch_turns":    [20],           # 확정. total_turns / 3 (기대수명 반올림이 아니다)
    "success_prob":   [0.3, 0.5, 0.7],
    "growth_coef":    [0.2],   # 0.3 은 유치국 혼자 91% 에 닿아 ★B 여유가 8.7% 뿐
    "surv_k":         [8],
    "surv_lambda":    [16.52],        # 왕복 하나에 두 턴이 든다 (8/19 에 8.26 에서 두 배)
    "wellness_gain":  [0.008],
    "facility_eff":   [1.0],
}
# initial_budget 은 사망마다 새로 지급되므로 **사망이 돈을 찍어낸다.**
# 0 과 50 을 함께 훑어 그 크기를 본다.
INITIAL_BUDGET = [0, 50]
GROWTH_SCALE_GENS = [0.5]        # 한 주기 국가 총소득의 배수
BUNKER_DEPTH_EPOCHS = [1.0, 2.0, 3.0]
INTERCEPT_POS = [0.15, 0.35, 0.55, 0.85, 0.95]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    ok, rejected = [], {}
    for vals in combos:
        d = dict(zip(keys, vals))
        epoch_income = d["income"] * 3 * d["epoch_turns"]
        for ib, gsg, bd, pos in itertools.product(
                INITIAL_BUDGET, GROWTH_SCALE_GENS, BUNKER_DEPTH_EPOCHS, INTERCEPT_POS):
            dd = dict(d, initial_budget=ib, growth_scale=epoch_income * gsg)
            probe = Cfg(interceptor=1, bunker=1, **dd)
            k = probe.facility_eff * probe.success_prob
            A, B, C, E = bounds(probe)
            lo = max(A, B, E)
            c = Cfg(**dict(dd, bunker=epoch_income * k * bd,
                           interceptor=lo + (C * POLICY_COEF - lo) * pos))
            good, why = passes_asserts(c)
            if not good:
                rejected[why.split("(")[0].strip()] = \
                    rejected.get(why.split("(")[0].strip(), 0) + 1
                continue
            ok.append(c)

    tried = (len(combos) * len(INITIAL_BUDGET) * len(GROWTH_SCALE_GENS)
             * len(BUNKER_DEPTH_EPOCHS) * len(INTERCEPT_POS))
    el = expected_life(GRID["surv_lambda"][0], GRID["surv_k"][0])
    print(f"세대 경계 없음 · {GRID['total_turns'][0]}턴 · 기대수명 {el:.1f}턴")
    print(f"후보 {tried}개 / assert 통과 {len(ok)}개")
    for kk, v in sorted(rejected.items(), key=lambda x: -x[1]):
        print(f"   {v:>4}  {kk}")
    if not ok:
        return

    scored = []
    for c in ok:
        m = evaluate(c, a.seeds)
        m["w_star"] = required_w(c)
        m["w_star_growth"], m["split"] = required_w_growth(c)
        scored.append((c, m))
    scored.sort(key=lambda x: (abs(x[1]["w_star"] - W_TARGET),
                               -x[1]["sensitivity"], x[1]["policy_var"]))

    print(f"\n{'='*104}")
    print(f"필요 기여율 w* 가 목표 {W_TARGET} 에 가까운 순")
    print(f"{'='*104}")
    print(f"{'p':>5}{'초기예산':>9}{'벙커깊이':>9}{'임계위치':>9}{'사망수':>8}  "
          f"{'w*':>6}{'민감도':>8}{'저조율':>7}{'고조율':>7}{'정책편차':>9}")
    for c, m in scored[:a.top]:
        kk = c.facility_eff * c.success_prob
        A, B, C, E = bounds(c)
        lo = max(A, B, E)
        pos = (c.interceptor - lo) / (C * POLICY_COEF - lo)
        epoch_prog = c.income * c.agents * c.epoch_turns * kk
        ws = "불가" if m["w_star"] > 1.0 else f"{m['w_star']:.2f}"
        print(f"{c.success_prob:>5.1f}{c.initial_budget:>9.0f}"
              f"{c.bunker/epoch_prog:>9.1f}{pos:>9.2f}{m['deaths']:>8.1f}  "
              f"{ws:>6}{m['sensitivity']:>8.2f}{m['lo']:>7.2f}{m['hi']:>7.2f}"
              f"{m['policy_var']:>9.2f}")

    best, bm = scored[0]
    print(f"\n{'='*104}\n추천 조합 — 정책별 회피율 (조율률 0.9)\n{'='*104}")
    for kk, v in bm["by_policy_hi"].items():
        print(f"  {kk:<12} {v:>5.2f}")
    A, B, C, E = bounds(best)
    kk = best.facility_eff * best.success_prob
    epoch_prog = best.income * best.agents * best.epoch_turns * kk
    print(f"\n  창 : A {A:.0f}  B {B:.0f}  E {E:.0f}  <  임계 {best.interceptor:.0f}"
          f"  <  C×{POLICY_COEF} {C*POLICY_COEF:.0f}")
    print(f"  현실 투자율에서 필요한 주기 수 : "
          f"{best.interceptor/(3*epoch_prog*POLICY_COEF):.1f} / "
          f"{best.total_turns//best.epoch_turns}")
    print(f"  벙커 : 한 주기 전력 → 생존 {1-math.exp(-epoch_prog/best.bunker):.0%}, "
          f"전 기간 → {1-math.exp(-epoch_prog*best.total_turns/best.epoch_turns/best.bunker):.0%}")
    print(f"  런당 평균 사망 {bm['deaths']:.1f}회 "
          f"(= 초기예산 {best.initial_budget:.0f} × {bm['deaths']:.0f} 만큼 돈이 새로 생김)")
    print("\n" + json.dumps(asdict(best), indent=2))


if __name__ == "__main__":
    main()
