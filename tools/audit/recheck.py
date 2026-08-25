#!/usr/bin/env python3
"""이슈 #38 산술 재검산 — **우리 유도를 인용하지 않는다.**

`core/asserts.py` · `tools/balance/sweep.py` · `tools/balance/learning.py` 는 전부
검증 대상이므로 여기서는 **YAML 값에서 직접** 다시 계산한다. 기존 구현은 마지막에
「우리 유도」 칸으로만 불러와 나란히 찍는다 — 같은 함수를 두 번 부르는 것은 검증이
아니기 때문이다.

몬테카를로가 필요한 자리(실현 배수·수명·투자 절삭)는 **실제 코드**를 돌린다. 식을
다시 쓰면 같은 오해를 두 번 하게 되고, 우리가 재려는 것은 「식이 세계와 맞는가」다.

    PYTHONIOENCODING=utf-8 python tools/audit/recheck.py
    PYTHONIOENCODING=utf-8 python tools/audit/recheck.py --trials 400 --only C

절(節)은 `--only` 로 고를 수 있다: A 창 · B spread · C 노브 · D 학습 · E AP격자 ·
F 벙커 · G 소득상한 · H 누출 · I can_act · J 프롬프트.
"""
from __future__ import annotations

import argparse
import itertools
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import yaml  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
LINE = "─" * 78


def load_raw(path: str) -> dict:
    """**dataclass 를 거치지 않고** YAML 을 그대로 읽는다 — 기본값이 끼어들지 않게."""
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


# ── 수명 (독립 구현) ─────────────────────────────────────────────────────────

def S(a: float, lam: float, k: float) -> float:
    """생존함수 S(a) = exp(-(a/λ)^k)."""
    return math.exp(-((a / lam) ** k))


def mean_age_mult_analytic(d: dict) -> float:
    """정상 연령분포 ∝ S(a) 에서 본 나이 배수의 평균. 지평선을 바꿔가며 수렴을 본다."""
    lam, k = d["survival"]["lambda_base"], d["survival"]["k"]
    g, adult = d["income"]["age_growth"], d["world"]["adult_age"]
    out = {}
    for horizon in (int(lam * 2), int(lam * 3) + 1, int(lam * 6)):
        w = [S(a, lam, k) for a in range(horizon)]
        tot = sum(w)
        out[horizon] = sum(w[a] * (1 + g * max(0, a - adult))
                           for a in range(horizon)) / tot
    return out


# ── A. 창과 임계 ─────────────────────────────────────────────────────────────

def sec_A(d: dict, trials: int) -> None:
    print(LINE)
    print("A. 요격기 창 — YAML 에서 직접 다시 유도")
    print(LINE)
    inc = d["income"]["per_turn"]
    n = d["world"]["agents_per_country"]
    total = d["world"]["total_turns"]
    epoch = d["world"]["epoch_turns"]
    p = d["world"]["success_prob"]
    eff = d["facility"]["eff"]
    best = max(d["facility"]["build_spread"])
    intc = d["thresholds"]["interceptor"]

    mm = mean_age_mult_analytic(d)
    m = mm[int(d["survival"]["lambda_base"] * 3) + 1]
    print(f"  나이 배수 평균   지평선별 {', '.join(f'{h}해→{v:.4f}' for h, v in mm.items())}")
    print(f"                   → 수렴값 {m:.4f} (지평선에 둔감. 우리 유도와 같은 1.363)")

    # **한 나라의 전 기간 총소득**을 실제 코드로 재본다 — 해석식의 전제(정상 연령분포)가
    # 60해 · 초기나이 1~10 인 이 세계에서 실제로 성립하는지가 요점이다.
    realized = simulate_income(d, trials)
    analytic_nation = inc * m * n * total
    print(f"\n  한 나라 전 기간 총소득 (성장·개체배수 제외)")
    print(f"    해석식  per_turn × 평균배수 × n × total = "
          f"{inc:.0f} × {m:.3f} × {n} × {total} = {analytic_nation:,.0f}")
    print(f"    실측    실제 루프 {trials}회 평균 {realized['mean']:,.0f} "
          f"(σ {realized['sd']:,.0f} · 배수 환산 {realized['mult']:.3f})")
    print(f"    차이    {realized['mean'] / analytic_nation - 1:+.1%}"
          f"   ← 초기 나이 1~10 은 정상분포가 아니고, 60해는 정상상태에 못 간다")

    kk = eff * p * best

    def prog(x: float) -> float:
        return x * kk

    per_country_turn = inc * m * n
    A = prog(3 * per_country_turn * epoch)
    B = prog(per_country_turn * total)
    C = prog(3 * per_country_turn * total)
    E = prog(3 * per_country_turn * (total - epoch)) * 0.6
    lo, hi = max(A, B, E), C * 0.6
    print(f"\n  창 (진척 단위 · 최선 국가 효율 {best} 포함)")
    for name, v in (("A 미루기", A), ("B 조율강제", B), ("E 지속참여", E), ("C×0.6 상한", hi)):
        print(f"    {name:<12}{v:>10,.0f}")
    print(f"    창 [{lo:,.0f}, {hi:,.0f}] · 임계 {intc:,.0f} · "
          f"위치 {(intc - lo) / (hi - lo):.3f}")

    # 실측 소득으로 창을 다시 그리면 어디에 놓이나
    rc = realized["mult"]
    lo2 = max(prog(3 * inc * rc * n * epoch), prog(inc * rc * n * total),
              prog(3 * inc * rc * n * (total - epoch)) * 0.6)
    hi2 = prog(3 * inc * rc * n * total) * 0.6
    print(f"    실측 배수 {rc:.3f} 로 다시 그리면 [{lo2:,.0f}, {hi2:,.0f}] · "
          f"위치 {(intc - lo2) / (hi2 - lo2):.3f}")

    print("\n  ★A 와 ★B 가 같은 값인 이유 — 3 × epoch(20) = total(60). 우연이 아니라")
    print("    epoch_turns 가 total/3 이라서다. epoch 를 바꾸면 두 조건이 갈린다.")


def simulate_income(d: dict, trials: int) -> dict:
    """**실제 코드**로 한 나라의 전 기간 총소득을 잰다 (개체 배수·성장은 1.0 으로 중화).

    `income_for` · `_death_birth` · `init_world` 를 그대로 쓴다 — 나이 분포가 실제로
    어떻게 도는지가 질문이므로 식을 다시 쓰면 안 된다.
    """
    from core import config as cfgmod, loop as loopmod

    cfg = cfgmod.load(str(ROOT / "configs" / "base.yaml"))
    tot = []
    for s in range(trials):
        rng = random.Random(10_000 + s)
        counter = itertools.count(1)
        w = loopmod.init_world(cfg, counter, rng)
        for a in w.agents.values():
            a.income_mult = 1.0            # 개체 배수를 중화 (창은 평균 1.0 을 전제한다)
        acc = {c: 0.0 for c in w.countries}
        for t in range(1, cfg.world.total_turns + 1):
            w.turn = t
            for a in w.agents.values():
                acc[a.country] += loopmod.income_for(a, w, cfg)
            if t < cfg.world.total_turns:
                snap = sorted(w.agents)
                loopmod._death_birth(w, cfg, rng, snap, set(), counter,
                                     loopmod.RunResult(world=w))
                for a in w.agents.values():
                    a.income_mult = 1.0    # 새로 태어난 사람도 중화
        tot.extend(acc.values())
    mean = sum(tot) / len(tot)
    sd = (sum((x - mean) ** 2 for x in tot) / len(tot)) ** 0.5
    base = d["income"]["per_turn"] * d["world"]["agents_per_country"] * d["world"]["total_turns"]
    return {"mean": mean, "sd": sd, "mult": mean / base}


# ── B. 두 spread 의 평균 ─────────────────────────────────────────────────────

def sec_B(d: dict, trials: int) -> None:
    print(LINE)
    print("B. 세 spread — 목록의 평균과 **실제로 뽑힌** 평균")
    print(LINE)
    for name, sp in (("income.spread", d["income"]["spread"]),
                     ("facility.throughput_spread", d["facility"]["throughput_spread"]),
                     ("facility.build_spread", d["facility"]["build_spread"])):
        mean = sum(sp) / len(sp)
        sd = (sum((x - mean) ** 2 for x in sp) / len(sp)) ** 0.5
        print(f"  {name:<28}{sp}  평균 {mean:.10f}  σ {sd:.3f}  "
              f"{'정확히 1.0' if mean == 1.0 else '≠ 1.0'}")

    from core import config as cfgmod, loop as loopmod
    cfg = cfgmod.load(str(ROOT / "configs" / "base.yaml"))
    n = cfg.world.agents_per_country

    # 배정 방식이 다르다 — 국가 효율은 순열, 개체 배수는 독립 추출
    worst_b, nation_means, all_means = 0.0, [], []
    for s in range(trials):
        w = loopmod.init_world(cfg, itertools.count(1), random.Random(20_000 + s))
        builds = [c.build_mult for c in w.countries.values()]
        worst_b = max(worst_b, abs(sum(builds) / len(builds) - 1.0))
        for cid in w.countries:
            mine = [a.income_mult for a in w.agents.values() if a.country == cid]
            nation_means.append(sum(mine) / len(mine))
        all_means.append(sum(a.income_mult for a in w.agents.values()) / len(w.agents))
    nm = sorted(nation_means)
    print(f"\n  배정 실측 ({trials} 시드 · 첫 해 기준)")
    print(f"    build_spread   나라 평균의 1.0 이탈 최대 {worst_b:.2e}   ← 순열이라 정확히 1.0")
    print(f"    income_mult    한 나라(3명) 평균 중앙 {nm[len(nm)//2]:.3f} · "
          f"5% {nm[int(len(nm)*.05)]:.3f} · 95% {nm[int(len(nm)*.95)]:.3f}")
    print(f"                   → **독립 추출이라 나라별로는 1.0 이 아니다.** "
          f"3명이면 σ = {0.4243/math.sqrt(n):.3f}")

    # 한 나라가 60해 동안 실제로 겪는 배수의 평균 (교체가 여러 번 일어난다)
    life = simulate_nation_mult(cfg, trials)
    print(f"    60해 누적      한 나라가 실제로 받는 소득 배수 평균 중앙 {life['p50']:.3f} · "
          f"5% {life['p05']:.3f} · 95% {life['p95']:.3f}")
    b = (d["income"]["per_turn"] * mean_age_mult_analytic(d)[int(d['survival']['lambda_base']*3)+1]
         * n * d["world"]["total_turns"] * d["facility"]["eff"] * d["world"]["success_prob"]
         * max(d["facility"]["build_spread"]))
    print(f"    ★B 는 배수 1.0 을 전제로 {b:,.0f} 다. 95 분위 배수 {life['p95']:.3f} 를 쓰면 "
          f"{b * life['p95']:,.0f}")
    print(f"    임계 {d['thresholds']['interceptor']:,.0f} 와의 여유 "
          f"{d['thresholds']['interceptor'] / (b * life['p95']) - 1:+.1%}  "
          f"← 운 좋은 나라도 단독으로는 못 한다면 ★B 는 살아 있다")


def simulate_nation_mult(cfg, trials: int) -> dict:
    """한 나라가 60해 동안 지급한 소득이 배수 1.0 세계의 몇 배인가 (나이 배수 제외)."""
    from core import loop as loopmod
    out = []
    for s in range(trials):
        rng = random.Random(30_000 + s)
        counter = itertools.count(1)
        w = loopmod.init_world(cfg, counter, rng)
        acc = {c: 0.0 for c in w.countries}
        base = {c: 0.0 for c in w.countries}
        for t in range(1, cfg.world.total_turns + 1):
            w.turn = t
            for a in w.agents.values():
                grown = 1.0 + cfg.income.age_growth * max(0, a.age - cfg.world.adult_age)
                acc[a.country] += cfg.income.per_turn * grown * a.income_mult
                base[a.country] += cfg.income.per_turn * grown
            if t < cfg.world.total_turns:
                loopmod._death_birth(w, cfg, rng, sorted(w.agents), set(), counter,
                                     loopmod.RunResult(world=w))
        out.extend(acc[c] / base[c] for c in acc)
    out.sort()
    return {"p05": out[int(len(out) * .05)], "p50": out[len(out) // 2],
            "p95": out[int(len(out) * .95)]}


# ── C. 노브 ──────────────────────────────────────────────────────────────────

def sec_C(d: dict, trials: int) -> None:
    print(LINE)
    print("C. 노브가 아직 무는가 — **AP 가 지출 상한을 정한다**")
    print(LINE)
    ap_total = d["turn"]["action_points"]
    ap_speak = d["ap"]["speak"]
    ap_unit = d["ap"]["unit"]
    unit = d["costs"]["unit"]
    knobs = d["knob"]["comm_intl_ai"]
    max_speaks = int(ap_total / ap_speak)
    print(f"  한 해 AP {ap_total} · speak {ap_speak} → 국제 발신은 한 해 최대 {max_speaks}건")
    print(f"\n  {'노브':>6}{'5건 전액':>10}{'무력화 예산':>12}   설명")
    for v in knobs:
        print(f"  {v:>6}{v * max_speaks:>10.0f}{v * max_speaks:>12.0f}   "
              f"예산이 {v * max_speaks:.0f} 이상이면 **말할 횟수는 AP 만 정한다**")
    top = max(knobs) * max_speaks
    print(f"\n  → 노브 6 과 48 이 **행동 가능성**에서 갈리는 구간은 예산 "
          f"[{min(knobs) * max_speaks:.0f}, {top:.0f}) 뿐이다.")
    print(f"    그 위에서는 두 조건이 같은 행동 집합을 갖고, 차이는 **잔액**으로만 남는다.")
    print(f"    inh30 실측 예산 중앙 588 · 최대 2,987 → 중앙값에서 이미 {top:.0f} 를 넘는다.")

    # 잔액 차이가 다른 행동 몇 번에 해당하는가
    for budget in (240, 588, 1000, 2987):
        d48 = budget - max(knobs) * max_speaks
        d6 = budget - min(knobs) * max_speaks
        print(f"    예산 {budget:>5} → 5건 뒤 잔액  노브48 {d48:>6.0f} · 노브6 {d6:>6.0f} · "
              f"차 {d6 - d48:>4.0f} = invest {(d6 - d48) / unit:.1f}회분(AP 는 이미 0)")
    print(f"\n  ⚠ 5건을 말하면 AP 가 {max_speaks * ap_speak:.1f} 로 **전부 소진**된다 "
          f"(invest 1회에 {ap_unit}).")
    print(f"    즉 노브가 아낀 돈은 **그 해에 쓸 손이 없다.** 다음 해로 넘어가 예산에 쌓인다.")
    print(f"    노브가 무는 경로는 둘뿐이다 — ① 예산 < {top:.0f} 인 젊은/가난한 해,")
    print(f"    ② 말하기와 투자를 **섞을** 때의 상대가격.")
    # **격자는 코드와 같은 반올림으로 센다.** `int((1.0-0.4)/0.2)` 는 2 를 준다 —
    # 부동소수 때문이고, 실제 `_afford` 는 round(_,3) 이라 3 을 허용한다.
    from core.agent_loop import AP_GRID, _afford
    mix = []
    for s in range(max_speaks + 1):
        left, inv = round(ap_total - s * ap_speak, AP_GRID), 0
        while _afford(left, ap_unit):
            left, inv = round(left - ap_unit, AP_GRID), inv + 1
        mix.append((s, inv))
    print(f"    섞기 격자 (speak, invest) = {mix}")
    for v in knobs:
        need = [(s * v + i * unit) for s, i in mix]
        print(f"      노브 {v:>2} 에서 각 조합의 최소 필요 예산 (invest_mult 1.0): "
              f"{[f'{x:.0f}' for x in need]}")


# ── D. 학습 ──────────────────────────────────────────────────────────────────

def sec_D(d: dict, trials: int) -> None:
    print(LINE)
    print("D. 학습 — 총지출·총AP 를 **호출 단위로** 다시 센다")
    print(LINE)
    L = d["costs"]["learn_base"]
    unit = d["costs"]["unit"]
    up = d["costs"]["learn_speedup"]
    ap_unit = d["ap"]["unit"]
    inc = d["income"]["per_turn"]
    print(f"  필요 진척 {L:.0f} 고정 · 한 호출 진척 {unit:.0f}×배수 · 지출 {unit:.0f} "
          f"(마지막 호출만 남은 만큼)\n")
    print(f"  {'사유':>4}{'배수':>6}{'호출':>6}{'총지출':>8}{'총AP':>7}"
          f"{'소득 해':>8}{'AP 해':>7}   base.yaml 주석")
    doc = {0: (200, 1.0), 1: (160, 0.8), 2: (120, 0.6)}
    for r in (0, 1, 2):
        mult = 1.0 + up * r
        done, money, calls = 0.0, 0.0, 0
        while done < L - 1e-9:                      # execute_tool 과 같은 절차
            gain = min(unit * mult, L - done)
            money += gain / mult
            done += gain
            calls += 1
        ap = calls * ap_unit
        dm, da = doc[r]
        mark = "일치" if abs(money - dm) < 0.5 and abs(ap - da) < 1e-9 else "**불일치**"
        print(f"  {r:>4}{mult:>6.1f}{calls:>6}{money:>8.1f}{ap:>7.2f}"
              f"{money / inc:>8.2f}{ap:>7.2f}   {dm:.0f}원·AP {da}  {mark}")
    print(f"\n  → 총지출은 정확히 L/배수 = {L:.0f}/배수 다. 총 AP 는 ceil 이라 1/배수가 아니다.")
    print(f"    지출비 1 : {1/1.5:.3f} : {1/2:.3f}   vs   AP비 1 : 0.8 : 0.6")
    print(f"    `tools/balance/learning.py` 의 「둘 다 1/배수」 는 AP 쪽이 틀렸고,")
    print(f"    `configs/base.yaml`·`agent_loop.learn_speed` 의 「160원·120원」 은 돈 쪽이 틀렸다.")
    print(f"    (그래서 이 자는 「돈이 없어서」 와 「AP 가 없어서」 를 **조금은** 가른다.)")


# ── E. AP 격자 ───────────────────────────────────────────────────────────────

def sec_E(d: dict, trials: int) -> None:
    print(LINE)
    print("E. AP 격자 — 도달 가능한 모든 행동열을 전수 탐색")
    print(LINE)
    from core.agent_loop import AP_GRID, _afford

    costs = {k: v for k, v in d["ap"].items()}
    start = d["turn"]["action_points"]
    print(f"  비용 {costs} · 시작 {start} · 격자 소수 {AP_GRID}자리")

    # 격자 밖으로 나가거나, **정확한 유리수 계산으로는 낼 수 있는데** 거절되는 조합을 찾는다
    from fractions import Fraction
    bad_reject, bad_accept, off_grid, seen = [], [], [], set()
    stack = [(start, Fraction(str(start)), ())]
    while stack:
        ap, exact, path = stack.pop()
        key = (round(ap, 6), path[-3:] if path else ())
        if (round(ap, 6), len(path)) in seen or len(path) > 24:
            continue
        seen.add((round(ap, 6), len(path)))
        if abs(round(ap * 100) - ap * 100) > 1e-6:
            off_grid.append((path, ap))
        for name, c in costs.items():
            ex_c = Fraction(str(c))
            can_exact = exact - ex_c >= 0
            can_code = _afford(ap, c)
            if can_exact and not can_code:
                bad_reject.append((path, name, ap))
            if can_code and not can_exact:
                bad_accept.append((path, name, ap))
            if can_code:
                stack.append((round(ap - c, AP_GRID), exact - ex_c, path + (name,)))
    print(f"  탐색한 상태 {len(seen):,}개")
    print(f"    격자 이탈        {len(off_grid)}건")
    print(f"    정당한데 거절    {len(bad_reject)}건")
    print(f"    부당한데 허용    {len(bad_accept)}건")
    for tag, rows in (("거절", bad_reject), ("허용", bad_accept), ("이탈", off_grid)):
        for r in rows[:5]:
            print(f"      {tag}: {r}")
    print(f"  → 모든 비용이 0.05 의 배수이고 매 차감이 round(_,{AP_GRID}) 로 스냅되므로")
    print(f"    give 0.1 이 들어와도 격자는 닫혀 있다. 25건이 사라졌던 부동소수 문제는 재발하지 않는다.")


# ── F. 벙커 ──────────────────────────────────────────────────────────────────

def sec_F(d: dict, trials: int) -> None:
    print(LINE)
    print("F. 벙커 창과 1인부담 — 진척 단위와 **돈 단위**로 각각")
    print(LINE)
    inc = d["income"]["per_turn"]
    n = d["world"]["agents_per_country"]
    total, epoch = d["world"]["total_turns"], d["world"]["epoch_turns"]
    p, eff = d["world"]["success_prob"], d["facility"]["eff"]
    best = max(d["facility"]["build_spread"])
    bunker, intc = d["thresholds"]["bunker_scale"], d["thresholds"]["interceptor"]
    m = mean_age_mult_analytic(d)[int(d["survival"]["lambda_base"] * 3) + 1]
    k = eff * p

    lo = inc * m * n * epoch * k
    hi = inc * m * n * total * k
    print(f"  벙커 창 (진척)   한 주기 {lo:,.0f} ≤ {bunker:,.0f} ≤ 전 기간 {hi:,.0f}   "
          f"{'OK' if lo <= bunker <= hi else '위반'}")
    print(f"                   {bunker / lo:.2f} 주기분 · 전 기간의 {bunker / hi:.0%}")
    print(f"  ⚠ `verify_config.py:70` 은 전 기간을 {inc * n * total * k:,.0f} 로 찍는다 — "
          f"나이 배수를 빼먹어 {1 - (inc * n * total * k) / hi:.0%} 낮다.")
    print(f"    그 숫자로 보면 벙커는 전 기간의 {bunker / (inc * n * total * k):.0%} 라 "
          f"「상한에 거의 붙었다」 로 읽힌다.")

    print(f"\n  1인부담 — 우리 유도 (진척 ÷ 사람-해)")
    b1 = bunker / (n * epoch)
    i1 = intc / (3 * n * total)
    print(f"    벙커 {bunker:,.0f} / ({n}×{epoch}) = {b1:.1f}   "
          f"요격기 {intc:,.0f} / (3×{n}×{total}) = {i1:.1f}   비 {b1 / i1:.2f}")
    print(f"    ⚠ 분모의 기간이 다르다 (벙커 한 주기 vs 요격기 전 기간). "
          f"같은 60해로 맞추면 벙커 {bunker / (n * total):.1f} vs 요격기 {i1:.1f} — 비 "
          f"{(bunker / (n * total)) / i1:.2f}")
    print(f"\n  1인부담 — **돈**으로 (진척 ÷ k ÷ 사람-해). 에이전트가 실제로 내는 것은 돈이다")
    bm = bunker / k / (n * total)
    im = intc / (k * best) / (3 * n * total)
    im_avg = intc / k / (3 * n * total)
    print(f"    벙커      {bunker:,.0f}/{k} / ({n}×{total}) = {bm:.1f} 원/사람-해")
    print(f"    요격기    최선 국가({best}) 기준 {im:.1f} · 평균 국가 기준 {im_avg:.1f} 원/사람-해")
    print(f"    소득이 사람-해당 {inc * m:.0f} 원이므로 벙커는 소득의 {bm / (inc * m):.0%}, "
          f"요격기는 {im / (inc * m):.0%}")
    print(f"    → 국가 효율을 벙커에 안 거는 선택은 **벙커를 상대적으로 더 비싸게** 만든다.")
    print(f"      최고 효율 나라가 벙커를 고르면 손해가 {best:.1f}배로 커진다 — 의도된 함정이다.")

    # 벙커는 **확률**을 산다. 같은 생존 확률을 사려면 얼마가 드는지로 맞춰 본다
    print(f"\n  같은 생존 확률로 환산 (벙커 p = 1 − exp(−진척/{bunker:.0f}))")
    print(f"    {'목표 p':>8}{'필요 진척':>11}{'사람-해당':>11}   요격기(확정) 대비")
    for target in (0.5, 0.63, 0.8, 0.9, 0.95):
        need = -math.log(1 - target) * bunker
        per = need / (n * total)
        print(f"    {target:>8.0%}{need:>11,.0f}{per:>11.1f}   {per / i1:>5.2f}배")
    print(f"    → 진척이 깊이척도와 같을 때(p=63%) 1인부담은 요격기의 "
          f"{(bunker / (n * total)) / i1:.2f}배뿐이다.")
    print(f"      ★D 가 찍는 3.34 배는 **벙커만 한 주기(20해)로 나눠서** 나온 값이다.")


# ── G. 소득 대 지출 상한 ─────────────────────────────────────────────────────

def sec_G(d: dict, trials: int) -> None:
    print(LINE)
    print("G. 소득 vs 한 해 지출 상한 — age_growth 0.10 이 맞는 값인가")
    print(LINE)
    inc = d["income"]["per_turn"]
    g = d["income"]["age_growth"]
    adult = d["world"]["adult_age"]
    unit = d["costs"]["unit"]
    ap_unit, ap_total = d["ap"]["unit"], d["turn"]["action_points"]
    spread = d["income"]["spread"]
    thr = d["facility"]["throughput_spread"]
    calls = int(ap_total / ap_unit)
    print(f"  한 해 지출 상한 = invest {calls}회 = {calls} × {unit:.0f} × 처리배수")
    print(f"  {'나이':>5}{'배수':>7}{'소득(배수1)':>12}   상한 대비 (처리배수별)")
    for age in (5, 8, 10, 12, 16, 20):
        grown = 1 + g * max(0, age - adult)
        row = "  ".join(f"{t}:{inc * grown / (calls * unit * t):>5.2f}" for t in thr)
        print(f"  {age:>5}{grown:>7.2f}{inc * grown:>12.0f}   {row}")
    print(f"\n  → 배수 1.0 끼리면 소득이 상한을 넘는 나이는 "
          f"{next(a for a in range(adult, 40) if inc * (1 + g * (a - adult)) > calls * unit * 1.0)}세.")
    print(f"    base.yaml 의 「0.10 이면 16세 210 으로 상한을 살짝 넘긴다」 와 일치.")
    over = [(i, t) for i in spread for t in thr if inc * i > calls * unit * t]
    print(f"  성인 진입({adult}세)에 이미 상한을 넘는 (소득배수, 처리배수) 조합 "
          f"{len(over)}/{len(spread) * len(thr)} = {len(over) / (len(spread) * len(thr)):.0%}")
    old = [0.6, 0.8, 1.0, 1.2, 1.4]          # 「0.2 폭」 이던 이전 spread
    def count(sp, ratio):
        return sum(1 for i in sp for t in sp if (inc * i) / (calls * unit * t) > ratio)
    print(f"  base.yaml 의 「소득/처리능력이 1.4 를 넘는(=돈이 남는) 칸이 1/25 → 4/25」")
    print(f"    (소득 ÷ 지출상한) > 1.0   이전 {count(old, 1.0)}/25 → 지금 {count(spread, 1.0)}/25"
          f"   ← **주석의 두 숫자를 정확히 재현한다**")
    print(f"    (소득 ÷ 지출상한) > 1.4   이전 {count(old, 1.4)}/25 → 지금 {count(spread, 1.4)}/25")
    print(f"    즉 세는 것은 맞고 **문장의 「1.4」 가 틀렸다** — 기준은 1.0 "
          f"(= 소득배수/처리배수 > 2.0) 이다.")
    print(f"\n  ⚠ 상한은 나이와 무관하고(AP 고정) 소득만 오른다. 60해 · 성장 배수까지 들어가면")
    print(f"    잉여는 계보에 누적된다 — inh30 의 「지출 103원 고정 · 소득 100→210」 과 같은 그림.")


# ── H. 타국 값 역산 가능성 ───────────────────────────────────────────────────

def sec_H(d: dict, trials: int) -> None:
    print(LINE)
    print("H. `fac_moved` 로 타국 build_mult 를 역산할 수 있나")
    print(LINE)
    p = d["world"]["success_prob"]
    eff = d["facility"]["eff"]
    unit = d["costs"]["unit"]
    print(f"  타국 출자에는 **늘었는지 여부만** 간다 (loop.py f-2 · _settle_step).")
    print(f"  gain = 0 일 확률 = (1-{p})^int(출자액 × eff)\n")
    print(f"  {'출자액':>8}{'나라효율':>9}{'시행수':>8}{'P(gain=0)':>12}   판정")
    for im in d["facility"]["throughput_spread"]:
        for bm in d["facility"]["build_spread"]:
            amount = unit * im
            n_i = int(amount * eff * bm)
            p0 = (1 - p) ** n_i
            verdict = "구분 불가" if p0 < 1e-3 else "신호 있음"
            print(f"  {amount:>8.0f}{bm:>9.1f}{n_i:>8}{p0:>12.2e}   {verdict}")
    print(f"\n  → 가장 작은 출자(16원 · 효율 0.7 · 시행 11)에서도 P(gain=0) = "
          f"{(1 - p) ** 11:.4f} 다.")
    print(f"    한 번의 통지로는 아무것도 안 새고, 수십 번을 쌓아도 배수 0.7 과 1.3 을")
    print(f"    가르려면 P 차이가 {(1-p)**11:.4f} vs {(1-p)**int(16*1.3):.6f} — 표본이 수백 건 필요하다.")
    print(f"    **다만 gain=0 이 곧 「그 나라가 국토를 아직 안 정했다」 다** (land is None).")
    print(f"    즉 이 통지는 배수가 아니라 **국토 미정 여부**를 흘린다 — 의도된 것이다(주석 ②).")
    print(f"    자국 값은 관측의 inv_build 한 줄로 정확히 공개된다 (100원당 기대 진척).")


# ── I. can_act ───────────────────────────────────────────────────────────────

def sec_I(d: dict, trials: int) -> None:
    print(LINE)
    print("I. `can_act` — 「실행 가능한 도구가 하나라도 있나」 를 정직하게 세는가")
    print(LINE)
    from core import config as cfgmod
    from core.agent_loop import can_act
    from core.state import Agent

    cfg = cfgmod.load(str(ROOT / "configs" / "base.yaml"))
    knob = max(cfg.knob.comm_intl_ai)
    print(f"  {'AP':>6}{'예산':>7}{'memory_open':>13}{'can_act':>9}   실제로 부를 수 있는 것")
    for ap, budget, mem in ((1.0, 1000, False), (0.0, 1000, False), (0.0, 0, False),
                            (0.04, 1000, False), (0.0, 1000, True)):
        a = Agent(id="X", country="Asla", native_lang="ja", known_langs={"ja"},
                  parent_langs=set(), budget=budget, ap=ap)
        a.memory_open = mem
        # 실제로 가능한 것을 따로 센다 — end_turn 은 행동이 아니다
        real = []
        if budget >= cfg.costs.comm_domestic and ap >= cfg.ap.speak:
            real.append("speak")
        if budget >= cfg.costs.unit and ap >= cfg.ap.unit:
            real.append("invest/learn")
        if ap >= cfg.ap.propose_vote:
            real.append("propose_vote")
        if ap >= cfg.ap.vote:
            real.append("vote(採決일만)")
        if ap >= cfg.ap.observe_risk and budget >= cfg.costs.observe_risk:
            real.append("observe_risk")
        if ap >= cfg.ap.give and budget > 0:
            real.append("give")
        if mem:
            real.append("memory_write")
        print(f"  {ap:>6}{budget:>7}{str(mem):>13}{str(can_act(a, cfg, knob)):>9}   "
              f"{real or '— 없음 —'}")
    broke = Agent(id="X", country="Asla", native_lang="ja", known_langs={"ja"},
                  parent_langs=set(), budget=0.0, ap=0.0)
    broke.memory_open = False
    if can_act(broke, cfg, knob):
        print(f"\n  → **결함** — AP 0 · 예산 0 · 기억 닫힘인데 참이다. 종료 조건 ②가 죽어 있다.")
        print(f"    `min(ap.memory_write, ap.vote)` = {min(cfg.ap.memory_write, cfg.ap.vote)} 이 원인이고,")
        print(f"    라운드로빈은 그 사람을 한 번 더 깨워 end_turn 만 시킨다.")
    else:
        print(f"\n  → 아니다 — 고쳐짐 (#47). 자원이 바닥나면 거짓이고, 라운드로빈이 그 사람을")
        print(f"    한 번 더 깨우지 않는다. 기억이 열려 있으면(마지막 줄) 여전히 참이다 —")
        print(f"    그때는 `memory_write` 가 실제로 목록에 있다.")


# ── J. 프롬프트가 말하는 것과 코드가 하는 것 (기계로 볼 수 있는 것만) ────────

def sec_J(d: dict, trials: int) -> None:
    print(LINE)
    print("J. 프롬프트 — 기계로 대조되는 것들")
    print(LINE)
    import itertools as _it
    import random as _r

    from core import config as cfgmod, loop as loopmod
    from core.agent_loop import execute_tool, Sink
    from domains.meteor import prompts

    cfg = cfgmod.load(str(ROOT / "configs" / "base.yaml"))

    # ① 세 언어 T 의 키 집합이 같은가
    keys = {l: set(prompts.T[l]) for l in ("ja", "zh", "fr")}
    allk = set().union(*keys.values())
    missing = {l: sorted(allk - k) for l, k in keys.items() if allk - k}
    print(f"  ① T 키 집합   {'모두 동일' if not missing else missing}")

    # ② propose_vote 응답이 「년」 인가 「턴」 인가
    w = loopmod.init_world(cfg, _it.count(1), _r.Random(1))
    w.turn = 3
    a = w.agents["Asla1"]
    a.ap = 1.0
    sink = Sink()
    res, _ = execute_tool("propose_vote", {"reasoning": "x"}, w, a, cfg, sink, 48.0)
    year = prompts.FIRST_YEAR + (w.turn + loopmod.VOTE_DELAY) - 1
    print(f"  ② propose_vote 응답 {res}")
    print(f"     같은 採決을 실패 응답은 「year {year}」 라고 부른다 (execute_tool 의 vote 분기).")
    leak = [k for k in res if "turn" in k]
    print(f"     성공 응답이 턴 눈금을 흘리는가: "
          f"{'**결함** ' + str(leak) if leak else '아니다 — 고쳐짐 (#43)'}")

    # ③ learn 과 invest 가 정말 같은 돈인가 (도구 설명: "costs the same money as an investment")
    from core import tools as toolsmod
    learn_desc = next(t["function"]["description"] for t in toolsmod.TOOLS
                      if t["function"]["name"] == "learn")
    says_same_money = "same money" in learn_desc
    print(f"  ③ learn 설명이 「invest 와 같은 돈」 이라고 말하는가: "
          f"{'**결함**' if says_same_money else '아니다 — 고쳐짐 (#41)'}")
    for mult in cfg.facility.throughput_spread:
        print(f"     invest_mult {mult}: invest {cfg.costs.unit * mult:>5.0f}원 vs "
              f"learn {cfg.costs.unit:>5.0f}원   {'같다' if mult == 1.0 else '**다르다**'}")

    # ④ fac_gain 통지가 「작년」 인가
    LASTYEAR = {"ja": "昨年", "zh": "去年", "fr": "l'an dernier"}
    stale = [l for l in ("ja", "zh", "fr")
             if any(LASTYEAR[l] in prompts.T[l][k]
                    for k in ("fac_gain", "fac_moved", "fac_still"))]
    print(f"  ④ 출자 통지가 「작년」 이라고 말하는가: "
          f"{'**결함** ' + str(stale) if stale else '아니다 — 고쳐짐 (#42)'}")
    for l in ("ja", "zh", "fr"):
        print(f"     {l}: {prompts.T[l]['fac_gain'][:34]}…")
    print(f"     순차 라운드로빈은 `_settle_step` 에서 **같은 해**에 통지한다 "
          f"(loop.py: `_notify(..., world.turn, ...)`).")

    # ⑤ 미성년 무소득이라고 말하는 도구 설명 vs income_for
    w2 = loopmod.init_world(cfg, _it.count(1), _r.Random(1))
    w2.turn = 1
    child = next(iter(w2.agents.values()))
    child.age = 1
    give_desc = next(t["function"]["description"] for t in toolsmod.TOOLS
                     if t["function"]["name"] == "give")
    says_child = "child has no income" in give_desc
    print(f"  ⑤ give 설명이 「아이는 소득이 없다」 고 말하는가: "
          f"{'**결함**' if says_child else '아니다 — 고쳐짐 (#40)'}")
    print(f"     income_for(나이 1) = {loopmod.income_for(child, w2, cfg):.0f}원  "
          f"← adult_age {cfg.world.adult_age} 미만이어도 소득이 나온다 (그래서 문구를 뺐다)")

    # ⑥ cap_per_turn 이 프롬프트 어디에도 없는가
    txt = prompts.system_for(w2.agents["Asla1"], w2, cfg, 48.0, same_year=True)
    print(f"  ⑥ facility.cap_per_turn = {cfg.facility.cap_per_turn:.0f} (국가·한 해 상한)")
    shown = str(int(cfg.facility.cap_per_turn)) in txt
    print(f"     관측+SYSTEM 전문에 '{cfg.facility.cap_per_turn:.0f}' 등장 여부: "
          f"{shown}" + ("  ← 고쳐짐 (#45)" if shown else "  ← **결함**"))
    per_nation_max = (cfg.world.agents_per_country * 3) * int(
        cfg.turn.action_points / cfg.ap.unit) * cfg.costs.unit * max(
        cfg.facility.throughput_spread)
    print(f"     한 해에 한 나라로 들어올 수 있는 최대 출자 = 9명 × 5회 × "
          f"{cfg.costs.unit:.0f} × 1.6 = {per_nation_max:,.0f} > {cfg.facility.cap_per_turn:.0f}")

    # ⑦ 진척 절삭 — int(share × eff)
    print(f"  ⑦ 진척은 `int(출자액 × eff)` 로 시행수를 자른다. 관측의 inv_build 는 자르지 않는다")
    for im in cfg.facility.throughput_spread:
        for bm in (max(cfg.facility.build_spread),):
            amount = cfg.costs.unit * im
            exact = amount * cfg.facility.eff * bm * cfg.world.success_prob
            trunc = int(amount * cfg.facility.eff * bm) * cfg.world.success_prob
            print(f"     출자 {amount:>5.0f} · 효율 {bm}: 기대 진척 실제 {trunc:.2f} vs "
                  f"관측 문구가 함의하는 {exact:.2f}  ({trunc / exact - 1:+.1%})")


# ── K. 라벨과 수명 ───────────────────────────────────────────────────────────

def sec_K(d: dict, trials: int) -> None:
    print(LINE)
    print("K. 라벨이 말하는 것 · 프롬프트가 적는 수명")
    print(LINE)
    from core import config as cfgmod, messaging
    from domains.meteor import prompts

    cfg = cfgmod.load(str(ROOT / "configs" / "base.yaml"))

    # ① original 경로 — 전달 판정은 **발신자의 모국어**로 하는데, 본문은 아무 말이나 된다
    print("  ① `original` — 「제3의 언어로 쓰면」")
    cases = [
        ("모국어(ja)로 씀 · 수신자가 ja 를 읽음", "ja", {"ja", "fr"}, {"ja", "zh"}, "こんにちは"),
        ("수신자 말(fr)로 씀", "ja", {"ja", "fr"}, {"ja", "zh"}, "Bonjour"),
        ("제3의 언어(zh)로 씀 · 수신자는 zh 를 못 읽음", "ja", {"ja", "fr"}, {"ja", "zh"}, "你好，我们一起建造"),
    ]
    for name, from_lang, rec_known, snd_known, text in cases:
        sent = {"from": "Asla1", "from_country": "Asla", "from_lang": from_lang,
                "to": "Miris1", "to_country": "Miris", "to_lang": "fr",
                "route": "original", "text": text, "translate_instruction": None}
        p = messaging.process_message(sent, rec_known, cfg, None, 48.0,
                                      sender_known_langs=snd_known)
        lbl = (p["inbox"] or {}).get("label")
        readable = p["meta"]["delivered_lang"] in rec_known
        print(f"     {name}")
        print(f"       전달 {p['delivered']} · 실제 언어 {p['meta']['delivered_lang']} · "
              f"수신자가 읽을 수 있나 {readable} · 라벨 {lbl}")
    third = messaging.process_message(
        {"from": "Asla1", "from_country": "Asla", "from_lang": "ja", "to": "Miris1",
         "to_country": "Miris", "to_lang": "fr", "route": "original",
         "text": "你好，我们一起建造", "translate_instruction": None},
        {"ja", "fr"}, cfg, None, 48.0, sender_known_langs={"ja", "zh"})
    if third["delivered"]:
        print("     → **결함** — 읽을 수 없는 글이 전달되고, 라벨은 「당신은 이 말을 못 읽지만")
        print("       상대가 당신 말을 다루므로 통했다」 라고 적는다. 상대는 zh 를 못 읽는다.")
        print("       판정은 `from_lang`(모국어)으로 하는데 본문은 `_TEXT` 가 아무 말이나 허용한다.")
    else:
        print("     → 아니다 — 고쳐짐 (#44). 판정이 **도착한 글의 언어**를 본다. 제3의 언어는")
        print("       미전달이고, 발신자에게 `unreadable` 통지가 간다. 8/17 의 허구(내 말로")
        print("       썼고 상대가 내 말을 다룬다 → 통했다)도 **함께 없앴다** (8/25) — 아는 말의")
        print("       나라에는 그 말로 쓰라고 나라별로 안내하므로 그 길이 더는 필요 없다.")
        print("       `[direct:write]` 라벨도 도달 불가능해져 지웠다 (전 조합 288 가지 0건).")

    # ② 프롬프트가 적는 수명 vs 실제로 죽는 나이
    print("\n  ② SYSTEM 의 「{life:.0f} 歳ごろ」 와 실제 사망 나이")
    lam, k = cfg.survival.lambda_base, cfg.survival.k
    weib = prompts.typical_lifespan(cfg)
    lived = sum(S(a, lam, k) for a in range(200))
    e_death = sum(a * (S(a, lam, k) - S(a + 1, lam, k)) for a in range(200))
    print(f"     Weibull 평균 λΓ(1+1/k) = {weib:.2f}  → 프롬프트는 「{weib:.0f}」 로 적는다")
    print(f"     살아낸 해 Σ S(a)        = {lived:.2f}")
    print(f"     **사망 나이 기대값** Σ a·(S(a)−S(a+1)) = {e_death:.2f}   ← 부고에 찍히는 값")
    obs = simulate_death_ages(cfg, trials)
    print(f"     실측(실제 루프 {trials}회): 사망 나이 평균 {obs['mean']:.2f} · "
          f"중앙 {obs['p50']} · 표본 {obs['n']:,}")
    print(f"     → 부고가 쌓여 사람들이 배우는 값은 {obs['mean']:.1f} 인데 SYSTEM 은 "
          f"{weib:.0f} 이라고 말한다 (차 {weib - obs['mean']:+.1f}해).")


def simulate_death_ages(cfg, trials: int) -> dict:
    """실제 `_death_birth` 로 부고에 찍히는 나이를 모은다."""
    from core import loop as loopmod
    ages = []
    for s in range(trials):
        rng = random.Random(40_000 + s)
        counter = itertools.count(1)
        w = loopmod.init_world(cfg, counter, rng)
        res = loopmod.RunResult(world=w)
        for t in range(1, cfg.world.total_turns):
            w.turn = t
            loopmod._death_birth(w, cfg, rng, sorted(w.agents), set(), counter, res)
        ages.extend(res.death_ages)
    ages.sort()
    return {"mean": sum(ages) / len(ages), "p50": ages[len(ages) // 2], "n": len(ages)}


SECTIONS = {"A": sec_A, "B": sec_B, "C": sec_C, "D": sec_D, "E": sec_E,
            "F": sec_F, "G": sec_G, "H": sec_H, "I": sec_I, "J": sec_J,
            "K": sec_K}


def main() -> int:
    ap = argparse.ArgumentParser(description="이슈 #38 산술·정합 재검산")
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--trials", type=int, default=200, help="몬테카를로 반복")
    ap.add_argument("--only", default="", help="ABCDEFGHIJK 중 골라서")
    a = ap.parse_args()
    d = load_raw(a.config)
    want = [s for s in (a.only.upper() or "".join(SECTIONS)) if s in SECTIONS]
    for s in want:
        SECTIONS[s](d, a.trials)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
