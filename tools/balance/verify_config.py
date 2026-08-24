#!/usr/bin/env python3
"""확정된 config 하나를 Phase 0 의 자로 다시 잰다. spec 7 · 12.4.

`sweep.py` 는 **격자를 훑어 후보를 고르는** 도구입니다. 후보가 이미 정해진 뒤에는
질문이 달라집니다 — *"우리가 못 박은 이 값이 아직 창 안에 있는가, `w*` 는 목표에서
얼마나 벗어났는가."* 코드가 바뀌었으니 그걸 다시 재는 것이 Phase 0 재실행입니다.

**캘리브레이션 목표는 회피율이 아니라 `w*` 입니다** (spec 12.4). 50턴 × 턴당 수십 번의
베르누이라 대수의 법칙으로 진척이 사실상 결정적이 되고, 회피율은 `coord` 를 그대로
복사해 파라미터로 움직이지 않습니다. `w*` 는 임계 위치를 따라 연속으로 움직입니다.

    python3 tools/balance/verify_config.py
    python3 tools/balance/verify_config.py --seeds 40 --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml  # noqa: E402

from tools.balance.sweep import (  # noqa: E402
    POLICIES, POLICY_COEF, W_TARGET, Cfg, bounds, evaluate, expected_life,
    mean_age_multiplier, passes_asserts, required_w, required_w_growth,
)

# 무성장 0.50 / 성장 0.30 — 못 박을 때 실측한 값 (todo · RESULTS.md).
# 코드가 바뀌어 여기서 벗어나면 세계의 난이도가 조용히 달라진 것이다.
#
# **0.45 / 0.20 에서 갱신했다** (8/23). 그 값은 임계 9558 · 100해 · adult_age 10 시절
# 것이고, `adult_age` 를 5 로 내린 이틀 전부터 이미 낡아 있었다 — 이 도구를 안 돌려서
# 몰랐다. 「숫자를 두 군데 적으면 하나가 낡는다」 의 여섯 번째다.
#
# 지금 값은 임계 13206 (창 [11483, 17225] 의 0.30 지점) · 국가 효율 최선 1.3 기준이다.
EXPECTED = {"w_star": 0.50, "w_star_growth": 0.30}
TOL = 0.05


def cfg_from_yaml(path: Path) -> Cfg:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Cfg(
        income=d["income"]["per_turn"],
        initial_budget=d["income"]["initial_budget"],
        total_turns=d["world"]["total_turns"],
        epoch_turns=d["world"]["epoch_turns"],
        success_prob=d["world"]["success_prob"],
        agents=d["world"]["agents_per_country"],
        growth_coef=d["growth"]["growth_coef"],
        growth_scale=d["growth"]["growth_scale"],
        surv_k=d["survival"]["k"],
        surv_lambda=d["survival"]["lambda_base"],
        wellness_gain=d["wellness"]["gain"],
        interceptor=d["thresholds"]["interceptor"],
        bunker=d["thresholds"]["bunker_scale"],
        facility_eff=d["facility"]["eff"],
        age_growth=d["income"].get("age_growth", 0.0),
        adult_age=d["world"].get("adult_age", 10),
        build_best=max(d["facility"].get("build_spread") or [1.0]),
    )


def report(c: Cfg, seeds: int) -> int:
    A, B, C, E = bounds(c)
    lo, hi = max(A, B, E), C * POLICY_COEF
    k = c.facility_eff * c.success_prob
    # **두 줄이 같은 자를 써야 한다** (#48). 아래 줄에만 나이 배수가 빠져 있어서 전 기간
    # 진척을 5,400 으로 찍었다 — 실제 7,361 보다 27% 낮다. 그 수로 보면 벙커 4,900 이
    # 전 기간의 91% 라 「상한에 거의 붙었다」 로 읽히는데, 실제로는 67% 다.
    # `core.asserts.check_all` 은 배수를 넣어 재고 있었으므로 **두 구현이 갈려 있었다.**
    eff = c.income * mean_age_multiplier(c)
    epoch_progress = eff * c.agents * c.epoch_turns * k
    whole_progress = eff * c.agents * c.total_turns * k

    print(f"세계   {c.total_turns}턴 · 주기 {c.epoch_turns} · 국가당 {c.agents}명 · "
          f"소득 {c.income:.0f}/턴 · 기대수명 {expected_life(c.surv_lambda, c.surv_k):.2f}턴")
    print(f"       eff {c.facility_eff} × success_prob {c.success_prob} = k {k}")
    print()

    print("요격기 임계가 놓여야 할 창 (진척 단위)")
    for name, v, rel in (("A 미루기 방지", A, ">"), ("B 조율 강제", B, ">"),
                         ("E 지속 참여", E, ">"), ("C 도달 가능", hi, "<")):
        ok = c.interceptor > v if rel == ">" else c.interceptor < v
        print(f"   {name:<14}{rel} {v:>9.0f}   {'OK' if ok else '위반'}")
    pos = (c.interceptor - lo) / (hi - lo) if hi > lo else float("nan")
    print(f"   창 [{lo:.0f}, {hi:.0f}] · 임계 {c.interceptor:.0f} · 위치 {pos:.2f}")
    print()

    print("벙커")
    print(f"   깊이척도 {c.bunker:.0f} · 한 주기 진척 {epoch_progress:.0f} · "
          f"전 기간 {whole_progress:.0f} · {c.bunker / epoch_progress:.2f} 주기분")
    # **같은 기간으로 나눈다** (#51). 벙커만 한 주기로 나누면 비가 3.34배로 찍히는데,
    # 맞추면 1.11배다 — 함정의 크기를 세 배로 읽고 있었다.
    b1 = c.bunker / (c.agents * c.total_turns)
    i1 = c.interceptor / (3 * c.agents * c.total_turns)
    print(f"   1인부담  벙커 {b1:.1f} > 요격기 {i1:.1f}  ({c.total_turns}해·사람당, 비 {b1 / i1:.2f})  "
          f"{'OK' if b1 > i1 else '위반 — 벙커가 더 싸면 함정이 아니다'}")
    print(f"   성장     growth_coef {c.growth_coef} · growth_scale {c.growth_scale:.0f} "
          f"(한 주기 국가소득의 {c.growth_scale / (c.income * c.agents * c.epoch_turns):.2f}배)")
    print()

    good, why = passes_asserts(c)
    if not good:
        print(f"✗ assert 실패 — {why}")
        return 1

    print(f"필요 기여율 w*   (목표 {W_TARGET}, 시드 {seeds})")
    w = required_w(c)
    wg, split = required_w_growth(c)
    for name, got, want in (("무성장", w, EXPECTED["w_star"]),
                            ("성장 포함", wg, EXPECTED["w_star_growth"])):
        d = None if got is None else got - want
        mark = "—" if d is None else ("OK" if abs(d) <= TOL else "이동")
        print(f"   {name:<10} {'—' if got is None else f'{got:.2f}'}  "
              f"(기록 {want:.2f}{'' if d is None else f' · 차 {d:+.2f}'})  {mark}")
    print(f"   성장 전환 턴 {split}")
    print()

    m = evaluate(c, seeds)
    print(f"민감도 — 재앙 회피율 (정책 {len(POLICIES)}종 평균, 시드 {seeds})")
    print(f"   조율 0.2 → {m['lo']:.2f}   조율 0.9 → {m['hi']:.2f}   "
          f"민감도 {m['sensitivity']:+.2f}")
    print(f"   정책 간 표준편차 {m['policy_var']:.2f} · 런당 사망 {m['deaths']:.1f}회")
    print("   정책별(조율 0.9)  " + " · ".join(
        f"{k} {v:.2f}" for k, v in sorted(m["by_policy_hi"].items(), key=lambda x: -x[1])))
    print("   ※ 회피율은 조율률의 복사본이라 **튜닝 대상이 아닙니다.** 민감도만 봅니다.")

    drift = [n for n, got, want in (("무성장", w, EXPECTED["w_star"]),
                                    ("성장", wg, EXPECTED["w_star_growth"]))
             if got is not None and abs(got - want) > TOL]
    if drift:
        print(f"\n⚠ w* 가 기록에서 벗어났습니다 ({', '.join(drift)}). "
              "세계 난이도가 바뀐 것이므로 원인을 찾기 전에는 본실험을 돌리지 마세요.")
        return 1
    print("\n창 안 · w* 기록과 일치. 본실험 파라미터를 그대로 씁니다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="확정 config 를 Phase 0 의 자로 재검증")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    return report(cfg_from_yaml(Path(a.config)), a.seeds)


if __name__ == "__main__":
    raise SystemExit(main())
