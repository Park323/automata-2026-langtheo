#!/usr/bin/env python3
"""RESULTS.md 의 「확정된 값」 표를 config 에서 뽑는다.

    ./.venv/bin/python tools/analysis/genvals.py

**손으로 적으면 낡는다.** 그 절이 실제로 `learn_base 300 · total_turns 50 · λ 8.26` 으로
남아 있었고, 그 셋은 지금 200 · 60 · 16.52 다. 값을 바꾸면 이걸 다시 돌려 붙인다.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from core import asserts, config  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    c = config.load(str(ROOT / "configs" / "base.yaml"))
    a, b, cc, e = asserts.window(c)
    L = c.length.message_max_chars
    # **수명에는 규약이 둘이다** — 이 프로젝트가 이미 그 혼동을 겪었다 (spec 2.2).
    #   살아낸 해 수      `expected_life` = Σ_(a≥0) S(a) = 16.06
    #   Weibull 평균      λ·Γ(1+1/k) = 15.56 — 프롬프트가 「16 歳ごろ」 로 적는 값
    # 둘 다 적는다. 하나만 적으면 다음에 또 섞인다.
    from core import survival as _s
    lived = _s.expected_life(c.survival.lambda_base, c.survival.k)
    weib = c.survival.lambda_base * __import__("math").gamma(1 + 1 / c.survival.k)
    print("```")
    # **모델 이름을 손으로 적지 않는다** (8/25). 여기 `qwen3.6-35b-a3b` 이 박혀 있었고
    # 실제 런은 deepseek-v4-flash 였다 — 「숫자를 두 군데 적으면 하나가 낡는다」.
    print(f"번역 모델   {c.llm.translate_model.split('/')[-1]:<32}"
          " 파일럿 확정은 mistral-small — fp8 1곳뿐이라 8/27 qwen3.6 교체 (RESULTS 인프라 절)")
    print(f"에이전트    {c.llm.agent_model.split('/')[-1]:<32} "
          f"사고 {c.llm.reasoning} · 도구 reasoning {c.llm.tool_reasoning}")
    print(f"{'':<12}{'':<32} max_tokens {c.llm.max_tokens} · "
          f"provider {c.llm.provider}")
    print()
    print(f"노브        ai 발신 AP {list(c.knob.comm_intl_ai_ap)}   ← 유일한 실험 변수")
    print(f"행동력      한 해 {c.turn.action_points} · speak {c.ap.speak} · unit {c.ap.unit} · "
          f"vote {c.ap.vote}")
    # propose_vote 는 지웠다 (8/26) — 採決을 시계가 연다
    print(f"            observe_risk {c.ap.observe_risk} · memory_write {c.ap.memory_write}")
    print(f"採決        {c.world.ballot_from}해부터 {c.world.ballot_every}해마다 전 국가 동시 · "
          f"전환 시 자국 몫만 승계 (kept_on_switch)")
    print(f"용량        한 사람 한 해 {asserts.capacity_per_year(c):.0f} "
          f"(= {c.turn.action_points:g}/{c.ap.unit:g}회 × unit {c.costs.unit:.0f})")
    print(f"학습        learn_base {c.costs.learn_base:.0f} (고정) · "
          f"speedup +{c.costs.learn_speedup} / 사유")
    print(f"수명        k {c.survival.k} · λ {c.survival.lambda_base}")
    print(f"            살아낸 해 {lived:.2f} · Weibull 평균 {weib:.2f} "
          f"(프롬프트는 후자를 반올림해 「{weib:.0f}」 로 적는다)")
    print(f"세계        total_turns {c.world.total_turns} · epoch {c.world.epoch_turns} · "
          f"success_prob {c.world.success_prob}")
    print(f"            첫 해 나이 1~{c.world.init_age_max}")
    print(f"            agents_per_country {c.world.agents_per_country}")
    print(f"임계        interceptor {c.thresholds.interceptor} · "
          f"bunker {c.thresholds.bunker}")
    print(f"            창 A {a:,.0f} · B {b:,.0f} · E {e:,.0f} < 임계 < C×0.6 {cc * 0.6:,.0f}")
    print(f"처리량      throughput_spread {list(c.facility.throughput_spread)} · "
          f"facility.eff {c.facility.eff}")
    print(f"국가효율    build_spread {list(c.facility.build_spread)} (요격기 전용 · 8/27부터 고정 배정 — 씨앗은 배치를 안 바꾼다)")
    print(f"성장        growth_coef {c.growth.growth_coef} / scale {c.growth.growth_scale:.0f} · "
          f"wellness.gain {c.wellness.gain}")
    print(f"문맥        context_limit {c.llm.context_limit} · warn_ratio {c.llm.warn_ratio} · "
          f"max_tokens {c.llm.max_tokens}")
    print(f"길이        fr {L['fr']} / ja {L['ja']} / zh {L['zh']}")
    print("```")
    fails = asserts.check_all(c)
    print()
    print("자가검증 ★A·B·C·E: " + ("통과" if not fails else " / ".join(fails)))


if __name__ == "__main__":
    main()
