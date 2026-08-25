"""설정이 세계의 전제를 만족하는지 검사한다. spec 7장.

각 함수는 통과하면 None, 실패하면 진단 문자열을 반환한다.
'왜 실패했는지'와 '무엇을 만져야 하는지'를 둘 다 담아라 —
숫자만 던지면 어디를 고쳐야 할지 알 수 없다.
"""
from __future__ import annotations


def window(cfg) -> tuple[float, float, float, float]:
    """요격기 임계가 놓여야 할 창. 전부 진척 단위로 환산해 반환한다. (A, B, C, E)

    A 미루기 방지 : 마지막 한 주기에 3국이 전력을 다해도 불가
    B 조율 강제   : 한 나라가 전 기간을 다 써도 불가
    C 도달 가능   : 세 나라가 모으면 가능
    E 지속 참여   : 한 주기가 통째로 쉬면 불가

    ⚠ 성장(multiplier)을 **빼고** 계산한다. 성장은 국가 투자의 결과이고
      그 돈은 요격기 투자와 경쟁하므로, 성장을 전제로 임계를 잡으면
      "국가 투자를 안 하면 구조적으로 도달 불가" 가 되어버린다. (Phase 0 결함 8번)

    ⚠ E 는 양변에 같은 정책계수(0.6)를 쓴다. 전력 기준으로 걸면 하한이
      (T−E)/T × C 가 되어 상한 C×0.6 을 넘고 **창이 닫힌다.**
    """
    # **나이 배수를 창에 반영한다** (8/22). 소득이 나이와 함께 오르므로 「한 나라의 전
    # 기간 총소득」 이 `per_turn × n × total` 보다 크다 — 그 값을 그대로 쓰면 창이 실제보다
    # 좁아지고, 임계가 「도달 가능」 쪽에 붙는다.
    #
    # 배수는 **정상 연령분포**에서 잡는다: 나이 a 에 살아 있을 확률 ∝ S(a).
    per_turn = cfg.income.per_turn * mean_age_multiplier(cfg)
    n = cfg.world.agents_per_country
    total = cfg.world.total_turns
    epoch = cfg.world.epoch_turns

    # 성장을 뺀 소득 (임계 유도용)
    nation_all = per_turn * n * total          # 한 나라의 전 기간 총소득
    nation_epoch = per_turn * n * epoch        # 한 나라의 한 주기 소득

    # **가장 효율 좋은 나라를 기준으로 잡는다** (8/23). 나라마다 요격기 진척 속도가
    # 다르므로(`facility.build_spread`), 네 조건이 모두 「최선의 나라에 몰아줬을 때」 로
    # 걸려야 한다.
    #
    #   ★B 가 결정적이다 — 「한 나라가 혼자서는 못 한다」 는 **가장 잘 짓는 나라**가
    #   혼자서도 못 해야 성립한다. 평균으로 잡으면 운 좋은 나라가 단독으로 해내고
    #   조율 강제가 무너진다.
    #
    #   ★C 도 같은 기준이어야 한다 — 「세 나라가 모으면 가능」 은 그들이 최선의 나라를
    #   고를 때의 얘기다. 네 조건이 같은 배수로 곱해지므로 창은 통째로 평행이동한다.
    best = max(cfg.facility.build_spread)

    def to_progress(income_amount: float) -> float:
        # 진척 단위 = 소득 × facility.eff × success_prob × 최선 국가 효율
        return income_amount * cfg.k * best

    a = to_progress(3 * nation_epoch)                      # 마지막 한 주기 3국 전력
    b = to_progress(nation_all)                            # 한 나라의 전 기간 총력
    c = to_progress(3 * nation_all)                        # 세 나라의 전 기간 총력
    e = to_progress(3 * (nation_all - nation_epoch)) * 0.6  # 한 주기를 통째로 쉬면
    return (a, b, c, e)


def mean_age_multiplier(cfg) -> float:
    """정상 연령분포에서 본 소득 나이 배수의 평균.

    나이 a 에 살아 있을 확률은 생존함수 S(a) 에 비례한다. 소득이 성인 나이 이후 한 해마다
    `age_growth` 씩 오르므로, 한 사람이 평균적으로 받는 배수는 그 가중평균이다.

        age_growth 0.20 · adult_age 10 → 평균 1.224
        age_growth 0.10 · adult_age  5 → 평균 1.363   ← 지금 값

    **손으로 적지 않는다.** `age_growth` 나 수명을 바꾸면 이 값이 따라 움직여야 하고,
    그러지 않으면 임계값 창이 조용히 어긋난다.
    """
    from core import survival as _s
    g = cfg.income.age_growth
    if g <= 0:
        return 1.0
    lam, k = cfg.survival.lambda_base, cfg.survival.k
    horizon = int(lam * 3) + 1
    w = [_s.survival(a, lam, k) for a in range(horizon)]
    tot = sum(w) or 1.0
    return sum(w[a] * (1 + g * max(0, a - cfg.world.adult_age))
               for a in range(horizon)) / tot


def check_all(cfg) -> list[str]:
    """전부 검사하고 실패 목록을 반환한다. 빈 리스트면 통과."""
    fails: list[str] = []
    a, b, c, e = window(cfg)
    intc = cfg.thresholds.interceptor
    bunker = cfg.thresholds.bunker_scale

    # ★A 미루기 방지 — 마지막 한 주기에 3국이 전력을 다해도 도달 불가
    if not (intc > a):
        fails.append(
            f"★A 미루기 방지: interceptor({intc}) 가 A({a:.0f}) 보다 커야 한다. "
            f"작으면 마지막에 몰아서 해결 가능 → 미루기가 옳은 전략이 된다. "
            f"thresholds.interceptor 를 올려라."
        )
    # ★B 조율 강제 — 한 나라가 전 기간을 다 써도 도달 불가
    if not (intc > b):
        fails.append(
            f"★B 조율 강제: interceptor({intc}) 가 B({b:.0f}) 보다 커야 한다. "
            f"작으면 한 나라가 혼자 해냄 → 조율이 무의미해진다. "
            f"thresholds.interceptor 를 올려라."
        )
    # ★C 도달 가능 — 세 나라가 모으면 가능. 0.6 은 정책계수(전액 투입은 비현실적)
    if not (intc < c * 0.6):
        fails.append(
            f"★C 도달 가능: interceptor({intc}) 가 C×0.6({c * 0.6:.0f}) 보다 작아야 한다. "
            f"크면 아무도 도달 못 함 → 전 조건에서 멸망. "
            f"thresholds.interceptor 를 내려라. "
            f"(success_prob 를 바꿨다면 임계도 함께 재계산했는지 확인.)"
        )
    # ★E 지속 참여 — 한 주기가 통째로 쉬면 도달 불가
    if not (intc > e):
        fails.append(
            f"★E 지속 참여: interceptor({intc}) 가 E({e:.0f}) 보다 커야 한다. "
            f"작으면 한 주기가 쉬어도 지어짐 → 지속 참여 압력이 사라진다. "
            f"thresholds.interceptor 를 올려라."
        )

    # 국가 효율 — 평균이 1.0 이 아니면 창이 어긋난다 (순열 배정이라 정확히 1.0 이어야)
    bs = list(cfg.facility.build_spread)
    if abs(sum(bs) / len(bs) - 1.0) > 1e-9:
        fails.append(
            f"국가 효율: facility.build_spread({bs}) 의 평균이 {sum(bs)/len(bs):.4f} 다. "
            f"정확히 1.0 이어야 한다 — 창이 `per_turn × n × total` 에서 나오므로 "
            f"평균이 1 이 아니면 임계 전체를 재계산해야 한다."
        )
    if len(bs) != len(cfg.world.countries):
        fails.append(
            f"국가 효율: facility.build_spread 가 {len(bs)}개인데 나라는 "
            f"{len(cfg.world.countries)}개다. 순열 배정이므로 같아야 한다."
        )

    # 벙커 깊이 창 — **국가 효율이 안 걸린다** (요격기 전용). 여기는 cfg.k 그대로다.
    eff = cfg.income.per_turn * mean_age_multiplier(cfg)
    nation_all = eff * cfg.world.agents_per_country * cfg.world.total_turns
    nation_epoch = eff * cfg.world.agents_per_country * cfg.world.epoch_turns
    bunker_lo = nation_epoch * cfg.k     # 한 주기 전력 진척
    bunker_hi = nation_all * cfg.k       # 전 기간 진척
    if not (bunker >= bunker_lo):
        fails.append(
            f"벙커↓: bunker_scale({bunker}) 가 한 주기 진척({bunker_lo:.0f}) 이상이어야 한다. "
            f"작으면 한 주기로 완성됨 → 함정이 함정이 아니게 된다. bunker_scale 를 올려라."
        )
    if not (bunker <= bunker_hi):
        fails.append(
            f"벙커↑: bunker_scale({bunker}) 가 전 기간 진척({bunker_hi:.0f}) 이하여야 한다. "
            f"크면 아무리 파도 의미가 없다. bunker_scale 를 내려라."
        )

    # ★D 1인부담 비교 — **분모의 기간을 맞춘다** (#51).
    #
    # 벙커는 한 주기(20해)로, 요격기는 전 기간(60해)으로 나누고 있었다. 사람 수가 다른
    # 것은 옳다(벙커는 한 나라 3명이 지고 요격기는 세 나라 9명이 나눠 진다) — **기간이**
    # 달랐다. 그래서 비가 3.34배로 찍혔는데, 같은 기간으로 재면 1.11배다.
    #
    # 조건은 그대로 통과한다. 다만 함정의 크기를 3배로 읽고 있었다 — 임계나 깊이척도를
    # 다시 만질 때 그 차이가 그대로 판단을 바꾼다.
    span = cfg.world.total_turns
    bunker_burden = bunker / (cfg.world.agents_per_country * span)
    intc_burden = intc / (3 * cfg.world.agents_per_country * span)
    if not (bunker_burden > intc_burden):
        fails.append(
            f"부담: 벙커 1인부담({bunker_burden:.1f}) 이 요격기 1인부담({intc_burden:.1f}) 보다 "
            f"커야 한다 (둘 다 {span}해·사람당). 벙커가 더 싸지면 아무도 요격기를 안 한다. "
            f"bunker_scale 를 올리거나 interceptor 를 조정하라."
        )

    # 노브 — 원문 경로가 AI 경로보다 싸야 경로 선택이 의미를 갖는다. 전 구간에서.
    learner = cfg.costs.comm_intl_learner
    for v in cfg.knob.comm_intl_ai:
        if not (v > learner):
            fails.append(
                f"노브: comm_intl_ai 의 값 {v} 가 comm_intl_learner({learner}) 보다 커야 한다. "
                f"원문 경로가 더 비싸지면 경로 선택이 무의미해진다. "
                f"knob.comm_intl_ai 의 최저값을 {learner} 위로 올려라."
            )

    return fails
