"""설정이 세계의 전제를 만족하는지 검사한다. spec 7장.

각 함수는 통과하면 None, 실패하면 진단 문자열을 반환한다.
'왜 실패했는지'와 '무엇을 만져야 하는지'를 둘 다 담아라 —
숫자만 던지면 어디를 고쳐야 할지 알 수 없다.
"""
from __future__ import annotations


def capacity_per_year(cfg) -> float:
    """한 사람이 한 해에 시설로 옮길 수 있는 **최대 투자량**.

    AP 를 전부 `invest` 에 쏟았을 때다. 돈이 사라진 뒤로 이것이 유일한 천장이다 —
    전에는 소득(`income.per_turn × 나이배수`)이 더 낮은 천장이었다.

        (action_points / ap.unit) × costs.unit = 5회 × 40 = 200

    개체 배수(`throughput_spread`)는 평균이 정확히 1.0 이라 여기 안 곱한다.
    """
    return (cfg.turn.action_points / cfg.ap.unit) * cfg.costs.unit


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
    # **창은 행동력 용량에서 나온다** (8/25 · AP 전면 통일). 전에는 소득이었다.
    #
    #   한 사람 한 해 = (action_points / ap.unit) 회 × costs.unit
    #                 = 5회 × 40 = 200 투자량
    #
    # 나이 배수(`mean_age_multiplier`)는 더는 안 곱한다 — `age_growth` 를 없앴다. AP 는
    # 매년 1.0 으로 리셋되므로 나이와 무관하게 누구나 같은 용량을 갖는다.
    #
    # 개체 배수(`throughput_spread`)는 평균이 정확히 1.0 이라 곱해도 그대로다.
    per_turn = capacity_per_year(cfg)
    n = cfg.world.agents_per_country
    total = cfg.world.total_turns
    epoch = cfg.world.epoch_turns

    nation_all = per_turn * n * total          # 한 나라가 전 기간에 옮길 수 있는 최대량
    nation_epoch = per_turn * n * epoch        # 한 주기

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

    def to_progress(amount: float) -> float:
        # 진척 단위 = 투자량 × facility.eff × success_prob × 최선 국가 효율
        return amount * cfg.k * best

    a = to_progress(3 * nation_epoch)                      # 마지막 한 주기 3국 전력
    b = to_progress(nation_all)                            # 한 나라의 전 기간 총력
    c = to_progress(3 * nation_all)                        # 세 나라의 전 기간 총력
    e = to_progress(3 * (nation_all - nation_epoch)) * 0.6  # 한 주기를 통째로 쉬면
    return (a, b, c, e)


def check_all(cfg) -> list[str]:
    """전부 검사하고 실패 목록을 반환한다. 빈 리스트면 통과."""
    fails: list[str] = []
    a, b, c, e = window(cfg)
    intc = cfg.thresholds.interceptor
    bunker = cfg.thresholds.bunker

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
    eff = capacity_per_year(cfg)
    nation_all = eff * cfg.world.agents_per_country * cfg.world.total_turns
    nation_epoch = eff * cfg.world.agents_per_country * cfg.world.epoch_turns
    bunker_lo = nation_epoch * cfg.k     # 한 주기 전력 진척
    bunker_hi = nation_all * cfg.k       # 전 기간 진척
    if not (bunker >= bunker_lo):
        fails.append(
            f"벙커↓: bunker({bunker}) 가 한 주기 진척({bunker_lo:.0f}) 이상이어야 한다. "
            f"작으면 한 주기로 완성됨 → 함정이 함정이 아니게 된다. bunker 를 올려라."
        )
    if not (bunker <= bunker_hi):
        fails.append(
            f"벙커↑: bunker({bunker}) 가 전 기간 진척({bunker_hi:.0f}) 이하여야 한다. "
            f"크면 아무리 파도 의미가 없다. bunker 를 내려라."
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
    # **벙커가 더 비싸야 한다.** 8/25 에 「정확히 같게」 로 바꿨다가 **되돌렸다** —
    # 같으면 개인에게 비용도 결과(내가 산다)도 같은데, 협력에는 추가 비용(말하기·학습 AP)과
    # 배신 위험이 얹힌다. 그러면 벙커가 지배하고 딜레마가 사라진다.
    #
    # 벙커의 유혹은 「싸다」 가 아니라 **「남이 필요 없다」** 다. 함정은 불신 때문에 더 비싼
    # 쪽을 택하는 것이고, 그 초과분이 불신의 가격이다.
    #
    # 초과분의 크기에는 뜻이 있다 — 요격기를 고르면 남는 AP 가 협력의 비용(국제 발신)을
    # 살 수 있어야 하고, **그것이 노브에 따라 갈려야** 한다. 7,200 에서 그 경계가 노브
    # 범위 안에 놓인다 (0.2 → 연 1.03통 · 0.5 → 0.41통).
    if not (bunker_burden > intc_burden):
        fails.append(
            f"부담: 벙커 1인부담({bunker_burden:.2f}) 이 요격기 1인부담({intc_burden:.2f}) 보다 "
            f"커야 한다 (둘 다 {span}해·사람당). 같거나 싸면 협력의 추가 비용(말하기·학습·"
            f"배신 위험)을 감수할 이유가 없어져 벙커가 지배한다. bunker 를 올려라."
        )

    # 노브 — **원문 경로보다 싸지면 경로 선택이 무의미해진다.** 전 구간에서.
    # **비교 대상은 `ap.speak_intl` 이다** (8/26). `ap.speak`(자국) 으로 걸면 `ai` 가
    # 국제 원문보다 싼 세계가 통과한다 — 그러면 배울 이유가 구조적으로 사라진다.
    aps = list(cfg.knob.comm_intl_ai_ap)
    for v in aps:
        if not (v >= cfg.ap.speak_intl):
            fails.append(
                f"노브: comm_intl_ai_ap 의 값 {v} 가 "
                f"ap.speak_intl({cfg.ap.speak_intl}) 이상이어야 한다. "
                f"AI 번역이 원문보다 싸지면 아무도 배우지 않는다 — 노브가 방향을 잃는다."
            )
    # **한 해에 한 통도 못 보내면 노브가 아니라 금지다.**
    if aps and max(aps) > cfg.turn.action_points:
        fails.append(
            f"노브: 최고값 {max(aps)} 이 한 해 AP({cfg.turn.action_points})를 넘는다. "
            f"그러면 AI 발신이 불가능해지고, 비싼 것과 없는 것이 구분되지 않는다."
        )
    # 오름차순이어야 `comm_intl_ai_ap[i]` 가 「i 번째로 비싼 노브」 라는 뜻을 갖는다.
    if aps != sorted(aps):
        fails.append(f"노브: comm_intl_ai_ap({aps}) 가 오름차순이 아니다.")

    return fails
