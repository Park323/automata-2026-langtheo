"""무엇을 누가 알 수 있는가. spec 4.1 을 **코드로** 옮긴 표.

에이전트에게는 이 파일의 어느 것도 보이지 않는다. 우리가 뒤에서 context 를 나눠 주는
규칙일 뿐이다.

    SECRET   아무도 모른다 — **행위자조차** 모른다
    PRIVATE  행위자만 안다
    PUBLIC   그 나라 안에서 공유된다
    GLOBAL   전 세계가 공유한다

## 왜 표로 만드는가

지금까지 **「누가 아는가」 가 호출부마다 흩어져 있었다.** 부고는 같은 나라를 훑는 루프로,
출자 결과는 출자자 한 명에게, 메시지는 수신자에게 — 각각 다른 곳에서 각각의 방식으로
청중을 정했다. 그러면 두 가지가 안 된다.

  ① **대조할 수 없다.** spec 4.1 의 은닉 목록과 코드가 어긋났는지 눈으로 봐야 한다.
     실제로 어긋난 것을 두 번 발견했다 — 타국 출자의 진척 증가분(액수를 쌓으면 상대국
     생산배수가 복원됐다)과 「wellness 는 수명을 늘린다」(λ 곡선을 말로 알려줬다).
  ② **새 사건을 추가할 때 청중을 다시 발명한다.** 그때마다 은닉이 새어 나갈 자리가 생긴다.

이제 사건은 등급을 **선언**하고, 청중은 `audience()` 하나가 정한다.
"""
from __future__ import annotations

from enum import Enum


class Vis(Enum):
    """공개 등급. 값은 넓이 순서라 비교할 수 있다 (`SECRET < PRIVATE < ...`)."""

    SECRET = 0
    PRIVATE = 1
    PUBLIC = 2
    GLOBAL = 3

    def __lt__(self, other):
        return self.value < other.value


# ─────────────────────────────────────────────────────────────────────────────
# 세계가 만들어 내는 모든 사실의 등급.
#
# **이 표에 없는 것을 에이전트에게 보내면 테스트가 실패한다.** 새 사건을 만들면 여기에
# 한 줄을 적어야 하고, 그 줄이 곧 「왜 이 사람들만 아는가」 의 근거다.
# ─────────────────────────────────────────────────────────────────────────────

FACTS: dict[str, tuple[Vis, str]] = {
    # ── SECRET — 아무도 모른다. 행위자조차 ──────────────────────────────────
    "success_prob":      (Vis.SECRET, "돈이 진척으로 바뀌는 확률. 알려주면 기대값 계산이 된다"),
    "lifespan_lambda":   (Vis.SECRET, "개인의 수명 척도 λ. **본인도 모른다** (4.1)"),
    "hazard_curve":      (Vis.SECRET, "나이→사망확률. 평균만 SYSTEM 에 적고 모양은 숨긴다"),
    "wellness_gain":     (Vis.SECRET, "wellness 가 λ 를 얼마 올렸나. 본인도 모른다"),
    "threshold_truth":   (Vis.SECRET, "요격기 임계의 진값. observe_risk 는 흐린 값만 준다"),
    "impact_turn_truth": (Vis.SECRET, "운석까지 남은 진짜 해수. 같다"),
    "growth_fn":         (Vis.SECRET, "생산배수 함수. 수입에서 추론할 뿐"),
    "inner_reasoning":   (Vis.SECRET, "타인의 내심. 로그에는 남지만 세계에는 없다"),
    "other_nation_state":(Vis.SECRET, "타국의 진척·예산·국토·언어 능력. **소통으로만** 안다"),

    # ── PRIVATE — 행위자만 ─────────────────────────────────────────────────
    "action_left":       (Vis.PRIVATE, "내 남은 행동력"),
    "lang_progress":     (Vis.PRIVATE, "내 언어 학습 진척"),
    "facility_invested": (Vis.PRIVATE, "내가 어느 나라에 얼마를 냈나"),
    "memory":            (Vis.PRIVATE, "내 메모"),
    "testament":         (Vis.PRIVATE, "앞사람이 죽으며 남긴 말. **뒷사람에게만** — "
                                       "기억에 심지 않고 들은 말로 온다. 옮겨 적을지는 "
                                       "본인이 고르고, 안 옮기면 대화에서 밀려 사라진다 "
                                       "(3.3 구전의 감쇠). 자연사는 예고가 없어 도구로는 "
                                       "남길 수 없으므로 죽는 그 순간에 우리가 묻는다"),
    "gift":              (Vis.PRIVATE, "누가 나에게 얼마를 주었나. **받는 이만** — "
                                       "예산은 PRIVATE 이고, 갑자기 늘어난 이유를 "
                                       "본인이 모르면 그 돈을 쓸 판단을 못 한다"),

    "risk_reading":      (Vis.PRIVATE, "내가 observe_risk 로 읽은 값. 남에게 알리려면 말해야 한다"),
    "fac_gain":          (Vis.PRIVATE, "내 출자가 진척을 얼마 올렸나 (타국이면 여부만)"),
    "delivery_failed":   (Vis.PRIVATE, "내가 보낸 말이 닿지 않았다"),
    "message":           (Vis.PRIVATE, "주고받은 말. **보낸 이와 받는 이만**"),
    "land":              (Vis.PUBLIC, "자국 국토"),
    "progress":          (Vis.PUBLIC, "자국 진척"),

    "proposal":          (Vis.PUBLIC, "열린 採決과 採決일"),
    "ballot_result":     (Vis.PUBLIC, "採決 결과와 **그때 사라진 진척**. 국토를 정하는 것은 "
                                      "그 나라 사람들이고, 그 대가도 그들이 함께 안다"),
    "progress_change":   (Vis.PUBLIC, "진척이 얼마 늘어 얼마가 됐나. **누가 냈는지는 없다** — "
                                      "자국민은 자기가 낸 것을 알므로 차이에서 타국 출자를 "
                                      "짐작할 수 있고, 그 짐작은 흘려도 되는 것이다"),
    "capital_change":    (Vis.PUBLIC, "국가 기술력이 올랐다. 수입·시설 전환율·관측 정확도가 "
                                      "다 여기 걸려 있어 국민 전원의 일이다"),
    "domestic_speaker":  (Vis.PUBLIC, "국내에 그 말을 하는 사람이 있는가. 학습가 할인으로 드러난다"),

    # ── GLOBAL — 전 세계 ───────────────────────────────────────────────────
    "year":              (Vis.GLOBAL, "올해"),
    "roster":            (Vis.GLOBAL, "누가 있는가. 없으면 서로를 부를 수 없다"),
    "rules":             (Vis.GLOBAL, "세계 규칙 (SYSTEM)"),
    "birth":             (Vis.GLOBAL, "누가 아이를 낳았고 누가 태어났나 — `roster` 가 "
                                      "GLOBAL 이므로 새 사람이 나타난 것은 어차피 보인다. "
                                      "**누구의 아이인가**만 새로 새는 것이고, 그건 "
                                      "세대 간 전달을 관측하려면 필요하다 (3.3)"),
    "obituary":          (Vis.GLOBAL, "누가 몇 살에 죽고 누가 그 자리에 왔나 — 8/20 에 "
                                      "PUBLIC 에서 올렸다. roster 가 이미 교체를 드러내므로 "
                                      "새로 새는 것은 **나이**뿐이고, 그것이 수명을 배우는 "
                                      "유일한 경로다 (곡선은 여전히 SECRET)"),
    # **선언은 하지만 전달되지 않는다.** 판정(`final_survival`)은 마지막 해 **뒤**에
    # 나오고, 그 뒤에 행동하는 사람이 없다 — context 에 넣어도 아무에게도 보내지지
    # 않는다. 등급을 적어 두는 것은 「모두가 겪는다」 가 사실이기 때문이고, 죽은 통지를
    # 만들지 않는 것은 그것이 전달될 자리가 없기 때문이다.
    "outcome":           (Vis.GLOBAL, "요격기 완성·운석 충돌. 모두가 겪지만 그 뒤에 아무도 "
                                      "행동하지 않으므로 통지할 자리가 없다"),
}


def level(fact: str) -> Vis:
    """등급을 읽는다. **표에 없으면 거절한다** — 청중을 즉흥으로 정하지 않게."""
    if fact not in FACTS:
        raise KeyError(
            f"공개 등급이 선언되지 않은 사실: {fact!r}. core/visibility.py 의 FACTS 에 "
            f"한 줄을 적으세요 — 그 줄이 「왜 이 사람들만 아는가」 의 근거입니다.")
    return FACTS[fact][0]


def audience(world, fact: str, actor: str | None = None,
             nation: str | None = None) -> list[str]:
    """이 사실을 알 수 있는 **살아 있는** 사람들. 정렬해 돌려준다 (결정론).

    `actor` 는 PRIVATE 의 임자, `nation` 은 PUBLIC 의 범위다. PUBLIC 인데 `nation` 이
    없으면 `actor` 의 나라로 본다.

    죽은 사람·교체된 슬롯은 알 수 없다 — 그 자리에 온 아이는 다른 사람이다 (3.2).
    """
    vis = level(fact)
    if vis is Vis.SECRET:
        return []
    if vis is Vis.PRIVATE:
        return [actor] if actor and actor in world.agents else []
    if vis is Vis.PUBLIC:
        home = nation or (world.agents[actor].country if actor in world.agents else None)
        if home is None:
            return []
        return sorted(a.id for a in world.agents.values()
                      if a.alive and a.country == home)
    return sorted(a.id for a in world.agents.values() if a.alive)
