"""더미 정책. 과제 2 에서 LLM 으로 교체된다.

인터페이스만 맞추는 것이 목적이다 — 반환 형식이 응답 스키마(spec 4.2)와
같아야 나중에 교체가 공짜가 된다.
"""
from __future__ import annotations

# 과제 명세(B-3)의 더미 규칙. 이 값은 정책의 것이지 세계의 것이 아니므로 여기 둔다.
PROCREATE_AGE = 7


def dummy_policy(world, agent, cfg, procreate_age: int | None = PROCREATE_AGE) -> dict:
    """actions 배열을 담은 dict 를 반환한다.

    최소 구현:
      - 예산의 절반을 자국 facility 에 invest
      - 나이가 `procreate_age` 이상이면 bear_child (8/21: 부모는 죽지 않는다)
      - speak 은 하지 않는다 (메시지 라우팅은 과제 2)

    ⚠ 배열 순서가 곧 우선순위다 (spec 4.2). invest 를 먼저 두어 절반을 시설에 쓴 뒤,
      procreate 로 **남은** 예산을 자식에게 넘긴다.

    `procreate_age=None` 이면 procreate 를 끈다 — 이때 사망은 오직 자연사(수명 모델)
    뿐이라, 수명 모델 자체의 캘리브레이션(사망 46~55, 수명 8.2~8.4)을 격리해서 잴 수
    있다. procreate 를 켜면 age≥7 에서 강제 교체되어 수명 분포가 잘린다.
    """
    actions: list[dict] = [
        {"type": "invest", "target": "facility", "amount": agent.budget / 2, "to": agent.country},
    ]
    if procreate_age is not None and agent.age >= procreate_age:
        actions.append({"type": "bear_child"})

    return {"reasoning": "dummy", "actions": actions, "received": []}
