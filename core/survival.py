"""확률적 수명. spec 2.2.

S(a) = exp(−(a/λ)^k). 세대 경계를 대신해 각자 다른 시점에 죽게 한다.
"""
from __future__ import annotations

import math


def survival(age: int, lam: float, k: float) -> float:
    """S(a) = exp(−(a/λ)^k).  나이 a 를 넘길 확률."""
    return math.exp(-((age / lam) ** k))


def hazard(age: int, lam: float, k: float) -> float:
    """나이 age 에서 age+1 로 가는 동안 죽을 확률.

    = 1 − S(age+1)/S(age).  조건부 확률이므로 S(age) 로 나눈다.
    """
    return 1.0 - survival(age + 1, lam, k) / survival(age, lam, k)


def expected_life(lam: float, k: float) -> float:
    """E[사망 나이] = Σ_(a≥0) S(a).

    ⚠ a=0 항(S(0)=1)을 빼먹으면 정확히 1턴이 모자란다.
      명세 초기 판이 8.3 을 7.3 으로 적었던 것이 이 실수였다.
    """
    total = 0.0
    age = 0
    while True:
        s = survival(age, lam, k)
        total += s
        if age > 0 and s < 1e-15:
            break
        age += 1
    return total
