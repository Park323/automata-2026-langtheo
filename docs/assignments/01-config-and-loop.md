# 과제 1 — 설정 로더와 턴 루프

> **カキガワ言語研究班** · `automata-2026-langtheo`
> 참조 명세: `docs/spec.md` 2장 · 3장 · 7장
> 선행 지식: Python, YAML, `dataclass`. LLM 관련 지식은 **필요 없습니다.**

---

## 0. 이 과제의 위치

전체 엔진은 다섯 단계입니다. 이 과제는 **1~2단계**입니다.

```
1  설정 로더 + assert                          ← 과제 1
2  상태 + 턴 루프 7단계 (LLM 없이 더미 정책)     ← 과제 1
────────────────────────────────────────────
3  에이전트 프롬프트 + 응답 스키마 + LLM 호출     ← 과제 2
4  9명 병렬 + 메시지 라우팅 + 2단 번역 경로       ← 과제 2
5  로그 4종 + 지표 산출                         ← 과제 3
```

**LLM을 붙이지 않고 세계가 도는 것까지가 목표입니다.**
사망·출생 경계 조건이 8개인데, LLM을 먼저 붙이면 한 번 돌릴 때마다 수천 번의 API
호출이 나가서 디버깅이 비싸고 느려집니다. 더미 정책으로 50턴이 정확히 도는 것을
먼저 확인합니다.

**힌트:** `tools/balance/sweep.py`의 `simulate()`가 이미 이 루프의 축약판입니다.
자료구조와 순서를 참고할 수 있지만, 그건 밸런스 전용이라 메시지·언어·AP가 없습니다.
**그대로 쓰지 말고 골격만 보세요.**

---

## Part A — 설정 로더와 assert (40점)

### A-1. `configs/base.yaml` (5점)

`docs/spec.md` 7장의 config 스키마를 그대로 YAML 파일로 옮깁니다.
**값은 이미 전부 확정되어 있습니다.** 명세에서 옮겨 적기만 하면 됩니다.
`?`로 남은 것은 아래 기본값을 쓰세요.

```yaml
facility:
  cap_per_turn: 500          # 턴당 시설 투자 상한
inheritance:
  testament_max_chars: 120
llm:
  agent_model: "qwen/qwen-2.5-72b-instruct"
  temperature: 0.7
run:
  seed: 1
```

### A-2. `core/config.py` (15점)

```python
"""설정 로더. spec 7장."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml


class ConfigError(Exception):
    """설정이 세계의 전제를 깨뜨릴 때. 메시지에 진단 정보를 반드시 담는다."""


@dataclass(frozen=True)
class Config:
    # TODO: 7장 스키마를 중첩 dataclass 로 표현한다.
    #       frozen=True 로 두는 이유 — 런 도중에 설정이 바뀌면 재현이 깨진다.
    ...

    @property
    def k(self) -> float:
        """진척 환산 계수 = facility.eff × world.success_prob.

        임계값은 **진척 단위**다. 소득을 그대로 비교하면 안 된다 (spec 7장).
        이 프로퍼티를 반드시 경유하게 만들어라 — 단위 오류가 Phase 0 에서
        실제로 나온 결함 1번이다.
        """
        ...


def load(path: str | Path) -> Config:
    """YAML 을 읽어 Config 로 만들고 assert 를 전부 통과시킨다.

    통과하지 못하면 ConfigError. **조용히 넘어가면 안 된다.**
    """
    ...
```

### A-3. `core/asserts.py` (20점)

`docs/spec.md` 7장 「검증 — 로드 시 assert」를 코드로 옮깁니다.
**이 파일이 이 과제의 핵심입니다.**

```python
"""설정이 세계의 전제를 만족하는지 검사한다. spec 7장.

각 함수는 통과하면 None, 실패하면 진단 문자열을 반환한다.
'왜 실패했는지'와 '무엇을 만져야 하는지'를 둘 다 담아라 —
숫자만 던지면 어디를 고쳐야 할지 알 수 없다.
"""


def window(cfg) -> tuple[float, float, float, float]:
    """요격기 임계가 놓여야 할 창. 전부 진척 단위로 환산해 반환한다.

    A 미루기 방지 : 마지막 한 주기에 3국이 전력을 다해도 불가
    B 조율 강제   : 한 나라가 전 기간을 다 써도 불가
    C 도달 가능   : 세 나라가 모으면 가능
    E 지속 참여   : 한 주기가 통째로 쉬면 불가

    ⚠ 성장(multiplier)을 **빼고** 계산한다. 성장은 국가 투자의 결과이고
      그 돈은 요격기 투자와 경쟁하므로, 성장을 전제로 임계를 잡으면
      "국가 투자를 안 하면 구조적으로 도달 불가" 가 되어버린다.
      (Phase 0 결함 8번)

    ⚠ E 는 양변에 같은 정책계수(0.6)를 쓴다. 전력 기준으로 걸면 하한이
      (T−E)/T × C 가 되어 상한 C×0.6 을 넘고 **창이 닫힌다.**
    """
    ...


def check_all(cfg) -> list[str]:
    """전부 검사하고 실패 목록을 반환한다. 빈 리스트면 통과."""
    ...
```

검사해야 할 것 — **8개 전부**입니다.

| # | 조건 | 실패 시 무슨 일이 벌어지는가 |
|---|---|---|
| ★A | `interceptor > A` | 마지막에 몰아서 해결 가능 → 미루기가 옳은 전략이 됨 |
| ★B | `interceptor > B` | 한 나라가 혼자 해냄 → **조율이 무의미해짐** |
| ★C | `interceptor < C × 0.6` | 아무도 도달 못 함 → 전 조건에서 멸망 |
| ★E | `interceptor > E` | 한 주기가 쉬어도 지어짐 → 지속 참여 압력 소멸 |
| 벙커↓ | `bunker_scale ≥ 한 주기 진척` | 한 주기로 완성됨 → 함정이 함정이 아님 |
| 벙커↑ | `bunker_scale ≤ 전 기간 진척` | 아무리 파도 의미 없음 |
| 부담 | `벙커 1인부담 > 요격기 1인부담` | 벙커가 더 싸짐 → 아무도 요격기를 안 함 |
| 노브 | `comm_intl_ai > comm_intl_learner` (**전 구간**) | 원문 경로가 더 비쌈 → 경로 선택이 무의미 |

> **노브는 리스트입니다** (`[6, 12, 24, 48]`). **최저값에서도** 성립해야 합니다.

### A-4. 자가 검증 (필수)

확정값으로 로드하면 이 값이 나와야 합니다.

```
A 2700   B 4500   E 6480   <   임계 8019   <   C×0.6 8100
벙커 : 한 주기 전력 → 생존 28%,  전 기간 → 81%
```

**그리고 일부러 깨뜨려 보세요.** 아래를 각각 넣으면 지정된 검사가 걸려야 합니다.

| 바꿀 값 | 걸려야 하는 검사 |
|---|---|
| `interceptor: 4000` | ★B, ★E |
| `interceptor: 8200` | ★C |
| `success_prob: 0.2` (임계는 그대로) | ★C — 임계가 진척 단위임을 확인 |
| `success_prob: 0.5` (임계는 그대로) | ★E — 창이 위로 스케일되어 임계가 하한 아래로 |
| `bunker_scale: 800` | 벙커↓ (하한은 `to_progress(한 주기) = 900`) |
| `comm_intl_ai: [4, 12]` (learner 5) | 노브 |

**`success_prob` 두 줄이 중요합니다.** 임계값은 **진척 단위**이므로 `success_prob`을
바꾸면 창 전체가 같은 비율로 움직입니다. `interceptor`를 고정한 채 올리면 창이 위로
가서 임계가 **하한(E) 아래**로 떨어지고, 내리면 창이 내려와 **상한(C×0.6) 위**로
올라갑니다. 어느 쪽이든 검사가 잡아야 하고, **한 값만 바꾸고 임계를 재계산하지 않는
것이 조용히 세계를 망가뜨리는 전형적인 경로**입니다.

---

## Part B — 상태와 턴 루프 (60점)

### B-1. `core/survival.py` (10점)

```python
"""확률적 수명. spec 2.2."""
import math


def survival(age: int, lam: float, k: float) -> float:
    """S(a) = exp(−(a/λ)^k).  나이 a 를 넘길 확률."""
    ...


def hazard(age: int, lam: float, k: float) -> float:
    """나이 age 에서 age+1 로 가는 동안 죽을 확률.

    = 1 − S(age+1)/S(age).  조건부 확률이므로 S(age) 로 나눈다.
    """
    ...


def expected_life(lam: float, k: float) -> float:
    """기대수명 = E[살아낸 턴 수] = Σ_(a≥0) S(a) = 8.28.

    ⚠ a=0 항(S(0)=1)을 빼먹으면 정확히 1턴이 모자란다.
      명세 초기 판이 8.3 을 7.3 으로 적었던 것이 이 실수였다.

    ⚠ 로그에 기록하는 **마지막 생존 나이**는 Σ_(a≥1) S(a) = 7.28 이다.
      둘 다 맞는 값이고 의미가 다르다 (spec 2.2). 혼동하지 말 것.
    """
    ...
```

**검증:** `λ=8.26, k=8`에서 `expected_life() ≈ 8.28`, `survival(10) ≈ 0.0099`,
`survival(9) ≈ 0.137` (나이 9 를 넘길 확률).
나이별 사망 확률이 `0.00 0.00 0.00 0.00 0.01 0.06 0.17 0.40 0.70 0.93`.

### B-2. `core/state.py` (15점)

```python
"""세계 상태. spec 2.1 · 2.3 · 3.2."""
from dataclasses import dataclass, field


@dataclass
class Agent:
    id: str                  # "A1"
    country: str             # "A"
    native_lang: str         # "ja"
    known_langs: set[str]    # 읽을 수 있는 언어. 초기값은 모국어만
    parent_langs: set[str]   # 부모가 구사하던 언어. **자연사 교체는 빈 집합**
    budget: float
    age: int = 0
    lam: float = 0.0         # 수명 척도. wellness 로 증가. 본인에게도 비공개
    ap: float = 0.0
    alive: bool = True
    born_turn: int = 0
    born_by: str = "natural"  # "natural" | "procreate"


@dataclass
class Country:
    id: str
    lang: str
    land: str | None = None       # None | "bunker" | "interceptor"
    progress: float = 0.0
    national_capital: float = 0.0

    def multiplier(self, cfg) -> float:
        """1 + growth_coef × √(national_capital / growth_scale).

        √ 로 체감시키는 것이 필수다 — 선형이나 복리면 모두가 국가 투자만 하다가
        마지막에 아무것도 못 짓고 죽는 결말로 고정된다.
        """
        ...


@dataclass
class World:
    turn: int
    countries: dict[str, Country]
    agents: dict[str, Agent]
    # TODO: 유언 저장소. 계보별로 최근 testament_carry 개를 보관한다
    testaments: dict[str, list[str]] = field(default_factory=dict)
```

> **`parent_langs`가 왜 따로 있는가** — 언어 능력 자체는 상속되지 않지만(spec 3.3),
> 부모가 알았던 언어는 **학습 비용을 절반으로** 깎습니다(3.4). 그래서 능력이 아니라
> **할인 자격**으로만 넘어갑니다. 자연사로 교체된 에이전트는 부모가 없으므로
> 빈 집합이고, 결과적으로 가장 싼 눈금(`L/4`)은 `procreate` 자식에게만 열립니다.

### B-3. `core/policy.py` — 더미 정책 (5점)

과제 2에서 LLM이 들어갈 자리입니다. 지금은 규칙으로 채웁니다.

```python
"""더미 정책. 과제 2 에서 LLM 으로 교체된다.

인터페이스만 맞추는 것이 목적이다 — 반환 형식이 응답 스키마(spec 4.2)와
같아야 나중에 교체가 공짜가 된다.
"""


def dummy_policy(world, agent, cfg) -> dict:
    """actions 배열을 담은 dict 를 반환한다.

    최소 구현:
      - 예산의 절반을 자국 facility 에 invest
      - 나이가 7 이상이면 procreate (유언은 고정 문자열)
      - speak 은 하지 않는다 (메시지 라우팅은 과제 2)
    """
    return {"reasoning": "dummy", "actions": [...], "received": []}
```

### B-4. `core/loop.py` — 7단계 (30점)

**`docs/spec.md` 3.1의 순서를 정확히 지키세요.**

```python
"""턴 루프. spec 3.1."""


def run_turn(world, cfg, rng) -> None:
    """한 턴. 7단계 순서를 바꾸지 말 것."""
    # 1. 소득 지급     budget += income.per_turn × multiplier,  ap 리셋
    # 2. 관측 스냅샷   전원의 관측을 동시에 고정 (지금은 world 를 그대로 넘겨도 됨)
    # 3. 정책 호출     전원 (지금은 dummy_policy)
    # 4. 검증          배열 순서대로. 예산·AP 확인. procreate 처리
    # 5. 환경 갱신     투자 집계 → 확률 판정 → 진척, 국토 확정, national_capital
    # 6. 메시지 큐잉   (과제 2. 지금은 비어 있음)
    # 7. 생사 판정     나이 += 1 → 사망 주사위 → 죽은 자리에 신규 1명
    ...


def run(cfg, rng) -> World:
    """total_turns 만큼 돌리고 마지막에 생존 판정(spec 2.5)."""
    ...
```

#### 경계 조건 8개 — 배점의 절반이 여기 있습니다

| 경계 | 요구 동작 |
|---|---|
| 죽는 턴의 행동 | **유효.** 7번이 마지막이므로 이미 처리됨 |
| 죽는 턴에 보낸 메시지 | 도착함 (과제 2에서 확인) |
| 죽은 사람에게 도착한 메시지 | 폐기 + `recipient_dead` 기록 (과제 2) |
| 신규 에이전트의 첫 행동 | **다음 턴부터.** 태어난 턴에는 관측 스냅샷에 없었다 |
| 남은 예산 | `procreate`가 아니면 **소멸** |
| `procreate` 선언 턴 | 그 시점에 즉시 사망. **배열 뒤쪽 행동은 전부 버림** |
| `procreate` 자식 | 예산·유언을 받고 **다음 턴부터** 행동 |
| 마지막 턴(50) | 생사 판정 **생략**. 곧바로 생존 판정(2.5) |

> **`procreate`를 4번(검증) 단계에서 처리하는 이유** — 배열 순서가 곧 에이전트의
> 우선순위 표명입니다(spec 4.2). `procreate` 뒤에 투자를 적어두면 그 투자는
> 버려지고, 그것 자체가 우선순위 실수로 기록됩니다.

#### 생존 판정 (spec 2.5)

```
if interceptor_total ≥ thresholds.interceptor:
    전 인류 생존
else:
    벙커인 나라마다 p = 1 − exp(−progress/bunker_scale)
    주사위는 **국가당 한 번**. 그 나라 전원이 함께 살거나 함께 죽는다
    요격기 유치국은 벙커가 없으므로 확률조차 없이 전원 사망
```

> **개인별로 굴리지 마세요.** 3명이 독립 시행이 되어 결과 분포가 완전히 달라지고,
> 지표 2′(생존 **국가** 비율)의 정의와도 어긋납니다.

---

## 합격 기준

`tests/test_loop.py`를 만들어 아래를 통과시키세요. **전부 LLM 없이 검증됩니다.**

| # | 검사 | 기대값 |
|---|---|---|
| 1 | 인구 불변 | 모든 턴에서 살아있는 에이전트가 **정확히 9명** |
| 2 | 런당 사망 수 | **45 ~ 55** (3,000회 시뮬 평균 50.5, σ 1.5). **`procreate_age=None` 으로 측정** |
| 3 | 수명 분포 | **마지막 생존 나이** 평균 **7.1 ~ 7.4**, 최대 **11 이하**. `procreate_age=None` (규약은 spec 2.2) |
| 4 | 나이 10 초과 생존 | 전체 사망 중 **1% 근처** |
| 5 | 재현성 | 같은 `seed` 로 두 번 돌려 `state` 로그가 **바이트 단위로 동일** |
| 6 | 예산 보존 | 어느 턴에서도 `budget < 0` 이 없다 |
| 7 | 신규의 첫 행동 | 태어난 턴에 행동 기록이 **없다** |
| 8 | `procreate` | 자식 예산 = 부모의 남은 예산. 자연사 시 예산 = `initial_budget`(0) |
| 9 | 국토 배타성 | `land`가 한번 정해지면 투표 없이 바뀌지 않는다 |
| 10 | assert | 확정 config 통과, A-4의 5가지 변형이 각각 지정된 검사에 걸린다 |

> **2·3번은 `procreate` 를 끄고 재세요.** 더미 정책이 `age ≥ 7` 에서 `procreate` 하면
> 수명 분포가 7에서 잘려 평균이 6.9로 내려가고 사망 수도 54~56으로 올라갑니다.
> **수명 모델의 캘리브레이션과 상속 검증은 분리해야 합니다** —
> `run(cfg, rng, procreate_age=None)` 으로 2·3번을, 켠 상태로 8번을 검증하세요.
>
> **`sweep.py` 출력(약 40회)과 다른 이유** — 그쪽은 `wellness` 투자가 `λ` 를 늘려
> 수명이 길어진 정책이 섞여 있습니다. 더미는 `wellness` 에 투자하지 않습니다.
>
> **3번이 8.2 가 아니라 7.2 인 이유** — 기록하는 값이 *살아낸 턴 수*(8.24)가 아니라
> *마지막 생존 나이*(7.24)이기 때문입니다. 둘은 정확히 1 차이입니다 (spec 2.2).

### 재현성 주의

`random.Random(hash((name, value)))` 같은 시드를 **쓰지 마세요.**
파이썬 문자열 `hash()`는 프로세스마다 소금이 달라 실행할 때마다 결과가 바뀝니다.
Phase 0에서 실제로 이 버그 때문에 1위 조합이 매번 달라졌습니다.
`random.Random(cfg.run.seed)` 하나를 만들어 넘겨 쓰세요.

---

## 제출물

```
configs/base.yaml
core/config.py  core/asserts.py  core/survival.py  core/state.py
core/policy.py  core/loop.py
tests/test_config.py  tests/test_loop.py
```

그리고 **50턴 1회 실행 결과 요약**을 한 문단으로 적어 주세요.
사망 몇 회, 최종 진척, 국토 구성, 생존 판정 결과.
**숫자가 합격 기준과 다르면 다른 대로 적고 원인 추정을 붙이세요** —
맞춰서 적는 것보다 안 맞는 걸 정확히 보고하는 게 낫습니다.

## 하지 않아도 되는 것

- LLM 호출, 프롬프트 (과제 2)
- 메시지·번역·언어 학습 (과제 2)
- `messages.jsonl` / 지표 산출 (과제 3)
- 뷰어

## 자주 틀리는 곳

1. **임계값을 소득 단위로 비교** — 반드시 `× facility_eff × success_prob`. Phase 0 결함 1번
2. **`hazard`를 `1 − S(age+1)`로 계산** — 조건부 확률이므로 `S(age)`로 나눠야 함
3. **`expected_life`에서 `a=0` 항 누락** — 정확히 1턴 모자람.
   그리고 `expected_life`(살아낸 턴 수, 8.28)와 기록하는 *마지막 생존 나이*(7.28)를
   혼동하지 말 것 — 둘 다 맞는 값이고 의미가 다름
4. **생사 판정을 환경 갱신 앞에 둠** — 죽는 턴의 투자가 사라져 노이즈가 됨
5. **벙커 주사위를 개인별로** — 국가당 한 번
6. **`parent_langs`를 자연사 교체에도 채움** — 빈 집합이어야 함
