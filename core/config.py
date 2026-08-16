"""설정 로더. spec 7장.

YAML 을 중첩 dataclass 로 표현한다. frozen=True 로 두는 이유 —
런 도중에 설정이 바뀌면 재현이 깨진다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(Exception):
    """설정이 세계의 전제를 깨뜨릴 때. 메시지에 진단 정보를 반드시 담는다."""


@dataclass(frozen=True)
class Knob:
    comm_intl_ai: tuple[int, ...]   # 유일한 실험 변수 (리스트)


@dataclass(frozen=True)
class Costs:
    comm_domestic: float
    comm_intl_learner: float
    ask_clarification: float
    learn_base: float
    propose_vote: float
    observe_risk: float = 20.0


@dataclass(frozen=True)
class Thresholds:
    interceptor: float
    bunker_scale: float


@dataclass(frozen=True)
class Income:
    per_turn: float
    initial_budget: float


@dataclass(frozen=True)
class TurnCfg:
    action_points: float


@dataclass(frozen=True)
class AP:
    speak: float
    ask: float
    learn: float
    propose_vote: float
    invest: float
    memory_write: float
    procreate: float
    observe_risk: float = 0.3


@dataclass(frozen=True)
class Growth:
    growth_coef: float
    growth_scale: float


@dataclass(frozen=True)
class SurvivalCfg:
    k: float
    lambda_base: float


@dataclass(frozen=True)
class Risk:
    """운석 충돌까지 남은 턴의 관측. 정확도는 국가 자본(기술력)에 비례한다."""
    base_error: float = 50.0        # 자본 0 일 때의 오차 폭 (턴). 전체 기간이 100턴


@dataclass(frozen=True)
class Wellness:
    gain: float


@dataclass(frozen=True)
class Facility:
    eff: float
    cap_per_turn: float
    transition_requires_vote: bool
    transition_forfeits_progress: bool


@dataclass(frozen=True)
class CountryDef:
    id: str
    lang: str


@dataclass(frozen=True)
class World:
    countries: tuple[CountryDef, ...]
    agents_per_country: int
    total_turns: int
    epoch_turns: int
    success_prob: float


@dataclass(frozen=True)
class Inheritance:
    testament_max_chars: int
    testament_carry: int


@dataclass(frozen=True)
class Length:
    message_max_chars: dict[str, int]
    understood_max_chars: int
    translate_instruction_max_chars: int
    on_overflow: str


@dataclass(frozen=True)
class LLM:
    context_limit: int
    warn_ratio: float
    repeat_guard: int
    agent_model: str
    translate_model: str
    temperature: float
    # 사후 채점 전용 (spec 6.2). 런에는 쓰이지 않으므로 없으면 번역 모델을 쓴다 —
    # 판정자는 6방향 언어쌍을 다 읽어야 해서 에이전트용 7B 로는 부족하다.
    judge_model: str | None = None
    # 응답 상한. 없으면 모델이 같은 문장을 반복하다 붕괴한다 — 실측 최악 40,935자.
    # reasoning 인자까지 담아야 하므로 넉넉히 잡는다 (메시지 상한은 fr 400자).
    max_tokens: int | None = 2048
    # 사고 예산. OpenRouter 통합 파라미터를 그대로 넘긴다 (effort / max_tokens /
    # enabled / exclude). **max_tokens 와 나눠 써야 한다** — 사고가 전부 먹으면
    # 도구 호출이 안 나온다 (실측: qwen3.7-flash 6/68 이 length 로 잘림).
    reasoning: dict | None = None
    # 도구마다 reasoning 인자를 받을 것인가. **사고형 모델에서는 끈다** — 모델이 이미
    # 사고를 하고 그건 api_reasoning 으로 남는다. 끄면 그 사고가 reasonings 스트림에
    # 들어가 지표 4 가 계속 읽을 수 있다 (spec 12.1).
    tool_reasoning: bool = True
    # 프로바이더 라우팅. OpenRouter 는 같은 모델을 여러 업체가 서빙하고 **가격이
    # 다르다.** {"order": [...]} 로 우선순위를, {"only": [...]} 로 고정한다.
    provider: dict | None = None


@dataclass(frozen=True)
class Run:
    seed: int


@dataclass(frozen=True)
class Config:
    knob: Knob
    costs: Costs
    thresholds: Thresholds
    income: Income
    turn: TurnCfg
    ap: AP
    growth: Growth
    survival: SurvivalCfg
    wellness: Wellness
    risk: Risk
    facility: Facility
    world: World
    inheritance: Inheritance
    length: Length
    llm: LLM
    run: Run

    @property
    def k(self) -> float:
        """진척 환산 계수 = facility.eff × world.success_prob.

        임계값은 **진척 단위**다. 소득을 그대로 비교하면 안 된다 (spec 7장).
        이 프로퍼티를 반드시 경유하게 만들어라 — 단위 오류가 Phase 0 에서
        실제로 나온 결함 1번이다.
        """
        return self.facility.eff * self.world.success_prob


def from_dict(d: dict) -> Config:
    """assert 없이 dict 를 Config 로 만든다. (break 테스트가 이걸로 변형본을 만든다)"""
    return Config(
        knob=Knob(comm_intl_ai=tuple(d["knob"]["comm_intl_ai"])),
        costs=Costs(**d["costs"]),
        thresholds=Thresholds(**d["thresholds"]),
        income=Income(**d["income"]),
        turn=TurnCfg(**d["turn"]),
        ap=AP(**d["ap"]),
        growth=Growth(**d["growth"]),
        survival=SurvivalCfg(**d["survival"]),
        wellness=Wellness(**d["wellness"]),
        risk=Risk(**(d.get("risk") or {})),
        facility=Facility(**d["facility"]),
        world=World(
            countries=tuple(CountryDef(**c) for c in d["world"]["countries"]),
            agents_per_country=d["world"]["agents_per_country"],
            total_turns=d["world"]["total_turns"],
            epoch_turns=d["world"]["epoch_turns"],
            success_prob=d["world"]["success_prob"],
        ),
        inheritance=Inheritance(**d["inheritance"]),
        length=Length(**d["length"]),
        llm=LLM(**d["llm"]),
        run=Run(**d["run"]),
    )


def load(path: str | Path) -> Config:
    """YAML 을 읽어 Config 로 만들고 assert 를 전부 통과시킨다.

    통과하지 못하면 ConfigError. **조용히 넘어가면 안 된다.**
    """
    from core import asserts  # 순환 방지: 로드 시점에만 가져온다

    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    cfg = from_dict(d)
    failures = asserts.check_all(cfg)
    if failures:
        joined = "\n  - ".join(failures)
        raise ConfigError(
            f"config '{path}' 가 세계의 전제를 {len(failures)}건 위반했습니다:\n  - {joined}"
        )
    return cfg
