"""설정 로더. spec 7장.

YAML 을 중첩 dataclass 로 표현한다. frozen=True 로 두는 이유 —
런 도중에 설정이 바뀌면 재현이 깨진다.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(Exception):
    """설정이 세계의 전제를 깨뜨릴 때. 메시지에 진단 정보를 반드시 담는다."""


@dataclass(frozen=True)
class Knob:
    # **유일한 실험 변수 — 이제 AP 다** (8/25). 돈으로 매기던 `comm_intl_ai` 를 지웠다.
    comm_intl_ai_ap: tuple

@dataclass(frozen=True)
class Costs:
    # **가격이 아니라 양이다** (8/25 · AP 전면 통일). 돈 항목 넷을 지웠다
    # (comm_domestic · comm_intl_learner · observe_risk · propose_vote) — `ap.*` 가 대신한다.
    learn_base: float          # 한 언어를 익히기까지 쌓아야 하는 **진척**
    learn_speedup: float       # 사유 하나당 배속 +
    unit: float                # 한 번의 invest·learn 이 옮기는 **양**

@dataclass(frozen=True)
class Thresholds:
    interceptor: float
    bunker: float


@dataclass(frozen=True)
class TurnCfg:
    action_points: float


@dataclass(frozen=True)
class AP:
    speak: float
    propose_vote: float
    memory_write: float
    # **기본값을 두지 않는다.** 넷 다 기본값이 있었고 셋이 yaml 과 달랐다
    # (give 0.2/0.1 · observe_risk 0.3/0.5 · unit 0.1/0.2). 「숫자를 두 군데 적으면
    # 하나가 낡는다」 — yaml 이 유일한 출처다.
    give: float
    observe_risk: float
    vote: float                    # 採決과 제안은 무게가 다르다 (4.4)
    # **한 번의 invest·learn 이 먹는 AP.** 금액은 costs.unit 으로 고정이라 금액별 계산이
    # 없다 — learn_full·invest_wellness·invest_per_ap 를 그래서 없앴다 (4.4).
    unit: float


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
    # 자본 0 일 때의 **표준편차**(상대). 남은 턴은 전체 기간의 이 비율이 σ 가 되고,
    # 임계는 그 값의 이 비율이 σ 가 된다. 정규분포라 꼬리에서는 크게 빗나간다 —
    # 의도된 것이다. 절대 턴 수로 두면 total_turns 를 줄인 런에서 깨진다.
    sigma_ratio: float = 0.25


@dataclass(frozen=True)
class Wellness:
    gain: float


@dataclass(frozen=True)
class Facility:
    eff: float
    transition_requires_vote: bool
    transition_forfeits_progress: bool
    # 개체별 「한 번에 옮기는 액수」 배수를 뽑는 단계들. 소득 배수와 **독립**이다.
    # 평균이 1.0 이어야 한다 — 국가의 돈→진척 전환 능력이 여기 걸린다.
    #
    # **기본값 있는 필드는 맨 뒤로.** 이 프로젝트에서 네 번째로 밟은 자리다 —
    # dataclass 는 기본값 뒤에 기본값 없는 필드를 두면 거절한다.
    throughput_spread: tuple = (1.0,)
    # **나라마다 요격기를 짓는 속도가 다르다** (8/23). 같은 돈이 나라에 따라 다른
    # 진척이 된다 — 개체별 `throughput_spread` 의 국가판이다. 국가 단위 비교우위를
    # 만들어 「어디에 몰아줄 것인가」 를 진짜 문제로 만든다.
    #
    # 나라 수만큼의 **순열**로 배정한다 — 독립 추출이면 평균이 1.0 에서 흔들리고,
    # 그러면 임계 창이 어긋난다.
    #
    # **벙커에는 안 걸린다.** 요격기에만 걸어야 「최고 효율 나라가 벙커를 골랐다」 가
    # 진짜 손실이 되고, 함정이 더 날카로워진다.
    build_spread: tuple = (1.0,)


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
    # 초기 나이를 1..이 값에서 뽑는다. 전원 0살이면 한꺼번에 죽어 세계가 백지가 된다.
    init_age_max: int = 10
    # **성인 나이.** 이 나이부터 아이를 낳을 수 있고, 이 나이부터 소득을 받는다.
    # 그전에는 부모가 주는 돈이 전부다.


@dataclass(frozen=True)
class Inheritance:
    """**유언과 언어 진척 상속을 없앴다** (8/21).

    부모가 살아 있으므로 세대 간 전달은 `speak` 로 한다 — 기존 메시지 채널로 관측되고
    여러 해에 걸쳐 반복할 수도 있다. 그리고 부모가 살아 있는 것 자체가 국내 구사자이므로
    아이의 학습이 이미 두 겹으로 싸다 (부모 −50 × 국내 −50). 진척 절반까지 얹으면 능력이
    사실상 상속된다.

    남는 것은 **부모 할인 자격**뿐이다.
    """

    # **필드가 없다.** 남는 것은 부모 할인 자격(`parent_langs`)뿐이고 그것은
    # `_bear_child` 가 직접 넘긴다 — 켜고 끄는 설정이 아니다. `parent_discount: true` 를
    # 두었다가 지웠다: **코드가 읽지 않는 설정은 있으면 거짓말을 한다.**


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
    # 도구 호출을 강제할 것인가. **"required" 는 프로바이더 선택을 덮어쓴다** (8/25) —
    # 업체 전부가 `supported_parameters` 에 `tool_choice` 를 신고하지만 실제로 `required`
    # 를 받는 곳은 적고, OpenRouter 가 실제 지원으로 걸러내면서 우리 `order` 를 통째로
    # 건너뛴다. 실측: 같은 요청에서 `required` → Sail Research 21초, 떼면 GMICloud 5.8초.
    #
    # 애초에 `required` 는 **사고를 껐을 때** 넣은 우회책이었다 (8/16 · d10ca2e) —
    # 「모델이 content 에 숙고를 쏟고 그대로 끝낸다 · 턴의 2~7% 가 날아갔다」. 사고를
    # 되켰으므로 숙고는 reasoning 채널로 간다. 안전망은 그대로다:
    # `_recover_tool_calls` 가 content 에서 도구 호출을 파싱하고, 실패하면 `no_tool_call`
    # 로 로그에 남는다 — 조용히 사라지지 않는다.
    tool_choice: str = "auto"
    # 프로바이더 라우팅. OpenRouter 는 같은 모델을 여러 업체가 서빙하고 **가격이
    # 다르다.** {"order": [...]} 로 우선순위를, {"only": [...]} 로 고정한다.
    provider: dict | None = None
    # **번역기는 따로 고정한다** (8/26 · Eddie). 에이전트 클라이언트만 `provider` 를
    # 받고 번역기는 안 받고 있었다 — 375콜이 DeepInfra 107 · Mistral 93 · Parasail 21
    # 로 흩어졌고 429 154건이 났다. 그런데 **429 보다 심각한 것은 양자화다**:
    #
    #     DeepInfra  fp8   ·  Parasail  bf16  ·  Mistral  unknown
    #
    # 번역 왜곡이 이 실험의 **종속변수**다 (지표 4c·4d·7). 그 왜곡을 만드는 기계가 런
    # 중간에 세 번 바뀌면 파일럿에서 「3언어 분산 최소」 로 이 모델을 고른 근거가 이
    # 런에 적용되지 않는다. 에이전트 쪽에는 그 논리를 적용해 놓고 번역기만 빠졌다.
    translate_provider: dict | None = None
    # **429 폭풍이면 런을 세운다** (8/26 · Eddie). 재시도를 다 쓴 429 가 이만큼 쌓이면
    # `RateLimitStorm` 으로 런을 끝낸다. 중간에 회수된 429 는 안 센다 — 시간만 먹었지
    # 데이터를 안 상하게 한다 (`260826-002-ai010`: 154건 중 실제 손실 23건).
    #
    # 죽은 번역은 **세계에 뚫린 구멍**이고 되돌릴 수 없다. 그럴 바에는 세우고 나중에
    # 이어한다 — 매해 복원점이 있으므로 이어붙이는 값이 싸다. 0 이면 끈다.
    rate_limit_abort: int = 5


@dataclass(frozen=True)
class Run:
    seed: int


@dataclass(frozen=True)
class Config:
    knob: Knob
    costs: Costs
    thresholds: Thresholds
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


def _world_from(d: dict) -> World:
    """`world` 절을 통째로 넘긴다. **모르는 키는 거절한다.**

    전에는 다섯 필드만 골라 넘겼다. 그래서 `adult_age`·`init_age_spread`·`init_age_max` 가
    **YAML 에서 읽히지 않았고**, 기본값과 우연히 같아서 드러나지 않았다 — config 를 고쳐도
    아무 일이 안 일어나는 상태다. 조용한 무시가 가장 나쁜 실패다.

    이제 dataclass 필드와 대조한다. 새 키를 yaml 에만 넣고 배선을 잊으면 **로드가 실패**한다.
    """
    known = {f.name for f in dataclasses.fields(World)}
    unknown = set(d) - known
    if unknown:
        raise ConfigError(
            f"world 절에 모르는 키: {sorted(unknown)}. "
            f"`World` 에 필드를 추가하거나 yaml 에서 지우세요 — "
            f"조용히 무시되면 config 를 고쳐도 아무 일이 안 일어납니다.")
    kw = dict(d)
    kw["countries"] = tuple(CountryDef(**c) for c in d["countries"])
    return World(**kw)


def from_dict(d: dict) -> Config:
    """assert 없이 dict 를 Config 로 만든다. (break 테스트가 이걸로 변형본을 만든다)"""
    return Config(
        knob=Knob(comm_intl_ai_ap=tuple(d["knob"]["comm_intl_ai_ap"])),
        costs=Costs(**d["costs"]),
        thresholds=Thresholds(**d["thresholds"]),
        turn=TurnCfg(**d["turn"]),
        ap=AP(**d["ap"]),
        growth=Growth(**d["growth"]),
        survival=SurvivalCfg(**d["survival"]),
        wellness=Wellness(**d["wellness"]),
        risk=Risk(**(d.get("risk") or {})),
        facility=Facility(**{**d["facility"], "throughput_spread":
                             tuple(d["facility"].get("throughput_spread", (1.0,)))}),
        world=_world_from(d["world"]),
        # `inheritance` 절은 없앴다 — 남은 것이 코드에 있고 설정할 것이 없다
        inheritance=Inheritance(),
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
