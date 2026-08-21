"""세계 상태. spec 2.1 · 2.3 · 3.2."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Agent:
    id: str                  # "Asla1". **재사용하지 않는다** — 죽으면 Asla4 가 온다
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
    uid: int = 0              # 인스턴스 고유 번호. id 도 유일하지만 하위 호환으로 남긴다
    memory: str = ""          # 기억 블록. memory_write 로 덮어쓴다 (spec 4.5)
    # 대화 이력. 태어나서 죽을 때까지 이어진다. 죽으면 소멸 — procreate 로도 안 넘어간다.
    # 넘기려면 1문장으로 압축해 유언에 옮겨야 하고, 그 압축이 곧 구전 감쇠다.
    convo: list = field(default_factory=list)
    last_prompt_tokens: int = 0   # 직전 호출의 실측 프롬프트 토큰 (압박 판정에 쓴다)
    budget_start: float = 0.0     # 이번 턴 **결정 시점**의 예산. x̂ 의 분모 (spec 8.4)
    # **이 해에 실제로 받은 수입.** 해 시작 문구가 이 값을 적는다.
    #
    # 전에는 문구가 렌더 시점에 다시 계산했다 — 순차 라운드로빈은 나중에 차례가 온 사람의
    # 오프닝이 남들이 national 에 넣은 **뒤**에 만들어지므로, 실제로 100 을 받았는데
    # 「+102」 라고 적혔다. 소득은 턴 시작에 한꺼번에 주어지는 그 해의 사실이다.
    income_this_year: float = 0.0
    wellness_spent: float = 0.0   # 생애 누적 wellness 출자. 본인에게는 비공개
    # 언어별 학습 진척 {lang: 누적 지불액}. 한 번에 다 낼 필요가 없고, 진척은
    # 관측에 그대로 보인다 (별도 관측 없이 투명). 완료 판정은 **그 순간의** 학습가로
    # 하므로, 국내 구사자가 생기면 필요액이 절반이 되어 즉시 완료될 수 있다 (3.4).
    lang_progress: dict = field(default_factory=dict)
    # **내가 그 나라 시설에 낸 누적액** {나라: 합계}. 생애 누적이고 상속되지 않는다.
    #
    # `learn` 은 누적을 돌려주는데(progress/required) `facility` 는 안 돌려주고 있었다.
    # 실측에서 13턴에 885원을 한 나라에 나눠 낸 에이전트가 **자기가 얼마 냈는지를
    # 메모로만** 알았고, memory_write 로 덮이면 그마저 사라진다.
    #
    # 내 행동의 합이라 상대 국가 정보를 흘리지 않는다. 그 나라의 총 진척이나 이번 턴
    # 그 나라에 모인 총액은 여전히 안 알려준다 — 전자는 4.1(타국 진척), 후자는 남의
    # 투자를 드러낸다.
    facility_invested: dict = field(default_factory=dict)
    # **이 사람이 표를 던진 採決의 해.** 없으면 -1.
    #
    # 3해 실측에서 Ranoa1 이 같은 採決에 두 번 던졌고, 두 표가 **둘 다 집계됐다** —
    # 세 사람 나라에서 interceptor 3표가 나왔지만 실제로 던진 사람은 둘이었고 Ranoa2 는
    # 던지지 않았다. 순차 라운드로빈은 한 해에 같은 사람을 두 번 방문할 수 있어
    # (메일로 깨우는 경로) 집계 쪽만 막아서는 부족하다.
    voted_turn: int = -1
    # **이번 콜에 기억 도구가 열려 있었나.** 목록을 고를 때 한 번 정하고 실행부가 그 값을
    # 읽는다 — `under_pressure()` 를 양쪽에서 각각 부르면 어긋난다. 목록은 **직전 콜의**
    # 프롬프트 크기로 정해지는데 실행은 **이번 콜의** 크기를 보게 되므로, 경계에서
    # 「목록에는 있는데 부르면 거절」 이 나온다. 그게 모델을 가장 헷갈리게 하는 모양이다.
    memory_open: bool = False
    # **아이를 낳았는가.** 생애 단 한 번이다 (spec 3.3, 8/21 개정). 나이가 아니라 이
    # 깃발로 막는다 — 나이는 매 해 오르지만 「이미 낳았다」 는 되돌아가지 않는다.
    has_borne: bool = False


@dataclass
class Country:
    id: str
    lang: str
    land: str | None = None       # None | "bunker" | "interceptor". **투표로만 정해진다**
    progress: float = 0.0
    national_capital: float = 0.0
    # 열린 제안 하나. {target, by, opened_turn, vote_turn}
    # 제안 → 3턴 유예(상의할 시간) → 네 번째 턴에 찬반 투표.
    proposal: dict | None = None

    def multiplier(self, cfg) -> float:
        """1 + growth_coef × √(national_capital / growth_scale).

        √ 로 체감시키는 것이 필수다 — 선형이나 복리면 모두가 국가 투자만 하다가
        마지막에 아무것도 못 짓고 죽는 결말로 고정된다.
        """
        return 1.0 + cfg.growth.growth_coef * math.sqrt(
            self.national_capital / cfg.growth.growth_scale
        )


@dataclass
class World:
    turn: int
    countries: dict[str, Country]
    agents: dict[str, Agent]
    # 유언 저장소. 계보(=slot id)별로 최근 testament_carry 개를 보관한다
    testaments: dict[str, list[str]] = field(default_factory=dict)
    # 메시지 큐. 이번 턴 발신은 다음 턴 도착 (spec 3.1). 각 원소:
    #   {"deliver_turn": t, "to": agent_id, "msg": inbox_dict}
    inbox_queue: list = field(default_factory=list)
    # 나라별 다음 번호. **id 를 재사용하지 않는다** — Asla1 이 죽고 그 자리에 다시
    # Asla1 이 나오면 「Asla1 이 죽었다」 는 부고 직후 명단에 Asla1 이 그대로 있게 된다.
    # Asla4, Asla5 … 로 이어 간다. 덤으로 id 가 곧 개체 식별자가 되어 로그 조인이 깔끔해진다.
    next_idx: dict = field(default_factory=dict)
