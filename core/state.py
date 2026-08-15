"""세계 상태. spec 2.1 · 2.3 · 3.2."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Agent:
    id: str                  # "A1" (슬롯. 죽으면 신규가 같은 id 를 물려받는다)
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
    uid: int = 0              # 인스턴스 고유 번호. id 는 슬롯이라 세대마다 재사용된다
    # ── 개체 기억 (spec 4.5). 인스턴스에 속하므로 죽음(=재생성)으로 전부 소멸한다 ──
    # 태어나서 죽을 때까지 이어지는 대화. 매 턴 관측이 뒤에 붙고 도구 왕복도 남는다.
    messages: list = field(default_factory=list)
    memory: str = ""          # memory_write 로 덮어쓰는 기억 블록. 관측에 [내 메모] 로 표시
    mem_pressure: bool = False  # 직전 턴이 warn_ratio 를 넘겼다 → 다음 관측 앞에 통지 한 줄
    last_prompt_tokens: int = 0  # 직전 응답 usage.prompt_tokens (압박 판정용, API 가 주면)


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
