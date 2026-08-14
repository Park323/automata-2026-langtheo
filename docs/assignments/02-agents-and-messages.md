# 과제 2 — 에이전트와 소통

> **カキガワ言語研究班** · `automata-2026-langtheo`
> 참조 명세: `docs/spec.md` 3.4 · 4장 · 5장
> 선행: 과제 1 (`core/config.py` `state.py` `loop.py`)
> 백엔드: **OpenRouter** (OpenAI 호환 `tools` 파라미터)

---

## 0. 이 과제의 위치

```
1  설정 로더 + assert                          ✅ 과제 1
2  상태 + 턴 루프 7단계 (더미 정책)              ✅ 과제 1
────────────────────────────────────────────
3  에이전트 프롬프트 + function calling loop     ← 과제 2
4  9명 병렬 + 메시지 라우팅 + 번역 경로           ← 과제 2
────────────────────────────────────────────
5  로그 4종 + 지표 산출                         과제 3
```

**과제 1 에서 더미 정책이 앉아 있던 자리에 LLM 을 넣습니다.**
그리고 이 세계에서 처음으로 **소통**이 생깁니다 — 지금까지는 아무도 말하지 않았습니다.

> **과제 1 의 `interceptor_best`(max) 수정이 들어간 뒤에는 더미 정책으로 거의 확실히
> 실패합니다.** 부지 하나당 약 4,200 인데 임계가 8,019 이니까요. 그게 정상입니다 —
> **조율 없이는 못 짓는 세계**가 맞고, 조율은 이 과제에서 처음 가능해집니다.

---

## Part A — Function calling agentic loop (45점)

### A-1. `core/llm.py` — 클라이언트 (10점)

```python
"""LLM 백엔드. OpenRouter (OpenAI 호환).

tools/pilot/run_pilot.py 가 이미 같은 엔드포인트를 stdlib urllib 로 호출한다.
재시도·레이트리밋 처리 패턴을 거기서 가져와라.
"""
from typing import Protocol


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None) -> dict:
        """OpenAI 호환 응답을 그대로 반환한다.

        반환에서 쓰는 것: choices[0].message  (content 또는 tool_calls)
        """
        ...


class OpenRouterClient:
    """실제 호출. POST https://openrouter.ai/api/v1/chat/completions

    - 키는 .env.local 의 OPENROUTER_API_KEY
    - 429/5xx 는 지수 백오프로 재시도. 파일럿에서 324콜 중 29건이 레이트리밋이었다
    - **모델이 tools 를 지원하는지 확인할 것.** OpenRouter 모델 목록의
      supported_parameters 에 tools 가 있어야 한다. 없으면 tool_calls 가
      절대 오지 않고 content 에 JSON 을 흉내낸 문자열이 온다
    """


class StubClient:
    """테스트용. 미리 정해둔 tool_call 시퀀스를 순서대로 돌려준다.

    ⚠ 이게 이 과제의 채점 가능성을 만든다. LLM 은 비결정적이라
      StubClient 없이는 합격 기준을 쓸 수 없다. **먼저 만들어라.**
    """
```

### A-2. `core/tools.py` — 도구 정의 (15점)

행동 하나가 도구 하나입니다. `docs/spec.md` 4.2 의 표를 OpenAI 함수 스키마로 옮깁니다.

```python
TOOLS = [
    # speak / ask / invest / learn / propose_vote / procreate / end_turn
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": "한 명에게 메시지를 보낸다. 다음 턴에 도착한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "수신자 id (예: B2)"},
                    "route": {"type": "string", "enum": ["original", "ai"],
                              "description": "국제 발신에만 유효. 자국민이면 무시된다"},
                    "text": {"type": "string", "description": "본문. 당신의 모국어로"},
                    "intent": {"type": "string", "description": "전하려 한 것 한 문장. 상대에게 전달되지 않는다"},
                    "translate_instruction": {"type": "string",
                              "description": "번역기에 줄 지시. 비우면 '번역하라' 만 쓰인다"},
                },
                "required": ["to", "text", "intent"],
            },
        },
    },
    # ... 나머지 6개
]
```

**주의할 것**

| 도구 | 반드시 지킬 것 |
|---|---|
| `speak` / `ask` | `intent` 는 **로그 전용**. 절대 수신자에게 가지 않는다 |
| `ask` | `reply_to` (msg_id) 추가. 비용은 `ask_clarification` + 경로 비용 |
| `invest` | `target` ∈ {`wellness`, `national`, `facility`}. `facility` 만 `to`(국가) 지정 |
| `learn` | `country` 를 받는다. **언어 코드가 아니라 국가** (spec 3.4) |
| `propose_vote` | 국내 전용. `target` ∈ {`bunker`, `interceptor`} |
| `procreate` | `testament` 하나. **호출되면 그 턴이 즉시 끝난다** |
| `end_turn` | 인자 없음. 루프 종료 신호 |

> **`end_turn` 을 반드시 두세요.** 없으면 모델이 할 일이 없을 때 아무 도구나 부르거나
> `content` 만 돌려주고, 루프 종료 조건이 애매해집니다.

### A-3. `core/agent_loop.py` — 루프 (20점)

```python
"""한 에이전트의 한 턴. spec 4.2.

  messages = [system, user(관측)]
  반복 (MAX_STEPS 회 이하):
      resp = client.chat(messages, tools=TOOLS)
      tool_calls 없으면 종료
      각 tool_call 실행 → 결과를 role="tool" 메시지로 append
"""

MAX_STEPS = 8   # AP 3건 + 투자 몇 번 + end_turn. 폭주 방지용 상한


def run_agent_turn(world, agent, cfg, client, sink) -> dict:
    """반환 = 로그용 논리 형식 {"reasoning", "actions", "received"} (spec 4.2).

    ⚠ 도구는 세계를 즉시 바꾸지 않는다. 자기 budget/ap 만 즉시 차감하고
      효과는 sink(=이 턴의 의도 큐)에 넣는다. 국토 확정·진척 판정·cap 배분은
      전원의 루프가 끝난 뒤 loop.py 5단계에서 정산한다.
    """
```

**도구 결과가 이 설계의 핵심입니다.** 에이전트는 결과를 보고 가격을 배웁니다.

```python
# 좋은 결과 — 사실만, 감춰야 할 것은 빼고
{"ok": True,  "charged": 150, "discount": "국내에 구사자가 있습니다 (×0.5)",
 "budget_left": 320, "ap_left": 0.0}
{"ok": True,  "queued": "B2 에게 다음 턴 도착합니다", "charged": 48, "ap_left": 0.7}
{"ok": False, "error": "예산이 부족합니다. 필요 300, 보유 120"}
{"ok": False, "error": "AP 가 부족합니다. speak 은 0.3 이 필요합니다"}
```

> **🔴 도구 결과로 감춰야 할 것을 흘리지 마세요.**
> `invest(facility)` 는 *"투자 접수"* 까지만 답합니다. 진척 증가분을 그 자리에서
> 알려주면 **한 턴 안에서 `success_prob` 을 역산**할 수 있습니다.
> `invest(wellness)` 도 `λ` 변화를 답하지 않습니다. 진척은 **다음 턴 관측**에서 봅니다.
>
> 반대로 **학습 할인 사유는 알려줘야 합니다.** 국내 구사자의 존재가 가격으로
> 드러나는 것은 의도된 설계입니다 (spec 3.4) — *"우리나라에 아직 저 말 아는 사람이
> 있는가"* 를 에이전트가 체감해야 합니다.

**프롬프트는 `domains/meteor/prompts.py` 에 둡니다.** `docs/spec.md` 4.1 의 관측 블록을
그대로 렌더링하세요. 비용표를 **반드시** 보여줘야 하고(비용을 모르면 선택이 불가능),
`4.1 절대 넣지 않는 것` 목록을 어기지 마세요 — 특히 **목적함수 금지**입니다.

---

## Part B — 메시지와 번역 (45점)

### B-1. `core/messaging.py` — 라우팅 (20점)

```
발신 → 길이 절단 → 경로 판정 → (필요시) 번역 → 큐 → 다음 턴 도착
```

**길이 절단 (spec 5.3)** — 원문에 **발신자 언어의 상한**을 적용합니다.
`fr 400 / ja 130 / zh 90`. 번역 **전에** 자릅니다.
`truncated`, `chars_cut` 을 남기고 **재시도하지 않습니다** (사전 고지했으므로).

**경로 3종 (spec 5.1)**

| 경로 | 조건 | 비용 | 실패 |
|---|---|---|---|
| 자국 내 | 수신자가 자국민 | `comm_domestic` | 없음 |
| `original` | 국제 | `comm_intl_learner` | **수신자가 발신 언어를 못 읽으면 전달 실패. 비용은 청구** |
| `ai` | 국제 | `comm_intl_ai` (노브) | 없음 |

**전달 실패 시** 수신자에게는 *"읽을 수 없는 메시지가 도착했다"* 만 통지합니다 —
**발신자와 도착 사실만, 본문은 안 보입니다.** 여기서 정보 교환이 시작됩니다.
`events.jsonl` 에 기록하고 지표 9(전달 실패율)로 셉니다.

**원문 병기 (spec 5.2 금지 항목)** — `route=ai` 인데 수신자가 발신 언어를 읽을 수
있으면 **번역문과 원문을 함께** 보여줍니다. **이것이 학습의 가치입니다** —
원문을 읽을 수 있는 사람만 왜곡을 검출할 수 있습니다.
발신 언어를 **모르는** 수신자에게 원문을 주는 것은 금지입니다.

**라벨 (spec 5.4)** — `ai` 경로 메시지에 `[AI 번역]` 을 붙입니다.

### B-2. `core/translate.py` — 번역 호출 (15점)

**에이전트 호출과 별개의 LLM 호출 1회.** 메시지 1건당 1회입니다.

```
L_발신자  →  L_수신자        직역. 중계 언어(pivot) 없음
```

> **pivot 을 두지 마세요.** 초기 설계는 영어를 거치는 2단 번역이었는데 폐기했습니다 —
> pivot 을 고른 이유가 *"손실이 두 배가 되어서"* 였고 그건 의도 주입입니다.
> 직역이라 **6개 방향이 각각 다른 손실·생성 프로필**을 갖고, 언어쌍이 대조군이 됩니다.

시스템 메시지는 **출력 형식 계약 하나뿐**입니다. 이것만 시스템이 붙일 수 있습니다.

```
You are a translation engine. Output ONLY the translated text.
No explanation, no alternatives, no quotes, no notes.
```

번역 방식 지시는 **발신 에이전트의 `translate_instruction`** 을 그대로 씁니다.
없으면 중립 기본값 **`"번역하라"` 만** 씁니다.

> **🔴 기본값에 "간결하게" 를 넣으면 즉시 위반입니다.** 압축은 에이전트가 요구할 때만
> 일어나야 합니다. 출력 형식 계약에도 *"정확하게"*, *"자연스럽게"* 같은 말을 넣지 마세요.
> 파일럿에서 **"간결하게" 지시가 실제로 손실을 만드는 것**이 확인됐습니다 —
> 그 효과는 에이전트의 선택이어야 합니다.

번역 모델은 `mistral-small-3.2-24b` 로 **고정**입니다. 조건마다 다른 모델을 쓰면
노브 효과와 모델 효과가 섞여 비교가 깨집니다. 에이전트 모델(Qwen 계열)과 계열이
달라야 하는 것도 의도된 것입니다.

`translate_logprob_mean` 은 지원되면 남기고, 안 되면 `null` 로 두고 넘어가세요.
**이것 때문에 백엔드를 바꾸지 마세요.**

### B-3. `core/loop.py` 수정 — 병렬과 정산 (10점)

```python
# 3. 정책 호출 — 9명을 **동시에**. I/O 바운드이므로 ThreadPoolExecutor
#    ⚠ 결과를 세계에 반영할 때는 반드시 agent id 정렬 순으로. 완료 순서로 하면
#      재현성이 깨진다
# 5. 환경 갱신 — 여기서 처음으로 세계가 바뀐다
#    cap_per_turn 초과분은 **비례 배분**. sorted(id) 순으로 소진하면 A1 이 항상 유리해져
#    spec 3.1 이 금지한 순서 편향이 된다 (과제 1 리뷰 지적)
```

---

## 합격 기준

**`StubClient` 로 전부 검증합니다.** 실제 API 는 마지막 1회 스모크 테스트만.

| # | 검사 | 기대 |
|---|---|---|
| 1 | 재현성 | 같은 `StubClient` 스크립트 + 같은 seed → `state` 로그 **바이트 동일** |
| 2 | AP 상한 | `speak` 4번째 호출이 `ok: False` (AP 0.3 × 3 = 0.9) |
| 3 | 예산 고갈 | 부족하면 `ok: False`, 예산이 **음수가 되지 않음** |
| 4 | `procreate` | 호출 즉시 턴 종료. 뒤의 tool_call 이 실행되지 않음 |
| 5 | 도착 지연 | 이번 턴 발신이 **다음 턴** 관측에 나타남. 같은 턴에는 없음 |
| 6 | `original` 실패 | 수신자가 발신 언어를 모르면 본문 미전달 + **비용은 청구** |
| 7 | 원문 병기 | 수신자가 발신 언어를 알면 `ai` 경로에서 번역문 + 원문 둘 다 |
| 8 | 절단 | `fr` 401자 → 400자로 잘리고 `chars_cut=1`, 번역 입력도 잘린 것 |
| 9 | 번역 지시 | `translate_instruction=None` → 번역 프롬프트에 "간결" 류 단어가 **없음** |
| 10 | 학습 할인 | 국내 구사자 있음 → 150, 부모까지 → 75, 둘 다 없음 → 300 |
| 11 | 정보 은닉 | 도구 결과·프롬프트에 `success_prob` · `λ` · 하자드 곡선 · 재앙까지 남은 턴이 **없음** |
| 12 | 순서 편향 | `cap_per_turn` 을 넘겨 투자할 때 국가별 배분이 **호출 순서와 무관** |
| 13 | 병렬 | 9명 동시 호출이 순차 대비 유의미하게 빠르고, 결과는 순차와 **동일** |
| 14 | 스모크 | 실제 OpenRouter 로 **3턴** 실행. `tool_calls` 가 실제로 오는지 확인 |

**11번이 가장 중요합니다.** 하나라도 새면 실험이 죽습니다.
프롬프트와 도구 결과 문자열 전체를 훑어 금지 항목이 없는지 검사하는 테스트를 쓰세요.

---

## 모델 선택 — 1M 토큰당 $1 미만

**테스트에 쓸 모델은 1M 토큰당 $1 미만으로 고르세요.** agentic loop 는 호출이 4~6배라
모델값이 그대로 곱해집니다. 그리고 조건 × 시드까지 곱해지므로, 비싼 모델을 쓰면
스모크 테스트만 하고 본실험을 못 돌립니다.

제약이 셋이고 **동시에** 만족해야 합니다.

```
① 1M 토큰당 $1 미만          (입력·출력 각각. 출력이 보통 더 비싸다)
② tools 파라미터 지원         없으면 tool_calls 가 아예 안 온다
③ 에이전트 ≠ 번역기 계열      같은 계열이면 왜곡이 상쇄될 수 있다
```

확인 방법 — OpenRouter 의 모델 목록에서 `pricing` 과 `supported_parameters` 를 함께 봅니다.

```bash
curl -s https://openrouter.ai/api/v1/models \
  | jq -r '.data[] | select(.supported_parameters | index("tools"))
      | select((.pricing.completion|tonumber) < 0.000001)
      | "\(.id)  in=\(.pricing.prompt)  out=\(.pricing.completion)"' | sort
```

> `pricing` 은 **토큰 1개당 달러**입니다. `1M당 $1` 은 `0.000001` 입니다.
> 위 필터는 출력 기준이니 입력도 함께 보세요.

**정해진 것과 고를 것**

| | 모델 | 근거 |
|---|---|---|
| 번역 | `mistral-small-3.2-24b` **고정** | 파일럿에서 3언어 분산 최소로 확정. 조건마다 바꾸면 노브 효과와 섞여 비교가 깨진다 |
| 에이전트 | Qwen 계열에서 **위 3조건을 만족하는 것** | `configs/base.yaml` 의 `qwen/qwen-2.5-72b-instruct` 는 잠정값이다. 값과 tools 지원을 직접 확인하고 바꿔도 된다 |

> **번역 모델이 $1 을 넘으면 조용히 바꾸지 말고 알려주세요.** 그건 파일럿에서 확정한
> 값이라 바꾸면 파일럿 결과의 근거가 사라집니다. 바꿀지 말지는 같이 판단합니다.

**고른 모델과 그 가격을 결과 문서에 적어 주세요.** 나중에 "왜 이 모델인가" 를
답해야 하고, 심사에서 실제로 묻는 항목입니다.

## 비용과 런타임

**agentic loop 는 배열 방식보다 에이전트 호출이 4~6배입니다.** 미리 알고 시작하세요.

```
에이전트 호출   9명 × 50턴 × (도구 호출 수 + 1)  ≈  1,800 ~ 2,700 회/런
번역 호출       국제 AI 메시지 1건당 1회          ≈  수백 회/런
```

`MAX_STEPS = 8` 이 상한을 잡아줍니다. **3턴 스모크 테스트로 실제 호출 수를 먼저
재고, 1런 비용을 추정한 다음 전체를 돌리세요.** 노브 4단계 × 시드까지 곱해집니다.

> 캐시나 배치가 필요해지면 그때 얘기합시다. **지금은 정확성이 먼저입니다.**

---

## 제출물

```
core/llm.py  core/tools.py  core/agent_loop.py  core/messaging.py  core/translate.py
core/loop.py                      (수정 — 병렬 · 정산 · cap 비례 배분)
domains/meteor/prompts.py
tests/test_agent_loop.py  tests/test_messaging.py  tests/test_translate.py
scripts/smoke_3turns.py
```

그리고 **3턴 스모크 실행 결과**를 적어 주세요 — 실제 호출 수, 소요 시간, 추정 비용,
그리고 **에이전트가 실제로 무엇을 했는지 한 문단**. 말을 걸었나요? 누구에게?
`reasoning` 에 뭐라고 썼나요? **그게 이 프로젝트의 첫 관측입니다.**

## 하지 않아도 되는 것

- `messages.jsonl` 전체 스키마 · 지표 산출 · 사후 채점 (과제 3)
- 화용 표지 카운터 (과제 3)
- 뷰어

## 자주 틀리는 곳

1. **`intent` 를 수신자에게 노출** — 로그 전용. 노출되면 왜곡이 즉시 드러나 실험이 죽는다
2. **번역 기본 지시에 "간결하게" 포함** — 왜곡 주입. 중립은 `"번역하라"` 뿐
3. **pivot(영어 경유) 번역** — 폐기된 설계. 직역이다
4. **도구 결과로 진척 증가분을 즉시 반환** — `success_prob` 역산 경로
5. **절단을 번역 후에 적용** — 원문 기준·발신 언어 상한·번역 전이다
6. **`original` 실패 시 비용 환불** — 실패해도 청구한다. 그래야 도박이다
7. **완료 순서로 세계 반영** — 재현성이 깨진다. `sorted(id)` 순으로
8. **모델이 `tools` 미지원인 것을 모르고 진행** — `tool_calls` 가 안 오고 `content` 에
   JSON 흉내가 온다. `supported_parameters` 를 먼저 확인
9. **비싼 모델로 테스트** — 1M 당 $1 미만. agentic loop 는 호출이 4~6배이고
   조건 × 시드까지 곱해진다. 스모크만 하고 본실험을 못 돌리게 된다
