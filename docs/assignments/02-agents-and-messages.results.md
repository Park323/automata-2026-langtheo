# 과제 2 — 구현·검증 결과

> `feat/02-agents-and-messages` · カキガワ言語研究班
> 검증 환경: Python 3.11.9 / pytest 9.1.1 / pyyaml 6.0.3 (Windows)
> 백엔드: OpenRouter · 에이전트 `qwen/qwen-2.5-7b-instruct` · 번역 `mistralai/mistral-small-3.2-24b-instruct`
> 실행: `py -3.11 -m pytest -q` → **47 passed** (StubClient, API 0원)

---

## 1. 합격 기준 14개 — 결과

13개는 **StubClient 로 결정론적 검증**(API 안 씀), #14 만 실제 API 3턴 스모크.

| # | 검사 | 방법 | 상태 |
|---|---|---|---|
| 1 | 재현성 (같은 seed → state 바이트 동일) | `test_loop_agentic` | ✅ |
| 2 | AP 상한 (speak 4번째 실패) | `test_agent_loop` | ✅ |
| 3 | 예산 고갈 (음수 안 됨) | `test_agent_loop` | ✅ |
| 4 | procreate 즉시 종료 | `test_agent_loop` | ✅ |
| 5 | 도착 지연 (다음 턴 관측) | `test_loop_agentic` | ✅ |
| 6 | original 실패 (본문 미전달·비용 청구) | `test_messaging` | ✅ |
| 7 | 원문 병기 (학습자만) | `test_messaging` | ✅ |
| 8 | 절단 (fr 401→400, 번역 전) | `test_messaging` | ✅ |
| 9 | 번역 지시 ("간결" 없음) | `test_translate` | ✅ |
| 10 | 학습 할인 (300/150/75) | `test_agent_loop` | ✅ |
| 11 | **정보 은닉** (success_prob·λ·재앙까지 남은 턴 없음) | `test_agent_loop`·`test_loop_agentic` | ✅ |
| 12 | cap 순서 편향 없음 (비례 배분) | `test_loop_agentic` | ✅ |
| 13 | 병렬 == 순차 | `test_loop_agentic` | ✅ |
| 14 | **실제 API 3턴 스모크** (tool_calls 옴) | `scripts/smoke_3turns.py` | ✅ |

---

## 2. 3턴 스모크 결과 (실제 API · 제출물)

`qwen-2.5-7b` (에이전트) + `mistral-small-3.2-24b` (번역), knob=48, seed=1.

```
소요 시간   106.5s  (35.5s/턴)
에이전트    94콜   (in 205,808 / out 13,902 토큰)
번역        18콜   (in 1,224 / out 847 토큰)
이번 run 비용  ~$0.024   (에이전트 in$0.10·out$0.20/1M, 번역 in$0.094·out$0.25/1M)
생존 판정   intercept_failed  (3턴으론 요격기 미완성 — 정상)
```

**에이전트가 실제로 한 것 — 조율이 창발했다:**
- **국제 소통 22건.** A2→B·C, A1→B, B2→C 등 서로 협력을 제안. reasoning 에 *"공동 방어"*,
  *"국가 B 와 협력"*, *"방어 시스템 공동 구축"* 이 반복됨.
- **언어 장벽이 작동.** `original`(원문 직통) 경로로 보낸 메시지 여러 개가 **전달 실패**
  (수신자가 그 언어를 못 읽음). 나머지는 `ai`(번역) 경로. 도박 메커니즘이 그대로 관측됨.
- **자원 희소성이 씹힘.** 다수 에이전트가 예산 부족으로 speak(국제 AI, 48)을 못 보냄 — 딜레마 작동.
- **`propose_vote`·`invest`·`speak`** 전부 실제 호출됨. A1·A2 가 시설 용도 투표를 발의.

> **본실험 비용 추정:** 3턴 94콜 → 50턴 ≈ 1,570콜, 1런 ≈ $0.4. 노브 4조건 × 시드 5 ≈ 20런 ≈ **$8~10**.
> `qwen-2.5-72b` 는 14s/콜이라 본실험이 며칠 걸려 부적합. 7b 로 1s/콜이 되어 현실화됨 (아래 3장).

---

## 3. 모델 선택 (리뷰어가 요구한 3조건 충족)

| | 모델 | 근거 |
|---|---|---|
| 에이전트 | `qwen/qwen-2.5-7b-instruct` | qwen 계열 · tools 지원 · out **$0.20/1M**(<$1) · **~1s/콜** |
| 번역 | `mistralai/mistral-small-3.2-24b-instruct` | 파일럿 확정 · **고정** · 에이전트와 다른 계열 |

- ① <$1/1M ✅  ② tools 지원 ✅ (`--check` 로 확인)  ③ 에이전트≠번역기 계열 ✅
- **잠정값 `qwen-2.5-72b` 는 14s/콜**(provider Novita)이라 3턴에 20분. 같은 qwen-2.5 계열의 7B 로
  바꿔 1s/콜(provider Phala)이 되어 본실험이 가능해짐. 리뷰어가 "에이전트 모델은 고를 것"이라 한 범위 내.
- `configs/base.yaml` 의 `translate_model` 이 과제1 때 넣은 잠정값(qwen)으로 **잘못돼 있어** mistral 로 정정.

---

## 4. 이번 세션에서 잡은 버그 (실행이 드러낸 것 포함)

| # | 버그 | 조치 |
|---|---|---|
| 1 | `learn` 이 `known_langs` 를 즉시 변경 → 병렬 레이스·재현성 파괴 | sink 로 이연, 예산·AP만 즉시 (자가리뷰에서 발견) |
| 2 | `invest(facility)` 의 `to` 미검증 → LLM 이 에이전트 id("B2")를 주면 정산 KeyError | 예산 차감 전 국가 검증 (**실제 API 가 드러냄**) |
| 3 | malformed tool_call(`function`/`name` 없음) → 미포착 KeyError 로 전체 run 사망 | 방어적 파싱 |
| 4 | 단일 chat 실패(HTTP 400)가 9명 병렬 전체를 죽임 | **에이전트별 예외 격리** — 그 에이전트만 턴 종료 |
| 5 | assistant `content=""` + tool_calls → 일부 프로바이더 400 | `None` 으로 |
| 6 | `amount`/`text` 타입 미보증 → `float()`/`truncate` 크래시 | 타입 방어 |
| 7 | `--turns` 가 config 에 미반영 → 3턴 대신 50턴 실행 | config 반영 |
| 8 | reasoning 의 한자 등으로 콘솔 인코딩 크래시 (cp949) | stdout utf-8 |

전부 회귀 테스트로 고정. 2·3·4·5 는 **StubClient 로는 안 잡히고 실제 LLM 에서만** 나온 것 —
스모크의 값어치.

---

## 5. 명세에 없어 임의로 정한 것 / 미완 (확인 필요)

| 항목 | 결정 | 근거 |
|---|---|---|
| 국토 최초 확정 용도 | `interceptor` (`loop.py` `DEFAULT_FACILITY_TYPE`) | `invest(facility)` 가 용도(bunker/interceptor)를 안 실음 — 스펙 미명시. 과제3 투표로 정하기 전 기본값 |
| `received` 캡처 | 빈 `[]` | `understood`(수신 이해) 캡처는 사후 채점(과제3)용. 합격기준 미검사 |
| `MAX_STEPS` | 8 (과제 스켈레톤 값 유지) | — |
| 모델 품질 | qwen-7b 가 reasoning 을 중국어로 하거나 tool_call 을 content 에 흘리는 경우 있음 | 작동엔 지장 없음(도구 정상 파싱). 본실험 품질은 더 큰 모델 검토 여지 |

---

## 6. 스펙 준수 재점검 (자주 틀리는 곳 · 금지 항목)

- ✅ `intent` 는 로그 전용 — 수신자 inbox 에 안 감
- ✅ 번역 기본 지시는 `"번역하라"` 뿐 ("간결/정확/자연" 없음). 출력 형식 계약만 시스템이 붙임
- ✅ pivot 없음 — `L_발신 → L_수신` 직역
- ✅ `invest(facility/wellness/national)` 결과에 진척 증가분·λ 변화·multiplier 함수 없음 (역산 방지)
- ✅ 절단은 **번역 전**, **발신 언어 상한**(fr400/ja130/zh90), 재시도 없음
- ✅ `original` 실패해도 비용 청구 (환불 없음)
- ✅ 정산은 `agent_id` 정렬 순 (완료 순서 아님) → 재현성
- ✅ 프롬프트·도구 결과에 success_prob·λ·하자드·재앙까지 남은 턴 없음 (정보 은닉 #11)
