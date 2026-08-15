# 과제 2 — 구현·검증 결과

> `feat/02-agents-and-messages` · カキガワ言語研究班
> 검증 환경: Python 3.11.9 / pytest 9.1.1 / pyyaml 6.0.3 (Windows)
> 백엔드: OpenRouter · 에이전트 `qwen/qwen-2.5-7b-instruct` · 번역 `mistralai/mistral-small-3.2-24b-instruct`
> 실행: `py -3.11 -m pytest -q` → **49 passed** (StubClient, API 0원)

## 리뷰 반영 (PR #6, Park323)

프롬프트(`domains/meteor/prompts.py`)에 3건 지적 → 전부 반영:

| 지적 | 조치 |
|---|---|
| **AI 번역의 "왜곡"을 프롬프트가 직접 알려줌 — 치명적 누수** | 제거. AI 경로는 "always reaches them"(채널 사실)만. 라벨도 `[AI translation]` 뿐 (spec 5.4) |
| **국가명 A/B/C 는 서열 편향** | 서열·실세계 연상 없는 가명 **Asla / Ranoa / Miris** 로 (언어 ja/zh/fr 는 고정). 언어도 나라 기준으로 참조("Ranoa's language") |
| **정보가 너무 많음 + 프롬프트는 영어** | SYSTEM 700자→319자 (시설·경로 설명 제거, 관측·도구가 대신). 전 프롬프트·도구 설명·도구 결과 영어화 |

> **트리밍 검증:** 시설이 뭔지 안 알려줬는데도 스모크에서 에이전트가 **스스로 요격기 전환을 발의**했다
> (Asla1·Asla3). "덜 알려주되 스스로 추론"이 실제로 작동 — 정보를 과하게 줄인 것이 아님을 확인.

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
아래는 **리뷰 반영(영어·가명·SYSTEM 축소) 후 코드**로 실행한 결과 (에러 0 완주).

```
소요 시간   37.0s  (12.3s/턴)
에이전트    102콜  (in 152,912 / out 7,367 토큰)
번역        5콜    (in 350 / out 172 토큰)
이번 run 비용  ~$0.016   (에이전트 in$0.10·out$0.20/1M, 번역 in$0.094·out$0.25/1M)
생존 판정   intercept_failed  (3턴으론 요격기 미완성 — 정상)
```

**에이전트가 실제로 한 것 — 목표를 스스로 추론했다:**
- **🔥 요격기 목표를 자력 추론.** 프롬프트에서 "interceptor 가 뭐하는지"를 뺐는데도, Asla1·Asla3 가
  **요격기로 시설 전환을 발의**(`propose_vote interceptor`). 운석 상황 + 도구 선택지만으로 목표를 찾아냄 —
  스펙 철학("알아서 추론")이 실제로 작동.
- **국제 소통 13건.** ai / domestic / original 혼용. `original` 은 전달 실패도 발생(언어 장벽), 발신자는
  다음 턴 실패 통지를 받음.
- **투자·투표·소통** 전부 호출. national(생산)·wellness(수명)·facility 투자, propose_vote, procreate.
- **에러 0.** 리뷰 전 세션에서 잡은 malformed/400/인코딩 수정이 모두 유효 — 3턴 전 구간 크래시 없이 완주.

> **본실험 비용 추정:** 3턴 ~100콜 → 50턴 ≈ 1,700콜, 1런 ≈ $0.3. 노브 4조건 × 시드 5 ≈ 20런 ≈ **$6~8**.
> `qwen-2.5-72b` 는 14s/콜이라 본실험이 며칠 걸려 부적합. 7b 로 ~1s/콜이 되어 현실화됨 (아래 3장).

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
| 9 | **발신자 실패 통지가 렌더 안 됨** → 프롬프트에 "None로부터" 쓰레기로 들어감 | `render_inbox` 에 `delivery_failed_to` 분기 추가 (심층 리뷰에서 발견) |
| 10 | `_state_line` 이 `lam`·`known_langs` 미기록 → 학습·wellness 비결정성을 재현성 검사가 못 잡음 | 두 필드 추가로 재현성 보장 강화 |
| 11 | 에이전트가 **자기 자신에게 메시지**를 보냄(무의미한 낭비) | `to == 자기 id` 거부 (리뷰 후 스모크에서 관측) |

전부 회귀 테스트로 고정. 2·3·4·5 는 **StubClient 로는 안 잡히고 실제 LLM 에서만** 나온 것 —
스모크의 값어치. 9 는 실제 스모크에서 original 전달 실패가 다수 발생해 발신자들이 깨진 통지를 받고 있었다.

---

## 5. 명세에 없어 임의로 정한 것 / 미완 (확인 필요)

| 항목 | 결정 | 근거 |
|---|---|---|
| 국토 최초 확정 용도 | `interceptor` (`loop.py` `DEFAULT_FACILITY_TYPE`) | `invest(facility)` 가 용도(bunker/interceptor)를 안 실음 — 스펙 미명시. 과제3 투표로 정하기 전 기본값 |
| `received` 캡처 | 빈 `[]` | `understood`(수신 이해) 캡처는 사후 채점(과제3)용. 합격기준 미검사 |
| "지금까지 알아낸 것" | 고정 "아직 없음" | 대화·유언으로 타인 언어능력을 누적 학습하는 이력은 과제3. 현재는 이번 턴 inbox 로만 앎 |
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
