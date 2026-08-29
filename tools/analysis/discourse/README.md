# discourse — 담화·think 전수 분류와 경제 분석

README·RESULTS 의 담화 3층 분류, think 동인, AP 원장·x̂ 표가 전부 여기서 나왔다.
**라벨 산출물을 동봉했으므로 재실행 없이 표를 재현할 수 있다.** 재실행하려면
OpenRouter 키(`.env.local`)가 필요하고, 분류기는 qwen3.6-35b-a3b(사고 OFF ·
temperature 0 · 조건 은닉 · 배치 25)를 쓴다. 모든 스크립트는 **리포 루트에서** 실행한다:

```bash
python3 tools/analysis/discourse/classify.py      # 출력은 이 디렉터리에 떨어진다
```

## 분류 스크립트 → 산출물

| 스크립트 | 산출물 | 행수 | 내용 |
|---|---|---|---|
| `classify.py` | `msg_labels.jsonl` | 17,340 | 본실험 20판 메시지 전수 — 장르 10범주 |
| `subclassify.py` | `msg_sublabels.jsonl` | 14,568 | ask·info 하위 장르 (6~8범주) |
| `hostprop.py` | `hostprop_targets.jsonl` | 3,462 | 숙주 제안의 지시 대상 (Asla/Ranoa/Miris/leader/none) |
| `thinkclass.py` | `think_labels.jsonl` | 4,310 | think 결정 동인 8범주 + confused (커버리지 ~60%) |
| `classify_dd.py` | `dd_*.jsonl` | 9,012 | 정전·여명 10판에 같은 4단계 연쇄 |

## 경제 분석 (LLM 불필요 — 동봉된 runs/ 만 읽는다)

| 스크립트 | 내용 |
|---|---|
| `ledger.py` | AP 원장(초과지출 분해)과 암묵 효용 x̂ 역산 — RESULTS 「AP 원장」 절 |
| `luxury.py` | 「잔돈 학습」 가설 검정 — 학습 시점 잔액 복원, 잔돈 순간의 메뉴 |
| `learn_think.py` | 학습 직전 think 전수 추출(`learn_thinks.json`)과 동기 키워드 히트율 |
| `table20.py` | 본실험 20판 판별 원표 — RESULTS 원표 |

라벨의 `label` 필드 범주 정의는 각 스크립트 상단의 프롬프트에 있다 — 범주를
어떻게 정했고 무엇을 기각했는지는 RESULTS 「측정 방법과 그 한계」 참조.
