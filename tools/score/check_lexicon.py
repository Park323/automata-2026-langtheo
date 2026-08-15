#!/usr/bin/env python3
"""화용 표지 사전 검증. spec 6.2.

파일럿 문장(`docs/pilot/sentences.json`)에는 문장마다 `tests` 가 있다 — **어느 자질을
시험하려고 쓴 문장인지가 정답으로 들어 있다.** 사전이 그 자질을 세 언어 모두에서
잡아내는지 확인한다.

여기서 놓치는 자질은 **지표 7(화용 표지 소실률·생성률)이 그만큼 눈이 먼다**는 뜻이다.
소실률의 분모가 `Σ sent` 이므로, 발신 언어에서 못 잡으면 그 자질은 아예 집계되지 않는다.

    python3 tools/score/check_lexicon.py
    python3 tools/score/check_lexicon.py --verbose     # 놓친 문장 전문
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.score import markers  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
PILOT = ROOT / "docs" / "pilot" / "sentences.json"
LANGS = ("ja", "zh", "fr")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    lex = markers.load_lexicon()
    data = json.loads(PILOT.read_text(encoding="utf-8"))
    sentences = data["sentences"]

    # 자질마다 "어느 언어에서 반드시 잡혀야 하는가" 가 다르다.
    #
    #   문법이 강제하는 자질(주어·시제)  →  fr 에서만 필수.
    #       ja/zh 에서 없는 것은 결함이 아니라 **관측 대상**이다 — 원문에 없던 것이
    #       fr 로 갈 때 생성되는 것이 지표 7 의 생성률이고, 사전 meta 가 "양성 대조" 로
    #       쓰라고 적어둔 그것이다.
    #   어휘적 자질(헤지·조건)          →  세 언어 모두 필수. 어느 언어든 단어로 표현된다.
    REQUIRED = {"subject": ("fr",), "tense": ("fr",),
                "hedge": LANGS, "condition": LANGS}
    # 파일럿의 tests 는 hedge 로 라벨돼 있지만 완곡 거절 문장(S2)은
    # euphemistic_refusal 하위 태그가 잡는다 — 둘 중 하나면 통과로 본다.
    ALSO = {"hedge": "euphemistic_refusal"}
    # 한 문장이 여러 자질을 달고 있을 때, 연결사가 없어 그 자질이 0 인 것은 사전 결함이
    # 아닐 수 있다 (S6: hedge+condition 인데 조건 연결사가 없는 문장). 같은 문장의 다른
    # tests 자질이 잡히면 "문장은 측정되고 있다" 로 보고 라벨 문제로 따로 보고한다.
    label_issues: list[tuple[str, str, str, str]] = []

    hit: dict[tuple[str, str], int] = {}
    tot: dict[tuple[str, str], int] = {}
    misses: list[tuple[str, str, str, str]] = []
    informational: list[tuple[str, str, str, str]] = []

    for s in sentences:
        for feat in s.get("tests", []):
            req = REQUIRED.get(feat, LANGS)
            for lang in LANGS:
                text = s.get(lang)
                if not text:
                    continue
                found = markers.count(text, lang, feat, lex) > 0
                if not found and feat in ALSO:
                    found = markers.count(text, lang, ALSO[feat], lex) > 0
                if lang not in req:
                    if not found:
                        informational.append((s["id"], feat, lang, text))
                    continue
                k = (feat, lang)
                tot[k] = tot.get(k, 0) + 1
                if found:
                    hit[k] = hit.get(k, 0) + 1
                else:
                    others = [f for f in s.get("tests", []) if f != feat
                              and (markers.count(text, lang, f, lex) > 0
                                   or markers.count(text, lang, ALSO.get(f, ""), lex) > 0)]
                    if others:
                        label_issues.append((s["id"], feat, lang, ",".join(others)))
                    else:
                        misses.append((s["id"], feat, lang, text))

    feats = sorted({f for f, _ in tot})
    print(f"파일럿 {len(sentences)}문장 × 3언어로 사전 검증")
    print(f"\n{'자질':<12}" + "".join(f"{l:>10}" for l in LANGS) + f"{'합계':>10}")
    print("-" * 54)
    worst = []
    for f in feats:
        row = f"{f:<12}"
        h = t = 0
        for lang in LANGS:
            k = (f, lang)
            if not tot.get(k):
                row += f"{'—':>10}"
                continue
            r = hit.get(k, 0) / tot[k]
            h += hit.get(k, 0); t += tot[k]
            row += f"{hit.get(k,0)}/{tot[k]} {r:>4.0%}".rjust(10)
        rate = h / t if t else 0
        row += f"{h}/{t} {rate:>4.0%}".rjust(10)
        print(row)
        if rate < 0.8:
            worst.append((f, rate))

    print()
    if worst:
        print("⚠ 커버리지가 낮은 자질 — 지표 7 이 그만큼 눈이 멉니다")
        for f, r in sorted(worst, key=lambda x: x[1]):
            print(f"    {f:<12} {r:.0%}")
    else:
        print("모든 자질이 80% 이상 잡힙니다.")

    if misses:
        print(f"\n놓친 것 {len(misses)}건 — 사전이 고쳐야 할 것")
        for sid, feat, lang, text in misses:
            print(f"  {sid} [{feat}/{lang}] {text}")
    if label_issues:
        print(f"\n라벨 검토 {len(label_issues)}건 — 사전이 아니라 코퍼스 쪽일 수 있음")
        print("  그 자질은 0 이지만 같은 문장의 다른 tests 자질은 잡힌다.")
        for sid, feat, lang, others in label_issues:
            print(f"  {sid} [{feat}/{lang}] 없음 · 대신 잡힌 것: {others}")
    if a.verbose and informational:
        print(f"\n참고 {len(informational)}건 — ja/zh 의 주어·시제 부재.")
        print("  결함이 아니라 관측 대상이다 (fr 로 갈 때 생성되는지가 지표 7 의 생성률).")
        for sid, feat, lang, text in informational:
            print(f"  {sid} [{feat}/{lang}] {text}")

    return 1 if worst else 0


if __name__ == "__main__":
    raise SystemExit(main())
