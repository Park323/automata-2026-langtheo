"""화용 표지 카운터. spec 6.2.

`text_sent`(발신 언어)와 `text_delivered`(수신 언어)에서 표지를 세어 소실률·생성률을
낸다. **LLM 판정을 쓰지 않는다** — 결정론적이고 완전히 재현 가능하다.

    소실률(자질) = Σ max(0, sent − delivered) / Σ sent
    생성률(자질) = Σ max(0, delivered − sent) / 메시지 수

⚠ **집계는 반드시 코퍼스 단위.** 메시지 단위로 나누면 대부분의 짧은 메시지가 특정
  표지를 아예 포함하지 않아 분모가 0 이 된다 (spec 6.2).

⚠ 두 텍스트는 **언어가 다르다.** 발신 언어 사전으로 sent 를, 수신 언어 사전으로
  delivered 를 센다. 같은 사전으로 세면 아무것도 안 잡힌다.

생성률이 이 프로젝트의 숨은 발견일 수 있다 — *"AI 가 유보를 지운다"* 보다
**"AI 가 없던 확신을 만들어낸다"** 가 훨씬 무섭고, 없던 것이 생긴 건 원문과 대조하지
않으면 절대 알 수 없다.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
LEXICON = ROOT / "domains" / "meteor" / "lexicon" / "markers.yaml"


def _wrap_word_boundary(tok: str) -> str:
    """라틴 문자만 단어 경계를 적용한다 (사전의 scoring.word_boundary).

    `j'` 처럼 문자가 아닌 것으로 끝나는 토큰은 뒤쪽 \\b 를 붙이지 않는다 —
    아포스트로피 뒤에 \\b 를 붙이면 매칭이 깨진다.
    """
    pat = re.escape(tok)
    if tok[:1].isalpha():
        pat = r"\b" + pat
    if tok[-1:].isalpha():
        pat = pat + r"\b"
    return pat


def load_lexicon(path: Path | None = None) -> dict:
    """{자질: {언어: 컴파일된 단일 패턴}}.

    사전의 `scoring` 규칙을 지킨다:
      · match: longest_first  — 긴 표지부터. 「我们」이 「我」로 이중 계수되면 안 된다
      · count_mode: occurrences — 등장 횟수 (유형 수가 아니다)
      · word_boundary: fr 만 적용
      · case_sensitive: false

    literal 을 길이 내림차순으로 정렬해 하나의 대안(alternation)으로 합치면,
    파이썬 정규식이 각 위치에서 가장 앞선 대안을 고르므로 **긴 것이 이기고**
    `finditer` 가 매칭 구간을 소비해 겹침이 자동으로 해소된다.
    """
    raw = yaml.safe_load((path or LEXICON).read_text(encoding="utf-8"))
    rules = raw.get("scoring") or {}
    wb = rules.get("word_boundary") or {}
    flags = 0 if rules.get("case_sensitive") else re.IGNORECASE

    # `euphemistic_refusal` 은 사전이 "별도 하위 태그로 센다" 고 적어둔 것이라
    # hedge 에 합치지 않고 **독립 자질**로 뽑는다 — 완곡 거절이 직설로 바뀌는지는
    # 유보가 깎이는 것과 다른 현상이다.
    SUBTAGS = {"euphemistic_refusal"}

    out: dict[str, dict[str, re.Pattern]] = {}
    for feature, langs in raw.items():
        if feature in ("meta", "scoring") or not isinstance(langs, dict):
            continue
        out.setdefault(feature, {})
        for sub in SUBTAGS:
            out.setdefault(sub, {})
        for lang, spec in langs.items():
            if not isinstance(spec, dict):
                continue
            for sub in SUBTAGS:                       # 하위 태그를 독립 자질로
                toks = [str(x) for x in (spec.get(sub) or [])]
                if toks:
                    toks.sort(key=len, reverse=True)
                    ps = [_wrap_word_boundary(t) if wb.get(lang) else re.escape(t) for t in toks]
                    out[sub][lang] = re.compile("|".join(ps), flags)
            lits: list[str] = []
            for key in ("literal", "plural_marker"):
                lits += [str(x) for x in (spec.get(key) or [])]
            lits.sort(key=len, reverse=True)              # ★ longest_first
            parts = [_wrap_word_boundary(t) if wb.get(lang) else re.escape(t) for t in lits]
            parts += [str(p) for p in (spec.get("regex") or [])]
            if parts:
                out[feature][lang] = re.compile("|".join(parts), flags)
    return {k: v for k, v in out.items() if v}


def count(text: str | None, lang: str, feature: str, lex: dict) -> int:
    """한 텍스트에서 그 자질의 표지 출현 수. 사전에 없는 언어면 0."""
    if not text:
        return 0
    pat = lex.get(feature, {}).get(lang)
    return len(pat.findall(text)) if pat else 0


def score_messages(messages: list[dict], lex: dict | None = None) -> dict:
    """메시지 목록에서 자질별·방향별 소실률·생성률.

    messages 는 `messages.jsonl` 의 행들. `meta.text_sent` / `meta.text_delivered` /
    `meta.src_lang` / `meta.dst_lang` 을 쓴다.

    반환 {"overall": {자질: {...}}, "by_direction": {"ja→zh": {자질: {...}}}, "n": ...}
    """
    lex = lex or load_lexicon()
    features = sorted(lex)
    buckets: dict[str, dict] = {}          # 키 "overall" 또는 "ja→zh"

    def bucket(key: str) -> dict:
        if key not in buckets:
            buckets[key] = {"n": 0, **{f: {"sent": 0, "lost": 0, "made": 0} for f in features}}
        return buckets[key]

    for m in messages:
        meta = m.get("meta") or {}
        src = meta.get("src_lang")
        # **도착한 글의 실제 언어**로 센다. 번역을 안 탄 경로(domestic·original)는
        # 발신 언어 그대로인데 dst_lang 으로 세면 같은 글을 다른 언어 사전으로 훑어
        # 표지가 통째로 "소실" 로 잡힌다.
        dst = meta.get("delivered_lang") or meta.get("dst_lang")
        sent_t, deliv_t = meta.get("text_sent"), meta.get("text_delivered")
        if not src or not dst or sent_t is None:
            continue
        if deliv_t is None:                # 전달 실패(original) — 채널을 안 건넜다
            continue
        keys = ["overall", f"{src}→{dst}"]
        for k in keys:
            bucket(k)["n"] += 1
        for f in features:
            s = count(sent_t, src, f, lex)          # 발신 언어 사전으로
            d = count(deliv_t, dst, f, lex)         # 수신 언어 사전으로
            for k in keys:
                b = buckets[k][f]
                b["sent"] += s
                b["lost"] += max(0, s - d)
                b["made"] += max(0, d - s)

    def rates(b: dict) -> dict:
        n = b["n"]
        out = {"n": n}
        for f in features:
            c = b[f]
            out[f] = {
                "sent": c["sent"],
                "loss_rate": round(c["lost"] / c["sent"], 4) if c["sent"] else None,
                "gen_rate": round(c["made"] / n, 4) if n else None,
            }
        return out

    return {
        "overall": rates(buckets["overall"]) if "overall" in buckets else {"n": 0},
        "by_direction": {k: rates(v) for k, v in sorted(buckets.items()) if k != "overall"},
    }


def score_numbers(messages: list[dict]) -> dict:
    """지표 6a/6b/6a' — 수치 소실·변조·추가. spec 8.2.

    **부분집합 검사**다. 집합 일치로 재면 표기 변환(단어 수사 → 아라비아 숫자)이
    "왜곡" 으로 잡혀 대조군이 오염된다 — 파일럿에서 실측으로 확인했다.
    """
    num = re.compile(r"\d+")

    def totals(rows: list[dict]) -> dict:
        lost = added = n = 0
        for m in rows:
            meta = m.get("meta") or {}
            s, d = meta.get("text_sent"), meta.get("text_delivered")
            if s is None or d is None:
                continue
            ns, nd = set(num.findall(s)), set(num.findall(d))
            n += 1
            if not ns <= nd:               # 부분집합이 아니면 소실·변조
                lost += 1
            if nd - ns:
                added += 1
        return {"n": n,
                "loss_rate": round(lost / n, 4) if n else None,
                "add_rate": round(added / n, 4) if n else None}

    ai = [m for m in messages if m.get("route") == "ai"]
    return {"6a_ai": totals(ai), "6b_all": totals(messages)}
