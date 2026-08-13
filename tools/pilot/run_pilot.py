#!/usr/bin/env python3
"""번역 파일럿 러너 — docs/spec.md 12.2

  ① 왕복 손실   6방향 × 10문장 → 화용 표지 소실률·생성률
  ② 지시별 차이 4문장 × 3지시
  ③ 길이         출력 글자 수로 message_max_chars 산출
  (④ 목표 지향성은 엔진 프롬프트가 필요해 별도)

의존성 없음 (stdlib 만). 결과는 runs/pilot/<ts>/ 에 저장.

  python3 tools/pilot/run_pilot.py --models mistralai/mistral-small-3.2-24b-instruct
  python3 tools/pilot/run_pilot.py --report runs/pilot/<ts>/raw.jsonl
"""
import argparse, json, os, pathlib, re, sys, time, urllib.error, urllib.request
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "pilot" / "sentences.json"
LANGS = ["ja", "zh", "fr"]
LANG_NAME = {"ja": "Japanese", "zh": "Chinese", "fr": "French"}

# 번역기에 주는 유일한 시스템 지시. 출력 형식 계약이며 번역 방식에 대한 지시가 아니다.
# (docs/spec.md 5.2 — 시스템은 번역 지시를 덧붙이지 않는다)
SYSTEM = ("You are a translation engine. Output ONLY the translated text. "
          "No explanation, no alternatives, no quotes, no notes.")


def load_key() -> str:
    for name in (".env.local", ".env"):
        p = ROOT / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY"):
                return line.split("=", 1)[1].strip().strip("\"'")
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    sys.exit("OPENROUTER_API_KEY 를 .env.local 에 넣으세요")


# ── 화용 표지 카운터 ────────────────────────────────────────────────
def count_markers(spec: dict, text: str) -> int:
    """longest_first — 긴 표지부터 매칭하고 겹친 구간은 소비한다.
    「我们」이 「我」로 이중 계수되면 subject 가 부풀려진다."""
    if not spec:
        return 0
    low, used, n = text.lower(), [False] * len(text), 0
    for lit in sorted(spec.get("literal", []), key=len, reverse=True):
        l, start = lit.lower(), 0
        while (i := low.find(l, start)) >= 0:
            if not any(used[i:i + len(l)]):
                n += 1
                used[i:i + len(l)] = [True] * len(l)
            start = i + 1
    for rx in spec.get("regex", []):
        for m in re.finditer(rx, text):
            if not any(used[m.start():m.end()]):
                n += 1
                used[m.start():m.end()] = [True] * (m.end() - m.start())
    return n


def numbers(text: str) -> list:
    return sorted(re.findall(r"\d+", text))


# ── API ────────────────────────────────────────────────────────────
def translate(key, model, src_lang, dst_lang, text, instruction=None, retries=3):
    prompt = f"Translate to {LANG_NAME[dst_lang]}.\n\n{text}"
    if instruction:
        prompt = f"Translate to {LANG_NAME[dst_lang]}. {instruction}\n\n{text}"
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "temperature": 0.2, "max_tokens": 400,
        "logprobs": True,
        "provider": {"require_parameters": True},   # logprobs 지원 프로바이더로 강제
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            ch = r["choices"][0]
            lp = ch.get("logprobs") or {}
            toks = lp.get("content") or []
            return {
                "text": ch["message"]["content"].strip(),
                "provider": r.get("provider"),
                "logprob_mean": (sum(t["logprob"] for t in toks) / len(toks)) if toks else None,
                "tokens": r.get("usage", {}).get("total_tokens"),
            }
        except urllib.error.HTTPError as e:
            if e.code == 429:                       # 레이트 리밋은 길게 물러난다
                time.sleep(min(60, 8 * (2 ** attempt)))
                continue
            if attempt == retries - 1:
                return {"text": None, "error": f"{e.code} {e.reason}", "provider": None,
                        "logprob_mean": None, "tokens": None}
            time.sleep(3 * (attempt + 1))
        except Exception as e:
            if attempt == retries - 1:
                return {"text": None, "error": str(e)[:200], "provider": None,
                        "logprob_mean": None, "tokens": None}
            time.sleep(3 * (attempt + 1))
    return {"text": None, "error": "retries exhausted", "provider": None,
            "logprob_mean": None, "tokens": None}


# ── 실행 ───────────────────────────────────────────────────────────
def run(models, outdir, limit=None, delay=1.2):
    key = load_key()
    d = json.loads(DATA.read_text())
    sents = d["sentences"][:limit] if limit else d["sentences"]
    variants = d["instruction_variants"]
    subset = set(d["instruction_test_subset"])
    markers = d["markers"]
    feats = [k for k in markers if not k.startswith("_")]

    jobs = []
    for model in models:
        for s in sents:
            for src in LANGS:
                for dst in LANGS:
                    if src == dst:
                        continue
                    jobs.append((model, s, src, dst, "null"))
                    if s["id"] in subset:
                        for v in ("precise", "concise"):
                            jobs.append((model, s, src, dst, v))

    outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / "raw.jsonl"
    print(f"총 {len(jobs)}회 호출 → {raw}\n")
    t0 = time.time()
    with raw.open("w") as f:
        for i, (model, s, src, dst, vkey) in enumerate(jobs, 1):
            res = translate(key, model, src, dst, s[src], variants[vkey])
            time.sleep(delay)
            rec = {
                "model": model, "sid": s["id"], "src": src, "dst": dst,
                "instruction": vkey, "provider": res.get("provider"),
                "text_src": s[src], "text_out": res["text"],
                "logprob_mean": res.get("logprob_mean"),
                "len_src": len(s[src]), "len_out": len(res["text"] or ""),
                "numbers_src": numbers(s[src]),
                "numbers_out": numbers(res["text"] or ""),
                "error": res.get("error"),
            }
            for feat in feats:
                rec[f"m_src_{feat}"] = count_markers(markers[feat].get(src), s[src])
                rec[f"m_out_{feat}"] = count_markers(markers[feat].get(dst), res["text"] or "")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el:.0f}s  (남은 예상 {el/i*(len(jobs)-i):.0f}s)")
    print(f"\n완료 {time.time()-t0:.0f}s")
    return raw


# ── 리포트 ─────────────────────────────────────────────────────────
def report(raw_path):
    rows = [json.loads(l) for l in pathlib.Path(raw_path).read_text().splitlines() if l.strip()]
    d = json.loads(DATA.read_text())
    feats = [k for k in d["markers"] if not k.startswith("_")]
    base = [r for r in rows if r["instruction"] == "null" and not r.get("error")]
    models = sorted({r["model"] for r in rows})

    def bar(title): print(f"\n{'='*78}\n{title}\n{'='*78}")

    # ── 선택 기준 1: 숫자 보존 (필수 하한)
    bar("① 수치 소실·변조 — 필수 하한. 어긋나면 지표 6a 가 대조군을 못 한다")
    print(f"{'모델':<46} {'소실·변조':>9} {'추가(표기변환)':>13}")
    for m in models:
        rs = [r for r in base if r["model"] == m and r["numbers_src"]]
        if not rs: continue
        # 부분집합 검사. 집합 일치로 재면 단어 수사 → 아라비아 숫자 변환에 오탐이 난다
        lost = sum(1 for r in rs if not set(r["numbers_src"]) <= set(r["numbers_out"]))
        added = sum(1 for r in rs if set(r["numbers_out"]) - set(r["numbers_src"]))
        print(f"{m:<46} {lost:>4}/{len(rs):<4} {added:>8}/{len(rs)}")
    print("\n  소실·변조가 0 이어야 지표 6a 가 대조군으로 성립. 추가는 개별 확인 대상")

    # ── 선택 기준 2: 3언어 균형
    bar("② 3언어 균형 — 방향 간 편차. 작을수록 좋다 (국가 비대칭 방지)")
    print(f"{'모델':<46} {'평균 소실률':>10} {'표준편차':>9}  ← 이걸로 고른다")
    balance = {}
    for m in models:
        per_dir = []
        for src in LANGS:
            for dst in LANGS:
                if src == dst: continue
                rs = [r for r in base if r["model"] == m and r["src"] == src and r["dst"] == dst]
                tot_s = sum(sum(r[f"m_src_{f}"] for f in feats) for r in rs)
                tot_o = sum(sum(r[f"m_out_{f}"] for f in feats) for r in rs)
                if tot_s: per_dir.append(max(0, tot_s - tot_o) / tot_s)
        if per_dir:
            mu = sum(per_dir) / len(per_dir)
            sd = (sum((x - mu) ** 2 for x in per_dir) / len(per_dir)) ** 0.5
            balance[m] = (mu, sd)
            print(f"{m:<46} {mu*100:>9.1f}% {sd*100:>8.1f}%")

    # ── 관측: 자질별 소실률·생성률 (선택 기준 아님)
    bar("③ 자질별 소실률 / 생성률 — 관측 결과. 선택 기준으로 쓰지 말 것")
    for m in models:
        print(f"\n[{m}]")
        print(f"{'':>10} " + " ".join(f"{f:>18}" for f in feats))
        for src in LANGS:
            for dst in LANGS:
                if src == dst: continue
                rs = [r for r in base if r["model"] == m and r["src"] == src and r["dst"] == dst]
                cells = []
                for f in feats:
                    s_ = sum(r[f"m_src_{f}"] for r in rs)
                    o_ = sum(r[f"m_out_{f}"] for r in rs)
                    loss = max(0, s_ - o_) / s_ * 100 if s_ else float("nan")
                    gen = max(0, o_ - s_) / len(rs) if rs else 0
                    cells.append(f"손{loss:>5.0f}% 생{gen:>4.1f}")
                print(f"{src}→{dst:<7} " + " ".join(f"{c:>18}" for c in cells))

    # ── S4 양성 대조
    bar("④ S4 주어 생성 — 양성 대조. →fr 에서 안 잡히면 사전이 틀린 것")
    for m in models:
        for dst in ["fr", "zh", "ja"]:
            rs = [r for r in base if r["model"] == m and r["sid"] == "S4" and r["dst"] == dst]
            if not rs: continue
            gen = sum(max(0, r["m_out_subject"] - r["m_src_subject"]) for r in rs)
            print(f"{m:<46} →{dst}  생성 {gen}/{len(rs)}  " +
                  ("★ 기대대로" if dst == "fr" and gen > 0 else ""))

    # ── 길이
    bar("⑤ 출력 길이 — message_max_chars 산출 근거")
    print(f"{'모델':<46} " + " ".join(f"{l:>8}" for l in LANGS))
    for m in models:
        cells = []
        for dst in LANGS:
            rs = [r for r in base if r["model"] == m and r["dst"] == dst]
            cells.append(sum(r["len_out"] for r in rs) / len(rs) if rs else 0)
        mx = max(cells) or 1
        print(f"{m:<46} " + " ".join(f"{c:>8.0f}" for c in cells))
        print(f"{'  비율 (최대=100)':<46} " + " ".join(f"{c/mx*100:>8.0f}" for c in cells))

    # ── 지시별 차이
    bar("⑥ 지시별 차이 — 차이가 없으면 translate_instruction 을 되돌린다")
    print(f"{'모델':<46} {'무지시':>9} {'정확히':>9} {'간결히':>9}")
    for m in models:
        cells = []
        for v in ("null", "precise", "concise"):
            rs = [r for r in rows if r["model"] == m and r["instruction"] == v
                  and not r.get("error") and r["sid"] in set(d["instruction_test_subset"])]
            s_ = sum(sum(r[f"m_src_{f}"] for f in feats) for r in rs)
            o_ = sum(sum(r[f"m_out_{f}"] for f in feats) for r in rs)
            cells.append(max(0, s_ - o_) / s_ * 100 if s_ else 0)
        print(f"{m:<46} " + " ".join(f"{c:>8.1f}%" for c in cells))

    # ── logprob
    bar("⑦ 번역 확신도 — 유창함이지 정확함이 아니다 (spec 6.1)")
    print(f"{'모델':<46} {'logprob 평균':>13} {'제공':>6}")
    for m in models:
        rs = [r for r in base if r["model"] == m and r["logprob_mean"] is not None]
        if rs:
            print(f"{m:<46} {sum(r['logprob_mean'] for r in rs)/len(rs):>13.4f} {len(rs):>6}")
        else:
            print(f"{m:<46} {'(미제공)':>13} {0:>6}")

    err = [r for r in rows if r.get("error")]
    if err:
        print(f"\n오류 {len(err)}건: {err[0]['error'][:100]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "mistralai/mistral-small-3.2-24b-instruct",
        "google/gemma-3-27b-it",
        "meta-llama/llama-3.3-70b-instruct",
    ])
    ap.add_argument("--limit", type=int, help="문장 수 제한 (연습용)")
    ap.add_argument("--delay", type=float, default=1.2, help="호출 간 간격(초)")
    ap.add_argument("--report", help="기존 raw.jsonl 로 리포트만")
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        out = ROOT / "runs" / "pilot" / time.strftime("%Y%m%d_%H%M%S")
        report(run(a.models, out, a.limit, a.delay))
