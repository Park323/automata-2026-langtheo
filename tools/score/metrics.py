"""지표 산출. spec 8.2.

런 하나를 읽어 지표를 전부 계산합니다. LLM 을 부르지 않습니다 — 지표 4 는
`judge.py` 가 미리 만들어 둔 `judged.jsonl` 을 읽기만 합니다. 없으면 그 칸은 `None` 이고,
**0 이 아닙니다.** 표본이 없는 것과 실패가 없는 것은 다릅니다.

    python3 tools/score/metrics.py runs/<run_id>              # 한 런
    python3 tools/score/metrics.py runs/*/ --out scored.jsonl  # 여러 런 → 조건별 표

`n` 과 `pair_dist` 는 **반드시 함께 봅니다.** 조건 간 언어쌍 구성이 다르면 4a 비교가
오염됩니다(spec 8.3). 그래서 이 둘은 옵션이 아니라 항상 산출에 들어갑니다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.score import judge, markers  # noqa: E402

LANGS = ("ja", "zh", "fr")


def _act_type(a) -> str:
    """행동 로그는 `{type, ...인자}` 다. 구버전 런은 종류만 담긴 문자열이라 둘 다 받는다."""
    return a.get("type") if isinstance(a, dict) else a


def country_of(agent_id: str) -> str:
    return re.sub(r"\d+$", "", agent_id or "")


def _rate(num: int, den: int) -> float | None:
    """표본이 없으면 None. **0.0 을 돌려주면 '실패가 없었다' 로 읽힌다.**"""
    return round(num / den, 4) if den else None


# ── 개별 지표 ───────────────────────────────────────────────────────────────────

def learner_rate(state: list[dict], window: int = 10) -> dict:
    """지표 1 — 마지막 `window` 턴 평균 학습자 비율. 나이 층화도 함께.

    `known_langs` 는 모국어를 포함하므로 **2 이상이 학습자**다. 나이 층화가 필요한 이유 —
    학습은 누적이라 늙은 개체일수록 높다. 층화하지 않으면 인구 구성이 바뀌기만 해도
    지표가 움직인다 (`x̂` 추정의 입력이기도 하다).
    """
    if not state:
        return {"rate": None, "n_agent_turns": 0, "turns": "—", "by_age": {}}
    last = max(r["turn"] for r in state)
    lo = max(1, last - window + 1)
    rows = [r for r in state if lo <= r["turn"] <= last and r.get("alive")]
    learners = sum(1 for r in rows if len(r.get("known_langs") or []) >= 2)

    by_age: dict[str, dict] = {}
    for r in rows:
        b = f"{(r.get('age') or 0) // 3 * 3}-{(r.get('age') or 0) // 3 * 3 + 2}"
        d = by_age.setdefault(b, {"n": 0, "learners": 0})
        d["n"] += 1
        d["learners"] += len(r.get("known_langs") or []) >= 2
    return {
        "rate": _rate(learners, len(rows)),
        "n_agent_turns": len(rows),
        "turns": f"{lo}-{last}",
        "by_age": {k: {"n": v["n"], "rate": _rate(v["learners"], v["n"])}
                   for k, v in sorted(by_age.items())},
    }


def outcome_metrics(summary: dict) -> dict:
    """지표 2 · 2' · 5 — 재앙 회피, 생존 인구 비율, 최종 진척."""
    fin = (summary or {}).get("final") or {}
    survived = fin.get("outcome") == "all_survive"
    survivors = fin.get("survivors") or []
    return {
        "2_avoided": 1 if survived else 0,
        "2p_survivor_share": 1.0 if survived else round(len(survivors) / 3, 4),
        "5_interceptor_best": fin.get("interceptor_best"),
        "outcome": fin.get("outcome"),
    }


def policy_shift(messages: list[dict], events: list[dict]) -> dict:
    """지표 3 — 국제 메시지를 받은 사람이 `propose_vote` 로 움직였는가.

    분모는 **국제 메시지를 1건 이상 받은 (사람, 턴) 쌍**이다. 메시지 수로 나누면
    AI 가 싼 조건에서 분모만 부풀어 지표가 기계적으로 내려간다.

    도착은 발신 다음 턴이므로 수신 턴 `r = turn + 1` 이다. 같은 턴에 반응하는 것
    (`3`)과 그 다음 턴에 반응하는 것(`3_lag`)을 **둘 다** 낸다 — 7B 에이전트가 읽은
    턴에 바로 움직이는지 한 턴 늦는지 실측 전에는 알 수 없다.
    """
    voted: set[tuple[int, str]] = set()
    for e in events:
        if e.get("type") == "agent_turn" and any(
                _act_type(a) == "propose_vote" for a in (e.get("actions") or [])):
            voted.add((e["turn"], e["agent"]))

    pairs: set[tuple[int, str]] = set()
    for m in messages:
        if m.get("route") in ("ai", "original") and m.get("delivered"):
            pairs.add((m["turn"] + 1, m["to"]))

    same = sum(1 for (t, a) in pairs if (t, a) in voted)
    lag = sum(1 for (t, a) in pairs if (t + 1, a) in voted)
    return {"3": _rate(same, len(pairs)), "3_lag": _rate(lag, len(pairs)),
            "n_pairs": len(pairs)}


def message_shape(messages: list[dict]) -> dict:
    """`n` · `pair_dist` · 지표 8 · 9 · 10 — 메시지 구성과 채널 자체의 상태.

    이것들이 조건 간에 크게 다르면 4a·7 의 비교가 오염된다. **먼저 본다.**
    """
    n = Counter(m.get("route") for m in messages)
    intl = [m for m in messages if m.get("route") in ("ai", "original")]
    pair = Counter(f"{(m.get('meta') or {}).get('src_lang')}→"
                   f"{(m.get('meta') or {}).get('dst_lang')}" for m in intl)

    # 8 자기검열 — 상한을 사전 고지받고도 넘겨 잘린 비율. 경로별·언어별.
    def censor(rows: list[dict]) -> dict:
        cut = [(m.get("meta") or {}).get("chars_cut") or 0 for m in rows]
        tr = sum(1 for m in rows if (m.get("meta") or {}).get("truncated"))
        return {"n": len(rows), "rate": _rate(tr, len(rows)),
                "mean_chars_cut": round(sum(cut) / len(rows), 2) if rows else None}

    by_lang = {l: censor([m for m in messages
                          if (m.get("meta") or {}).get("src_lang") == l]) for l in LANGS}
    by_route = {r: censor([m for m in messages if m.get("route") == r])
                for r in ("domestic", "ai", "original")}

    orig = [m for m in messages if m.get("route") == "original"]
    failed = sum(1 for m in orig if not m.get("delivered"))
    # 번역 호출 실패는 **엔진 장애**다. 지표 9(전달 실패율)에 섞으면 "읽을 수 없어서
    # 못 받았다" 로 오독된다. 조건 간 빈도가 다르면 4a·7 도 오염되므로 따로 센다.
    tr_failed = sum(1 for m in messages if (m.get("meta") or {}).get("translate_failed"))

    return {
        "engine_translate_failed": {"n": tr_failed,
                                    "rate": _rate(tr_failed, len(messages))},
        "n": {"total": len(messages), **{k: v for k, v in sorted(n.items()) if k}},
        "pair_dist": {k: round(v / len(intl), 4) for k, v in sorted(pair.items())} if intl else {},
        "pair_counts": dict(sorted(pair.items())),
        "8_self_censor": {"overall": censor(messages), "by_route": by_route,
                          "by_lang": by_lang},
        "9_delivery_failure": {"n": len(orig), "rate": _rate(failed, len(orig))},
        "10_original_attempt": {"n_intl": len(intl), "rate": _rate(len(orig), len(intl))},
    }


def reply_metrics(messages: list[dict]) -> dict:
    """답장률 — **A→B 가 있었을 때 B→A 가 같은 턴이나 다음 턴에 있었는가.**

    `speak` 에 `reply_to` 인자를 두고 그것을 세려 했는데, 그 인자는 **도구 스키마에
    없었다** — 배관만 여섯 곳에 깔려 있고 모델에게 준 적이 없어서 항상 0 이었다.
    그런데 그 0 이 「대화가 죽었다」 의 근거로 쓰였다. 실제로는 근거로 답장을 하고
    있었다 (*"Responding to Asla1's inquiry"*).

    그래서 필드를 없애고 **로그에서 센다.** 내용은 안 본다 — 오간 방향과 시점만으로
    「말을 걸었을 때 돌아왔는가」 를 재고, 그것이 우리가 알고 싶은 것이다.

        A→B (턴 t)  에 대해  B→A 가 턴 t 또는 t+1 에 있으면 답장으로 센다

    같은 턴을 포함하는 이유 — 순차 라운드로빈은 같은 턴에 배달되므로 그 턴에 답이
    올 수 있다. 병렬은 다음 턴이 가장 이른 답이다. **둘을 한 지표로 비교하려면 창이
    둘을 다 덮어야 한다.**

    한 번의 A→B 가 여러 B→A 를 끌어냈어도 **한 번으로 센다** (짝이 아니라 발신 기준).
    자기 나라 안(domestic)과 국제를 나눠 내는 이유는, 국제 답장률이 언어 채널의 상태를
    직접 재고 그것이 노브에 반응하는 값이기 때문이다.
    """
    by_dir: dict = {}
    for m in messages:
        key = (m.get("from"), m.get("to"))
        by_dir.setdefault(key, []).append(m.get("turn"))

    def rate(rows: list[dict]) -> dict:
        answered = 0
        for m in rows:
            back = by_dir.get((m.get("to"), m.get("from")), ())
            t = m.get("turn")
            if t is not None and any(t <= u <= t + 1 for u in back):
                answered += 1
        return {"n": len(rows), "answered": answered, "rate": _rate(answered, len(rows))}

    intl = [m for m in messages if m.get("route") in ("ai", "original")]
    dom = [m for m in messages if m.get("route") == "domestic"]
    return {"overall": rate(messages), "domestic": rate(dom), "international": rate(intl),
            "by_route": {r: rate([m for m in messages if m.get("route") == r])
                         for r in ("domestic", "ai", "original")}}


def intent_metrics(judged: list[dict]) -> dict:
    """지표 4a/4b/4c/4d — `judge.py` 의 결과를 읽기만 한다. 없으면 전부 None."""
    if not judged:
        return {"4a": {"n": 0, "fail_rate": None}, "4b": {"n": 0, "fail_rate": None},
                "4c": {"n": 0, "fail_rate": None}, "4d": {"n": 0, "fail_rate": None},
                "4a_minus_4c": None, "mention_rate": {}, "judged": False}
    return {**judge.aggregate(judged), "judged": True}


def engine_health(metrics_rows: list[dict], summary: dict) -> dict:
    """엔진이 정상이었는가. 지표가 아니라 **그 지표를 믿어도 되는가**의 근거다.

    LLM 실패율이 조건 간에 다르면 행동 분포 차이가 번역 효과로 오독된다.
    """
    turns = len(metrics_rows)
    fails = sum(r.get("llm_failures") or 0 for r in metrics_rows)
    turns_agents = sum(r.get("agent_turns") or 0 for r in metrics_rows)
    return {
        "turns": turns,
        "llm_failure_rate": _rate(fails, turns_agents),
        "raw_calls": ((summary or {}).get("raw_calls") or {}).get("raw"),
        "raw_errors": ((summary or {}).get("raw_calls") or {}).get("errors"),
        "prompt_tokens_max": max((r.get("prompt_tokens_max") or 0
                                  for r in metrics_rows), default=0),
        "pressured_turns": sum(1 for r in metrics_rows if r.get("pressured")),
        "memory_writes": sum(r.get("memory_writes") or 0 for r in metrics_rows),
        "deaths": (summary or {}).get("deaths"),
    }


# ── 런 하나 ─────────────────────────────────────────────────────────────────────

def score_run(run_dir: Path) -> dict:
    """런 하나의 전체 지표. 파일이 없으면 그 칸만 비고 나머지는 낸다."""
    rd = Path(run_dir)
    messages = judge.read_jsonl(rd / "messages.jsonl")
    events = judge.read_jsonl(rd / "events.jsonl")
    # **`step` 이 붙은 줄은 해 도중의 스냅샷이다** (8/26). 지표는 턴 끝 줄만 읽어야
# 한다 — 안 그러면 한 해가 여러 번 세어진다.
    state = [r for r in judge.read_jsonl(rd / "state.jsonl") if r.get("step") is None]
    mrows = judge.read_jsonl(rd / "metrics.jsonl")
    judged = judge.read_jsonl(rd / "judged.jsonl")
    sp = rd / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}

    knob = seed = None
    cs = rd / "config_snapshot.yaml"
    if cs.exists():
        import yaml
        snap = yaml.safe_load(cs.read_text(encoding="utf-8")) or {}
        cfg = snap.get("config") or {}
        # 런 인자가 먼저다. config 의 knob 은 스윕할 **목록**이라 조건을 특정하지 못한다.
        knob = snap.get("knob_ai")
        if knob is None:
            k = (cfg.get("knob") or {}).get("comm_intl_ai_ap")
            knob = k[0] if isinstance(k, list) and len(k) == 1 else k
        seed = snap.get("seed")
        if seed is None:
            seed = (cfg.get("run") or {}).get("seed")

    out: dict = {
        "run_id": rd.name, "knob": knob, "seed": seed,
        "1_learner": learner_rate(state),
        **outcome_metrics(summary),
        **policy_shift(messages, events),
        **message_shape(messages),
        # 답장률 — reply_to 필드가 아니라 **오간 방향과 시점**으로 센다 (reply_metrics)
        "reply": reply_metrics(messages),
        "4": intent_metrics(judged),
        "6": markers.score_numbers(messages),
        "7": markers.score_messages(messages),
        "engine": engine_health(mrows, summary),
    }
    return out


# ── 표 ──────────────────────────────────────────────────────────────────────────

def _pct(v) -> str:
    return "—" if v is None else f"{v:.0%}"


def format_run(r: dict) -> str:
    L = [f"{r['run_id']}   노브 {r['knob']}  시드 {r['seed']}", "─" * 66]
    e = r["engine"]
    L.append(f"엔진   {e['turns']}턴 · 호출 {e['raw_calls']} · 실패율 "
             f"{_pct(e['llm_failure_rate'])} · 사망 {e['deaths']} · "
             f"프롬프트 최대 {e['prompt_tokens_max']} · memory_write {e['memory_writes']}")
    n = r["n"]
    L.append(f"메시지 {n['total']}건  " + " · ".join(f"{k} {v}" for k, v in n.items() if k != "total"))
    if r["pair_dist"]:
        L.append("       언어쌍 " + " · ".join(f"{k} {v:.0%}" for k, v in r["pair_dist"].items()))
    L.append("")

    L.append(f"1  학습자 비율        {_pct(r['1_learner']['rate'])}  "
             f"(턴 {r['1_learner']['turns']}, n={r['1_learner']['n_agent_turns']})")
    L.append(f"2  재앙 회피          {r['2_avoided']}   [{r['outcome']}]")
    L.append(f"2' 생존 국가 비율      {r['2p_survivor_share']:.0%}")
    L.append(f"3  정책 전환 유발      {_pct(r['3'])}  (한 턴 뒤 {_pct(r['3_lag'])}, n={r['n_pairs']})")
    L.append(f"5  최종 진척          {r['5_interceptor_best']}")

    f4 = r["4"]
    if f4["judged"]:
        L.append("")
        for k, name in (("4a", "AI 경로"), ("4c", "국내 (기저선)"),
                        ("4d", "국제·원문직통"), ("4b", "전체")):
            mr = f4["mention_rate"].get(k, {})
            L.append(f"{k} {name:<14} {_pct(f4[k]['fail_rate']):>5}  n={f4[k]['n']:<4}"
                     f"언급률 {_pct(mr.get('rate'))} (n={mr.get('n')})")
        d = f4["4a_minus_4c"]
        L.append(f"   4a − 4c        {'—' if d is None else f'{d:+.1%}'}   ← 번역 고유의 오해 증분")
    else:
        L.append("\n4  판정 없음 — judge.py 를 먼저 돌리세요")

    L.append("")
    six = r["6"]
    L.append(f"6a 수치 소실 (AI)     {_pct(six['6a_ai']['loss_rate'])}  n={six['6a_ai']['n']}"
             f"   6a' 추가 {_pct(six['6a_ai']['add_rate'])}")
    L.append(f"6b 수치 소실 (전체)    {_pct(six['6b_all']['loss_rate'])}  n={six['6b_all']['n']}")

    ov = r["7"]["overall"]
    L.append(f"\n7  화용 표지  (n={ov.get('n', 0)})")
    for feat, v in ov.items():
        if feat == "n" or not isinstance(v, dict) or not v["sent"]:
            continue
        L.append(f"     {feat:<20} sent {v['sent']:>4}  소실 {_pct(v['loss_rate'])}"
                 f"  생성 {_pct(v['gen_rate'])}")

    c = r["8_self_censor"]["overall"]
    L.append(f"\n8  자기검열           {_pct(c['rate'])}  평균 절단 {c['mean_chars_cut']}자")
    L.append(f"9  전달 실패          {_pct(r['9_delivery_failure']['rate'])}  "
             f"n={r['9_delivery_failure']['n']}")
    L.append(f"10 원문 직통 시도      {_pct(r['10_original_attempt']['rate'])}  "
             f"n={r['10_original_attempt']['n_intl']}")
    tf = r["engine_translate_failed"]
    if tf["n"]:
        L.append(f"\n⚠ 번역 호출 실패 {tf['n']}건 ({_pct(tf['rate'])}) — **엔진 장애**입니다. "
                 "지표 9 와 별개이고,\n  조건 간 빈도가 다르면 4a·7 비교가 오염됩니다.")
    return "\n".join(L)


def format_table(rows: list[dict]) -> str:
    """조건별 표. **`n` 없이 4a·6a 를 읽지 않습니다** (spec 8.4)."""
    by_knob: dict = defaultdict(list)
    for r in rows:
        by_knob[r["knob"]].append(r)

    L = ["", f"{'노브':>5}{'런':>4}{'회피':>7}{'학습자':>8}{'4a':>8}{'4c':>8}"
             f"{'4a−4c':>9}{'6a':>7}{'전환':>7}{'n(ai)':>8}"]
    L.append("─" * 71)

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    for knob in sorted(by_knob, key=lambda x: (x is None, x)):
        rs = by_knob[knob]
        a = avg([r["4"]["4a"]["fail_rate"] for r in rs])
        c = avg([r["4"]["4c"]["fail_rate"] for r in rs])
        L.append(
            f"{str(knob):>5}{len(rs):>4}"
            f"{_pct(avg([r['2_avoided'] for r in rs])):>7}"
            f"{_pct(avg([r['1_learner']['rate'] for r in rs])):>8}"
            f"{_pct(a):>8}{_pct(c):>8}"
            f"{'—' if a is None or c is None else f'{a - c:+.1%}':>9}"
            f"{_pct(avg([r['6']['6a_ai']['loss_rate'] for r in rs])):>7}"
            f"{_pct(avg([r['3'] for r in rs])):>7}"
            f"{sum(r['4']['4a']['n'] for r in rs):>8}")
    L.append("\n※ 4a·6a 는 n 과 함께 읽습니다. 언어쌍 구성이 조건 간에 다르면 4a 비교가 오염됩니다.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="지표 산출 (spec 8.2)")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default=None, help="scored.jsonl 경로 (런당 1행)")
    ap.add_argument("--table", action="store_true", help="런별 상세 없이 조건별 표만")
    a = ap.parse_args()

    dirs = [Path(d) for d in a.run_dirs if (Path(d) / "messages.jsonl").exists()]
    if not dirs:
        print("messages.jsonl 을 가진 런 디렉터리가 없습니다", file=sys.stderr)
        return 2

    rows = [score_run(d) for d in sorted(dirs)]
    if not a.table:
        for r in rows:
            print(format_run(r))
            print()
    if len(rows) > 1 or a.table:
        print(format_table(rows))
    if a.out:
        Path(a.out).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n",
            encoding="utf-8")
        print(f"\n{a.out} · {len(rows)}행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
