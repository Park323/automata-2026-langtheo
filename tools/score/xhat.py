"""`x̂` — 학습의 암묵효용 하한. spec 7장 · 8.4.

`x` 는 우리가 넣는 값이 아니라 **읽어내는 값**입니다. 에이전트가 학습비를 실제로
지불했다면 그에게 학습의 가치는 최소한 그만큼이었다는 뜻입니다.

    학습이 일어남  ⟺  x ≥ 실제 학습비 ∈ {L, L/2, L/4}

**스윕이 필요 없습니다.** 할인 구조(spec 3.4)가 눈금 셋을 *같은 런 안에* 동시에
만들어 줍니다 — 국내 구사자 유무 × 부모 구사 여부. 그래서 조건을 하나도 늘리지 않고
`x` 를 `<L/4 / L/4~L/2 / L/2~L / ≥L` 네 구간으로 좁힙니다.

### 무엇을 세는가 — **분할 납부라 둘로 나뉩니다** (8/17)

학습은 한 번에 다 내는 것이 아니라 낸 만큼 쌓입니다. 그래서 턴 단위의 결정은
*"배웠는가"* 가 아니라 **"냈는가"** 이고, 습득은 여러 턴에 걸쳐 일어납니다.

    납부율(눈금 r)  그 턴에 낼 돈이 있었던 (사람, 턴) 중 **한 푼이라도 낸** 비율
    습득(눈금 r)    누적이 그 눈금에 닿아 실제로 읽게 된 건수

전액 기준으로 분모를 잡으면 거의 비어 버립니다 — 턴 시작 예산이 대개 100~200원인데
눈금은 150~600 입니다. **한 푼이라도 낼 수 있으면 시작할 수 있고, `x` 가 0 이면 한 푼도
안 냅니다.** 그것이 드러난 선호입니다.

**눈금은 `required`(그때의 필요액)에서 읽습니다.** 낸 액으로 읽으면 200원 납부가
`L/3` 처럼 잡힙니다 — 눈금은 셋뿐입니다.

**AP 도 금액에 비례합니다** — `ap.learn_full × (금액 ÷ learn_base)`. 그래서 관측되는
`x` 에 섞이는 AP 기회비용이 **눈금에 비례합니다**: `L` 을 끝내면 1.0, `L/2` 는 0.5,
`L/4` 는 0.25. 눈금별 납부율을 비교할 때 이 항이 눈금과 같이 움직이므로, 낮은 눈금이
켜지고 높은 눈금이 꺼지는 경계는 **돈과 시간을 합친 값**의 경계입니다. `x̂` 를 순수한
금전 지불의사로 읽으면 그만큼 과대평가하게 됩니다.

### 나이로 층화합니다 — 이게 최대 노이즈원입니다

늙은 에이전트는 회수 기간이 없어 같은 눈금도 사실상 더 비쌉니다. `income 100 / L 300`
에서 나이 6이면 남은 소득이 249 라 `L/2` 이하만 감당 가능하고, 나이 8이면 132 라 `L/4`
만 가능합니다. **층화하지 않으면 "안 배운 것" 이 `x` 가 작아서인지 늙어서인지 구분되지
않습니다.**

### 절대값보다 `Δx` 를 믿으세요

    Δx = x̂(노브 최저) − x̂(노브 최고)      ← 연쇄 1~2칸의 정량 진술

노브가 바뀌어도 AP 기회비용은 대체로 같아 **차분에서 상쇄**됩니다. 절대값에는 그
기회비용이 통째로 섞여 있습니다.

    python3 tools/score/xhat.py runs/<run_id> ...
    python3 tools/score/xhat.py runs/*/ --by-age
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.score import judge  # noqa: E402

# **눈금은 `learn_speedup` 에서 유도한다** (8/23). 8/22 에 학습 할인이 **가속**으로
# 바뀌면서 이 상수가 조용히 거짓이 됐다:
#
#   할인 모델   필요액이 L · L/2 · L/4 로 달라졌다 → rung = required / L 이 눈금이었다
#   가속 모델   필요액은 **늘 L** 이고 회당 수확이 배속을 탄다
#               → 총 지출은 L / 배속 이고, rung 은 **1/배속** 이다
#
# 그날부터 `required` 가 상수라 `_snap_rung` 이 **언제나 1.0** 을 냈다. 눈금이 하나로
# 붕괴해 x̂ 가 네 구간이 아니라 「x ≥ L」/「x < L」 두 갈래만 낼 수 있었다 (spec 7 파괴).
# 뷰어의 「L×1」 이 그 증상이었다.
def rungs_for(speedup: float, max_reasons: int = 2) -> tuple[float, ...]:
    """사유 수(0..max_reasons)별 **총 지출 / L**. 사유가 많을수록 작다."""
    return tuple(sorted({round(1.0 / (1.0 + speedup * r), 4)
                         for r in range(max_reasons + 1)}))


def rung_name(r: float) -> str:
    """1.0 → "L", 0.6667 → "L/1.5", 0.5 → "L/2".

    **역수를 두 자리로 끊는다** — `rungs_for` 가 눈금을 4자리로 반올림하므로 그대로
    나누면 1/0.6667 = 1.49993 이 되어 「L/1.49993」 이 찍힌다.
    """
    if abs(r - 1.0) < 1e-9:
        return "L"
    return f"L/{round(1.0 / r, 2):g}"


DEFAULT_SPEEDUP = 0.5             # config 를 못 읽을 때만. 정상 경로는 스냅샷에서 읽는다
AGE_BANDS = ((0, 2), (3, 5), (6, 99))
BAND_NAME = {(0, 2): "0-2", (3, 5): "3-5", (6, 99): "6+"}


def band_of(age: int) -> tuple[int, int]:
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return (lo, hi)
    return AGE_BANDS[-1]


# ── 관측 만들기 ─────────────────────────────────────────────────────────────────

def observations(run_dir: Path) -> tuple[list[dict], dict, list[dict]]:
    """(관측 목록, 진단, 습득 목록). 관측 1건 = (사람, 턴) 하나의 학습 결정.

    각 관측은 그 사람이 그 턴에 마주한 **가장 싼 눈금**과, 그 턴에 **한 푼이라도
    냈는지**를 담는다. 분할 납부라 "배웠다/안 배웠다" 가 아니라 **"냈다/안 냈다"** 가
    턴 단위의 결정이다. 습득은 여러 턴에 걸쳐 일어나므로 따로 모은다.

    ⚠ 반사실(배우지 않은 경우)의 눈금은 상태에서 **다시 계산**한다. 그 순간 국내에
      구사자가 있었는지가 관건인데, `state.jsonl` 이 턴 끝 상태라 그 턴에 배운 사람이
      이미 반영돼 있다. 그래서 **직전 턴** 상태로 국내 구사자를 판정한다 — 같은 턴에
      배운 사람을 할인 근거로 쓰면 없던 할인이 생긴다.
    """
    rd = Path(run_dir)
    state = judge.read_jsonl(rd / "state.jsonl")
    events = judge.read_jsonl(rd / "events.jsonl")
    diag: dict = {"no_budget_start": 0, "learns": 0, "opportunities": 0}

    base = None
    speedup = DEFAULT_SPEEDUP
    langs: dict[str, str] = {}          # 국가 → 언어
    cs = rd / "config_snapshot.yaml"
    if cs.exists():
        import yaml
        snap = yaml.safe_load(cs.read_text(encoding="utf-8")) or {}
        costs = (snap.get("config") or {}).get("costs") or {}
        base = costs.get("learn_base")
        speedup = costs.get("learn_speedup", DEFAULT_SPEEDUP)
        for c in ((snap.get("config") or {}).get("countries") or []):
            if isinstance(c, dict) and c.get("id"):
                langs[c["id"]] = c.get("lang")
    if not base:
        diag["error"] = "config_snapshot 에 learn_base 가 없습니다"
        return [], diag, []

    # **분할 납부다** (8/17). 한 턴의 `learn` 은 납부 1건이고, 습득은 누적이 그 순간의
    # 필요액에 닿았을 때 따로 기록된다 (`kind: "acquired"`).
    #
    #   type=learn, kind 없음   → 납부.   charged = 이번에 낸 액, required = 그때의 필요액
    #   type=learn, kind=acquired → 습득. charged = 총 지불액, required = 완료 시 필요액
    #
    # 눈금은 **required** 에서 읽는다. charged 로 읽으면 200원을 낸 것이 L/3 눈금처럼
    # 잡힌다 — 눈금은 셋뿐이다 (L · L/2 · L/4).
    paid: dict[tuple[int, str], list] = defaultdict(list)
    acquired: list[dict] = []
    for e in events:
        if e.get("type") != "learn":
            continue
        if e.get("kind") == "acquired":
            acquired.append(e)
        else:
            paid[(e["turn"], e["agent"])].append(e)
            diag["learns"] += 1
    diag["acquired"] = len(acquired)

    by_turn: dict[int, list[dict]] = defaultdict(list)
    for r in state:
        by_turn[r["turn"]].append(r)
    # 국가별 언어는 스냅샷에 없을 수도 있다 — 상태의 모국어 분포에서 채운다.
    country_lang = {r["country"]: (r.get("known_langs") or [None])[0]
                    for r in state if r.get("born_turn") == 0}
    country_lang.update({k: v for k, v in langs.items() if v})

    out: list[dict] = []
    for turn in sorted(by_turn):
        prev = by_turn.get(turn - 1, by_turn[turn])      # 첫 턴은 자기 자신
        # 직전 턴 기준 "국내에 그 언어를 아는 사람" (자기 자신 제외는 아래에서)
        speakers: dict[tuple[str, str], set[str]] = defaultdict(set)
        for r in prev:
            if r.get("alive"):
                for lg in (r.get("known_langs") or []):
                    speakers[(r["country"], lg)].add(r["agent"])

        for r in by_turn[turn]:
            evs = paid.get((turn, r["agent"]), [])
            if not r.get("alive") and not evs:
                continue
            bs = r.get("budget_start")
            if bs is None:
                diag["no_budget_start"] += 1
                continue
            known = set(r.get("known_langs") or [])
            parent = set(r.get("parent_langs") or [])
            if evs:                                   # 이번 턴에 냈다 — 눈금은 로그에서
                for ev in evs:
                    out.append({
                        "turn": turn, "agent": r["agent"], "country": r["country"],
                        "age": ev.get("age", r.get("age", 0)),
                        # **`speed` 에서 낸다.** `required` 는 늘 L 이라 눈금이 안 된다.
                        "rung": round(1.0 / (ev.get("speed") or 1.0), 4),
                        "cost": base / (ev.get("speed") or 1.0), "paid": ev["charged"],
                        "budget_start": bs, "put_in": True})
                continue
            # 안 냈다 — 그가 고를 수 있었던 **가장 싼** 눈금을 다시 계산한다
            best = None
            for cid, lg in country_lang.items():
                if not lg or lg in known:
                    continue
                dom = bool(speakers[(r["country"], lg)] - {r["agent"]})
                # **할인의 곱이 아니라 가속의 합이다** (8/23). `(0.5 if dom)*(0.5 if parent)`
                # 는 8/22 에 없어진 규칙이었다 — 사유는 배속에 **더해지고**, 총 지출은
                # 그 역수다: 사유 0·1·2 → 1.0 · 1/1.5 · 1/2.
                reasons = int(dom) + int(lg in parent)
                mult = round(1.0 / (1.0 + speedup * reasons), 4)
                if best is None or mult < best:
                    best = mult
            if best is None:                          # 배울 게 없다 (이미 3개국어)
                continue
            out.append({"turn": turn, "agent": r["agent"], "country": r["country"],
                        "age": r.get("age", 0), "rung": best, "cost": base * best,
                        "paid": 0.0, "budget_start": bs, "put_in": False})
    diag["opportunities"] = len(out)
    diag["learn_base"] = base
    diag["learn_speedup"] = speedup
    return out, diag, acquired


# ── 추정 ────────────────────────────────────────────────────────────────────────

def take_rates(obs: list[dict], by_age: bool = False) -> dict:
    """눈금별 **납부율**. 분모는 그 턴에 낼 돈이 있었던 (사람, 턴).

    **분할 납부라 기준이 바뀌었다.** 전에는 "전액을 감당할 수 있었는데 배웠는가" 였다.
    이제 한 푼이라도 낼 수 있으면 시작할 수 있으므로, 턴 단위의 결정은 **"냈는가"** 다.
    x 가 0 이면 한 푼도 안 낸다 — 그 자체가 드러난 선호다.

    `budget_start > 0` 을 기회로 본다. 전액 기준으로 두면 분모가 거의 비어 버린다
    (실측에서 턴 시작 예산이 대개 100~200원인데 눈금은 150~600 이다).
    """
    cells: dict = defaultdict(lambda: {"n": 0, "put_in": 0, "paid": 0.0})
    for o in obs:
        if o["budget_start"] <= 0:                   # 낼 돈이 없다 — 기회가 아니다
            continue
        keys = [(o["rung"], "all")]
        if by_age:
            keys.append((o["rung"], BAND_NAME[band_of(o["age"])]))
        for k in keys:
            cells[k]["n"] += 1
            cells[k]["put_in"] += bool(o["put_in"])
            cells[k]["paid"] += o.get("paid", 0.0)
    return {k: {"n": v["n"], "rate": round(v["put_in"] / v["n"], 4) if v["n"] else None,
                "put_in": v["put_in"], "paid": round(v["paid"], 1)}
            for k, v in cells.items()}


def acquisitions(acq: list[dict], base: float, by_age: bool = False) -> dict:
    """습득한 눈금별 건수. **가장 비싼 습득 눈금이 `x` 의 하한**이다.

    납부율(위)은 *"시작했는가"* 를 재고, 이것은 *"끝냈는가"* 를 잰다. 분할 납부에서는
    둘이 갈릴 수 있다 — 반쯤 내다 죽거나, 값이 싸져 예상보다 일찍 끝나거나.
    """
    cells: dict = defaultdict(int)
    for a in acq:
        rung = round(1.0 / (a.get("speed") or 1.0), 4)
        cells[(rung, "all")] += 1
        if by_age:
            cells[(rung, BAND_NAME[band_of(a.get("age", 0))])] += 1
    return dict(cells)


def bracket(rates: dict, band: str = "all", threshold: float = 0.5,
            speedup: float = DEFAULT_SPEEDUP) -> dict:
    """`x` 를 구간으로 좁힌다. spec 7장의 네 구간.

    **가장 비싼 "켜진" 눈금이 하한**입니다 — 그걸 실제로 지불했으니 `x` 는 최소 그만큼.
    그 위 눈금이 꺼져 있으면 그게 상한이 됩니다.

    ⚠ 표본이 없는 눈금은 **꺼진 것이 아닙니다.** 가장 싼 눈금(사유 둘)은 부모가 그 언어를
      알고 국내에도 구사자가 있을 때만 나오므로 런 초반에는 존재하지 않습니다. 그 경우
      구간 한쪽이 열린 채로 보고합니다 — 닫힌 것처럼 적으면 없는 정밀도를 지어냅니다.
    """
    on, off, seen = [], [], []
    for r in rungs_for(speedup):
        c = rates.get((r, band))
        if not c or not c["n"]:
            continue
        seen.append(r)
        (on if c["rate"] >= threshold else off).append(r)

    lo = max(on) if on else None
    hi = min((r for r in off if lo is None or r > lo), default=None)
    if lo is None and off:
        hi = min(off)
    return {
        "band": band, "lower": lo, "upper": hi,
        "rungs_seen": [rung_name(r) for r in sorted(seen)],
        "label": _label(lo, hi),
        "n": sum(rates[(r, band)]["n"] for r in seen),
    }


def _label(lo: float | None, hi: float | None) -> str:
    if lo is None and hi is None:
        return "표본 없음"
    if lo is None:
        return f"x < {rung_name(hi)}"
    if hi is None:
        return f"x ≥ {rung_name(lo)}"
    return f"{rung_name(lo)} ≤ x < {rung_name(hi)}"


def estimate(run_dirs: list[Path], by_age: bool = True) -> dict:
    """여러 런을 합쳐 하나의 `x̂` 를 낸다. 같은 노브의 런들을 함께 넣으세요."""
    obs: list[dict] = []
    acq: list[dict] = []
    diags = []
    base = None
    speedup = DEFAULT_SPEEDUP
    for d in run_dirs:
        o, diag, a = observations(d)
        obs += o
        acq += a
        base = diag.get("learn_base") or base
        speedup = diag.get("learn_speedup", speedup)
        diags.append({"run": Path(d).name, **diag})
    rates = take_rates(obs, by_age=by_age)
    ac = acquisitions(acq, base or 1.0, by_age=by_age)
    bands = ["all"] + ([BAND_NAME[b] for b in AGE_BANDS] if by_age else [])
    return {"rates": {f"{rung_name(r)}|{b}": v for (r, b), v in sorted(rates.items())},
            "acq": {f"{rung_name(r)}|{b}": v for (r, b), v in sorted(ac.items())},
            "brackets": {b: bracket(rates, b, speedup=speedup) for b in bands},
            "n_obs": len(obs), "n_acq": len(acq), "diag": diags}


def delta_x(by_knob: dict[float, dict]) -> dict:
    """`Δx = x̂(노브 최저) − x̂(노브 최고)`. 연쇄 1~2칸의 정량 진술.

    양쪽 하한이 다 있어야 뺄 수 있습니다. 한쪽이 열려 있으면 **방향만** 보고합니다 —
    없는 수를 지어내는 것보다 "이쪽이 더 낮다" 가 정직합니다.
    """
    if len(by_knob) < 2:
        return {"delta": None, "note": "노브가 둘 이상 필요합니다"}
    lo_knob, hi_knob = min(by_knob), max(by_knob)
    a = by_knob[lo_knob]["brackets"]["all"]["lower"]
    b = by_knob[hi_knob]["brackets"]["all"]["lower"]
    if a is None or b is None:
        return {"delta": None, "knob_low": lo_knob, "knob_high": hi_knob,
                "note": f"하한이 열려 있습니다 (저 {a} · 고 {b})"}
    return {"delta": round(a - b, 4), "knob_low": lo_knob, "knob_high": hi_knob,
            "unit": "L 배수",
            "note": "양수면 AI 가 쌀수록 학습의 암묵효용이 낮다는 뜻입니다"}


# ── 출력 ────────────────────────────────────────────────────────────────────────

def format_estimate(est: dict, knob=None) -> str:
    L = [f"x̂ 추정   노브 {knob}   관측 {est['n_obs']}건 · 습득 {est['n_acq']}건", "─" * 72]
    L.append(f"{'눈금':<6}{'층':<8}{'기회':>6}{'납부':>6}{'납부율':>9}{'총액':>10}{'습득':>7}")
    for key, v in est["rates"].items():
        rung, band = key.split("|")
        rate = "—" if v["rate"] is None else f"{v['rate']:.0%}"
        L.append(f"{rung:<6}{band:<8}{v['n']:>6}{v['put_in']:>6}{rate:>9}"
                 f"{v['paid']:>10.0f}{est.get('acq', {}).get(f'{rung}|{band}', 0):>7}")
    L.append("")
    for band, b in est["brackets"].items():
        if not b["n"]:
            continue
        L.append(f"  {band:<6} {b['label']:<18} (눈금 {'·'.join(b['rungs_seen'])}, n={b['n']})")
    bad = [d for d in est["diag"] if d.get("no_budget_start") or d.get("error")]
    if bad:
        L.append("\n⚠ " + " · ".join(
            d.get("error") or f"{d['run']}: budget_start 없는 상태행 {d['no_budget_start']}"
            for d in bad))
        L.append("  budget_start 는 x̂ 의 분모입니다. 이 필드 이전의 런은 x̂ 를 낼 수 없습니다.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="x̂ 추정 (spec 7 · 8.4)")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--no-age", action="store_true", help="나이 층화 없이")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dirs = [Path(d) for d in a.run_dirs if (Path(d) / "state.jsonl").exists()]
    if not dirs:
        print("state.jsonl 을 가진 런이 없습니다", file=sys.stderr)
        return 2

    # 노브별로 묶는다 — x̂ 는 노브마다 하나이고, Δx 는 그 둘의 차다.
    groups: dict = defaultdict(list)
    for d in dirs:
        knob = None
        cs = d / "config_snapshot.yaml"
        if cs.exists():
            import yaml
            knob = (yaml.safe_load(cs.read_text(encoding="utf-8")) or {}).get("knob_ai")
        groups[knob].append(d)

    ests = {}
    for knob, ds in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0])):
        est = estimate(ds, by_age=not a.no_age)
        ests[knob] = est
        print(format_estimate(est, knob))
        print()

    numeric = {k: v for k, v in ests.items() if isinstance(k, (int, float))}
    if len(numeric) >= 2:
        d = delta_x(numeric)
        print(f"Δx = {'—' if d['delta'] is None else d['delta']}   "
              f"(노브 {d.get('knob_low')} vs {d.get('knob_high')})")
        print(f"     {d['note']}")

    if a.out:
        Path(a.out).write_text(
            json.dumps({str(k): v for k, v in ests.items()},
                       ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n{a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
