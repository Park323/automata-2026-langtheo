"""한 해에 API 비용과 시간이 얼마 드는가.

    ./.venv/bin/python tools/analysis/cost.py runs/vis3c [runs/...]

**값을 추정하지 않는다.** OpenRouter 가 호출마다 실제 과금액(`usage.cost`)을 돌려주므로
그것을 그대로 합친다 — 단가표를 코드에 적으면 모델을 바꿀 때 조용히 낡는다.

벽시계는 `response.created`(초 단위 유닉스 시각)의 최소·최대로 잡는다. 순차 라운드로빈은
호출이 겹치지 않으므로 지연시간 합과 대체로 같지만, **대체로 같다는 것도 재서 보인다**
— 벌어지면 그 차이가 우리 쪽 오버헤드다.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def load(run: pathlib.Path) -> list[dict]:
    p = run / "raw_calls.jsonl"
    if not p.exists():
        raise SystemExit(f"{p} 가 없습니다")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _cost(r: dict) -> float:
    u = (r.get("response") or {}).get("usage") or {}
    return float(u.get("cost") or 0.0)


def _tok(r: dict) -> tuple[int, int]:
    u = (r.get("response") or {}).get("usage") or {}
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def report(run: pathlib.Path) -> dict:
    rows = load(run)
    ok = [r for r in rows if r.get("response")]
    turns = sorted({r["turn"] for r in rows if r.get("turn") is not None})

    per = defaultdict(lambda: dict(calls=0, cost=0.0, pt=0, ct=0, ms=0,
                                   t0=None, t1=None))
    kinds = defaultdict(lambda: dict(calls=0, cost=0.0))
    no_cost = 0
    for r in ok:
        d = per[r.get("turn")]
        d["calls"] += 1
        c = _cost(r)
        if c == 0.0:
            no_cost += 1
        d["cost"] += c
        pt, ct = _tok(r)
        d["pt"] += pt
        d["ct"] += ct
        d["ms"] += int(r.get("latency_ms") or 0)
        created = (r.get("response") or {}).get("created")
        if created:
            d["t0"] = created if d["t0"] is None else min(d["t0"], created)
            d["t1"] = created if d["t1"] is None else max(d["t1"], created)
        k = kinds[r.get("kind", "?")]
        k["calls"] += 1
        k["cost"] += c

    model = next((r["request"].get("model") for r in ok if r.get("kind") == "agent"), "?")
    tr = next((r["request"].get("model") for r in ok if r.get("kind") != "agent"), None)

    print(f"\n═══ {run.name} ═══")
    print(f"에이전트 모델: {model}" + (f"   번역 모델: {tr}" if tr else ""))
    if no_cost:
        print(f"⚠ 과금액이 안 실린 호출 {no_cost}건 — 아래 합계에서 빠졌습니다")

    print("\n  해   호출     비용($)   프롬프트토큰  생성토큰    API초   벽시계초")
    tot = dict(calls=0, cost=0.0, pt=0, ct=0, ms=0, wall=0)
    for t in turns:
        d = per[t]
        wall = (d["t1"] - d["t0"]) if d["t0"] is not None else 0
        print(f"  {t:>3} {d['calls']:>6}  {d['cost']:>10.4f}  {d['pt']:>12,}"
              f"  {d['ct']:>9,}  {d['ms']/1000:>7.1f}  {wall:>9}")
        for k in ("calls", "cost", "pt", "ct", "ms"):
            tot[k] += d[k]
        tot["wall"] += wall

    n = len(turns) or 1
    print(f"  합계 {tot['calls']:>6}  {tot['cost']:>10.4f}  {tot['pt']:>12,}"
          f"  {tot['ct']:>9,}  {tot['ms']/1000:>7.1f}  {tot['wall']:>9}")
    print(f"  해당 {tot['calls']/n:>6.1f}  {tot['cost']/n:>10.4f}  {tot['pt']/n:>12,.0f}"
          f"  {tot['ct']/n:>9,.0f}  {tot['ms']/1000/n:>7.1f}  {tot['wall']/n:>9.0f}")

    # ── 프롬프트 캐시 ──────────────────────────────────────────────────────
    #
    # 대화가 append-only 라 앞부분이 그대로 남는다 — 그 몫은 싸게 청구된다. `cost` 는
    # 이미 그것을 반영한 실제 과금액이므로 **따로 깎지 않는다.** 여기 적는 이유는
    # 정상상태 추정이 「토큰에 선형」 을 가정하기 때문이다: 캐시율이 해마다 오르면 그
    # 추정은 과대가 되고, 평평하면 맞는다. 3해에서는 평평했다 (47 / 63 / 46%).
    cd = sum(((r.get("response") or {}).get("usage") or {})
             .get("prompt_tokens_details", {}).get("cached_tokens", 0) for r in ok)
    if cd:
        unc = tot["pt"] - cd
        print(f"\n  프롬프트 캐시  {cd:,} / {tot['pt']:,} ({cd/tot['pt']:.0%}) 가 캐시에서 "
              f"나왔다\n                 미캐시 토큰당 ${tot['cost']/unc*1e6:.3f}/1M")

    print("\n  종류별")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]["cost"]):
        share = 100 * v["cost"] / tot["cost"] if tot["cost"] else 0
        print(f"    {k:10} 호출 {v['calls']:>4}   ${v['cost']:.4f}  ({share:.0f}%)")

    per_turn_cost, per_turn_wall = tot["cost"] / n, tot["wall"] / n

    # ── 정상상태 ───────────────────────────────────────────────────────────
    #
    # **앞 몇 해의 값을 그대로 곱하면 과소추정한다.** 대화가 죽을 때까지 이어지므로
    # 프롬프트가 매 해 불어나고, `evict` 가 도는 지점에서야 멈춘다. 3해 런에서 호출당
    # 프롬프트가 3.4k → 4.2k → 5.2k 였다 — 아직 천장에 닿지도 않았다.
    #
    # 천장은 계산할 수 있다: evict 가 `convo + 도구스키마 ≤ context_limit` 을 지키므로
    #
    #     호출당 프롬프트(천장) = system + 도구스키마 + (context_limit − 도구스키마)
    #                           = system + context_limit
    #
    # 비용은 프롬프트가 지배하므로(아래 실측 비율) 그 비율만 늘려 잡는다.
    last = per[turns[-1]]
    calls_last = last["calls"] or 1
    pt_now = last["pt"] / calls_last
    ratio = _ceiling(rows) / pt_now if pt_now else 1.0
    # 프롬프트가 비용에서 차지하는 몫 — cost_details 에 실려 온다
    pshare = _prompt_share(ok)
    grow = pshare * ratio + (1 - pshare)
    ss_cost = (last["cost"] / calls_last) * grow * calls_last
    ss_wall = (last["ms"] / 1000 / calls_last) * grow * calls_last

    print(f"\n  정상상태 추정 — 호출당 프롬프트 {pt_now:,.0f} → 천장 "
          f"{_ceiling(rows):,.0f} 토큰 ({ratio:.2f}배), 프롬프트가 비용의 {pshare:.0%}")
    print(f"    한 해   ${ss_cost:.4f}   {ss_wall:.0f}초   "
          f"(마지막 해 실측 ${last['cost']:.4f} · {last['ms']/1000:.0f}초의 {grow:.2f}배)")

    print("\n  환산 — 왼쪽은 지금 실측 평균, 오른쪽은 정상상태")
    for label, turns_n, runs_n in (("100해 1런", 100, 1),
                                   ("100해 × 노브 4단", 100, 4),
                                   ("100해 × 노브 4단 × 시드 3", 100, 12)):
        k = turns_n * runs_n
        print(f"    {label:26} ${per_turn_cost*k:>7.2f} / ${ss_cost*k:>7.2f}   "
              f"{per_turn_wall*k/3600:>5.1f} / {ss_wall*k/3600:>5.1f}시간")
    print("\n  ⚠ 인구는 9명 고정이지만 **대화량은 늘 수 있다** — 호출 수가 늘면 위 값도"
          "\n    같이 는다. 3해에서 해당 호출이 53 → 63 → 63 이었다.")
    return dict(cost_per_turn=per_turn_cost, wall_per_turn=per_turn_wall,
                steady_cost=ss_cost, steady_wall=ss_wall)


def _ceiling(rows: list[dict]) -> float:
    """`evict` 가 허용하는 호출당 프롬프트 토큰의 천장.

    evict 는 `convo + 도구스키마 ≤ context_limit` 만 지킨다. system 은 convo 에 없고 매
    호출 앞에 붙으므로 천장 밖이다. 따라서 천장은 `system + context_limit` 이다.
    """
    from core import agent_loop, config
    cfg = config.load("configs/base.yaml")
    sysmsg = next((m for r in rows for m in r.get("request", {}).get("messages", [])
                   if m.get("role") == "system"), None)
    sys_tok = agent_loop.estimate_tokens([sysmsg]) if sysmsg else 0
    return sys_tok + cfg.llm.context_limit


def _prompt_share(rows: list[dict]) -> float:
    """비용 중 프롬프트가 차지하는 몫. `cost_details` 에 실려 온다."""
    pc = cc = 0.0
    for r in rows:
        d = ((r.get("response") or {}).get("usage") or {}).get("cost_details") or {}
        pc += float(d.get("upstream_inference_prompt_cost") or 0.0)
        cc += float(d.get("upstream_inference_completions_cost") or 0.0)
    return pc / (pc + cc) if (pc + cc) else 0.8


if __name__ == "__main__":
    args = sys.argv[1:] or ["runs/vis3c"]
    for a in args:
        report(pathlib.Path(a))
