#!/usr/bin/env python3
"""조율의 암묵 비용 — 세계가 준 예산과 실제로 쓴 소통비의 차 (Eddie 착안, 8/25).

    ./.venv/bin/python -m tools.score.coordination runs/<run> [runs/<run2> ...]

## 왜 이 차가 뜻을 갖나

합리적인 나라는 이렇게 고른다.

    조율 비용 = 요격기 기여율 + 소통비        (두 나라를 믿어야 한다)
    단독 비용 = 벙커 기여율                   (아무도 안 믿어도 된다)

    조율을 고른다  ⟺  소통비 < 벙커 기여율 − 요격기 기여율

그 우변이 **세계가 조율에 내준 예산**이다. 실제로 쓴 소통비를 빼면 남는 것이 잔차다.

    잔차 > 0 · 성공    여유가 있었다
    잔차 > 0 · 실패    **예산이 있었는데 안 썼다** — 이 크기가 불신·오해의 값이다
    잔차 < 0           세계가 조율을 감당 불가로 만들었다 (호스트 오선택 또는 노브 과다)

## ⚠ 단위

1인부담(`threshold / (사람 × 해)`)은 **진척**이고 소통비는 **AP** 다. 그대로 빼면 안 된다.
환산은 `1 AP → capacity_per_year × k × build_mult` 이고, **`build_mult` 이 요격기에만
걸리므로 계수가 목표마다 다르다** — 그래서 예산은 실제 호스트의 효율로 계산한다.

    호스트 효율 1.3 → +0.207 AP/해 · 1.0 → +0.069 · 0.7 → −0.188

## 잔차가 뭉개는 것 넷 — 로그로 갈릴 수 있다

    ① 안 시도했다 (불신)        국제 발신 건수가 0 에 가깝다
    ② 시도했는데 왜곡으로 낭비    미전달(`unreadable`) 건수 · 지표 6a 의 번역 손실
    ③ 호스트를 잘못 골랐다       build_mult 과 실제 투자처가 어긋난다
    ④ 조율이 싸다는 걸 몰랐다     사고 로그에 그 비교가 없다

`x̂`(학습의 암묵 **효용**)와 쌍이다 — 이쪽은 불신의 암묵 **비용**이고, 자가 다르므로
서로를 교차 검증할 수 있다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from core import asserts, config  # noqa: E402


def _rows(d: pathlib.Path, name: str) -> list[dict]:
    p = d / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def measure(run_dir: str | pathlib.Path) -> dict:
    d = pathlib.Path(run_dir)
    snap = __import__("yaml").safe_load((d / "config_snapshot.yaml").read_text(encoding="utf-8"))
    # **AP 이전의 런은 거부한다.** 돈이 있고 벙커가 확률이던 세계에서는 이 지표의 두 항이
    # 둘 다 다른 뜻이다 — 대충 환산해 숫자를 내면 없는 비교를 지어내는 것이다.
    if "comm_intl_ai_ap" not in (snap.get("config") or {}).get("knob", {}):
        raise SystemExit(f"{d.name}: AP 전면 통일(8/25) 이전의 런이다 — 이 지표를 낼 수 없다.\n"
                         f"  그 세계에는 돈이 있었고 벙커가 임계가 아니라 확률이었다.")
    cfg = config.from_dict(snap["config"])
    knob = snap.get("knob_ai")
    n, T = cfg.world.agents_per_country, cfg.world.total_turns
    cap, k = asserts.capacity_per_year(cfg), cfg.k

    metrics = _rows(d, "metrics")
    turns = max((m["turn"] for m in metrics), default=0)
    last = [m for m in metrics if m["turn"] == turns]
    land = last[0]["land"] if last else {}
    build = last[0].get("build_mult") or {}
    prog = last[0]["progress"] if last else {}

    # **호스트는 진척이 가장 많은 요격기 나라다.** 아무도 안 지으면 예산을 못 정한다.
    hosts = [(c, prog.get(c, 0.0)) for c, l in land.items() if l == "interceptor"]
    host = max(hosts, key=lambda x: x[1])[0] if hosts else None
    host_mult = build.get(host, 1.0) if host else None

    # 기여율 = 1인부담(진척) ÷ (1 AP 가 낳는 진척)
    bunker_rate = cfg.thresholds.bunker / (n * T) / (cap * k)
    intc_rate = (cfg.thresholds.interceptor / (3 * n * T) / (cap * k * host_mult)
                 if host else None)
    budget = None if intc_rate is None else bunker_rate - intc_rate

    # 실제 소통비 — 학습 AP + 국제 발신 AP
    learns = [e for e in _rows(d, "events") if e.get("type") == "learn" and "kind" not in e]
    msgs = _rows(d, "messages")
    intl = [m for m in msgs if m.get("route") != "domestic"]
    learn_ap = len(learns) * cfg.ap.unit
    send_ap = sum((knob if m["route"] == "ai" else cfg.ap.speak) for m in intl)
    people_years = sum(1 for s in _rows(d, "state") if s.get("alive")) or 1
    comm = (learn_ap + send_ap) / people_years

    return {
        "run": d.name, "turns": turns, "knob": knob,
        "host": host, "host_mult": host_mult, "land": land,
        "bunker_rate": bunker_rate, "intc_rate": intc_rate, "budget": budget,
        "learn_ap": learn_ap, "send_ap": send_ap, "comm_per_person_year": comm,
        "residual": None if budget is None else budget - comm,
        "intl_msgs": len(intl), "undelivered": sum(1 for m in intl if not m.get("delivered")),
        "outcome": (json.loads((d / "summary.json").read_text(encoding="utf-8"))
                    .get("final", {}).get("outcome") if (d / "summary.json").exists() else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    print(f"{'run':<18}{'해':>4}{'노브':>7}{'호스트':>12}{'예산':>8}{'소통비':>8}"
          f"{'잔차':>8}{'국제':>6}{'미전달':>7}  판정")
    for r in a.runs:
        m = measure(r)
        b = "—" if m["budget"] is None else f"{m['budget']:+.3f}"
        res = "—" if m["residual"] is None else f"{m['residual']:+.3f}"
        hostm = "—" if m["host"] is None else f"{m['host']}×{m['host_mult']}"
        # **`or 0` 을 쓰지 않는다.** `None` 은 「AI 가 없는 세계」 고 `0` 은 「공짜 AI」 다 —
        # 프롬프트에서 같은 실수를 이미 한 번 했다 (8/25 · `knob_ai or 0.0`). 표에서
        # 둘이 같은 모양이면 대조군과 최저 눈금 런을 구분할 수 없다.
        kn = "없음" if m["knob"] is None else f"{m['knob']:.2f}"
        print(f"{m['run']:<18}{m['turns']:>4}{kn:>7}{hostm:>12}{b:>8}"
              f"{m['comm_per_person_year']:>8.3f}{res:>8}{m['intl_msgs']:>6}"
              f"{m['undelivered']:>7}  {m['outcome']}")
        if m["host"] is None:
            print("     ⚠ 요격기를 짓는 나라가 없다 — 예산이 정의되지 않는다 "
                  f"(국토 {m['land']})")


if __name__ == "__main__":
    main()
