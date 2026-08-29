import json, pathlib, collections, re
def nat(a): return "".join(c for c in a if not c.isdigit())
KN = {"noai": "no-ai", "ai006": "0.06", "ai012": "0.12", "ai024": "0.24"}
rows = []
for p in sorted(pathlib.Path("runs").iterdir()):
    if not p.is_dir() or not (p.name.startswith("260827-0") or p.name.startswith("260828-0")): continue
    if "dark" in p.name or "dawn" in p.name or p.name.endswith(("003-ai012", "004-ai024", "002-ai006")): continue
    if not (p/"metrics.jsonl").exists(): continue
    mt = [r for r in [json.loads(l) for l in (p/"metrics.jsonl").read_text().splitlines() if l.strip()]
          if r.get("step") is None]
    if not mt or mt[-1]["turn"] < 30: continue
    ev = [json.loads(l) for l in (p/"events.jsonl").read_text().splitlines() if l.strip()]
    inv = collections.Counter()
    for e in ev:
        for a in e.get("actions", []):
            if a["type"] == "invest" and a.get("target") == "facility":
                inv[a.get("to") or nat(e["agent"])] += 1
    host = max(("Asla", "Ranoa", "Miris"), key=lambda c: inv[c])
    m = mt[-1]; land = m["land"]; P = m["progress"]
    acq = sum(1 for e in ev if e.get("type") == "learn" and e.get("kind") == "acquired")
    sd = int(re.search(r"^seed: (\d+)", (p/"config_snapshot.yaml").read_text(), re.M).group(1))
    rows.append((KN[p.name.split("-")[-1]], sd, p.name, host, max(P.values()),
                 sum(1 for c in land if land[c] == "bunker"), acq))
order = {"0.06": 0, "0.12": 1, "0.24": 2, "no-ai": 3}
rows.sort(key=lambda r: (order[r[0]], r[1]))
print("| 노브 | 씨앗 | 런 | 숙주(최다투자국) | 최고 진척 /10,899 | 벙커 이탈 | 언어 습득 |")
print("|---|---|---|---|---|---|---|")
for k, sd, run, host, best, bunk, acq in rows:
    star = " ★" if host == "Ranoa" else ""
    print(f"| {k} | {sd} | `{run}` | {host}{star} | {best:,.0f} | {bunk} | {acq} |")
