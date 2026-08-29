"""학습은 사치재인가 — 학습 시점의 잔여 AP 와 대안 메뉴.

각 agent_turn 의 행동열에 가격표를 얹어 러닝 잔액을 재구성한다.
가격: speak 자국 0.04 / 국제원문 0.06 / 국제ai knob / vote 0.04 /
     memory_write 0.04 / invest 0.10 / learn 0.10 / observe_risk 0.30
"""
import json, pathlib, collections, statistics as St
def nat(a): return "".join(c for c in a if not c.isdigit())
KN = {"noai":"no-ai","ai006":"0.06","ai012":"0.12","ai024":"0.24"}
KNOB = {"no-ai":None,"0.06":.06,"0.12":.12,"0.24":.24}
OBS = 0.30
def cost(a, actor, knob):
    t = a["type"]
    if t == "speak":
        if nat(a.get("to","")) == nat(actor): return 0.04
        return knob if (a.get("route")=="ai" and knob) else 0.06
    return {"vote":0.04,"memory_write":0.04,"invest":0.10,"learn":0.10,
            "observe_risk":OBS}.get(t, 0.0)

R = collections.defaultdict(lambda: collections.defaultdict(list))
MENU = collections.defaultdict(collections.Counter)   # 잔액 0.10~0.14 시점의 선택
for p in sorted(pathlib.Path("runs").iterdir()):
    if not p.is_dir() or not (p.name.startswith("260827-0") or p.name.startswith("260828-0")): continue
    if "dark" in p.name or "dawn" in p.name or p.name.endswith(("003-ai012","004-ai024","002-ai006")): continue
    if not (p/"metrics.jsonl").exists(): continue
    mt=[r for r in [json.loads(l) for l in (p/"metrics.jsonl").read_text().splitlines() if l.strip()] if r.get("step") is None]
    if not mt or mt[-1]["turn"]<30: continue
    k = KN[p.name.split("-")[-1]]; knob = KNOB[k]
    for l in (p/"events.jsonl").read_text().splitlines():
        e = json.loads(l)
        if e.get("type")!="agent_turn" or not e.get("actions"): continue
        acts = e["actions"]; left = 1.0
        n = len(acts)
        for i,a in enumerate(acts):
            c = cost(a, e["agent"], knob)
            if a["type"]=="learn":
                R[k]["left"].append(left)
                R[k]["is_last"].append(i==n-1)
                R[k]["pos"].append((i+1)/n)
                R[k]["exhaust"].append(left-c < 0.04)   # 학습 후 아무것도 못 함
            if 0.10-1e-9 <= left <= 0.14+1e-9:          # 「잔돈 0.1」 순간의 메뉴 선택
                MENU[k][a["type"]] += 1
            left -= c

K = ["0.06","0.12","0.24","no-ai"]
print("=== 학습 행동의 잔액 프로필 ===")
print(f"{'':<30}"+"".join(f"{k:>9}" for k in K))
rows = [("학습 시점 잔액 중앙값", lambda v: St.median(v["left"])),
        ("잔액 ≤0.20 에서 학습 %", lambda v: 100*sum(1 for x in v["left"] if x<=0.20)/len(v["left"])),
        ("잔액 ≥0.50 에서 학습 %", lambda v: 100*sum(1 for x in v["left"] if x>=0.50)/len(v["left"])),
        ("그 해의 마지막 행동 %", lambda v: 100*sum(v["is_last"])/len(v["is_last"])),
        ("행동열 내 위치 중앙값", lambda v: St.median(v["pos"])),
        ("학습 후 빈털터리 %", lambda v: 100*sum(v["exhaust"])/len(v["exhaust"])),
        ("(학습 행동 수)", lambda v: len(v["left"]))]
for name,f in rows:
    line=f"{name:<30}"
    for k in K:
        x=f(R[k]); line += f"{x:>9.0f}" if name.startswith("(") else f"{x:>9.2f}"
    print(line)
print()
print("=== 잔액이 정확히 잔돈(0.10~0.14)인 순간, 무엇을 골랐나 (%) ===")
alltypes = sorted({t for c in MENU.values() for t in c})
print(f"{'':<16}"+"".join(f"{k:>9}" for k in K))
for t in alltypes:
    line=f"{t:<16}"
    for k in K:
        tot=sum(MENU[k].values()); line+=f"{100*MENU[k][t]/tot:>9.1f}"
    print(line)
print(f"{'(순간 수)':<16}"+"".join(f"{sum(MENU[k].values()):>9}" for k in K))
