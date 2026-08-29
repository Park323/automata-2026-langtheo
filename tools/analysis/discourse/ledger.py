"""언어 학습의 AP 원장과 암묵 효용 x̂ 역산.

원장 규칙 (이 세계의 가격표):
  original 0.06 — 내가 상대 언어를 알면 확실 배달(writer), 모르면 도박(reader 여부)
  ai       knob — 항상 배달. 0.06 조건에서는 original 과 동가
  learn    0.10/회 · 총 0.40/0.30/0.20 (사유 0/1/2)

학습의 금전 회수(발신측)는 「도박을 확실로 바꾼 것」뿐이다:
  절약/통 = 0.06 × (1/p − 1),  p = 도박 배달률(그 조건 실측)
0.06 조건에서는 ai 가 같은 값에 확실하므로 발신측 회수 = 0 (구조적).
수신·정보 가치는 AP 원장 밖 — 그것이 x̂ 로 흡수된다.
"""
import json, pathlib, collections, statistics as St
def nat(a): return "".join(c for c in a if not c.isdigit())
KN = {"noai": "no-ai", "ai006": "0.06", "ai012": "0.12", "ai024": "0.24"}
KNOB = {"no-ai": None, "0.06": .06, "0.12": .12, "0.24": .24}
runs = []
for p in sorted(pathlib.Path("runs").iterdir()):
    if not p.is_dir() or not (p.name.startswith("260827-0") or p.name.startswith("260828-0")): continue
    if "dark" in p.name or "dawn" in p.name or p.name.endswith(("003-ai012","004-ai024","002-ai006")): continue
    if not (p/"metrics.jsonl").exists(): continue
    mt = [r for r in [json.loads(l) for l in (p/"metrics.jsonl").read_text().splitlines() if l.strip()] if r.get("step") is None]
    if mt and mt[-1]["turn"] >= 30: runs.append(p)

AGG = collections.defaultdict(lambda: collections.defaultdict(list))
XHAT = collections.defaultdict(list)
for p in runs:
    k = KN[p.name.split("-")[-1]]
    ev = [json.loads(l) for l in (p/"events.jsonl").read_text().splitlines() if l.strip()]
    ms = [json.loads(l) for l in (p/"messages.jsonl").read_text().splitlines() if l.strip()]
    # ── 원장 ──
    learn_calls = sum(1 for e in ev for a in e.get("actions", []) if a["type"] == "learn")
    learnAP = 0.1 * learn_calls
    intl = [m for m in ms if nat(m["from"]) != nat(m["to"])]
    orig = [m for m in intl if m["route"] == "original"]
    ai   = [m for m in intl if m["route"] == "ai"]
    intlAP = 0.06 * len(orig) + (KNOB[k] or 0) * len(ai)
    deliv = sum(1 for m in intl if m["delivered"])
    AGG[k]["learnAP"].append(learnAP)
    AGG[k]["intlAP"].append(intlAP)
    AGG[k]["deliv"].append(deliv)
    AGG[k]["cost_per"].append((learnAP + intlAP) / max(deliv, 1))
    # ── 도박 배달률 p (writer 아닌 원문 = 도박) ──
    gam = [m for m in orig if (m["meta"].get("direct_by") or "") != "writer"]
    gd = sum(1 for m in gam if m["delivered"])
    if gam: AGG[k]["p_gamble"].append(gd / len(gam))
    # ── 습득별: 비용과 금전 회수 → x̂ 하한 = 비용 − 회수 ──
    p_g = gd / len(gam) if gam else 0.5
    acq = [e for e in ev if e.get("type") == "learn" and e.get("kind") == "acquired"]
    NLANG = {"Asla": "ja", "Ranoa": "zh", "Miris": "fr"}
    for e in acq:
        import math
        calls = math.ceil(e["required"] / (20 * e.get("speed", 1.0)) - 1e-9)
        cost = 0.1 * calls
        # 습득 후, 그 언어로 확실 배달한 원문 (writer 덕 배달)
        m_used = sum(1 for m in orig
                     if m["from"] == e["agent"] and nat(m["to"]) == e["target"]
                     and m["turn"] >= e["turn"] and m["delivered"]
                     and (m["meta"].get("direct_by") or "") == "writer")
        # 0.06+ 조건: ai 가 knob 가격에 확실 → 절약/통 = max(0, knob − 0.06)=0 (0.06),
        #             knob>0.06 이면 ai 대비 절약 = (knob − 0.06)/통
        # no-ai: 도박 대비 절약 = 0.06×(1/p − 1)/통
        if KNOB[k] is None:
            save = m_used * 0.06 * (1/p_g - 1)
        else:
            save = m_used * max(0.0, KNOB[k] - 0.06)
        XHAT[k].append(dict(cost=cost, save=save, x=cost - save, used=m_used))

K = ["0.06", "0.12", "0.24", "no-ai"]
print("=== AP 원장 (판당 평균) ===")
print(f"{'':<26}" + "".join(f"{k:>9}" for k in K))
for name, key in (("학습 지출", "learnAP"), ("국제 발신 지출", "intlAP"),
                  ("배달된 국제 통수", "deliv"), ("배달 1통 총비용(학습포함)", "cost_per")):
    print(f"{name:<26}" + "".join(f"{St.mean(AGG[k][key]):>9.3f}" for k in K))
print(f"{'도박 배달률 p':<26}" + "".join(f"{St.mean(AGG[k]['p_gamble']):>9.3f}" for k in K))
print()
print("=== 습득별 경제성과 x̂ 하한 (조건별 습득 전수) ===")
print(f"{'':<26}" + "".join(f"{k:>9}" for k in K))
rows = [("습득 수", lambda v: len(v)),
        ("평균 학습비", lambda v: St.mean(x["cost"] for x in v)),
        ("평균 금전 회수", lambda v: St.mean(x["save"] for x in v)),
        ("회수≥비용 (경제로 정당화)", lambda v: sum(1 for x in v if x["save"] >= x["cost"]) / len(v)),
        ("x̂ 하한 중앙값", lambda v: St.median(x["x"] for x in v)),
        ("x̂ 하한 평균", lambda v: St.mean(x["x"] for x in v))]
for name, f in rows:
    line = f"{name:<26}"
    for k in K:
        val = f(XHAT[k])
        line += f"{val:>9.3f}" if isinstance(val, float) else f"{val:>9}"
    print(line)
print()
# 판당 합계: 학습에 태운 AP 중 금전으로 못 돌아온 몫 = 세계가 지불한 암묵 효용
print("=== 판당 순지출 (학습비 − 금전회수) = 드러난 암묵 효용 총량 ===")
for k in K:
    tot_c = sum(x["cost"] for x in XHAT[k]) / 5
    tot_s = sum(x["save"] for x in XHAT[k]) / 5
    print(f"  {k:<7} 비용 {tot_c:6.2f} − 회수 {tot_s:6.2f} = {tot_c-tot_s:6.2f} AP/판"
          f"  (한 해 AP의 {(tot_c-tot_s)/1.0*100/9/30*100:.1f}bp… = 세계 노동력의 {(tot_c-tot_s)/270*100:.1f}%)")
