"""학습 직전 think 전수 추출 (0.06 + no-ai) — 동기 키워드 스캔."""
import json, pathlib, collections, re
KN = {"noai":"no-ai","ai006":"0.06","ai012":"0.12","ai024":"0.24"}
OUT = collections.defaultdict(list)
for p in sorted(pathlib.Path("runs").iterdir()):
    if not p.is_dir() or not (p.name.startswith("260827-0") or p.name.startswith("260828-0")): continue
    if "dark" in p.name or "dawn" in p.name or p.name.endswith(("003-ai012","004-ai024","002-ai006")): continue
    if not (p/"metrics.jsonl").exists(): continue
    mt=[r for r in [json.loads(l) for l in (p/"metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if r.get("step") is None]
    if not mt or mt[-1]["turn"]<30: continue
    k = KN[p.name.split("-")[-1]]
    if k in ("0.12","0.24"): continue
    for l in (p/"events.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(l)
        if e.get("type")!="agent_turn": continue
        rs = e.get("reasonings",[])
        for i,r in enumerate(rs):
            if r.get("tool")=="learn" and i>0 and rs[i-1].get("source")=="thinking":
                OUT[k].append(dict(run=p.name, agent=e["agent"], turn=e["turn"],
                                   think=rs[i-1]["reasoning"]))
KW = {
 "잔돈/소진":  r"余|残り|使い切|剩|花完|用完|rest[ae]|épuiser",
 "읽기/수신":  r"読め|読む|理解でき|看懂|读懂|能懂|comprendre|lire",
 "직접/원문":  r"直接|原文|生の|直訳なし|directement",
 "AI대비":    r"AI|翻訳機|翻译|traduct",
 "신뢰/관계":  r"信頼|信任|関係|友好|confiance|relation",
 "비용계산":   r"0\.1|コスト|費用|成本|coût",
 "완성임박":   r"あと1回|もう一度|80|進捗|进度|complét",
}
import sys
for k in ("0.06","no-ai"):
    print(f"── {k}: 학습 직전 think {len(OUT[k])}건, 키워드 히트율 %")
    for name,pat in KW.items():
        n = sum(1 for x in OUT[k] if re.search(pat, x["think"]))
        print(f"   {name:<8} {100*n/len(OUT[k]):5.1f}")
json.dump(dict(OUT), open(pathlib.Path(__file__).parent/"learn_thinks.json","w",encoding="utf-8"), ensure_ascii=False)
