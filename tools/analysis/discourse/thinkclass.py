"""think 분류 — 판당 150개 층화 표본(해 균등)에 「결정 동인」 라벨 + 혼란 플래그."""
import json, pathlib, re, sys, time, urllib.request, threading, queue, collections
sys.path.insert(0, "/Users/supergene/Documents/eddie/Personal/park-world-sim")
from core import llm
KEY = llm.load_key()
ROOT = pathlib.Path("/Users/supergene/Documents/eddie/Personal/park-world-sim")
HERE = pathlib.Path(__file__).parent
OUT = HERE / "think_labels.jsonl"

TAX = """arith: 도달 가능성 산수 — 임계·페이스·남은 해 계산이 판단을 이끈다
budget: 행동력 배분 — 값·잔여 AP 계산이 판단을 이끈다
strat_host: 전략 — 어디(어느 나라)에 모을지, 시설을 바꿀지의 판단
social: 타인 해석 — 상대의 말·의도를 믿을지, 누가 했는지 추론
lang: 언어·경로 — 무슨 말로 쓸지, 배울지, 어느 경로로 보낼지
rule: 규칙 해석 — 세계 규칙·표결·정산이 어떻게 되는 건지 궁리
legacy: 전승 — 유언·기억·후손에게 남길 것
routine: 절차 — 특별한 판단 없이 할 일을 나열하고 실행"""
SYS = ("Each numbered item is an agent's private reasoning from a simulation. "
       "Label the PRIMARY driver of its decision with ONE code, and set conf=1 if the "
       "agent shows confusion (misreading the year, rules, or state), else 0. "
       'Output ONLY JSON like [{"i":1,"c":"arith","conf":0},...]\n\n' + TAX)

KN = {"noai": "no-ai", "ai006": "0.06", "ai012": "0.12", "ai024": "0.24"}
items = []
for p in sorted(ROOT.joinpath("runs").iterdir()):
    if not p.is_dir(): continue
    if not (p.name.startswith("260827-0") or p.name.startswith("260828-0")): continue
    if "dark" in p.name or "dawn" in p.name: continue
    if p.name.endswith(("003-ai012", "004-ai024", "002-ai006")): continue
    q = p / "raw_calls.jsonl"
    if not q.exists(): continue
    knob = KN[p.name.split("-")[-1]]
    per_run = []
    with q.open() as f:
        for line in f:
            if '"reasoning"' not in line: continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get("kind") != "agent": continue
            th = (((r.get("response") or {}).get("choices") or [{}])[0].get("message", {}).get("reasoning") or "").strip()
            if len(th) < 200: continue
            per_run.append(dict(run=p.name, knob=knob, turn=r.get("turn"),
                                agent=r.get("agent"), text=th[:600]))
    # 해 균등 층화 — 앞뒤로 치우치지 않게
    per_run.sort(key=lambda x: (x["turn"], x["agent"]))
    step = max(1, len(per_run)//150)
    items += per_run[::step][:150]
print(f"표본 {len(items)}개", flush=True)

done = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        try:
            r=json.loads(l)
            if r.get("c")!="unk": done.add(r["gid"])
        except Exception: pass

B = 5
LOCK = threading.Lock(); Q = queue.Queue()
for s0 in range(0, len(items), B):
    if any(gid not in done for gid in range(s0, min(s0+B, len(items)))): Q.put(s0)
TOT = Q.qsize(); print(f"배치 {TOT}", flush=True)

def one(fout, s0):
    batch = [(gid, items[gid]) for gid in range(s0, min(s0+B, len(items))) if gid not in done]
    if not batch: return
    listing = "\n\n".join(f"[{k+1}] {m['text']}" for k, (gid, m) in enumerate(batch))
    body = {"model": "qwen/qwen3.6-35b-a3b", "temperature": 0,
            "reasoning": {"enabled": False}, "max_tokens": 600,
            "provider": {"order": ["DeepInfra","Venice","AkashML","Parasail","AtlasCloud","Io Net"],
                         "allow_fallbacks": False},
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": listing}]}
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=150).read())
            arr = json.loads(re.search(r"\[.*\]", d["choices"][0]["message"]["content"], re.S).group(0))
            lab = {int(x["i"]): x for x in arr}
            with LOCK:
                for k, (gid, m) in enumerate(batch):
                    x = lab.get(k+1, {})
                    fout.write(json.dumps({"gid": gid, "run": m["run"], "knob": m["knob"],
                        "turn": m["turn"], "agent": m["agent"],
                        "c": str(x.get("c", "unk")), "conf": int(x.get("conf", 0) or 0)},
                        ensure_ascii=False) + "\n")
                fout.flush()
            return
        except Exception:
            time.sleep(3 * (attempt + 1))
    with LOCK:
        for gid, m in batch:
            fout.write(json.dumps({"gid": gid, "run": m["run"], "knob": m["knob"],
                "turn": m["turn"], "agent": m["agent"], "c": "unk", "conf": 0}) + "\n")
        fout.flush()

def work(fout):
    while True:
        try: s0 = Q.get_nowait()
        except queue.Empty: return
        one(fout, s0)
        if Q.qsize() % 40 == 0: print(f"  남은 {Q.qsize()}/{TOT}", flush=True)

with OUT.open("a") as fout:
    ts = [threading.Thread(target=work, args=(fout,)) for _ in range(8)]
    for t in ts: t.start()
    for t in ts: t.join()
print("완료", flush=True)
