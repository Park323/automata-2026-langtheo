"""dark·dawn 의 16~30해를 같은 3층 + think 로 분류한다. 단계 연쇄 실행."""
import json, pathlib, re, sys, time, urllib.request, threading, queue
sys.path.insert(0, "/Users/supergene/Documents/eddie/Personal/park-world-sim")
from core import llm
KEY = llm.load_key()
ROOT = pathlib.Path("/Users/supergene/Documents/eddie/Personal/park-world-sim")
HERE = pathlib.Path(__file__).parent

CATS_MSG = """info: 사실 전달 — 관측치·진척·자기 상태 보고
ask: 요청·제안 — 특정 행동을 요구하거나 전략을 제안 (투자해달라, 한 나라에 모으자)
promise: 약속·보증 — 자기/자국의 미래 행동을 확약 (바꾸지 않겠다, 계속하겠다)
verify: 확인 요청 — 상대의 말·상태의 진위를 물음 (정말인가, 확인해달라)
suspect: 의심·추궁 — 비난, 사보타주/배신 의심, 해명 요구
apology: 사과·해명 — 사과하거나 오해를 풀려는 발화
vote: 표결 정치 — 採決 관련 설득·표 계획·투표 요청
teach: 지식 전수 — 세계 규칙·역사·요령을 가르침 (신인 교육, 유언 인용)
social: 인사·유대 — 인사, 감사, 환영, 격려 등 실질 내용 없는 관계 발화
other: 어디에도 안 맞음"""
TAX_SUB = {
"ask": """invest_req: 특정 시설·나라에 투자해 달라는 요청
host_prop: 어느 나라에 집중할지(숙주) 제안·설득
info_req: 정보를 알려달라는 요청 (관측치·진척·의향을 물음)
deal: 조건부 교환·협상 (~하면 ~하겠다)
learn_urge: 언어를 배우라는 권유
switch_pers: 시설 전환(벙커/요격기) 설득
coord_misc: 그 밖의 행동 조율
other: 어디에도 안 맞음""",
"info": """obs_share: 자기 관측치 공유
prog_report: 진척·투자 활동 보고
status: 자기·자국 상태 전달
correction: 남의 틀린 정보를 바로잡음
relay: 제3자에게 들은 것을 전달
other: 어디에도 안 맞음""",
}
TAX_THINK = """arith: 도달 가능성 산수
budget: 행동력 배분·가격 계산
strat_host: 어디에 모을지·시설을 바꿀지 전략
social: 타인의 말·의도 해석
lang: 언어·경로 선택 (무슨 말로, 배울지, ai/original)
rule: 세계 규칙·표결·정산 해석
legacy: 유언·기억·전승
routine: 특별한 판단 없는 절차"""

def nat(a): return "".join(c for c in a if not c.isdigit())
def call(sysmsg, listing, max_tokens=1100):
    body = {"model": "qwen/qwen3.6-35b-a3b", "temperature": 0,
            "reasoning": {"enabled": False}, "max_tokens": max_tokens,
            "provider": {"order": ["DeepInfra","Venice","AkashML","Parasail","AtlasCloud","Io Net"],
                         "allow_fallbacks": False},
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": listing}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=150).read())
    return json.loads(re.search(r"\[.*\]", d["choices"][0]["message"]["content"], re.S).group(0))

def runpool(tasks, fn, workers=8):
    LOCK = threading.Lock(); Q = queue.Queue()
    for t in tasks: Q.put(t)
    def work():
        while True:
            try: t = Q.get_nowait()
            except queue.Empty: return
            fn(t, LOCK)
            if Q.qsize() % 40 == 0: print(f"    남은 {Q.qsize()}", flush=True)
    ts = [threading.Thread(target=work) for _ in range(workers)]
    for t in ts: t.start()
    for t in ts: t.join()

RUNS = [f"260828-0{o+i}-{k}{i}" for k, o in (("dark", 23), ("dawn", 28)) for i in range(1, 6)]
GROUP = {r: ("dark" if "dark" in r else "dawn") for r in RUNS}

# ── A. 메시지 (16해~) ─────────────────────────────────────────────────────
msgs = []
for rid in RUNS:
    p = ROOT/"runs"/rid
    for m in (json.loads(l) for l in (p/"messages.jsonl").read_text().splitlines() if l.strip()):
        if m["turn"] < 16: continue
        t = (m["meta"].get("text_written") or "").strip()
        if not t: continue
        msgs.append(dict(run=rid, grp=GROUP[rid], turn=m["turn"], frm=m["from"], to=m["to"],
                         intl=nat(m["from"]) != nat(m["to"]), route=m["route"],
                         delivered=bool(m["delivered"]), text=t[:200]))
print(f"A. 메시지 {len(msgs)}통", flush=True)
OUTA = HERE/"dd_msg_labels.jsonl"
done = set()
if OUTA.exists():
    for l in OUTA.read_text().splitlines():
        try:
            r = json.loads(l)
            if r.get("cat") != "unk": done.add(r["id"])
        except Exception: pass
SYSA = ("You label messages from a simulation. For each numbered message, choose exactly ONE "
        "category code (primary intent). Output ONLY a JSON array like "
        '[{"i":1,"c":"info"},...]. No other text.\n\n' + CATS_MSG)
B = 25
def do_msg(s0, LOCK):
    batch = [(g, msgs[g]) for g in range(s0, min(s0+B, len(msgs))) if g not in done]
    if not batch: return
    listing = "\n".join(f"[{k+1}] ({'국제' if m['intl'] else '국내'}) {m['text']}"
                        for k, (g, m) in enumerate(batch))
    for attempt in range(4):
        try:
            lab = {int(x["i"]): str(x["c"]) for x in call(SYSA, listing)}
            with LOCK, OUTA.open("a") as f:
                for k, (g, m) in enumerate(batch):
                    f.write(json.dumps({"id": g, **m, "cat": lab.get(k+1, "unk")},
                                       ensure_ascii=False) + "\n")
            return
        except Exception:
            time.sleep(3*(attempt+1))
    with LOCK, OUTA.open("a") as f:
        for g, m in batch:
            f.write(json.dumps({"id": g, **m, "cat": "unk"}, ensure_ascii=False) + "\n")
runpool([s for s in range(0, len(msgs), B)], do_msg)
print("A 완료", flush=True)

lab = {}
for l in OUTA.read_text().splitlines():
    r = json.loads(l)
    if r["id"] not in lab or lab[r["id"]]["cat"] == "unk": lab[r["id"]] = r

# ── B. ask/info 하위 ─────────────────────────────────────────────────────
OUTB = HERE/"dd_msg_sublabels.jsonl"
doneb = set()
if OUTB.exists():
    for l in OUTB.read_text().splitlines():
        try:
            r = json.loads(l)
            if r.get("sub") != "unk": doneb.add(r["id"])
        except Exception: pass
for parent in ("ask", "info"):
    items = [r for r in lab.values() if r["cat"] == parent and r["id"] not in doneb]
    print(f"B. {parent} {len(items)}통", flush=True)
    SYSB = (f"All messages were classified as '{parent}'. Choose ONE finer code each. "
            'Output ONLY JSON like [{"i":1,"c":"code"}].\n\n' + TAX_SUB[parent])
    def do_sub(s0, LOCK, items=items, SYSB=SYSB):
        batch = items[s0:s0+B]
        if not batch: return
        listing = "\n".join(f"[{k+1}] {r['text']}" for k, r in enumerate(batch))
        for attempt in range(4):
            try:
                m = {int(x["i"]): str(x["c"]) for x in call(SYSB, listing)}
                with LOCK, OUTB.open("a") as f:
                    for k, r in enumerate(batch):
                        f.write(json.dumps({"id": r["id"], "sub": m.get(k+1, "unk")}) + "\n")
                return
            except Exception:
                time.sleep(3*(attempt+1))
        with LOCK, OUTB.open("a") as f:
            for r in batch: f.write(json.dumps({"id": r["id"], "sub": "unk"}) + "\n")
    runpool([s for s in range(0, len(items), B)], do_sub)
print("B 완료", flush=True)

sub = {}
for l in OUTB.read_text().splitlines():
    r = json.loads(l)
    if r["id"] not in sub or sub[r["id"]] == "unk": sub[r["id"]] = r["sub"]

# ── C. host_prop 대상 ─────────────────────────────────────────────────────
OUTC = HERE/"dd_hostprop.jsonl"
donec = set()
if OUTC.exists():
    for l in OUTC.read_text().splitlines():
        try: donec.add(json.loads(l)["id"])
        except Exception: pass
items = [lab[i] for i, s in sub.items() if s == "host_prop" and i not in donec]
print(f"C. host_prop {len(items)}통", flush=True)
SYSC = ("Each message proposes a host nation. Sender nation in parentheses. Answer one of "
        "Asla|Ranoa|Miris|leader|none per item. leader='whichever is ahead' without a name. "
        'Output ONLY JSON like [{"i":1,"t":"Ranoa"}].')
def do_host(s0, LOCK):
    batch = items[s0:s0+B]
    if not batch: return
    listing = "\n".join(f"[{k+1}] (sender: {nat(r['frm'])}) {r['text']}" for k, r in enumerate(batch))
    for attempt in range(4):
        try:
            m = {int(x["i"]): str(x["t"]) for x in call(SYSC, listing, 900)}
            with LOCK, OUTC.open("a") as f:
                for k, r in enumerate(batch):
                    f.write(json.dumps({"id": r["id"], "t": m.get(k+1, "unk")}) + "\n")
            return
        except Exception:
            time.sleep(3*(attempt+1))
    with LOCK, OUTC.open("a") as f:
        for r in batch: f.write(json.dumps({"id": r["id"], "t": "unk"}) + "\n")
runpool([s for s in range(0, len(items), B)], do_host)
print("C 완료", flush=True)

# ── D. think (16해~, 판당 75 표본) ────────────────────────────────────────
OUTD = HERE/"dd_think_labels.jsonl"
doned = set()
if OUTD.exists():
    for l in OUTD.read_text().splitlines():
        try:
            r = json.loads(l)
            if r.get("c") != "unk": doned.add(r["gid"])
        except Exception: pass
titems = []
for rid in RUNS:
    per = []
    with (ROOT/"runs"/rid/"raw_calls.jsonl").open() as f:
        for line in f:
            if '"reasoning"' not in line: continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get("kind") != "agent" or r.get("turn", 0) < 16: continue
            th = (((r.get("response") or {}).get("choices") or [{}])[0].get("message", {}).get("reasoning") or "").strip()
            if len(th) < 200: continue
            per.append(dict(run=rid, grp=GROUP[rid], turn=r["turn"], agent=r.get("agent"), text=th[:600]))
    per.sort(key=lambda x: (x["turn"], x["agent"]))
    step = max(1, len(per)//75)
    titems += per[::step][:75]
print(f"D. think 표본 {len(titems)}", flush=True)
SYSD = ("Label each agent-reasoning item: ONE code for the PRIMARY decision driver, and conf=1 "
        "if the agent shows confusion about year/rules/state else 0. "
        'Output ONLY JSON like [{"i":1,"c":"arith","conf":0}].\n\n' + TAX_THINK)
TB = 5
def do_think(s0, LOCK):
    batch = [(g, titems[g]) for g in range(s0, min(s0+TB, len(titems))) if g not in doned]
    if not batch: return
    listing = "\n\n".join(f"[{k+1}] {m['text']}" for k, (g, m) in enumerate(batch))
    for attempt in range(4):
        try:
            arr = call(SYSD, listing, 600)
            m2 = {int(x["i"]): x for x in arr}
            with LOCK, OUTD.open("a") as f:
                for k, (g, m) in enumerate(batch):
                    x = m2.get(k+1, {})
                    f.write(json.dumps({"gid": g, "run": m["run"], "grp": m["grp"],
                        "turn": m["turn"], "c": str(x.get("c", "unk")),
                        "conf": int(x.get("conf", 0) or 0)}) + "\n")
            return
        except Exception:
            time.sleep(3*(attempt+1))
    with LOCK, OUTD.open("a") as f:
        for g, m in batch:
            f.write(json.dumps({"gid": g, "run": m["run"], "grp": m["grp"],
                                "turn": m["turn"], "c": "unk", "conf": 0}) + "\n")
runpool([s for s in range(0, len(titems), TB)], do_think)
print("전체 완료", flush=True)
