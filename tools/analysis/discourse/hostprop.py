"""host_prop 드릴 — 「어느 나라에 모으자」 고 했는지 추출.

이름 매칭으로 안 된다 — 「우리나라에」 「너희 쪽에」 「제일 앞선 곳에」 같은 지시가
많다. LLM 에게 발신자 국적을 주고 제안 대상국을 뽑게 한다.
  Asla / Ranoa / Miris : 특정 나라 지목
  leader               : 「지금 제일 앞선 나라」 식 — 이름 없이 선두 지목
  none                 : 대상 불특정 (그냥 「한 곳에 모으자」)
"""
import json, pathlib, re, sys, time, urllib.request, threading, queue
sys.path.insert(0, "/Users/supergene/Documents/eddie/Personal/park-world-sim")
from core import llm
KEY = llm.load_key()
HERE = pathlib.Path(__file__).parent
OUT = HERE / "hostprop_targets.jsonl"

SYS = ("Each numbered message proposes where to concentrate investment (which nation should "
       "host the interceptor). The sender's own nation is given in parentheses. Decide which "
       "nation the message proposes as the host:\n"
       "Asla | Ranoa | Miris : a specific nation (resolve '우리나라/notre pays/我们国' "
       "using the sender's nation)\n"
       "leader : proposes 'whichever nation is furthest ahead' without naming one\n"
       "none : no specific target (just 'we should concentrate somewhere')\n"
       'Output ONLY a JSON array like [{"i":1,"t":"Ranoa"},...]')

lab = {}
for l in (HERE/"msg_labels.jsonl").read_text().splitlines():
    r = json.loads(l); lab[r["id"]] = r
sub = {}
for l in (HERE/"msg_sublabels.jsonl").read_text().splitlines():
    r = json.loads(l); sub[r["id"]] = r["sub"]
todo = [r for i, r in lab.items() if sub.get(i) == "host_prop"]
print(f"host_prop {len(todo)}통", flush=True)

done = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        try: done.add(json.loads(l)["id"])
        except Exception: pass
todo = [r for r in todo if r["id"] not in done]

def nat(a): return "".join(c for c in a if not c.isdigit())
B = 25
LOCK = threading.Lock(); Q = queue.Queue()
for s0 in range(0, len(todo), B): Q.put(s0)
TOT = Q.qsize(); print(f"배치 {TOT}", flush=True)

def one(fout, s0):
    batch = todo[s0:s0+B]
    if not batch: return
    listing = "\n".join(f"[{k+1}] (sender: {nat(r['frm'])}) {r['text']}" for k, r in enumerate(batch))
    body = {"model": "qwen/qwen3.6-35b-a3b", "temperature": 0,
            "reasoning": {"enabled": False}, "max_tokens": 1000,
            "provider": {"order": ["DeepInfra","Venice","AkashML","Parasail","AtlasCloud","Io Net"],
                         "allow_fallbacks": False},
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": listing}]}
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=120).read())
            arr = json.loads(re.search(r"\[.*\]", d["choices"][0]["message"]["content"], re.S).group(0))
            m = {int(x["i"]): str(x["t"]) for x in arr}
            with LOCK:
                for k, r in enumerate(batch):
                    fout.write(json.dumps({"id": r["id"], "t": m.get(k+1, "unk")}) + "\n")
                fout.flush()
            return
        except Exception:
            time.sleep(3 * (attempt + 1))
    with LOCK:
        for r in batch: fout.write(json.dumps({"id": r["id"], "t": "unk"}) + "\n")
        fout.flush()

def work(fout):
    while True:
        try: s0 = Q.get_nowait()
        except queue.Empty: return
        one(fout, s0)
        if Q.qsize() % 30 == 0: print(f"  남은 {Q.qsize()}/{TOT}", flush=True)

with OUT.open("a") as fout:
    ts = [threading.Thread(target=work, args=(fout,)) for _ in range(8)]
    for t in ts: t.start()
    for t in ts: t.join()
print("완료", flush=True)
