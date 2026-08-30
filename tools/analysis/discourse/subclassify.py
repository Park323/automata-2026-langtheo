"""2차 분류 — ask 와 info 를 하위 장르로 쪼갠다. 국내·국제 전부."""
import json, pathlib, re, sys, time, urllib.request, threading, queue
sys.path.insert(0, "/Users/supergene/Documents/eddie/Personal/park-world-sim")
from core import llm
KEY = llm.load_key()
HERE = pathlib.Path(__file__).parent
SRC = HERE / "msg_labels.jsonl"
OUT = HERE / "msg_sublabels.jsonl"

TAX = {
"ask": """invest_req: 특정 시설·나라에 투자해 달라는 요청
host_prop: 어느 나라에 집중할지(숙주) 제안·설득
info_req: 정보를 알려달라는 요청 (관측치·진척·의향을 물음)
deal: 조건부 교환·협상 (~하면 ~하겠다, 대가를 걸음)
learn_urge: 언어를 배우라는 권유
switch_pers: 시설 전환(벙커/요격기) 설득
coord_misc: 그 밖의 행동 조율 (관측 분담, 국가투자 권유 등)
other: 어디에도 안 맞음""",
"info": """obs_share: 자기 관측치 공유 (남은 해·임계 추정)
prog_report: 진척·투자 활동 보고 (내가/우리가 얼마 했다)
status: 자기·자국 상태 전달 (탄생·죽음·언어·계획)
correction: 남의 틀린 정보를 바로잡음 (그 값이 아니라 이것이다)
relay: 제3자에게 들은 것을 전달 (아무개가 이렇게 말했다)
other: 어디에도 안 맞음""",
}
SYSFMT = ("You label messages from a simulation. All messages were already classified as "
          "'{parent}'. For each numbered message choose exactly ONE finer category code. "
          "Output ONLY a JSON array like [{{\"i\":1,\"c\":\"code\"}},...].\n\n{tax}")

rows = {}
for l in SRC.read_text(encoding="utf-8").splitlines():
    r = json.loads(l); rows[r["id"]] = r
todo = [r for r in rows.values() if r["cat"] in ("ask", "info")]
print(f"대상 {len(todo)}통 (ask {sum(1 for r in todo if r['cat']=='ask')} · info {sum(1 for r in todo if r['cat']=='info')})", flush=True)

done = set()
if OUT.exists():
    for l in OUT.read_text(encoding="utf-8").splitlines():
        try: done.add(json.loads(l)["id"])
        except Exception: pass
todo = [r for r in todo if r["id"] not in done]

B = 25
LOCK = threading.Lock()
Q = queue.Queue()
groups = {"ask": [r for r in todo if r["cat"]=="ask"], "info": [r for r in todo if r["cat"]=="info"]}
for parent, items in groups.items():
    for s0 in range(0, len(items), B): Q.put((parent, s0))
TOT = Q.qsize(); print(f"배치 {TOT}", flush=True)

def one(fout, parent, s0):
    batch = groups[parent][s0:s0+B]
    if not batch: return
    listing = "\n".join(f"[{k+1}] {r['text']}" for k, r in enumerate(batch))
    body = {"model": "qwen/qwen3.6-35b-a3b", "temperature": 0,
            "reasoning": {"enabled": False}, "max_tokens": 1200,
            "provider": {"order": ["DeepInfra","Venice","AkashML","Parasail","AtlasCloud","Io Net"],
                         "allow_fallbacks": False},
            "messages": [{"role": "system", "content": SYSFMT.format(parent=parent, tax=TAX[parent])},
                         {"role": "user", "content": listing}]}
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=120).read())
            arr = json.loads(re.search(r"\[.*\]", d["choices"][0]["message"]["content"], re.S).group(0))
            lab = {int(x["i"]): str(x["c"]) for x in arr}
            with LOCK:
                for k, r in enumerate(batch):
                    fout.write(json.dumps({"id": r["id"], "sub": lab.get(k+1, "unk")},
                                          ensure_ascii=False) + "\n")
                fout.flush()
            return
        except Exception:
            time.sleep(3 * (attempt + 1))
    with LOCK:
        for r in batch:
            fout.write(json.dumps({"id": r["id"], "sub": "unk"}) + "\n")
        fout.flush()

def work(fout):
    while True:
        try: parent, s0 = Q.get_nowait()
        except queue.Empty: return
        one(fout, parent, s0)
        if Q.qsize() % 50 == 0: print(f"  남은 {Q.qsize()}/{TOT}", flush=True)

with OUT.open("a", encoding="utf-8") as fout:
    ts = [threading.Thread(target=work, args=(fout,)) for _ in range(8)]
    for t in ts: t.start()
    for t in ts: t.join()
print("완료", flush=True)
