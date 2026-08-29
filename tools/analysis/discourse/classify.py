"""대화 전수 분류 — 20판(토박이) 모든 메시지에 장르 하나씩. 워커 8 병렬."""
import json, pathlib, re, sys, time, urllib.request, threading, queue
sys.path.insert(0, "/Users/supergene/Documents/eddie/Personal/park-world-sim")
from core import llm
KEY = llm.load_key()
ROOT = pathlib.Path("/Users/supergene/Documents/eddie/Personal/park-world-sim")
OUT = pathlib.Path(__file__).parent / "msg_labels.jsonl"

CATS = """info: 사실 전달 — 관측치·진척·자기 상태 보고
ask: 요청·제안 — 특정 행동을 요구하거나 전략을 제안 (투자해달라, 한 나라에 모으자)
promise: 약속·보증 — 자기/자국의 미래 행동을 확약 (바꾸지 않겠다, 계속하겠다)
verify: 확인 요청 — 상대의 말·상태의 진위를 물음 (정말인가, 확인해달라)
suspect: 의심·추궁 — 비난, 사보타주/배신 의심, 해명 요구
apology: 사과·해명 — 사과하거나 오해를 풀려는 발화
vote: 표결 정치 — 採決 관련 설득·표 계획·투표 요청
teach: 지식 전수 — 세계 규칙·역사·요령을 가르침 (신인 교육, 유언 인용)
social: 인사·유대 — 인사, 감사, 환영, 격려 등 실질 내용 없는 관계 발화
other: 어디에도 안 맞음"""
SYS = ("You label messages from a simulation. For each numbered message, choose exactly ONE "
       "category code from the list below (the primary intent of the message). "
       "Output ONLY a JSON array like [{\"i\":1,\"c\":\"info\"},...]. No other text.\n\n" + CATS)

def nat(a): return "".join(c for c in a if not c.isdigit())
KN = {"noai": "no-ai", "ai006": "0.06", "ai012": "0.12", "ai024": "0.24"}
msgs = []
for p in sorted(ROOT.joinpath("runs").iterdir()):
    if not p.is_dir(): continue
    if not (p.name.startswith("260827-0") or p.name.startswith("260828-0")): continue
    if "dark" in p.name or "dawn" in p.name: continue
    if p.name.endswith(("003-ai012", "004-ai024", "002-ai006")): continue
    if not (p/"messages.jsonl").exists(): continue
    knob = KN[p.name.split("-")[-1]]
    for m in (json.loads(l) for l in (p/"messages.jsonl").read_text().splitlines() if l.strip()):
        t = (m["meta"].get("text_written") or "").strip()
        if not t: continue
        msgs.append(dict(run=p.name, knob=knob, turn=m["turn"], frm=m["from"], to=m["to"],
                         intl=nat(m["from"]) != nat(m["to"]), route=m["route"], text=t[:200]))
print(f"메시지 {len(msgs)}통", flush=True)

done = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        try: done.add(json.loads(l)["id"])
        except Exception: pass
print(f"이미 라벨 {len(done)}통 — 건너뜀", flush=True)

B = 25
LOCK = threading.Lock()
Q = queue.Queue()
for s0 in range(0, len(msgs), B):
    if any(gid not in done for gid in range(s0, min(s0+B, len(msgs)))):
        Q.put(s0)
TOT = Q.qsize()
print(f"배치 {TOT}개", flush=True)

def one(fout, s0):
    batch = [(gid, msgs[gid]) for gid in range(s0, min(s0+B, len(msgs))) if gid not in done]
    if not batch: return
    listing = "\n".join(f"[{k+1}] ({'국제' if m['intl'] else '국내'}) {m['text']}"
                        for k, (gid, m) in enumerate(batch))
    body = {"model": "qwen/qwen3.6-35b-a3b", "temperature": 0,
            "reasoning": {"enabled": False}, "max_tokens": 1200,
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
            lab = {int(x["i"]): str(x["c"]) for x in arr}
            with LOCK:
                for k, (gid, m) in enumerate(batch):
                    fout.write(json.dumps({"id": gid, **m, "cat": lab.get(k+1, "unk")},
                                          ensure_ascii=False) + "\n")
                fout.flush()
            return
        except Exception:
            time.sleep(3 * (attempt + 1))
    with LOCK:
        for gid, m in batch:
            fout.write(json.dumps({"id": gid, **m, "cat": "unk"}, ensure_ascii=False) + "\n")
        fout.flush()

def work(fout):
    n = 0
    while True:
        try: s0 = Q.get_nowait()
        except queue.Empty: return
        one(fout, s0); n += 1
        if Q.qsize() % 50 == 0:
            print(f"  남은 배치 {Q.qsize()}/{TOT}", flush=True)

with OUT.open("a") as fout:
    ts = [threading.Thread(target=work, args=(fout,)) for _ in range(8)]
    for t in ts: t.start()
    for t in ts: t.join()
print("완료", flush=True)
