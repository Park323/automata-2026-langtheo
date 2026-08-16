"""의도 전달 판정 — 지표 4a/4b/4c/4d. spec 6.2 · 12.1.

**2단계입니다.** 메시지에 귀속된 `understood` 를 받는 도구를 폐기했기 때문입니다
(모델이 한 번도 부르지 않았습니다 — spec 12.1). 대신 수신자가 **다음 턴에 실제로 한
행동의 `reasoning`** 에서 이해를 역추적합니다.

    ① 수신자의 그 턴 reasoning 에서 이 메시지에 대한 이해를 추출할 수 있는가
    ② 추출된 이해가 text_sent 와 같은 뜻인가

    분모는 ①이 가능했던 메시지만. ①이 안 된 것은 버리지 않고 **언급률**로 따로 냅니다.

언급률은 결측이 아니라 **관측**입니다. AI 가 싸져서 메시지가 쏟아지면 언급률이 떨어질
텐데, 그게 곧 *"많이 받지만 덜 읽는다"* 라는 발견입니다.

> ⚠️ **선택 효과가 남습니다.** 실패율의 분모가 *언급된 메시지* 라서, 잘 이해한 것일수록
> 근거에 적힐 가능성이 높다면 실패율이 낮게 나옵니다. 6턴 실측에서 언급률이 경로별로
> 달랐습니다(AI 47% · 국내 35%). 언급률을 **4a 와 나란히 반드시 함께 보고**해야 하고,
> 경로 간 언급률 차이가 크면 4a 비교를 그만큼 깎아서 읽어야 합니다.

### 판정자가 무엇을 보는가 — 두 단계가 보는 것이 다릅니다

    ①  text_delivered  +  reasoning      수신자가 실제로 본 것과 실제로 쓴 것
    ②  text_sent       +  understood     발신자가 보낸 것과 수신자가 가져간 것

②에 `text_delivered` 를 절대 주지 않습니다. 주면 판정자가 *"번역이 잘 됐는가"* 를
재게 되는데, 그건 다른 질문입니다. 우리가 재는 것은 **원문의 의도가 사람에게 도달했는가**
입니다 — 번역이 완벽해도 수신자가 딴 뜻으로 가져갔으면 실패입니다.

### 판정자가 지어내는 것을 막는 장치

①에서 `evidence` 를 **reasoning 원문에서 그대로 복사**하게 하고, 코드가 실제 부분
문자열인지 확인합니다. 없으면 그 판정은 버립니다(`evidence_missing`). 판정자에게
메시지 본문을 함께 보여주므로, 아무 근거 없이 메시지를 그대로 베껴 *"이해했다"* 고
답할 위험이 실재합니다. 이 검사가 그걸 막습니다.

### 실행

    python3 tools/score/judge.py runs/<run_id>                # 판정 (이어서 하기)
    python3 tools/score/judge.py runs/<run_id> --limit 20     # 표본만
    python3 tools/score/judge.py runs/<run_id> --report       # 호출 없이 집계만

산출은 `runs/<run_id>/judged.jsonl` 이고 LLM 호출 원본은 `judge_raw.jsonl` 에
전부 남습니다 (spec 9장 — 파생 로그는 raw 에서 재생성 가능해야 합니다).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core import config, llm  # noqa: E402

LANG_NAME = {"ja": "Japanese", "zh": "Chinese", "fr": "French"}

# 판정 대상 밖으로 밀려나는 이유들. 버리지 않고 전부 센다 —
# 어떤 이유로 몇 건이 빠졌는지 모르면 4a 의 n 을 믿을 수 없다.
SKIP_UNREADABLE = "unreadable"        # 못 읽었다 (지표 9 의 몫)
SKIP_NO_TURN = "no_receiver_turn"     # 수신자가 다음 턴에 없다 (사망·마지막 턴)
SKIP_NO_REASONING = "no_reasoning"    # 턴은 있었으나 근거를 한 줄도 안 썼다
# 번역 호출이 실패했다. **엔진 장애지 세계의 사건이 아니다** — 못 읽어서 못 받은 것과
# 섞으면 지표 9 가 부풀고, 조건별 빈도가 다르면 4a 도 오염된다.
SKIP_TRANSLATE_FAILED = "translate_failed"


# ── 링크 ────────────────────────────────────────────────────────────────────────

def link(messages: list[dict], events: list[dict]) -> list[dict]:
    """메시지 ↔ 수신자의 **다음 턴** reasoning 을 잇는다.

    턴 t 에 보낸 메시지는 t+1 에 인박스로 도착한다 (`deliver_turn = turn + 1`).
    따라서 그 메시지를 읽고 한 행동의 근거는 t+1 의 `agent_turn` 에 있다.
    """
    by_agent_turn: dict[tuple[int, str], dict] = {}
    for e in events:
        if e.get("type") == "agent_turn":
            by_agent_turn[(e["turn"], e["agent"])] = e

    out = []
    for m in messages:
        meta = m.get("meta") or {}
        rec = {
            "msg_id": m.get("msg_id"), "turn": m.get("turn"),
            "from": m.get("from"), "to": m.get("to"), "route": m.get("route"),
            "src_lang": meta.get("src_lang"), "dst_lang": meta.get("dst_lang"),
            "text_sent": meta.get("text_sent"),
            "text_delivered": meta.get("text_delivered"),
            # `reader` 는 "수신자가 발신 언어를 읽을 수 있는가" 다 — 전달 여부가 아니다.
            # **원문 병기는 폐지됐다** (5.1 개정): ai 를 고른 순간 원문은 볼 수 없다.
            # 그래서 판정자도 번역문만 본다. reader 는 남겨둔다 — "읽을 수 있었는데도
            # ai 로 받았다" 를 사후에 가르는 데 쓴다.
            "saw_original": False,
            "reasonings": [], "skip": None,
        }
        # 못 읽은 것은 route=original 에서 delivered=False 로 이미 표시된다 (지표 9 의 몫).
        if meta.get("translate_failed"):
            rec["skip"] = SKIP_TRANSLATE_FAILED
            out.append(rec)
            continue
        if not m.get("delivered") or not meta.get("text_delivered"):
            rec["skip"] = SKIP_UNREADABLE
            out.append(rec)
            continue
        ev = by_agent_turn.get((m["turn"] + 1, m["to"]))
        if ev is None:
            rec["skip"] = SKIP_NO_TURN
            out.append(rec)
            continue
        texts = [str(r.get("reasoning") or "").strip()
                 for r in (ev.get("reasonings") or [])]
        texts = [t for t in texts if t]
        if not texts:
            rec["skip"] = SKIP_NO_REASONING
            out.append(rec)
            continue
        rec["reasonings"] = texts
        rec["actions"] = ev.get("actions")
        out.append(rec)
    return out


# ── 판정 ────────────────────────────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(content: str) -> dict | None:
    """모델 출력에서 JSON 하나를 건진다. 코드펜스·앞뒤 잡담을 견딘다."""
    if not content:
        return None
    for candidate in (content, *(m.group(1) for m in _FENCE.finditer(content))):
        s = candidate.strip()
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try:
                v = json.loads(s[i:j + 1])
                if isinstance(v, dict):
                    return v
            except json.JSONDecodeError:
                continue
    return None


def _norm(s: str) -> str:
    """공백·따옴표만 지우고 비교한다. evidence 검사가 서식 차이로 실패하면 안 된다."""
    return re.sub(r"[\s\"'`「」『』«»]", "", s or "").lower()


STAGE1_SYSTEM = (
    "You are a strict annotator. You answer only with a single JSON object. "
    "You never infer beyond the text you are given."
)

STAGE2_SYSTEM = (
    "You are a strict annotator comparing meaning across languages. "
    "You answer only with a single JSON object."
)


def stage1_prompt(rec: dict) -> str:
    who = rec["to"]
    lines = "\n".join(f"{i}. {t}" for i, t in enumerate(rec["reasonings"], 1))
    body = rec["text_delivered"]      # 수신자가 본 것 전부. 원문 병기는 폐지됐다 (5.1)
    return (
        f"Agent {who} received this message. This is exactly what {who} saw "
        f"(in {LANG_NAME.get(rec['dst_lang'], rec['dst_lang'])}):\n\n"
        f"<message>\n{body}\n</message>\n\n"
        f"On the following turn, {who} wrote these justifications for their own actions, "
        f"verbatim:\n\n<reasoning>\n{lines}\n</reasoning>\n\n"
        f"Question: does the reasoning show how {who} understood THAT message?\n\n"
        "Rules:\n"
        "- Judge from the reasoning ALONE. Never use the message to fill in what the "
        "reasoning does not say.\n"
        "- If the reasoning does not engage with this message's content, answer "
        "mentioned=false.\n"
        "- \"evidence\" must be copied word for word from the reasoning, not from the "
        "message. Empty string if mentioned=false.\n"
        "- \"understood\" states in English what the agent took the message to mean. "
        "One or two sentences. Cover every part of the message the reasoning engages "
        "with — do not compress it further than the reasoning does. "
        "Empty string if mentioned=false.\n\n"
        'Answer with JSON only: {"mentioned": true or false, "evidence": "...", '
        '"understood": "..."}'
    )


def stage2_prompt(rec: dict, understood: str) -> str:
    """②는 **주된 뜻**만 묻습니다 — 완전성을 물으면 요약 손실이 오해로 잡힙니다.

    첫 판에서 실측한 오판정입니다.

        A "동의합니다. 방어 계획을 논의할 회의를 제안합시다"
        B "회의를 조직하자는 제안"          → 실패 ("방어 계획 누락")
        B "즉시 대피소를 세워야 한다"        → 실패 ("'즉시' 가 A 에 없다")

    B 는 근거에서 뽑은 요약이라 원문보다 짧은 것이 정상입니다. 이걸 실패로 세면
    **4c(번역 없는 기저선)가 80% 로 천장에 붙어** 번역 효과가 보일 자리가 없어집니다.
    """
    return (
        "Statement A is what a sender sent (in "
        f"{LANG_NAME.get(rec['src_lang'], rec['src_lang'])}).\n"
        "Statement B is what the recipient took away from it. B is a brief summary, so "
        "it is EXPECTED to be shorter, vaguer, and less detailed than A.\n\n"
        f"<A>\n{rec['text_sent']}\n</A>\n\n<B>\n{understood}\n</B>\n\n"
        "Did the recipient get A's main point?\n"
        "- true if B captures A's main point, even when B omits details, examples, "
        "secondary requests, or qualifiers.\n"
        "- true if B is vaguer or more general than A.\n"
        "- false if B contradicts or reverses A.\n"
        "- false if B is about a different subject, or misses A's main point entirely.\n"
        "- false if B asserts something substantive that A does not support.\n"
        "Ignore style, politeness, verbosity, and language differences.\n\n"
        'Answer with JSON only: {"same": true or false, "why": "at most 15 words"}'
    )


class Judge:
    """두 단계를 순서대로 돌린다. `client` 는 OpenRouterClient 또는 StubClient."""

    def __init__(self, client, temperature: float = 0.0):
        self.client = client
        self.temperature = temperature

    def _ask(self, system: str, user: str) -> tuple[dict | None, str]:
        resp = self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
        )
        content = (resp["choices"][0]["message"].get("content") or "").strip()
        return parse_json(content), content

    def stage1(self, rec: dict) -> dict:
        obj, raw = self._ask(STAGE1_SYSTEM, stage1_prompt(rec))
        if obj is None:
            return {"mentioned": False, "understood": "", "evidence": "",
                    "error": "unparsed", "raw": raw}
        mentioned = bool(obj.get("mentioned"))
        understood = str(obj.get("understood") or "").strip()
        evidence = str(obj.get("evidence") or "").strip()
        out = {"mentioned": mentioned, "understood": understood,
               "evidence": evidence, "error": None}
        if mentioned:
            if not understood:
                out.update(mentioned=False, error="empty_understood")
            elif _norm(evidence) not in _norm(" ".join(rec["reasonings"])):
                # 판정자가 근거를 지어냈다. 메시지 본문을 베꼈을 수 있으므로 버린다.
                out.update(mentioned=False, error="evidence_missing")
        return out

    def stage2(self, rec: dict, understood: str) -> dict:
        obj, raw = self._ask(STAGE2_SYSTEM, stage2_prompt(rec, understood))
        if obj is None:
            return {"same": None, "why": "", "error": "unparsed", "raw": raw}
        return {"same": bool(obj.get("same")), "why": str(obj.get("why") or "")[:120],
                "error": None}

    def judge(self, rec: dict) -> dict:
        """한 메시지의 판정. skip 이면 호출하지 않는다."""
        if rec.get("skip"):
            return {**rec, "mentioned": None, "same": None}
        s1 = self.stage1(rec)
        out = {**rec, **{f"s1_{k}": v for k, v in s1.items()},
               "mentioned": s1["mentioned"], "same": None}
        if not s1["mentioned"]:
            return out
        s2 = self.stage2(rec, s1["understood"])
        out.update({f"s2_{k}": v for k, v in s2.items()}, same=s2["same"])
        return out


# ── 집계 ────────────────────────────────────────────────────────────────────────

def is_intl(rec: dict) -> bool:
    """국적이 다른가. 국가 접두는 에이전트 id 에서 숫자를 뗀 것이다 (Ranoa2 → Ranoa)."""
    a = re.sub(r"\d+$", "", rec.get("from") or "")
    b = re.sub(r"\d+$", "", rec.get("to") or "")
    return bool(a and b and a != b)


def layer(rec: dict) -> str | None:
    """지표 4 의 어느 칸인가. spec 8.2 의 삼중 대조."""
    r = rec.get("route")
    if r == "ai":
        return "4a"
    if r == "domestic":
        return "4c"
    if r == "original" and is_intl(rec):
        return "4d"       # 원문 직통인데 상대가 외국인 — 번역 없이 국적만 다른 조건
    return None


def aggregate(judged: list[dict]) -> dict:
    """지표 4a/4b/4c/4d + 언급률 + 제외 사유별 건수.

    실패율의 분모는 **`same` 이 결정된 메시지**뿐이다. 언급 안 됨은 분모 밖이고,
    그 자체가 언급률이라는 별도 지표가 된다.
    """
    cells = {"4a": [], "4c": [], "4d": [], "4b": []}
    mention = {"4a": [0, 0], "4c": [0, 0], "4d": [0, 0], "4b": [0, 0]}   # [언급, 후보]
    skips: dict[str, int] = {}
    unparsed = 0

    for r in judged:
        if r.get("skip"):
            skips[r["skip"]] = skips.get(r["skip"], 0) + 1
            continue
        cell = layer(r)
        keys = [k for k in (cell, "4b") if k]
        for k in keys:
            mention[k][1] += 1
        if r.get("s1_error") == "unparsed" or r.get("s2_error") == "unparsed":
            unparsed += 1
        if not r.get("mentioned"):
            continue
        for k in keys:
            mention[k][0] += 1
        if r.get("same") is None:
            continue
        for k in keys:
            cells[k].append(0 if r["same"] else 1)

    def rate(v: list[int]) -> dict:
        return {"n": len(v),
                "fail_rate": round(sum(v) / len(v), 4) if v else None}

    out = {k: rate(v) for k, v in cells.items()}
    out["mention_rate"] = {
        k: {"n": t, "rate": round(c / t, 4) if t else None}
        for k, (c, t) in mention.items()
    }
    out["skipped"] = skips
    out["unparsed"] = unparsed
    # ★ 핵심 수치. 기저선이 없으면 4a 는 해석 불가능한 절대값이다 (spec 8.3).
    if out["4a"]["fail_rate"] is not None and out["4c"]["fail_rate"] is not None:
        out["4a_minus_4c"] = round(out["4a"]["fail_rate"] - out["4c"]["fail_rate"], 4)
    else:
        out["4a_minus_4c"] = None
    return out


# ── 실행 ────────────────────────────────────────────────────────────────────────

def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def judge_run(run_dir: Path, client, limit: int | None = None,
              resume: bool = True, progress=None) -> list[dict]:
    """런 하나를 판정해 `judged.jsonl` 에 이어붙인다.

    이미 판정된 `msg_id` 는 건너뛴다 — 판정은 비싸고, 중간에 죽어도 거기까지는 남는다.
    """
    out_path = run_dir / "judged.jsonl"
    done = {r["msg_id"] for r in read_jsonl(out_path)} if resume else set()
    if not resume and out_path.exists():
        out_path.unlink()

    recs = link(read_jsonl(run_dir / "messages.jsonl"),
                read_jsonl(run_dir / "events.jsonl"))
    todo = [r for r in recs if r["msg_id"] not in done]
    if limit is not None:
        skips = [r for r in todo if r.get("skip")]
        live = [r for r in todo if not r.get("skip")][:limit]
        todo = skips + live

    judge = Judge(client)
    written = []
    with out_path.open("a", encoding="utf-8") as f:
        for i, rec in enumerate(todo, 1):
            res = judge.judge(rec)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            written.append(res)
            if progress:
                progress(i, len(todo), res)
    return read_jsonl(out_path)


def _fmt(agg: dict) -> str:
    L = []
    L.append(f"{'층':<6}{'이름':<22}{'n':>5}{'실패율':>9}{'언급률':>9}")
    L.append("─" * 51)
    names = {"4a": "AI 경로", "4c": "국내 (기저선)", "4d": "국제·원문직통",
             "4b": "전체"}
    for k in ("4a", "4c", "4d", "4b"):
        c, m = agg[k], agg["mention_rate"][k]
        fr = "—" if c["fail_rate"] is None else f"{c['fail_rate']:.0%}"
        mr = "—" if m["rate"] is None else f"{m['rate']:.0%} ({m['n']})"
        L.append(f"{k:<6}{names[k]:<22}{c['n']:>5}{fr:>9}{mr:>11}")
    L.append("")
    d = agg["4a_minus_4c"]
    L.append(f"4a − 4c  = {'—' if d is None else f'{d:+.1%}'}   ← 번역 고유의 오해 증분")
    if agg["skipped"]:
        L.append("\n판정 밖 " + " · ".join(f"{k} {v}" for k, v in sorted(agg["skipped"].items())))
    if agg["unparsed"]:
        L.append(f"⚠ 파싱 실패 {agg['unparsed']}건 — 판정 모델 출력 형식 확인")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="의도 전달 2단계 판정 (지표 4)")
    ap.add_argument("run_dir")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default=None, help="판정 모델 (기본: config 의 judge_model)")
    ap.add_argument("--limit", type=int, default=None, help="판정할 메시지 수 상한")
    ap.add_argument("--fresh", action="store_true", help="judged.jsonl 을 지우고 다시")
    ap.add_argument("--report", action="store_true", help="LLM 호출 없이 기존 결과만 집계")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    if not (run_dir / "messages.jsonl").exists():
        print(f"messages.jsonl 이 없습니다: {run_dir}", file=sys.stderr)
        return 2

    if a.report:
        judged = read_jsonl(run_dir / "judged.jsonl")
        if not judged:
            print("judged.jsonl 이 비어 있습니다. --report 없이 먼저 돌리세요.", file=sys.stderr)
            return 2
        print(_fmt(aggregate(judged)))
        return 0

    cfg = config.load(a.config)
    model = a.model or cfg.llm.judge_model or cfg.llm.translate_model

    raw_path = run_dir / "judge_raw.jsonl"
    raw_f = raw_path.open("a", encoding="utf-8")

    def recorder(rec: dict) -> None:
        raw_f.write(json.dumps({"kind": "judge", **rec}, ensure_ascii=False, default=str) + "\n")
        raw_f.flush()

    client = llm.OpenRouterClient(model, temperature=0.0, recorder=recorder)

    recs = link(read_jsonl(run_dir / "messages.jsonl"), read_jsonl(run_dir / "events.jsonl"))
    live = sum(1 for r in recs if not r.get("skip"))
    print(f"메시지 {len(recs)}건 · 판정 대상 {live}건 · 모델 {model}")
    t0 = time.time()

    def progress(i, n, res):
        if res.get("skip"):
            return
        mark = "·" if res.get("mentioned") is False else ("○" if res.get("same") else "×")
        print(f"  [{i}/{n}] {mark} msg {res['msg_id']} {res['route']:<9}"
              f"{res['src_lang']}→{res['dst_lang']}", flush=True)

    judged = judge_run(run_dir, client, limit=a.limit, resume=not a.fresh,
                       progress=progress)
    raw_f.close()
    print(f"\n{time.time() - t0:.1f}s · judged.jsonl {len(judged)}행 · raw {raw_path.name}\n")
    print(_fmt(aggregate(judged)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
