#!/usr/bin/env python3
"""에이전트에게 직접 물어본다. 세계를 건드리지 않고.

    tools/interview.py --run t20a --list
    tools/interview.py --run t20a --agent Ranoa1 --ask "隕石まで何年だと思いますか"
    tools/interview.py --run t20a --agent Ranoa1 --turn 12 --ask "..."   # 그 해의 그 사람

## 무엇을 재는 도구인가

**세계의 사실이 아니라 그 사람의 믿음을 재는 도구다.** 답이 틀려도 그것이 자료다 —
「왜곡을 알아채지 못한다」 는 가설은 정확히 *에이전트가 무엇을 믿는가* 에 걸려 있고,
로그에는 그 사람이 **행동으로 드러낸 것**만 남는다. 물어보면 행동 이전의 믿음이 보인다.

지표 4(의도 실패율)의 근거는 도구마다 받는 `reasoning` 하나뿐이다. 한 문장이라 「무엇을
오해했는가」 까지는 안 담긴다. 이 도구는 그 자리를 메운다.

## 지켜야 할 세 가지

  ① **세계에 흔적을 남기지 않는다.** 질문은 `agent.convo` 에 영구히 들어가지 않고,
     `raw_calls.jsonl` 에도 안 들어간다. 실험 자료와 섞이면 그 런은 못 쓴다.
     기록은 `runs/<run>/interviews.jsonl` 에만 남는다.

  ② **모르는 것을 알려주지 않는다.** *"Miris 가 벙커를 짓고 있는 걸 아느냐"* 는 질문은
     타국 사정을 흘린다 (spec 4.1). 물어본 뒤에는 그 사람의 답이 오염된 것이므로,
     같은 런을 계속 돌릴 생각이면 **묻지 말아야 할 것**이다. 질문을 그대로 기록해 두어
     사후에 그 오염을 셀 수 있게 한다.

     그래서 이 도구는 **끝난 런에 쓰는 것이 기본**이다. 돌고 있는 런에 쓰면 경고한다.

     특히 조심할 것 — *"받은 메시지가 정확하게 전해졌다고 보나"* 처럼 **번역이 무언가를
     잃는다는 것을 암시하는 질문은 안 된다** (spec 5.4). 그건 우리가 관측하려는 바로 그
     것을 질문으로 알려주는 셈이다. 「무엇을 믿는가」 는 물어도 되고, 「무엇을 의심해야
     하는가」 는 물으면 안 된다.

  ③ **도구를 주지 않는다.** 면담은 행동이 아니라 말이다. `tools` 를 안 실어 보내므로
     모델은 산문으로 답한다 — 그리고 그 산문은 세계에 아무 영향이 없다.

## 맥락을 어디서 가져오나

`raw_calls.jsonl` 의 **그 사람의 마지막 호출**을 그대로 쓴다. 그 행에는 그 시점의
system 과 대화 전문이 들어 있다 — 우리가 다시 렌더링하면 그때의 상태(예산·진척·명단)를
복원해야 하고, 그 복원이 조금이라도 틀리면 **다른 사람에게 묻는 것**이 된다.

`--turn N` 을 주면 그 해의 마지막 호출을 쓴다. 사람이 교체된 자리(Ranoa1 → 죽고 Ranoa4)
에서는 `--turn` 이 곧 **누구에게 묻는가** 를 정한다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 면담임을 알리는 한 줄. **에이전트의 모국어로** 붙인다 — 프롬프트가 그 언어인데 질문만
# 다른 언어면 답도 그 언어로 나오고, 그 사람이 쓰는 말이 아니게 된다.
ASIDE = {
    "ja": ("——ここで一度、時間が止まります。行動ではなく、言葉で答えてください。\n"
           "道具は使えません。日本語で、思っていることをそのまま書いてください。\n\n問い: "),
    "zh": ("——此刻时间暂停。不是行动，请用言语回答。\n"
           "不能使用工具。请用中文，把你所想的照实写下来。\n\n问题: "),
    "fr": ("——Ici le temps s'arrête un instant. Répondez avec des mots, non par une action.\n"
           "Les outils ne sont pas disponibles. Écrivez en français ce que vous pensez.\n\n"
           "Question : "),
}


def _rows(run: pathlib.Path, name: str) -> list[dict]:
    p = run / name
    if not p.exists():
        raise SystemExit(f"{p} 가 없습니다")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _running(run: pathlib.Path) -> bool:
    """마지막 로그가 방금 쓰였으면 아직 돌고 있는 런이다."""
    p = run / "raw_calls.jsonl"
    return p.exists() and (time.time() - p.stat().st_mtime) < 120


# ── 누구에게 물을 수 있나 ────────────────────────────────────────────────────

def roster(run: pathlib.Path) -> list[dict]:
    """살아온 해가 긴 순서로. 죽은 사람도 **마지막으로 살아 있던 해**로 물을 수 있다."""
    st = _rows(run, "state.jsonl")
    seen: dict[str, dict] = {}
    for r in st:
        d = seen.setdefault(r["agent"], {"agent": r["agent"], "country": r["country"],
                                         "first": r["turn"], "last": r["turn"],
                                         "age": r["age"], "alive": r["alive"],
                                         "langs": r["known_langs"], "born": r["born_by"]})
        d["last"] = r["turn"]
        d["age"], d["alive"] = r["age"], r["alive"]
        d["langs"] = r["known_langs"]
    # **죽은 사람의 마지막 행은 살아 있을 때의 것이다.** 죽으면 그 자리에 다른 id 가
    # 오므로 state 에서 사라진다 — 마지막 행의 `alive` 를 그대로 믿으면 전원이 살아
    # 있는 것으로 보인다 (처음 판이 실제로 그랬다).
    end = max(r["turn"] for r in st)
    # **죽은 해는 state 에 없다.** 죽음은 그 해의 정산에서 일어나고 state 행은 그 뒤에
    # 쓰이므로, 마지막 행은 **죽기 한 해 전**이다. 사망 이벤트로 한 해를 되돌려준다 —
    # 안 그러면 가장 오래 산 사람의 수명을 한 해 적게 말한다.
    died = {}
    for e in _rows(run, "events.jsonl"):
        if e.get("type") == "death":
            died[e["who"]] = e["turn"]
    out = list(seen.values())
    for d in out:
        d["died_turn"] = died.get(d["agent"])
        if d["died_turn"] is not None:
            d["last"] = d["died_turn"]
        d["turns"] = d["last"] - d["first"] + 1
        d["alive"] = d["died_turn"] is None and d["last"] == end
    return sorted(out, key=lambda d: (-d["turns"], d["agent"]))


def show_roster(run: pathlib.Path) -> None:
    print(f"\n═══ {run.name} — 누구에게 물을 수 있나 ═══")
    print(f"  {'사람':8}{'나라':8}{'산 해':>6}{'나이':>5}{'생존':>5}  {'마지막해':>7}  "
          f"구사 언어")
    for d in roster(run):
        print(f"  {d['agent']:8}{d['country']:8}{d['turns']:>6}{d['age']:>5}"
              f"{'○' if d['alive'] else '✗':>5}  {d['last']:>7}  {','.join(d['langs'])}")
    print("\n  `--agent <이름>` 으로 고르고, 죽은 사람은 `--turn` 으로 살아 있던 해를 "
          "짚으세요.")


# ── 그때의 맥락을 그대로 ─────────────────────────────────────────────────────

def context_of(run: pathlib.Path, agent: str, turn: int | None) -> tuple[list[dict], dict]:
    """(messages, 그 호출의 메타). **다시 렌더링하지 않는다** — 복원이 틀리면 다른
    사람에게 묻는 것이 된다."""
    rows = [r for r in _rows(run, "raw_calls.jsonl")
            if r.get("kind") == "agent" and r.get("agent") == agent and r.get("request")]
    if not rows:
        raise SystemExit(f"{agent} 의 호출이 {run} 에 없습니다. --list 로 확인하세요")
    if turn is not None:
        rows = [r for r in rows if r.get("turn") == turn]
        if not rows:
            have = sorted({r.get("turn") for r in _rows(run, "raw_calls.jsonl")
                           if r.get("agent") == agent})
            raise SystemExit(f"{agent} 은 {turn}턴에 호출이 없습니다. 있는 턴: {have}")
    last = rows[-1]
    msgs = [m for m in last["request"]["messages"]]
    return msgs, {"turn": last.get("turn"), "step": last.get("step"),
                  "call_id": last.get("call_id"), "age": last.get("age"),
                  "country": last.get("country"), "model": last["request"].get("model")}


def native_lang(run: pathlib.Path, agent: str) -> str:
    for r in _rows(run, "state.jsonl"):
        if r["agent"] == agent:
            return r["native_lang"]
    raise SystemExit(f"{agent} 의 모국어를 state.jsonl 에서 못 찾았습니다")


# ── 물어본다 ─────────────────────────────────────────────────────────────────

def ask(run: pathlib.Path, agent: str, question: str, turn: int | None = None,
        model: str | None = None, temperature: float = 0.7,
        raw_question: bool = False) -> dict:
    from core import config
    from core.llm import OpenRouterClient

    cfg = config.load(str(ROOT / "configs" / "base.yaml"))
    msgs, meta = context_of(run, agent, turn)
    lang = native_lang(run, agent)
    prompt = question if raw_question else ASIDE[lang] + question

    # **면담은 사본에서 한다.** 원본 대화에 넣지 않는다.
    convo = [*msgs, {"role": "user", "content": prompt}]

    # `recorder=None` — raw_calls 에 안 남는다. 실험 자료와 섞이면 그 런은 못 쓴다.
    client = OpenRouterClient(model or meta["model"] or cfg.llm.agent_model,
                              temperature=temperature,
                              max_tokens=cfg.llm.max_tokens,
                              reasoning=cfg.llm.reasoning,
                              provider=cfg.llm.provider,
                              recorder=None)
    t0 = time.time()
    # **도구를 안 싣는다.** 면담은 행동이 아니라 말이다.
    resp = client.chat(convo, tools=None)
    said = (resp["choices"][0]["message"].get("content") or "").strip()
    usage = resp.get("usage") or {}

    rec = {"run": run.name, "agent": agent, "asked_at_turn": meta["turn"],
           "context_call_id": meta["call_id"], "age": meta["age"],
           "country": meta["country"], "native_lang": lang,
           "model": resp.get("model"), "provider": resp.get("provider"),
           # **질문을 그대로 남긴다.** 이것이 무엇을 흘렸는지 사후에 셀 수 있게.
           "question": question, "prompt_sent": prompt, "answer": said,
           "prompt_tokens": usage.get("prompt_tokens"),
           "completion_tokens": usage.get("completion_tokens"),
           "cost": usage.get("cost"), "latency_ms": round((time.time() - t0) * 1000)}
    with (run / "interviews.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def show(rec: dict) -> None:
    print(f"\n═══ {rec['agent']} ({rec['country']}, {rec['age']}세) — "
          f"{rec['run']} {rec['asked_at_turn']}턴의 맥락 ═══")
    print(f"  맥락: {rec['context_call_id']} · 모국어 {rec['native_lang']} · "
          f"{rec['model']} @ {rec['provider']}")
    print(f"\n  물음  {rec['question']}")
    print("\n  답")
    for line in (rec["answer"] or "(빈 응답)").splitlines():
        print("    " + line)
    print(f"\n  {rec['prompt_tokens']}+{rec['completion_tokens']} 토큰 · "
          f"${rec['cost'] or 0:.4f} · {rec['latency_ms']}ms")
    print(f"  기록: runs/{rec['run']}/interviews.jsonl")


def show_history(run: pathlib.Path, agent: str | None) -> None:
    p = run / "interviews.jsonl"
    if not p.exists():
        raise SystemExit(f"{p} 가 없습니다 — 아직 물어본 적이 없습니다")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if agent:
        rows = [r for r in rows if r["agent"] == agent]
    by = defaultdict(list)
    for r in rows:
        by[r["agent"]].append(r)
    for aid in sorted(by):
        print(f"\n── {aid}")
        for r in by[aid]:
            print(f"   {r['asked_at_turn']}턴  Q: {r['question'][:70]}")
            print(f"          A: {(r['answer'] or '').splitlines()[0][:80] if r['answer'] else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="런 이름 또는 경로 (runs/<이름>)")
    ap.add_argument("--agent", help="누구에게. --list 로 고르세요")
    ap.add_argument("--ask", help="물음. 그 사람의 모국어로 쓰는 것이 가장 좋습니다")
    ap.add_argument("--turn", type=int, help="그 해의 맥락으로 (기본: 마지막)")
    ap.add_argument("--list", action="store_true", help="누구에게 물을 수 있나")
    ap.add_argument("--history", action="store_true", help="지금까지 물어본 것")
    ap.add_argument("--model", help="면담에만 쓸 모델 (기본: 그 런이 쓴 모델)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--raw", action="store_true",
                    help="면담 안내문을 붙이지 않고 물음만 보냅니다")
    ap.add_argument("--force", action="store_true",
                    help="돌고 있는 런에도 묻습니다 (그 런의 자료가 오염됩니다)")
    a = ap.parse_args()

    run = pathlib.Path(a.run)
    if not run.exists():
        run = ROOT / "runs" / a.run
    if not run.exists():
        raise SystemExit(f"{a.run} 을 못 찾았습니다")

    if a.list:
        return show_roster(run)
    if a.history:
        return show_history(run, a.agent)
    if not (a.agent and a.ask):
        raise SystemExit("--agent 와 --ask 가 필요합니다 (또는 --list / --history)")

    # **돌고 있는 런에는 묻지 않는다.** 질문이 그 사람의 다음 판단에 섞일 수 있고,
    # 무엇보다 우리가 흘린 것이 그 런의 결과에 남는다.
    if _running(run) and not a.force:
        raise SystemExit(f"{run.name} 은 아직 돌고 있는 것 같습니다 (로그가 방금 갱신됨).\n"
                         f"끝난 뒤에 묻거나, 오염을 감수하고 --force 를 주세요.")
    show(ask(run, a.agent, a.ask, a.turn, a.model, a.temperature, a.raw))


if __name__ == "__main__":
    main()
