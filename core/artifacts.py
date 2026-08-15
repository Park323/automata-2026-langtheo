"""실험 산출물 기록. spec 9.

    runs/{run_id}/
      config_snapshot.yaml   재현용 (코드 커밋 해시 포함)
      raw_calls.jsonl        모든 LLM 호출의 요청·응답 원본 (재시도 각각 한 행)
      state.jsonl            턴별 에이전트 상태
      messages.jsonl         메시지 1건당 1행 (understood 조인 후)
      events.jsonl           출생·투표 (그 외 granular 이벤트는 이후 과제)
      metrics.jsonl          턴별 집계 (생존 수·종료 사유 분포·llm 실패율)
      summary.json           run 요약

파생 로그는 전부 raw_calls.jsonl 에서 재생성할 수 있어야 한다 — 정의는 나중에 바뀌고,
원본이 없으면 그때 다시 돌려야 한다. **크래시 내성**: 턴이 끝날 때마다 append 한다.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 에이전트-턴의 이 비율을 넘겨 LLM 호출이 실패하면 그 런을 무효로 표시한다 (#8). 실패는
# 조용한 편향이라(비싼 조건일수록 429 가 늘 수 있음) 조율 실패를 번역 왜곡과 혼동하게 만든다.
# 세계 규칙이 아니라 분석 임계값이므로 config 가 아니라 여기 둔다.
FAILURE_INVALID_RATE = 0.02


def code_commit() -> str | None:
    """현재 코드의 커밋 해시. git 이 없거나 리포가 아니면 None."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def make_run_id(knob, seed) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_knob{knob:g}_seed{seed}"


class RawCallSink:
    """raw_calls.jsonl 한 행 = LLM 호출 한 번(재시도 각각). 병렬 에이전트가 공유하므로 락."""

    def __init__(self, fh, run_id: str):
        self._fh = fh
        self.run_id = run_id
        self._lock = threading.Lock()

    def write(self, kind: str, meta: dict, request, response, error,
              latency_ms: int, attempt: int) -> None:
        rec = {
            "run_id": self.run_id, "kind": kind,
            "turn": meta.get("turn"), "agent": meta.get("agent"), "step": meta.get("step"),
            "attempt": attempt, "latency_ms": latency_ms,
            "request": request, "response": response, "error": error,
        }
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()


def _agent_state(turn: int, a) -> dict:
    return {"turn": turn, "agent": a.id, "country": a.country, "age": a.age,
            "lambda": round(a.lam, 6), "known_langs": sorted(a.known_langs),
            "parent_langs": sorted(a.parent_langs), "budget": round(a.budget, 6),
            "born_turn": a.born_turn, "born_by": a.born_by, "alive": a.alive}


class RunWriter:
    """run 하나의 산출물 디렉토리를 열고, 턴마다 append 한다.

    사용:
        w = RunWriter(run_id, config_dict, knob=48, seed=1)
        client = OpenRouterClient(..., raw_sink=w.raw_sink)
        run_agentic(..., on_turn_end=w.on_turn_end)
        w.finish(result); w.close()
    """

    def __init__(self, run_id: str, config_dict: dict, knob=None, seed=None,
                 runs_dir: Path | None = None):
        self.run_id = run_id
        self.dir = (runs_dir or (ROOT / "runs")) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        commit = code_commit()

        # config_snapshot.yaml — 재현용. 커밋 해시를 상단 주석 + _meta 로 함께 남긴다.
        import yaml
        snap = dict(config_dict)
        snap["_meta"] = {"run_id": run_id, "commit": commit, "knob": knob, "seed": seed,
                         "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            f.write(f"# run_id: {run_id}\n# commit: {commit}\n")
            yaml.safe_dump(snap, f, allow_unicode=True, sort_keys=False)

        self._raw_fh = open(self.dir / "raw_calls.jsonl", "a", encoding="utf-8")
        self.raw_sink = RawCallSink(self._raw_fh, run_id)
        self._state_fh = open(self.dir / "state.jsonl", "a", encoding="utf-8")
        self._msg_fh = open(self.dir / "messages.jsonl", "a", encoding="utf-8")
        self._events_fh = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._metrics_fh = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")
        # 이미 써낸 파생 로그의 커서 (증분 append 용)
        self._msg_written = 0
        self._births_written = 0
        self._votes_written = 0
        self._learn_written = 0
        self._death_written = 0

    # ── 턴별 append (크래시 내성) ────────────────────────────────────────────
    def on_turn_end(self, turn: int, result) -> None:
        # state — 현재 전 에이전트
        for aid, a in sorted(result.world.agents.items()):
            self._write(self._state_fh, _agent_state(turn, a))

        # messages — understood 조인이 T+1 에 일어나므로 한 턴 늦춰 쓴다 (전 턴 것까지).
        mlog = result.messages_log
        while self._msg_written < len(mlog) and mlog[self._msg_written]["turn"] <= turn - 1:
            self._write(self._msg_fh, mlog[self._msg_written])
            self._msg_written += 1

        # events — 이번 턴 출생·투표·학습·사망 (6.1). learn 의 지불액·할인은 x̂ 눈금.
        for b in result.births[self._births_written:]:
            self._write(self._events_fh, {"turn": b["turn"], "event": "birth", **b})
        self._births_written = len(result.births)
        for v in result.votes_log[self._votes_written:]:
            self._write(self._events_fh, {"turn": v["turn"], "event": "vote", **v})
        self._votes_written = len(result.votes_log)
        for e in result.learn_log[self._learn_written:]:
            self._write(self._events_fh, {"event": "learn", **e})
        self._learn_written = len(result.learn_log)
        for e in result.death_log[self._death_written:]:
            self._write(self._events_fh, {"event": "death", **e})
        self._death_written = len(result.death_log)

        # metrics — 이번 턴 집계 (종료 사유 분포·llm 실패율)
        turn_log = result.agent_logs[-1] if result.agent_logs else {}
        end_reasons = Counter(lg.get("end_reason") for lg in turn_log.values())
        n = len(turn_log) or 1
        errors = sum(1 for lg in turn_log.values() if lg.get("error"))
        missing = sum(1 for lg in turn_log.values() if lg.get("reasoning_missing"))
        acted = sum(1 for lg in turn_log.values() if lg.get("actions"))
        self._write(self._metrics_fh, {
            "turn": turn,
            "alive": sum(1 for a in result.world.agents.values() if a.alive),
            "acted": acted, "messages_total": len(result.messages_log),
            "end_reasons": dict(end_reasons),
            "llm_failures": errors, "llm_failure_rate": round(errors / n, 4),
            "reasoning_missing": missing,
        })

    def finish(self, result) -> None:
        """남은 메시지(마지막 턴 것, understood 는 null 로 확정) flush + summary.json."""
        mlog = result.messages_log
        while self._msg_written < len(mlog):
            self._write(self._msg_fh, mlog[self._msg_written])
            self._msg_written += 1
        all_reasons: Counter = Counter()
        agent_turns = 0
        failures = 0
        for tl in result.agent_logs:
            all_reasons.update(lg.get("end_reason") for lg in tl.values())
            agent_turns += len(tl)
            failures += sum(1 for lg in tl.values() if lg.get("error"))
        failure_rate = round(failures / agent_turns, 4) if agent_turns else 0.0

        # 무응답률 (#7): 전달된 메시지 중 수신자가 report_understanding 을 안 한 비율.
        # 조건 간에 크게 다르면 그 자체가 편향 신호 — 무응답을 오해로 세면 안 되므로 별도로 남긴다.
        delivered = sum(1 for m in result.messages_log if m.get("delivered"))
        reported = sum(1 for m in result.messages_log
                       if m.get("delivered") and m.get("understood") is not None)
        no_response_rate = round(1 - reported / delivered, 4) if delivered else None

        summary = {
            "run_id": self.run_id,
            "turns": len(result.state_lines),
            "deaths": result.deaths,
            "final": result.final,
            "messages": len(result.messages_log),
            "delivered": delivered,
            "understood_reported": reported,
            "no_response_rate": no_response_rate,
            "agent_turns": agent_turns,
            "llm_failures": failures,
            "llm_failure_rate": failure_rate,
            # 실패율이 임계를 넘으면 이 런은 무효 — 결과 해석에서 제외해야 한다 (#8)
            "invalid_high_failure": failure_rate > FAILURE_INVALID_RATE,
            "end_reasons": dict(all_reasons),
        }
        with open(self.dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _write(fh, obj) -> None:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()

    def close(self) -> None:
        for fh in (self._raw_fh, self._state_fh, self._msg_fh,
                   self._events_fh, self._metrics_fh):
            try:
                fh.close()
            except Exception:
                pass
