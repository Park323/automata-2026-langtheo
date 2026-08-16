"""산출물 기록. spec 9장.

  runs/{run_id}/
    config_snapshot.yaml   설정 + 코드 커밋 해시
    raw_calls.jsonl        ★ LLM 호출 전문 (요청·응답 원본)
    state.jsonl            턴별 에이전트 상태
    messages.jsonl         6.1 스키마
    events.jsonl           사망·출생·학습·투표·검증 거부
    metrics.jsonl          턴별 집계
    summary.json           run 전체 요약

**파생 로그는 전부 raw_calls.jsonl 에서 재생성할 수 있어야 한다.** 파일럿이 raw.jsonl 을
남긴 덕에 지표 6a 의 정의를 사후에 바꿀 수 있었다. 정의는 나중에 바뀐다 — 원본이 없으면
그때 다시 돌려야 한다.

턴마다 append 한다. 50턴 × ~1,700콜 런이 45턴에서 죽어도 거기까지는 남는다.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


class RunWriter:
    """한 run 의 산출물. 스레드 안전 (에이전트 호출이 병렬이라 raw 가 동시에 들어온다)."""

    def __init__(self, run_id: str, cfg_raw: dict | None = None, root: Path | None = None,
                 overwrite: bool = False, knob_ai: float | None = None,
                 seed: int | None = None):
        self.run_id = run_id
        self.dir = (root or ROOT / "runs") / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        # 같은 run_id 로 다시 돌리면 이전 런 위에 **이어붙는다.** 두 런이 한 파일에
        # 섞이면 지표가 조용히 오염된다 (실측에서 실제로 겪었다).
        existing = list(self.dir.glob("*.jsonl"))
        if existing and not overwrite:
            raise FileExistsError(
                f"{self.dir} 에 이미 로그가 있습니다. run_id 를 바꾸거나 overwrite=True. "
                f"({', '.join(p.name for p in existing)})")
        for f in existing:
            f.unlink()
        self._lock = threading.Lock()
        self._files: dict[str, object] = {}
        self.counts = {"raw": 0, "errors": 0, "retries": 0}
        if cfg_raw is not None:
            import yaml
            # ★ knob_ai 는 config 가 아니라 **런 인자**다. config 에는 스윕할 목록
            # [6,12,24,48] 이 들어 있어서, 이걸 따로 안 적으면 산출물만 보고는 그 런이
            # 어느 조건이었는지 알 수 없다 — 조건별 표를 만들 수 없다는 뜻이다.
            snap = {"config": cfg_raw, "code_commit": git_commit(), "run_id": run_id,
                    "knob_ai": knob_ai, "seed": seed}
            (self.dir / "config_snapshot.yaml").write_text(
                yaml.safe_dump(snap, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # ── 저수준 ────────────────────────────────────────────────────────────────
    def _append(self, name: str, obj: dict) -> None:
        with self._lock:
            f = self._files.get(name)
            if f is None:
                f = open(self.dir / f"{name}.jsonl", "a", encoding="utf-8")
                self._files[name] = f
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            f.flush()          # 크래시해도 거기까지는 남아야 한다

    def close(self, summary: dict | None = None) -> None:
        if summary is not None:
            (self.dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")
        with self._lock:
            for f in self._files.values():
                f.close()
            self._files.clear()

    # ── raw 호출 ──────────────────────────────────────────────────────────────
    def raw(self, rec: dict) -> None:
        """LLM 호출 1회(재시도 각각). request/response 를 가공 없이 남긴다."""
        rec.setdefault("run_id", self.run_id)
        self.counts["raw"] += 1
        if rec.get("error"):
            self.counts["errors"] += 1
        if rec.get("attempt", 1) > 1:
            self.counts["retries"] += 1
        self._append("raw_calls", rec)

    def recorder(self, **tag):
        """클라이언트에 붙일 기록 콜백. tag 는 {kind, agent, ...}."""
        def _rec(rec: dict) -> None:
            self.raw({**tag, **rec})
        return _rec

    # ── 턴별 ──────────────────────────────────────────────────────────────────
    def on_turn_end(self, turn: int, result) -> None:
        """run_agentic 의 on_turn_end 훅으로 그대로 넘길 수 있다."""
        world = result.world
        for aid in sorted(world.agents):
            a = world.agents[aid]
            self._append("state", {
                "turn": turn, "agent": aid, "country": a.country, "age": a.age,
                "lambda": round(a.lam, 4), "known_langs": sorted(a.known_langs),
                "parent_langs": sorted(a.parent_langs), "budget": round(a.budget, 4),
                "budget_start": round(a.budget_start, 4),
                "wellness_spent": round(a.wellness_spent, 4),
                "born_turn": a.born_turn, "born_by": a.born_by, "alive": a.alive,
                "uid": a.uid,
            })
        for m in result.messages_log:
            if m.get("turn") == turn and not m.get("_written"):
                m["_written"] = True
                self._append("messages", {k: v for k, v in m.items() if k != "_written"})
        logs = result.agent_logs[turn - 1] if len(result.agent_logs) >= turn else {}
        for aid in sorted(logs):
            lg = logs[aid]
            self._append("events", {
                "turn": turn, "type": "agent_turn", "agent": aid,
                "reasonings": lg.get("reasonings"), "api_reasoning": lg.get("api_reasoning"),
                "ended_by": lg.get("ended_by"), "error": lg.get("error"),
                "reasoning_missing": lg.get("reasoning_missing"),
                "steps": lg.get("steps"), "prompt_tokens": lg.get("prompt_tokens"),
                "pressured": lg.get("pressured"), "evicted_blocks": lg.get("evicted_blocks"),
                "memory_len": lg.get("memory_len"),
                # 한 사람이 한 턴을 사는 데 걸린 시간. llm_ms 가 elapsed 의 거의 전부여야
                # 정상이고, 갈리면 우리 코드가 병목이라는 뜻이다.
                "recovered_calls": lg.get("recovered_calls"),
                "no_tool_content": lg.get("no_tool_content"),
                "elapsed_ms": lg.get("elapsed_ms"), "llm_ms": lg.get("llm_ms"),
                "ms_per_step": lg.get("ms_per_step"),
                "actions": [a.get("type") for a in lg.get("actions", [])],
            })
        for lr in result.learns_log:
            if lr.get("turn") == turn and not lr.get("_written"):
                lr["_written"] = True
                self._append("events", {"type": "learn",
                                        **{k: v for k, v in lr.items() if k != "_written"}})
        # ★ 아래 넷은 **RunResult 에만 있고 파일에는 안 남고 있었다.** 투표·국토 전환·
        #   부고·진척 기여가 전부 그랬다 — 오늘 만든 규칙이 통째로 관측 불가였다는 뜻이다.
        #   metrics 의 land 로 전환은 역산되지만 찬반 수·소실 진척은 복구되지 않는다.
        for name, rows in (("vote", result.votes_log),
                           ("land_change", result.land_changes),
                           ("death", result.deaths_log),
                           ("facility_gain", result.facility_gains),
                           ("risk_observe", result.risk_log)):
            for r in rows:
                if r.get("turn") != turn or r.get("_written"):
                    continue
                r["_written"] = True
                # type 은 **마지막에** 넣는다 — 행 안에 같은 키가 있으면 덮어써 버린다
                self._append("events", {**{k: v for k, v in r.items() if k != "_written"},
                                        "type": name})
        for b in result.births:
            if b.get("turn") == turn and not b.get("_written"):
                b["_written"] = True
                self._append("events", {"turn": turn, "type": "birth",
                                        **{k: v for k, v in b.items() if k != "_written"}})
        failed = sum(1 for lg in logs.values() if lg.get("error"))
        ends: dict[str, int] = {}
        for lg in logs.values():
            ends[lg.get("ended_by", "?")] = ends.get(lg.get("ended_by", "?"), 0) + 1
        self._append("metrics", {
            "turn": turn,
            "alive": sum(1 for a in world.agents.values() if a.alive),
            "progress": {c.id: round(c.progress, 3) for c in world.countries.values()},
            "land": {c.id: c.land for c in world.countries.values()},
            "national_capital": {c.id: round(c.national_capital, 3)
                                 for c in world.countries.values()},
            "messages_this_turn": sum(1 for m in result.messages_log if m.get("turn") == turn),
            "agent_turns": len(logs), "llm_failures": failed,
            "llm_failure_rate": round(failed / len(logs), 4) if logs else 0.0,
            "ended_by": ends,
            "prompt_tokens_max": max((lg.get("prompt_tokens") or 0 for lg in logs.values()), default=0),
            # 턴 벽시계는 9명 병렬이라 **가장 느린 사람**이 정한다
            "turn_wall_ms": max((lg.get("elapsed_ms") or 0 for lg in logs.values()), default=0),
            "agent_ms_sum": sum(lg.get("elapsed_ms") or 0 for lg in logs.values()),
            "pressured": sum(1 for lg in logs.values() if lg.get("pressured")),
            # 도구 채널로 안 나온 호출을 몇 건 주웠나 / 끝내 못 주운 턴은 몇인가
            "recovered_calls": sum(lg.get("recovered_calls") or 0 for lg in logs.values()),
            "no_tool_call": sum(1 for lg in logs.values()
                                if lg.get("ended_by") == "no_tool_call"),
            "memory_writes": sum(1 for lg in logs.values()
                                 for a in lg.get("actions", []) if a.get("type") == "memory_write"),
            "raw_calls_total": self.counts["raw"], "raw_errors": self.counts["errors"],
        })
