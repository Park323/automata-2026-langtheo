"""산출물 기록. spec 9장.

  runs/{run_id}/
    config_snapshot.yaml   설정 + 코드 커밋 해시
    raw_calls.jsonl        ★ LLM 호출 전문 (call_id + 요청·응답 원본 + turn·agent·step·msg_id)
    state.jsonl            턴별 에이전트 상태 (메모·학습 진척·AP 잔량 포함)
    messages.jsonl         6.1 스키마 (원문·번역 프롬프트·도착문)
    events.jsonl           사망·출생·학습·투표·관측 + agent_turn(호출 전문)
    metrics.jsonl          턴별 집계 (국토·진척·자본·열린 제안)
    summary.json           run 전체 요약

**파생 로그는 전부 raw_calls.jsonl 에서 재생성할 수 있어야 한다.** 파일럿이 raw.jsonl 을
남긴 덕에 지표 6a 의 정의를 사후에 바꿀 수 있었다. 정의는 나중에 바뀐다 — 원본이 없으면
그때 다시 돌려야 한다.

턴마다 append 한다. 50턴 × ~1,700콜 런이 45턴에서 죽어도 거기까지는 남는다.

## 빠뜨리는 방식은 늘 같았다

누락은 **분석하려고 파일을 열어 봤을 때** 발견됐고, 그때는 이미 그 런을 다시 돌려야
했다. 8/18 전수 검토에서 나온 다섯 건이 전부 그런 것이었다.

    memory_write 의 본문   *"본문은 messages 에 있으므로 뺀다"* 가 speak 에만 맞는
                          말이었는데 종류를 안 가리고 잘랐다. 세 런 60건이 전부
                          {"type": "memory_write"} 로만 남았다
    procreate 의 유언      같은 이유로 잘렸다. 부고에도 없었다 — 아이의 기억
                          초기값으로만 흘러가서, 아이가 덮어쓰면 원문이 사라졌다.
                          하필 그 덮어쓰기가 spec 3.3 이 관측하려는 구전의 감쇠다
    도구 호출의 실패 사유   성공만 actions 에 남고 실패는 이름과 ok=False 로만 남았다.
                          AP 부족인지 국가 이름을 틀렸는지 구분이 안 됐다
    실제 과금·절삭        actions 는 **요청한 값**이다. 9,999 를 냈는데 AP 가 300 으로
                          잘라도 로그에는 9,999 가 남았다
    호출의 임자·시각       kind 는 클라이언트를 만들 때 붙는 고정 태그라 turn·agent 를
                          담을 수 없었다. raw_calls 를 events 와 이어붙일 키가 없어
                          호출 단위 분석이 통째로 막혀 있었다

**호출마다 `call_id` 가 붇는다** (`c00001` …). 로그를 읽다 한 호출을 가리켜야 할 때
(turn, agent, step, attempt) 네 개를 나열하는 대신 한 마디로 말할 수 있다. 순차 라운드
로빈에서는 호출 순서가 결정론이라 같은 시드·같은 코드면 같은 번호가 나오고, 병렬 경로는
순서가 비결정론이라 런마다 다르다 — 그때는 (turn, agent, step) 으로 찾는다.

그래서 `tests/test_logging_complete.py` 가 **덮개를 지킨다.** `Agent`·`Country` 에 필드가
늘면 로그에 안 들어간 채로는 통과하지 못한다 — 면제하려면 이유를 적어야 하고, 이유를
적을 수 없으면 그건 빠뜨린 것이다.
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
    except (OSError, subprocess.SubprocessError):
        # git 이 없거나 저장소가 아닌 환경. 그것 말고는 통과시킨다.
        return None


def _redact(a: dict) -> dict:
    """행동 인자에서 **중복인 것만** 뺀다.

    전에는 `text`·`testament` 를 종류와 무관하게 잘랐다. 근거는 *"본문은 messages 에
    있으므로"* 였는데 **그건 `speak` 에만 맞는 말이었다.** `memory_write` 의 본문과
    `procreate` 의 유언은 messages 에도 events 에도 없어서 **통째로 사라지고 있었다** —
    세 런 60건의 memory_write 가 전부 `{"type": "memory_write"}` 로만 남았다.

    무엇을 적어둘 가치가 있다고 봤는지, 무엇을 남기고 죽었는지는 이 시뮬레이션이
    내놓는 자료 중 가장 해석이 필요한 축이다 (spec 3.3 — 유언은 자유 텍스트이고
    거기가 창발 지점이다). 그것을 로그에서 버리면 사후에 복구할 방법이 없다.

    `reasoning` 은 같은 이벤트의 `reasonings` 에 이미 있으므로 계속 뺀다.
    `speak` 의 `text` 도 `messages.jsonl` 에 원문·도착문이 함께 있으므로 계속 뺀다.
    """
    drop = {"reasoning"}
    if a.get("type") == "speak":
        drop.add("text")
    return {k: v for k, v in a.items() if k not in drop}


class RunWriter:
    """한 run 의 산출물. 스레드 안전 (에이전트 호출이 병렬이라 raw 가 동시에 들어온다)."""

    def __init__(self, run_id: str, cfg_raw: dict | None = None, root: Path | None = None,
                 overwrite: bool = False, knob_ai: float | None = None,
                 seed: int | None = None, append: bool = False):
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
        # 이어할 때는 **지우지 않는다.** 앞 구간의 로그가 곧 그 런의 절반이다.
        if not append:
            for f in existing:
                f.unlink()
        # **재진입 락.** raw() 가 카운터를 잡은 채 _append 를 부른다 — 한 락 안에서
        # 끝내야 `call_id` 순서와 파일 순서가 어긋나지 않는다.
        self._lock = threading.RLock()
        self._files: dict[str, object] = {}
        self.counts = {"raw": 0, "errors": 0, "retries": 0}
        # **이어할 때는 이미 쓴 줄 수에서 이어 센다.** 0 에서 다시 시작하면 `call_id` 가
        # 앞 구간과 충돌하고, raw_calls_total 도 이어붙인 구간만 센 값이 된다.
        if append:
            prev = self.dir / "raw_calls.jsonl"
            if prev.exists():
                with prev.open(encoding="utf-8") as f:
                    self.counts["raw"] = sum(1 for line in f if line.strip())
        self.last_turn = 0                    # 크래시 행을 놓을 자리
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

    def crash(self, exc: BaseException, where: str = "run") -> None:
        """**터진 자리를 디스크에 남긴다.** 그전에는 트레이스백이 stderr 로만 갔다.

        `summary.json` 의 `aborted` 는 `"KeyError: dst_lang"` 300자뿐이었다. 야간
        배치나 nohup 이면 스택이 그대로 사라지고, 남은 것으로는 어디서 터졌는지 알 수
        없다 — 예외를 좁혀 버그를 드러내기로 한 뜻이 절반만 이뤄진다.

        `events.jsonl` 에 넣는 이유는 **턴 순서 안에 놓이기 때문**이다. 마지막 정상
        턴 바로 뒤에 붙으므로, raw_calls 의 마지막 호출(turn·agent·step 이 붙어 있다)과
        나란히 읽으면 어느 호출 뒤에 무엇이 터졌는지가 이어진다.
        """
        import traceback
        # **턴을 두 값으로 적는다.** `last_turn` 하나만 적으면 오해를 부른다 —
        # 1턴 정산 중에 터지면 on_turn_end(1) 이 아직 안 돌아서 0 이 찍히고, 마치
        # 시작도 못 한 것처럼 읽힌다. 실제로는 그 다음 턴이 돌던 중이다.
        self._append("events", {
            "turn": self.last_turn + 1, "last_completed_turn": self.last_turn,
            "type": "crash", "where": where,
            "exc": type(exc).__name__, "message": str(exc)[:2000],
            "notes": list(getattr(exc, "__notes__", []) or []),
            "traceback": "".join(traceback.format_exception(exc))[:20000],
        })
        with self._lock:                      # 곧 죽는다 — 버퍼에 남기면 안 된다
            for f in self._files.values():
                f.flush()

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
        """LLM 호출 1회(재시도 각각). request/response 를 가공 없이 남긴다.

        **호출마다 `call_id` 를 붇인다** (`c00001` …). 로그를 읽다 한 호출을 가리켜야 할
        때 (turn, agent, step, attempt) 네 개를 나열하는 대신 한 마디로 말할 수 있다 —
        재시도까지 갈리므로 같은 스텝의 1차·2차 시도도 서로 다른 이름을 갖는다.

        번호는 그 런에서 이 파일에 쓴 줄 수다. 순차 라운드로빈에서는 호출 순서가
        결정론이라 같은 시드·같은 코드면 같은 번호가 나온다. 병렬 경로는 순서가
        비결정론이므로 번호도 런마다 다르다 — 그때는 (turn, agent, step) 으로 찾는다.

        카운터를 락 안에서 올린다. 전에는 밖이라 병렬 경로에서 **집계가 새고 있었다.**
        """
        with self._lock:
            self.counts["raw"] += 1
            if rec.get("error"):
                self.counts["errors"] += 1
            if rec.get("attempt", 1) > 1:
                self.counts["retries"] += 1
            # call_id 를 맨 앞에 둔다 — 한 줄을 눈으로 훑을 때 먼저 보이게
            row = {"call_id": f"c{self.counts['raw']:05d}", **rec}
            row.setdefault("run_id", self.run_id)
            self._append("raw_calls", row)

    def recorder(self, **tag):
        """클라이언트에 붙일 기록 콜백. tag 는 {kind, agent, ...}."""
        def _rec(rec: dict) -> None:
            self.raw({**tag, **rec})
        return _rec

    # ── 턴별 ──────────────────────────────────────────────────────────────────
    def on_turn_end(self, turn: int, result) -> None:
        """run_agentic 의 on_turn_end 훅으로 그대로 넘길 수 있다."""
        self.last_turn = turn          # 크래시 행을 마지막 정상 턴 뒤에 놓기 위해
        world = result.world
        for aid in sorted(world.agents):
            a = world.agents[aid]
            self._append("state", {
                "turn": turn, "agent": aid, "country": a.country, "age": a.age,
                "lambda": round(a.lam, 4), "known_langs": sorted(a.known_langs),
                "parent_langs": sorted(a.parent_langs), "budget": round(a.budget, 4),
                "budget_start": round(a.budget_start, 4),
                "income_this_year": round(a.income_this_year, 4),
                "wellness_spent": round(a.wellness_spent, 4),
                "born_turn": a.born_turn, "born_by": a.born_by, "alive": a.alive,
                "uid": a.uid, "native_lang": a.native_lang,
                # **AP 가 남아 있지 않았다.** 이 세계의 진짜 예산인데 턴 끝 잔량이
                # 어디에도 없어서 "무엇을 포기했는가" 를 사후에 볼 수 없었다.
                "ap_left": round(a.ap, 4),
                # **지금 이 사람의 메모.** 쓴 턴에만 남기면 "무엇을 들고 다니는가" 를
                # 볼 수 없다 — 유언을 물려받고 한 번도 안 고친 아이가 특히 그렇다.
                "memory": a.memory,
                "lang_progress": {k: round(v, 2) for k, v in (a.lang_progress or {}).items()},
                "facility_invested": {k: round(v, 2)
                                      for k, v in (a.facility_invested or {}).items()},
                # **이 사람이 표를 던진 採決의 해.** vote 이벤트로도 복원되지만, 두 표가
                # 둘 다 집계된 적이 있어(3해 실측) 「한 사람 한 표」 를 상태에서 바로
                # 대조할 수 있게 남긴다.
                "voted_turn": a.voted_turn,
                # 기억 도구가 열려 있었나 — 압박과 함께 움직이지만, 어긋난 적이 있어
                # 상태에서 바로 대조되게 남긴다
                "memory_open": a.memory_open,
                # 생애 1회 — 아이를 낳았나 (8/21)
                "has_borne": a.has_borne,
                # 부모 — 살아 있는지가 무소득 판정이다 (8/22)
                "parent_id": a.parent_id,
            })
        for m in result.messages_log:
            if m.get("turn") == turn and not m.get("_written"):
                m["_written"] = True
                self._append("messages", {k: v for k, v in m.items() if k != "_written"})
        # on_turn_end 은 이번 턴 로그를 append 한 **직후** 호출되므로 마지막 것이 이 턴이다.
        # 절대 인덱스(turn-1)로 잡으면 resume 시 result 가 새로 시작해 어긋난다 —
        # 이어받은 턴의 agent_turn 이벤트가 통째로 안 써졌다.
        logs = result.agent_logs[-1] if result.agent_logs else {}
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
                # **인자까지 남긴다.** 전에는 종류만 남아서 "누가 어디에 냈는가" 를
                # 사후에 볼 수 없었다 — raw_calls 에도 agent id 가 없어 호출 단위
                # 분석이 통째로 막혔다.
                "actions": [_redact(a) for a in lg.get("actions", [])],
                # **호출 전문.** actions 는 성공한 것만, 요청한 값 그대로다.
                # calls 는 실패까지 남기고 실제 결과(과금·절삭·오류 사유)를 담는다.
                "calls": lg.get("calls"),
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
            # **열린 제안.** vote 이벤트로만 남아 있어서, 제안이 열려 있던 구간을
            # 사후에 복원하려면 이벤트를 되짚어야 했다 — 採決 전에 무슨 말이 오갔는지를
            # 보려면 매 턴의 상태가 있어야 한다.
            "proposal": {c.id: c.proposal for c in world.countries.values()},
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
