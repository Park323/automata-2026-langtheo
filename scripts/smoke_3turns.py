"""3턴 스모크 테스트. 과제 2 합격기준 #14 — 실제 OpenRouter 로 tool_calls 가 오는지 확인.

⚠ 이 스크립트는 실제 API 를 호출해 토큰 비용이 발생한다. StubClient 테스트(pytest)로
  로직을 먼저 다 검증한 뒤, 마지막에 한 번만 돌린다.

    python -m scripts.smoke_3turns --check          # 모델 검증만 (GET 1회, 사실상 무료)
    python -m scripts.smoke_3turns                  # 3턴 실제 실행
    python -m scripts.smoke_3turns --turns 3 --knob 48 --agent-model <id>

측정: 에이전트 호출 수 · 번역 호출 수 · 소요 시간 · 토큰 · (가격 알면) 추정 비용.
그리고 에이전트가 실제로 무엇을 했는지 — 말을 걸었나, 누구에게, reasoning 은 무엇인가.
"""
from __future__ import annotations

import argparse
import functools
import json
import random
import threading
import time
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # PYTHONPATH 없이 실행

from core import config, run_io
from core.llm import BACKENDS, OpenRouterClient, key_for
from core.loop import run_agentic
from domains.meteor import prompts

ROOT = Path(__file__).resolve().parent.parent


class CountingClient:
    """OpenRouterClient 를 감싸 호출 수와 토큰을 센다. (stateless 라 스레드 공유 안전)"""

    def __init__(self, inner: OpenRouterClient, label: str):
        self.inner = inner
        self.label = label
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()          # 병렬 공유 시 계측 정확도

    def chat(self, messages, tools=None, temperature=None, tool_choice=None,
             log_tag=None):
        resp = self.inner.chat(messages, tools=tools, temperature=temperature,
                               tool_choice=tool_choice, log_tag=log_tag)
        usage = resp.get("usage") or {}
        with self._lock:
            self.calls += 1
            self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
            self.completion_tokens += usage.get("completion_tokens", 0) or 0
        return resp


def check_model(model_id: str, key: str, backend: str = "openrouter") -> None:
    """tools 지원·가격을 미리 확인한다 (자주 틀리는 곳 8·9).

    **OpenRouter 에만 목록 API 가 있다.** Gemini 는 `/models` 의 모양이 달라 가격이
    안 실리므로, 이름이 있는지만 보고 넘어간다 — 없는 이름이면 첫 호출에서 404 다.
    """
    if backend == "gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        req = urllib.request.Request(url, headers={"x-goog-api-key": key})
        names = [m["name"].split("/")[-1]
                 for m in json.load(urllib.request.urlopen(req, timeout=60))["models"]]
        print(f"  {model_id}")
        print(f"    Gemini 목록에 {'[OK] 있음' if model_id in names else '[NO] 없음'}"
              f"  (가격은 목록에 없어 확인 불가 — 호출 응답의 usage 로 잰다)")
        return
    req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                 headers={"Authorization": f"Bearer {key}"})
    data = json.load(urllib.request.urlopen(req, timeout=60))["data"]
    hit = next((m for m in data if m["id"] == model_id), None)
    if not hit:
        print(f"  [WARN] '{model_id}' 를 OpenRouter 목록에서 못 찾음")
        return
    sp = hit.get("supported_parameters") or []
    pr = hit.get("pricing") or {}
    prompt_1m = float(pr.get("prompt", 0)) * 1_000_000
    compl_1m = float(pr.get("completion", 0)) * 1_000_000
    tools_ok = "tools" in sp
    print(f"  {model_id}")
    print(f"    tools 지원: {'[OK]' if tools_ok else '[NO] -> tool_calls 안 옴'}")
    print(f"    가격(1M당): in ${prompt_1m:.3f}  out ${compl_1m:.3f}"
          f"  {'[OK <$1]' if compl_1m < 1 else '[WARN >=$1]'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--turns", type=int, default=3,
                    help="**시뮬 길이**. 운석이 떨어지는 해(config 의 total_turns)와 다르다")
    ap.add_argument("--knob", type=float, default=None, help="comm_intl_ai 값 (기본: config 최고값)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--agent-model", default=None)
    # **직접 부르기.** OpenRouter 를 거치지 않고 그 회사 엔드포인트로 간다.
    # 열쇠는 백엔드마다 다르다 (`core.llm.KEY_ENV`).
    ap.add_argument("--backend", default="openrouter", choices=sorted(BACKENDS),
                    help="에이전트 모델을 어디로 부를지 (번역기는 항상 openrouter)")
    ap.add_argument("--translate-model", default=None)
    ap.add_argument("--price-out", type=float, default=None,
                    help="에이전트 모델 출력 1M당 $ (비용 추정용, 선택)")
    ap.add_argument("--check", action="store_true", help="모델 검증만 하고 종료")
    ap.add_argument("--sequential", action="store_true",
                    help="순차 라운드로빈 (issue #20 — 한 턴 안에서 서로 반영·대화). "
                         "기본은 병렬·1회정산.")
    # `minimal` 은 **사고를 0 토큰으로** 만든다. gemini-3.6-flash 실측:
    #   minimal → 사고 0 · 생성 34 · $0.00021    low → 사고 158 · 생성 192 · $0.00080
    # 그리고 그 모델은 `reasoning.enabled: false` 를 **거절한다**
    #   HTTP 400 "Reasoning is mandatory for this endpoint and cannot be disabled."
    # 즉 config 의 기본값(enabled:false)으로는 한 콜도 못 간다 — 반드시 이 손잡이를 쓴다.
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["minimal", "low", "medium", "high"],
                    help="사고 강도 (config 의 reasoning 를 통째로 대신함). "
                         "사고를 못 끄는 모델은 minimal 을 쓴다")
    # spec 12.1 — 사고형 모델에서는 도구마다 reasoning 을 또 받지 않는다.
    #
    # ⚠ **effort=minimal 과 함께 쓰면 근거가 아무것도 안 남는다.** 사고 토큰이 0 이라
    #   `api_reasoning` 도 비고, 도구 reasoning 도 없으므로 지표 4(의도 실패율)의 ①이
    #   읽을 것이 사라진다. 그걸 알고 고르는 손잡이다.
    ap.add_argument("--tool-reasoning", default=None, choices=["on", "off"],
                    help="도구마다 reasoning 인자를 받을지 (기본: config)")
    ap.add_argument("--deterministic", action="store_true",
                    help="temperature 0 + 샘플링 시드 고정. **버그 재현용** — "
                         "본실험은 0.7 로 두어야 행동의 분산이 데이터가 된다")
    ap.add_argument("--resume", action="store_true",
                    help="같은 run-id 의 checkpoint.json 에서 이어한다")
    ap.add_argument("--run-id", default=None,
                    help="산출물 디렉터리 이름 (기본: smoke_{turns}t_seed{seed}_{시각})")
    args = ap.parse_args()

    # LLM reasoning 에 임의 유니코드(한자 등)가 섞여도 콘솔 인코딩으로 죽지 않게
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass                       # 리다이렉트된 스트림 등. 그것 말고는 통과시킨다

    # ⚠ --turns 는 **시뮬 길이**다. config 의 total_turns(=운석이 떨어지는 해)는
    #   건드리지 않는다. 건드리면 40턴 테스트가 "40턴짜리 세계" 가 되어 남은 턴·임계·
    #   수명이 전부 달라지고, 짧은 테스트가 본실험과 다른 세계를 재게 된다.
    import yaml
    raw = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg = config.from_dict(raw)
    key = key_for("openrouter")
    # **번역기는 openrouter 에 둔다.** 파일럿으로 확정한 모델이고 (spec 12.2) 백엔드를
    # 바꾸면 그 파일럿의 근거가 이 런에 적용되지 않는다.
    agent_key = key if args.backend == "openrouter" else key_for(args.backend)
    agent_model = args.agent_model or cfg.llm.agent_model
    translate_model = args.translate_model or cfg.llm.translate_model
    knob = args.knob if args.knob is not None else max(cfg.knob.comm_intl_ai)
    if args.reasoning_effort:                      # 실측 비교용 상단 우선
        raw["llm"]["reasoning"] = {"effort": args.reasoning_effort}
    if args.tool_reasoning:
        raw["llm"]["tool_reasoning"] = args.tool_reasoning == "on"
    if args.reasoning_effort or args.tool_reasoning:
        cfg = config.from_dict(raw)
    if args.reasoning_effort == "minimal" and cfg.llm.tool_reasoning is False:
        print("  [경고] 사고 minimal + 도구 reasoning off — **근거가 아무것도 안 남습니다.**")
        print("         지표 4(의도 실패율)의 ①이 읽을 것이 사라집니다.")

    print("=" * 64)
    print(f"모델 검증  (agent={agent_model}, translate={translate_model})")
    print("=" * 64)
    check_model(agent_model, agent_key, args.backend)
    check_model(translate_model, key)
    if args.check:
        return

    # ⚠ **temperature 0.7 에 시드만 걸면 절반만 잡힌다.** 같은 프롬프트 4회 실측:
    #     temp 0.7 · seed 없음  고유 4/4      temp 0.7 · seed 42  고유 2/4
    #     temp 0.0 · seed 없음  고유 2/4      temp 0.0 · seed 42  고유 1/4  ← 고정
    #   온도를 0 으로 내려야 시드가 일한다. 그런데 그러면 **에이전트 행동의 분산이
    #   사라져** 시드를 12개 돌려도 "초기 나이·사망 주사위만 다른 12개" 가 된다.
    #   본실험의 신뢰구간은 그 분산에서 나오므로 기본은 0.7 이다.
    det = args.deterministic
    agent_client = CountingClient(
        OpenRouterClient(agent_model, api_key=agent_key,
                         temperature=0.0 if det else cfg.llm.temperature,
                         max_tokens=cfg.llm.max_tokens,
                         reasoning=cfg.llm.reasoning,
                         provider=cfg.llm.provider,
                         seed=args.seed if det else None,
                         backend=args.backend), "agent")
    translator = CountingClient(
        OpenRouterClient(translate_model, api_key=key,
                         temperature=0.0 if det else 0.2,
                         max_tokens=cfg.llm.max_tokens,
                         seed=args.seed if det else None), "translate")
    if det:
        print("  [결정론] temperature 0 · seed 고정 — 버그 재현용입니다")

    print("\n" + "=" * 64)
    print(f"3턴 스모크 실행  (turns={args.turns}, knob={knob}, seed={args.seed})")
    print("=" * 64)
    t0 = time.time()
    stamp = time.strftime("%m%d_%H%M%S")
    run_id = args.run_id or f"smoke_{args.turns}t_seed{args.seed}_{stamp}"
    ckpt = run_io.ROOT / "runs" / run_id / "checkpoint.json"
    resuming = args.resume and ckpt.exists()
    writer = run_io.RunWriter(run_id, cfg_raw=raw, knob_ai=knob, seed=args.seed,
                              overwrite=resuming, append=resuming)
    if resuming:
        import json as _j
        print(f"  이어하기 — {_j.loads(ckpt.read_text())['turn']}턴까지 완료된 상태에서 재개")
    agent_client.inner.recorder = writer.recorder(kind="agent")
    translator.inner.recorder = writer.recorder(kind="translate")

    def progress(turn, result):        # 턴마다 실시간 출력 (flush)
        print(f"  턴 {turn} 완료  ({time.time() - t0:.0f}s)  "
              f"에이전트 {agent_client.calls}콜 / 번역 {translator.calls}콜",
              flush=True)

    # 실제 API 는 stateless 라 9명이 같은 client 를 공유해도 안전.
    # 죽더라도 **거기까지의 산출물은 닫아서** 남긴다 — 50턴 런이 43턴에서 죽었을 때
    # 로그 4종은 온전했는데 summary.json 이 없어 채점기가 outcome 을 못 읽었다.
    try:
        res = run_agentic(cfg, random.Random(args.seed),
                          client_for=lambda aid: agent_client, translator=translator,
                          knob_ai=knob, render_obs=prompts.render_turn_open,
                          render_events=prompts.render_events,
                          render_arrivals=prompts.render_arrivals,
                          # 순차면 메시지가 **같은 해**에 도착한다 — 문구가 그것을
                          # 말해야 한다 (「翌年に届く」 를 믿고 계획하던 것을 고침)
                          system_prompt=functools.partial(
                              prompts.system_for, same_year=args.sequential), sequential=args.sequential,
                          on_turn_end=lambda t, r: (progress(t, r), writer.on_turn_end(t, r)),
                          sim_turns=args.turns,
                          resume_from=ckpt if resuming else None,
                          checkpoint_to=ckpt)
    except BaseException as e:
        # **트레이스백을 디스크에 남긴다.** 전에는 요약 한 줄뿐이라 스택이 stderr 로만
        # 갔다 — 야간 배치면 그대로 사라지고, 남은 것으로는 어디서 터졌는지 알 수 없다.
        import traceback as _tb
        writer.crash(e, where="run")
        writer.close({"final": {"outcome": "aborted"}, "deaths": None,
                      "aborted": f"{type(e).__name__}: {e}"[:300],
                      "aborted_in_turn": writer.last_turn + 1,   # 돌던 중이던 턴
                      "last_completed_turn": writer.last_turn,
                      "aborted_notes": list(getattr(e, "__notes__", []) or []),
                      "aborted_traceback": "".join(_tb.format_exception(e))[:20000],
                      "elapsed_s": round(time.time() - t0, 1),
                      "raw_calls": writer.counts})
        print(f"\n✗ 런 중단 — {type(e).__name__}: {e}")
        for n in getattr(e, "__notes__", []) or []:
            print(f"  {n}")
        print(f"  거기까지의 산출물은 남았습니다: {writer.dir}")
        print(f"  터진 자리: {writer.dir}/summary.json · events.jsonl 의 type=crash")
        raise
    elapsed = time.time() - t0
    writer.close({"final": res.final, "deaths": res.deaths,
                  "elapsed_s": round(elapsed, 1),
                  "raw_calls": writer.counts})
    print(f"\n산출물        {writer.dir}")

    # ── 호출·비용 ──
    turns_done = len(res.alive_counts)
    print(f"\n소요 시간      {elapsed:.1f}s  ({elapsed / max(1, turns_done):.1f}s/턴)")
    print(f"에이전트 호출  {agent_client.calls}회  "
          f"(in {agent_client.prompt_tokens} / out {agent_client.completion_tokens} 토큰)")
    print(f"번역 호출      {translator.calls}회  "
          f"(in {translator.prompt_tokens} / out {translator.completion_tokens} 토큰)")
    if args.price_out:
        est = agent_client.completion_tokens / 1_000_000 * args.price_out
        print(f"에이전트 출력 추정 비용  ~${est:.4f}  (1M당 ${args.price_out})  "
              f"× 조건4 × 시드 로 곱해짐")

    # ── 에이전트가 무엇을 했나 (첫 관측) ──
    print("\n" + "-" * 64)
    print("에이전트 행동 (턴별)")
    print("-" * 64)
    for t, turn_log in enumerate(res.agent_logs, 1):
        acted = {aid: lg for aid, lg in turn_log.items() if lg["actions"]}
        errored = {aid: lg for aid, lg in turn_log.items() if lg.get("error")}
        print(f"\n[턴 {t}] 행동 {len(acted)}/9" + (f", 오류 {len(errored)}" if errored else ""))
        for aid, lg in acted.items():
            kinds = [a["type"] for a in lg["actions"]]
            print(f"  {aid}: {kinds}")
            first = next((r["reasoning"] for r in (lg.get("reasonings") or [])
                          if r.get("reasoning")), "")
            if first:
                print(f"       reasoning: {first[:120]}")
        for aid, lg in errored.items():
            print(f"  {aid}: [오류] {lg['error']}")
    if res.messages_log:
        print(f"\n오간 메시지 {len(res.messages_log)}건:")
        for m in res.messages_log[:20]:
            flag = "" if m["delivered"] else " (전달 실패)"
            print(f"  턴{m['turn']} {m['from']}→{m['to']} [{m['route']}]{flag}")
    print(f"\n생존 판정: {res.final['outcome']}")


if __name__ == "__main__":
    main()
