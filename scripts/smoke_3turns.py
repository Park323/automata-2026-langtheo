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
import json
import random
import threading
import time
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # PYTHONPATH 없이 실행

from core import config, run_io
from core.llm import OpenRouterClient, load_key
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

    def chat(self, messages, tools=None, temperature=None):
        resp = self.inner.chat(messages, tools=tools, temperature=temperature)
        usage = resp.get("usage") or {}
        with self._lock:
            self.calls += 1
            self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
            self.completion_tokens += usage.get("completion_tokens", 0) or 0
        return resp


def check_model(model_id: str, key: str) -> None:
    """OpenRouter /models 에서 tools 지원·가격을 확인한다 (자주 틀리는 곳 8·9)."""
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
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--knob", type=float, default=None, help="comm_intl_ai 값 (기본: config 최고값)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--agent-model", default=None)
    ap.add_argument("--translate-model", default=None)
    ap.add_argument("--price-out", type=float, default=None,
                    help="에이전트 모델 출력 1M당 $ (비용 추정용, 선택)")
    ap.add_argument("--check", action="store_true", help="모델 검증만 하고 종료")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["low", "medium", "high"],
                    help="사고 강도 (config 의 reasoning.max_tokens 를 대신함)")
    ap.add_argument("--run-id", default=None,
                    help="산출물 디렉터리 이름 (기본: smoke_{turns}t_seed{seed}_{시각})")
    args = ap.parse_args()

    # LLM reasoning 에 임의 유니코드(한자 등)가 섞여도 콘솔 인코딩으로 죽지 않게
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # --turns 를 config 에 실제로 반영한다 (base.yaml 은 total_turns=50)
    import yaml
    raw = yaml.safe_load(open(args.config, encoding="utf-8"))
    raw["world"]["total_turns"] = args.turns
    cfg = config.from_dict(raw)
    key = load_key()
    agent_model = args.agent_model or cfg.llm.agent_model
    translate_model = args.translate_model or cfg.llm.translate_model
    knob = args.knob if args.knob is not None else max(cfg.knob.comm_intl_ai)
    if args.reasoning_effort:                      # 실측 비교용 상단 우선
        raw["llm"]["reasoning"] = {"effort": args.reasoning_effort}
        cfg = config.from_dict(raw)

    print("=" * 64)
    print(f"모델 검증  (agent={agent_model}, translate={translate_model})")
    print("=" * 64)
    check_model(agent_model, key)
    check_model(translate_model, key)
    if args.check:
        return

    agent_client = CountingClient(
        OpenRouterClient(agent_model, api_key=key, temperature=cfg.llm.temperature,
                         max_tokens=cfg.llm.max_tokens,
                         reasoning=cfg.llm.reasoning,
                         provider=cfg.llm.provider), "agent")
    translator = CountingClient(
        OpenRouterClient(translate_model, api_key=key, temperature=0.2,
                         max_tokens=cfg.llm.max_tokens), "translate")

    print("\n" + "=" * 64)
    print(f"3턴 스모크 실행  (turns={args.turns}, knob={knob}, seed={args.seed})")
    print("=" * 64)
    t0 = time.time()
    stamp = time.strftime("%m%d_%H%M%S")
    run_id = args.run_id or f"smoke_{args.turns}t_seed{args.seed}_{stamp}"
    writer = run_io.RunWriter(run_id, cfg_raw=raw, knob_ai=knob, seed=args.seed)
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
                          knob_ai=knob, render_obs=prompts.render_observation,
                          system_prompt=prompts.system_for, parallel=not args.sequential,
                          on_turn_end=lambda t, r: (progress(t, r), writer.on_turn_end(t, r)))
    except BaseException as e:
        writer.close({"final": {"outcome": "aborted"}, "deaths": None,
                      "aborted": f"{type(e).__name__}: {e}"[:300],
                      "elapsed_s": round(time.time() - t0, 1),
                      "raw_calls": writer.counts})
        print(f"\n✗ 런 중단 — {type(e).__name__}: {e}")
        print(f"  거기까지의 산출물은 남았습니다: {writer.dir}")
        raise
    elapsed = time.time() - t0
    writer.close({"final": res.final, "deaths": res.deaths,
                  "elapsed_s": round(elapsed, 1),
                  "raw_calls": writer.counts})
    print(f"\n산출물        {writer.dir}")

    # ── 호출·비용 ──
    print(f"\n소요 시간      {elapsed:.1f}s  ({elapsed / args.turns:.1f}s/턴)")
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
