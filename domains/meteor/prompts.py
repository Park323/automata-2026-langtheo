"""프롬프트. spec 4.1 의 관측 블록을 그대로 렌더링한다.

절대 넣지 않는 것 (4.1):
  재앙까지 남은 턴 · success_prob · 나이→사망확률 · wellness→수명 · 타국 진척/예산/국토/언어능력
  · 다른 에이전트의 내심 · 목적함수("살아남아라" 등)
비용표는 반드시 보여준다 (비용을 모르면 선택이 불가능).
"""
from __future__ import annotations

from core.agent_loop import learn_cost

LANG_KO = {"ja": "일본어", "zh": "중국어", "fr": "프랑스어"}

# 세계 규칙만. 목표는 주지 않는다 (spec 4.1). 운석의 존재는 사실이라 진술 가능.
SYSTEM = """당신은 이 세계에 사는 한 사람입니다. 아래는 세계의 규칙입니다 — 지시가 아니라 사실입니다.

- 큰 운석이 이 행성으로 접근하고 있습니다.
- 세 나라(A·B·C)가 있고 서로 다른 언어를 씁니다. 당신은 자국어만 읽을 수 있습니다(배우면 늘어납니다).
- 각 나라는 국토 한 칸에 '요격기' 또는 '벙커' 중 하나를 지을 수 있습니다(공존 불가).
  요격기는 부지별로 독립이며, 한 부지가 충분히 완성되면 전 인류가 삽니다.
  벙커는 판 깊이에 따라 자국만 확률적으로 삽니다.
- 소통은 1:1이고 다음 턴에 도착합니다. 국제 소통은 원문 직통(상대가 읽을 줄 알아야 전달)
  또는 AI 번역(항상 전달되나 왜곡)을 고를 수 있습니다.
- 투자·소통·학습에는 예산과 행동력(AP)이 듭니다.

무엇을 할지는 전적으로 당신이 정합니다. 제공된 도구로 행동하세요."""


def render_costs(world, agent, cfg, knob_ai: float) -> str:
    lines = ["행동 비용",
             f"  speak (자국)       {cfg.costs.comm_domestic:g}",
             f"  speak (국제·원문)   {cfg.costs.comm_intl_learner:g}   상대가 못 읽으면 전달되지 않음. 비용은 그대로 나감",
             f"  speak (국제·AI)    {knob_ai:g}",
             f"  ask               {cfg.costs.ask_clarification:g}   + 위 경로 비용"]
    for c in world.countries.values():
        if c.id != agent.country:
            cost, _ = learn_cost(agent, c.id, world, cfg)
            note = ""
            if cost < cfg.costs.learn_base:
                note = "   우리 나라에 구사자가 있어 쌉니다" if cost == cfg.costs.learn_base * 0.5 else "   할인 적용"
            lines.append(f"  learn ({c.id}국 말)   {cost:g}{note}")
    lines.append(f"  propose_vote      {cfg.costs.propose_vote:g}")
    lines.append("  invest            지정한 만큼")
    lines.append("  procreate         0    아이를 남기고 당신은 죽습니다")
    return "\n".join(lines)


def render_inbox(inbox: list[dict]) -> str:
    if not inbox:
        return "이번 턴 도착한 메시지: 없음"
    out = ["이번 턴 도착한 메시지:"]
    for i, m in enumerate(inbox, 1):
        mid = m.get("msg_id", i)
        if m.get("unreadable"):
            out.append(f"  [{mid}] {m['from']}로부터 — 읽을 수 없는 메시지가 도착했습니다")
            continue
        label = f" {m['label']}" if m.get("label") else ""
        out.append(f"  [{mid}] {m['from']}로부터{label}")
        out.append(f'      "{m.get("text", "")}"')
        if m.get("original"):
            out.append(f'      [원문] "{m["original"]}"')
    return "\n".join(out)


def render_observation(world, agent, cfg, knob_ai: float,
                       inbox: list[dict] | None = None,
                       income_this_turn: float | None = None) -> str:
    """spec 4.1 관측 블록."""
    c = world.countries[agent.country]
    land = {None: "미정", "interceptor": "요격기", "bunker": "벙커"}[c.land]
    mult = c.multiplier(cfg)
    cap = cfg.length.message_max_chars[agent.native_lang]
    testaments = world.testaments.get(agent.id, [])

    parts = [
        f"당신: {agent.id} (국가 {agent.country})",
        f"읽을 수 있는 언어: {', '.join(sorted(agent.known_langs))}",
        f"예산: {agent.budget:.0f}",
        f"나이: {agent.age}턴",
        "",
        f"자국 국토: {land}",
        f"자국 진척: {c.progress:.0f}",
        f"자국 생산 배수: {mult:.2f}",
        "",
        "지금까지 알아낸 것 (누가 어떤 언어를 읽을 수 있는지)",
        "  아직 아는 것이 없습니다",     # TODO(과제3): 대화·유언으로 알아낸 타인 언어능력 이력
        "",
        f"이번 턴 수입: +{(income_this_turn if income_this_turn is not None else cfg.income.per_turn * mult):.0f} (자국 생산 배수 반영)",
        "예산이 허락하는 한 여러 행동을 할 수 있습니다. 메시지는 한 턴에 3건까지.",
        "",
        render_costs(world, agent, cfg, knob_ai),
        "",
        "invest 효과",
        "  wellness   내 수명이 늘어납니다 (얼마나인지는 알 수 없습니다)",
        "  national   자국 생산 배수가 오릅니다. 그 나라 사람 전원이 받습니다",
        "  facility   시설 진척에 기여합니다. 자국 또는 지정한 나라",
        "",
        f"메시지는 {cap}자까지 전달됩니다. 그 이후는 전달되지 않습니다.",
        "",
    ]
    if testaments:
        parts.append("앞사람이 남긴 말:")
        for t in testaments:
            parts.append(f'  "{t}"')
        parts.append("")
    parts.append(render_inbox(inbox or []))
    return "\n".join(parts)
