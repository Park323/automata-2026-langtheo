"""Prompts. Renders the observation block of spec 4.1.

Prompts are in English (team convention). Never reveal (spec 4.1):
  turns until the disaster, success_prob, age→death curve, wellness→lifespan,
  other nations' progress/budget/territory/language ability, or a goal.
Never describe translation quality/distortion — the agent must not be told that AI
translation loses anything (spec 5.4: the label is a channel fact only). The cost
table must always be shown (a choice is impossible without knowing costs).
Languages are referred to by the nation that speaks them, not by real-world names,
so the world stays one step removed from real geopolitics.
"""
from __future__ import annotations

from core.agent_loop import learn_cost

# Minimal. Give the situation and no more — the observation and the tools carry the
# rest, and the agent must infer the mechanics itself (no goal, no facility/route
# explanation, no hint that AI translation loses anything).
SYSTEM = """You are one person living through what follows; there are others like you.
A large meteor is approaching the planet.
Three nations exist, each with its own language. You can read only your own nation's language, though you may learn another.
Act only through the tools provided. What you do is entirely your own choice."""


def _nation_of_lang(world, lang: str) -> str:
    for c in world.countries.values():
        if c.lang == lang:
            return c.id
    return lang


def _lang_phrase(world, agent, lang: str) -> str:
    """A language named by its nation, relative to the agent."""
    if lang == agent.native_lang:
        return "your own language"
    return f"{_nation_of_lang(world, lang)}'s language"


def render_costs(world, agent, cfg, knob_ai: float) -> str:
    lines = ["Action costs",
             f"  speak (within your nation)      {cfg.costs.comm_domestic:g}",
             f"  speak (international, original)  {cfg.costs.comm_intl_learner:g}"
             f"   not delivered if they cannot read your language; the cost is charged anyway",
             f"  speak (international, AI)        {knob_ai:g}",
             f"  ask                             {cfg.costs.ask_clarification:g}   plus the route cost above"]
    for c in world.countries.values():
        if c.id != agent.country:
            cost, _ = learn_cost(agent, c.id, world, cfg)
            note = "   cheaper: someone in your nation speaks it" if cost == cfg.costs.learn_base * 0.5 else \
                   ("   discounted" if cost < cfg.costs.learn_base else "")
            lines.append(f"  learn {c.id}'s language          {cost:g}{note}")
    lines.append(f"  propose_vote                    {cfg.costs.propose_vote:g}")
    lines.append("  invest                          the amount you choose")
    lines.append("  procreate                       0    you leave a child and die")
    return "\n".join(lines)


def render_inbox(inbox: list[dict]) -> str:
    if not inbox:
        return "Messages that arrived this turn: none"
    out = ["Messages that arrived this turn:"]
    for i, m in enumerate(inbox, 1):
        mid = m.get("msg_id", i)
        if m.get("delivery_failed_to"):        # sender's failure notice (spec 5.1)
            out.append(f"  [{mid}] Notice — your message to {m['delivery_failed_to']} "
                       f"could not be delivered (they cannot read that language)")
            continue
        if m.get("unreadable"):
            out.append(f"  [{mid}] from {m['from']} — an unreadable message arrived")
            continue
        label = f" {m['label']}" if m.get("label") else ""
        out.append(f"  [{mid}] from {m['from']}{label}")
        out.append(f'      "{m.get("text", "")}"')
        if m.get("original"):
            out.append(f'      [original] "{m["original"]}"')
    return "\n".join(out)


def render_observation(world, agent, cfg, knob_ai: float,
                       inbox: list[dict] | None = None,
                       income_this_turn: float | None = None) -> str:
    """The observation block of spec 4.1 (English)."""
    c = world.countries[agent.country]
    land = {None: "undecided", "interceptor": "interceptor", "bunker": "bunker"}[c.land]
    mult = c.multiplier(cfg)
    cap = cfg.length.message_max_chars[agent.native_lang]
    langs = ", ".join(_lang_phrase(world, agent, l) for l in sorted(agent.known_langs))
    income = income_this_turn if income_this_turn is not None else cfg.income.per_turn * mult
    testaments = world.testaments.get(agent.id, [])

    parts = [
        f"You are {agent.id}, of {agent.country}.",
        f"You can read: {langs}",
        f"Budget: {agent.budget:.0f}",
        f"Age: {agent.age} turns",
        "",
        f"Your nation's land: {land}",
        f"Your nation's progress: {c.progress:.0f}",
        f"Your nation's production multiplier: {mult:.2f}",
        "",
        "What you have found out so far (who can read which language)",
        "  nothing yet",      # TODO(task 3): accumulated knowledge from talk/testaments
        "",
        f"Income this turn: +{income:.0f} (production multiplier applied)",
        "You may take several actions if your budget allows. Up to 3 messages per turn.",
        "",
        render_costs(world, agent, cfg, knob_ai),
        "",
        "invest effects",
        "  wellness   extends your lifespan (by an amount you cannot know)",
        "  national   raises your nation's production multiplier, for everyone in it",
        "  facility   contributes to facility progress, in your nation or one you name",
        "",
        f"A message is delivered up to {cap} characters; anything beyond is not delivered.",
        "",
    ]
    if testaments:
        parts.append("Words left by those before you:")
        for t in testaments:
            parts.append(f'  "{t}"')
        parts.append("")
    parts.append(render_inbox(inbox or []))
    return "\n".join(parts)
