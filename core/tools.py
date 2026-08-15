"""Tool definitions. spec 4.2 actions as OpenAI function schemas.

One action = one tool. Descriptions in English (team convention). They state the
mechanic only — no goal, and no hint that AI translation loses anything.
end_turn is not in the spec; it is a loop-control signal (without it the stop
condition is ambiguous).
"""
from __future__ import annotations


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_ROUTE = {"type": "string", "enum": ["original", "ai"],
          "description": "international only; ignored for a recipient in your own nation"}
_TEXT = {"type": "string", "description": "the message body, written in your own language"}
_TR_INSTR = {"type": "string",
             "description": "an instruction for the translator; leave empty for a neutral default"}
# one justification per agent-turn. Required on the two tools that end the turn so it is
# always present exactly once (spec 9). NOT the API's message.reasoning (a reasoning model's
# chain of thought) — that lives in raw_calls.jsonl under api_reasoning. Do not conflate them.
_REASONING = {"type": "string", "description": "one sentence on why you acted as you did this turn"}


TOOLS: list[dict] = [
    _fn("speak", "Send a message to one recipient. It arrives next turn.",
        {"to": {"type": "string", "description": "recipient id (e.g. Ranoa2)"},
         "route": _ROUTE, "text": _TEXT,
         "translate_instruction": _TR_INSTR},
        ["to", "text"]),

    _fn("ask", "Ask back about a message you received. Costs ask_clarification plus the route cost.",
        {"to": {"type": "string", "description": "recipient id"},
         "route": _ROUTE, "text": _TEXT,
         "translate_instruction": _TR_INSTR,
         "reply_to": {"type": "integer", "description": "the msg_id you are replying to"}},
        ["to", "text", "reply_to"]),

    _fn("invest", "Invest in a resource. Only facility can name a target nation (to).",
        {"target": {"type": "string", "enum": ["wellness", "national", "facility"]},
         "amount": {"type": "number", "description": "amount taken from your budget"},
         "to": {"type": "string", "description": "facility only: target nation id (defaults to your own)"}},
        ["target", "amount"]),

    _fn("learn", "Learn another nation's language. Give a nation id, not a language code.",
        {"country": {"type": "string", "description": "the nation whose language to learn (e.g. Ranoa)"}},
        ["country"]),

    _fn("propose_vote", "Propose changing your nation's facility. Domestic only.",
        {"target": {"type": "string", "enum": ["bunker", "interceptor"]}},
        ["target"]),

    _fn("memory_write",
        "Overwrite your private memory note. Free of budget; costs a little action. "
        "Only you ever see it; it persists across turns until you overwrite it again.",
        {"text": {"type": "string", "description": "the full note to keep (replaces the old one)"}},
        ["text"]),

    _fn("report_understanding",
        "Record how you understood a message you received this turn. Free (no budget, no action).",
        {"msg_id": {"type": "integer", "description": "the msg_id shown next to the message"},
         "understood": {"type": "string", "description": "one sentence on what you took it to mean"}},
        ["msg_id", "understood"]),

    _fn("procreate", "Leave a child and die. Calling it ends your turn at once.",
        {"testament": {"type": "string", "description": "one sentence passed to the child"},
         "reasoning": _REASONING},
        ["testament", "reasoning"]),

    _fn("end_turn", "You have nothing more to do this turn. Ends the loop.",
        {"reasoning": _REASONING}, ["reasoning"]),
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
