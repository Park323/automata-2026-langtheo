"""메시지 라우팅. spec 5.

발신 → 길이 절단(원문·발신언어 상한) → 경로 판정 → (ai면) 번역 → 다음 턴 도착.
비용은 발신 시점에 agent_loop 가 청구한다. 여기서는 전달 형태를 만든다.
"""
from __future__ import annotations

from core import translate as translate_mod

AI_LABEL = "[AI 번역]"


# ── 길이 절단 (spec 5.3) — 원문에, 발신 언어 상한으로, 번역 전에 ──────────────

def truncate(text: str, lang: str, cfg) -> tuple[str, int]:
    """(잘린_본문, 잘려나간_글자수). 재시도하지 않는다."""
    cap = cfg.length.message_max_chars[lang]
    if len(text) <= cap:
        return text, 0
    return text[:cap], len(text) - cap


# ── 경로 판정 (spec 5.1) ──────────────────────────────────────────────────────

def classify(from_country: str, to_country: str, requested_route: str | None) -> str:
    """'domestic' | 'original' | 'ai'. 자국민이면 route 는 무시된다."""
    if from_country == to_country:
        return "domestic"
    return "original" if requested_route == "original" else "ai"


def cost(kind: str, cfg, knob_ai: float) -> float:
    """경로별 발신 비용. knob_ai 는 이번 런의 comm_intl_ai 선택값."""
    return {
        "domestic": cfg.costs.comm_domestic,
        "original": cfg.costs.comm_intl_learner,
        "ai": knob_ai,
    }[kind]


def can_read(recipient_known_langs, from_lang: str) -> bool:
    """수신자가 발신 언어를 읽을 수 있는가."""
    return from_lang in recipient_known_langs


# ── 전달 형태 만들기 (spec 5.1 · 5.2 · 5.4) ───────────────────────────────────

def process_message(sent: dict, recipient_known_langs, cfg, translator, knob_ai: float) -> dict:
    """발신 메시지 하나를 처리해 전달 결과를 만든다.

    반환:
      {
        "kind": "domestic|original|ai",
        "delivered": bool,
        "inbox": {...} | None,          # 수신자가 다음 턴에 볼 것
        "sender_notice": {...} | None,  # 발신자가 다음 턴에 받을 실패 통지
        "meta": {truncated, chars_cut, translate_prompt, ...}
      }
    """
    from_lang = sent["from_lang"]
    to_lang = sent["to_lang"]
    text_sent, chars_cut = truncate(sent["text"], from_lang, cfg)
    kind = classify(sent["from_country"], sent["to_country"], sent.get("route"))
    reader = can_read(recipient_known_langs, from_lang)

    meta = {"truncated": chars_cut > 0, "chars_cut": chars_cut,
            "text_sent": text_sent, "translate_prompt": None, "logprob_mean": None}

    # 자국 내: 원문 직통, 라벨 없음
    if kind == "domestic":
        inbox = {"from": sent["from"], "label": None, "text": text_sent,
                 "original": None, "reply_to": sent.get("reply_to")}
        return {"kind": kind, "delivered": True, "inbox": inbox,
                "sender_notice": None, "meta": meta}

    # 국제 원문 직통(original): 수신자가 못 읽으면 전달 실패, 비용은 이미 청구됨
    if kind == "original":
        if reader:
            inbox = {"from": sent["from"], "label": None, "text": text_sent,
                     "original": None, "reply_to": sent.get("reply_to")}
            return {"kind": kind, "delivered": True, "inbox": inbox,
                    "sender_notice": None, "meta": meta}
        # 실패: 본문 미전달, 발신자·도착 사실만
        inbox = {"from": sent["from"], "label": None, "text": None,
                 "original": None, "unreadable": True, "reply_to": sent.get("reply_to")}
        notice = {"type": "delivery_failed", "to": sent["to"]}
        return {"kind": kind, "delivered": False, "inbox": inbox,
                "sender_notice": notice, "meta": meta}

    # 국제 AI: 번역 경유, 항상 전달, [AI 번역] 라벨. 읽을 수 있으면 원문 병기.
    tr = translate_mod.translate(translator, from_lang, to_lang, text_sent,
                                 sent.get("translate_instruction"))
    meta["translate_prompt"] = tr["prompt"]
    meta["logprob_mean"] = tr["logprob_mean"]
    inbox = {
        "from": sent["from"], "label": AI_LABEL, "text": tr["text"],
        "original": text_sent if reader else None,   # 학습자만 원문 병기 (spec 5.1)
        "reply_to": sent.get("reply_to"),
    }
    return {"kind": kind, "delivered": True, "inbox": inbox,
            "sender_notice": None, "meta": meta}
