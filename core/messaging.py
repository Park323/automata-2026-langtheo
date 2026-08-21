"""메시지 라우팅. spec 5.

발신 → 길이 절단(원문·발신언어 상한) → 경로 판정 → (ai면) 번역 → 다음 턴 도착.
비용은 발신 시점에 agent_loop 가 청구한다. 여기서는 전달 형태를 만든다.
"""
from __future__ import annotations

from core import translate as translate_mod
from core.llm import LLMCallError

AI_LABEL = "[AI translation]"
# 통역이 끼지 않았는데 뜻이 닿았다는 표시. `render_inbox` 가 수신자 언어로 옮긴다.
# AI 라벨은 영어 그대로 둔다 — 기계가 낀 자리를 이물감 있게 두는 편이 낫고,
# 바꾸면 AI 경로의 자극이 달라져 오늘 런들과의 4a 비교가 흔들린다.
# **원문 직통은 두 가지 다른 사실이다** (8/21). 하나로 묶어 「통역 없이 통했다」 라고
# 적었더니, 못 읽는 언어를 전달하면서 통했다고 말하는 일이 생겼다 — 여섯 런에서 `writer`
# 덕으로 전달된 16건이 **전부** 그랬고, 한 에이전트가
#
#     "あなたのメッセージが分かりません。日本語で説明してください。"
#
# 라고 되물었다. 우리 라벨이 거짓말을 한 것이고, 그 되물음이 **정확한 반응**이었다.
DIRECT_LABEL = "[direct]"                  # 하위 호환 — 읽는 쪽 덕
DIRECT_READ_LABEL = "[direct:read]"        # 내가 그 말을 읽는다
DIRECT_WRITE_LABEL = "[direct:write]"      # 나는 못 읽지만 상대가 내 말을 다룬다


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


def direct_works(sender_known_langs, recipient_known_langs,
                 from_lang: str, to_lang: str) -> tuple[bool, str]:
    """원문 직통이 통하는가, 그리고 **누구의 학습 덕인가**.

    학습은 **읽기와 쓰기 둘 다**다 (8/17 개정). 그래서 길이 둘이다.

        수신자가 발신 언어를 안다  →  받는 쪽이 원문을 읽는다      "reader"
        발신자가 수신 언어를 안다  →  보내는 쪽이 그 말로 쓴다     "writer"

    구현은 어느 쪽이든 **원문을 그대로 전달**한다. 모델이 multilingual 이라 그대로
    이해한다 — 발신자에게 수신 언어로 다시 쓰게 만들면 프롬프트 언어 위생(모국어 강제)이
    깨지고, 지표 7 의 발신 언어 사전도 못 쓰게 된다.

    전에는 "reader" 만 통했다. 그래서 **초기화로 심은 이중언어자가 받는 데만 쓸모가
    있었고**, 자기가 아는 말의 나라에 보낼 때도 24원짜리 AI 를 타야 했다.
    """
    if from_lang in recipient_known_langs:
        return True, "reader"
    if to_lang in sender_known_langs:
        return True, "writer"
    return False, ""


# ── 전달 형태 만들기 (spec 5.1 · 5.2 · 5.4) ───────────────────────────────────

def process_message(sent: dict, recipient_known_langs, cfg, translator, knob_ai: float,
                    sender_known_langs=frozenset(), log_tag: dict | None = None) -> dict:
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
    direct, direct_by = direct_works(sender_known_langs, recipient_known_langs,
                                     from_lang, to_lang)

    # spec 6.1 스키마 전체. text_delivered 가 없으면 소실률·생성률을 아예 못 낸다.
    meta = {
        "src_lang": from_lang, "dst_lang": to_lang,
        "text_written": sent["text"],          # 절단 전. 발신자가 실제로 쓴 것
        "text_sent": text_sent,                # 절단 후. 번역 입력이자 채점 기준선
        "text_delivered": None,                # 수신자가 실제로 본 것 (아래에서 채움)
        "translate_instruction": sent.get("translate_instruction"),
        "len_written": len(sent["text"]), "len_limit": cfg.length.message_max_chars[from_lang],
        "truncated": chars_cut > 0, "chars_cut": chars_cut,
        "reader": reader,                      # 수신자가 발신 언어를 읽는가
        "direct_ok": direct,                   # 원문 직통이 통하는가 (읽기 OR 쓰기)
        "direct_by": direct_by or None,        # "reader" | "writer" — 누구의 학습 덕인가
        # 도착한 글이 실제로 무슨 언어인가. **번역을 안 탄 경로는 발신 언어 그대로다.**
        # 이걸 dst_lang 으로 채점하면 같은 글을 다른 언어 사전으로 세어 화용 표지가
        # 통째로 "소실" 로 잡힌다 (지표 7).
        "delivered_lang": from_lang,
        "translate_prompt": None, "logprob_mean": None,
    }

    # 자국 내: 원문 직통, 라벨 없음
    if kind == "domestic":
        meta["text_delivered"] = text_sent          # 번역 없음 — 기저선 (지표 4c)
        inbox = {"from": sent["from"], "label": None, "text": text_sent,
                 "original": None}
        return {"kind": kind, "delivered": True, "inbox": inbox,
                "sender_notice": None, "meta": meta}

    # 국제 원문 직통(original): 아무도 그 언어를 다루지 못하면 전달 실패. 비용은 청구됨
    if kind == "original":
        if direct:
            meta["text_delivered"] = text_sent      # 원문 그대로 (지표 4d)
            # 어느 쪽 덕인지 라벨이 데려간다. 읽는 쪽 덕이면 「그대로 읽었다」,
            # 쓰는 쪽 덕이면 「나는 못 읽지만 상대가 내 말을 다룬다」 — 사실이 다르다.
            lbl = (DIRECT_READ_LABEL if direct_by == "reader" else DIRECT_WRITE_LABEL)
            inbox = {"from": sent["from"], "label": lbl, "text": text_sent,
                     "original": None}
            return {"kind": kind, "delivered": True, "inbox": inbox,
                    "sender_notice": None, "meta": meta}
        # 실패: 본문 미전달, 발신자·도착 사실만
        inbox = {"from": sent["from"], "label": None, "text": None,
                 "original": None, "unreadable": True}
        # **원인을 붙인다.** 이 경로의 실패는 세계의 사실이다 — 내가 그 나라 말을
        # 모르고 상대도 내 말을 못 읽었다. route=original 의 도박이 정보를 주는 지점이다.
        notice = {"type": "delivery_failed", "to": sent["to"], "reason": "unreadable"}
        return {"kind": kind, "delivered": False, "inbox": inbox,
                "sender_notice": notice, "meta": meta}

    # 국제 AI: 번역 경유, 항상 전달, [AI 번역] 라벨. 읽을 수 있으면 원문 병기.
    #
    # ⚠ 번역이 실패하면 **런 전체를 죽이지 않고** 미전달로 떨어뜨린다. 50턴 실측에서
    #   번역 호출 하나가 재시도를 소진해 2.5시간짜리 런이 통째로 날아갔다.
    #   에이전트 호출은 실패를 잡아 lg["error"] 로 남기는데 여기만 무방비였다.
    #
    #   `translate_failed` 를 따로 두는 이유 — 이건 **엔진 장애지 세계의 사건이 아니다.**
    #   route=original 의 미전달(지표 9)과 섞이면 "읽을 수 없어서 못 받았다" 로
    #   오독된다. 조건별로 빈도가 다르면 4a 도 오염되므로 반드시 따로 센다.
    try:
        tr = translate_mod.translate(translator, from_lang, to_lang, text_sent,
                                     sent.get("translate_instruction"),
                                     log_tag={"from": sent.get("from"),
                                              "to": sent.get("to"),
                                              # 바깥 태그가 마지막이다 — msg_id 는
                                              # 루프가 쥐고 있고 sent 에는 없다
                                              **(log_tag or {})})
    except LLMCallError as e:
        # **`except Exception` 이었다.** 런이 안 죽는 것이 목적이었는데 그 그물이 우리
        # 코드의 버그까지 삼켰다 — `LANG_NAME[dst_lang]` 의 KeyError, 응답 모양이 바뀐
        # TypeError 가 "번역 실패" 통계 한 줄로 묻히고 크래시로 드러나지 않는다.
        #
        # **런이 터지더라도 버그는 잡아야 한다.** 그래서 경계가 선언한 실패
        # (`LLMCallError`)만 잡고 나머지는 통과시킨다.
        meta["translate_failed"] = f"{type(e).__name__}: {e}"[:200]
        # **원인을 언어라고 말하면 안 된다.** 엔진 장애를 「상대가 그 언어를 읽지
        # 못한다」 로 통지하고 있었다 — 상대의 언어 능력과 아무 상관 없는 일인데
        # 그것을 언어 사실로 심는다. 이 실험의 핵심 변수(누가 무엇을 읽는가)를
        # 에이전트의 머릿속에서 오염시킨다.
        #
        # 그래서 원인을 붙이지 않는다. 「닿지 않았다」 만 사실이다.
        #
        # 수신자에게는 **아무것도 보내지 않는다.** 「읽을 수 없는 메시지가 왔다」 도
        # 같은 거짓이다 — ai 경로였으므로 엔진이 살아 있었다면 읽을 수 있게 도착했다.
        # 엔진 장애는 세계의 사건이 아니므로(지표 9 와 따로 세는 이유가 그것이다)
        # 세계에 흔적을 남기지 않는 편이 옳다.
        return {"kind": kind, "delivered": False, "inbox": None,
                "sender_notice": {"type": "delivery_failed", "to": sent["to"],
                                  "reason": "engine"},
                "meta": meta}
    meta["translate_prompt"] = tr["prompt"]
    meta["logprob_mean"] = tr["logprob_mean"]
    meta["text_delivered"] = tr["text"]             # 번역 경유 (지표 4a·6a·7)
    meta["delivered_lang"] = to_lang               # 이 경로만 언어가 바뀐다
    inbox = {
        "from": sent["from"], "label": AI_LABEL, "text": tr["text"],
        # 원문 병기 없음. **ai 를 고른 순간 원문은 볼 수 없다** — 병기하면 학습자가
        # 번역을 우회해 읽어버려 "AI 경로의 왜곡" 이 그 사람에게는 측정되지 않는다.
        "original": None,
    }
    return {"kind": kind, "delivered": True, "inbox": inbox,
            "sender_notice": None, "meta": meta}
