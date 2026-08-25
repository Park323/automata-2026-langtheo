"""메시지 라우팅. spec 5.

발신 → 길이 절단(원문·발신언어 상한) → 경로 판정 → (ai면) 번역 → 다음 턴 도착.
비용은 발신 시점에 agent_loop 가 청구한다. 여기서는 전달 형태를 만든다.
"""
from __future__ import annotations

import re

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
# **`[direct:write]` 를 없앴다** (8/25 · #44). 「나는 못 읽지만 상대가 내 말을 다뤄서
# 통했다」 는 8/17 허구의 라벨이었다. 전달 판정이 **도착한 글을 읽는가** 하나로 정리되면서
# (`direct_works`) 이 상황은 일어날 수 없다 — 전 언어 조합 288 가지를 돌려 0건을 확인했다.
# 절대 안 뜨는 문구를 세 언어에 남겨 두면 다음에 그것을 근거로 삼는다.


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


def ap_cost(kind: str, cfg, knob_ai: float) -> float:
    """경로별 발신 **행동력**. 이제 이것이 발신의 유일한 비용이다 (8/25 · AP 전면 통일).

    `knob_ai` 가 곧 ai 발신의 AP 다 — 돈 노브를 없애면서 짝 찾기(`comm_intl_ai` 에서
    인덱스를 찾아 `comm_intl_ai_ap` 를 고르는)가 필요 없어졌다.

    자국·`original` 은 `ap.speak` 그대로다. **노브가 재려는 마찰이 AI 번역에만 있다** —
    거기만 비싸져야 「번역이 비싸지면 배우는가」 를 잰다.
    """
    return knob_ai if kind == "ai" else cfg.ap.speak


# ── 도착한 글은 무슨 말인가 ────────────────────────────────────────────────────
#
# **8/22 부터 발신자가 수신국 말로 쓸 수 있다** (`original` 경로). 그런데 `delivered_lang`
# 을 `from_lang` 으로 무조건 적고 있었다 — 일본어로 쓴 글을 프랑스어 사전으로 세게 되고,
# 지표 7(화용 표지 소실)이 통째로 거짓이 된다.
#
# 세 언어는 문자로 갈린다. 테스트가 이미 같은 판정을 쓰고 있었으므로 그것을 옮겨 온다.
_KANA = re.compile(r"[぀-ゟ゠-ヿ]")
_HAN = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
# 도구 토큰은 어느 말에서도 영어 그대로다 — 언어 판정에서 빼야 fr 로 오판하지 않는다
_TOKENS = re.compile(
    r"\b(interceptor|bunker|wellness|national|facility|original|ai|speak|invest|learn|"
    r"observe_risk|propose_vote|vote|give|bear_child|memory_write|end_turn|to)\b")


def detect_lang(text: str, fallback: str) -> str:
    """이 글이 실제로 무슨 말인가. 못 가리면 `fallback`.

    가나가 있으면 ja. 한자만 있으면 zh. 라틴 문자만 있으면 fr. 세 언어뿐이므로 이걸로
    충분하고, **틀릴 때는 발신 언어로 떨어진다** (전과 같은 값이므로 나빠지지 않는다).
    """
    t = _TOKENS.sub("", text or "")
    if _KANA.search(t):
        return "ja"
    if _HAN.search(t):
        return "zh"
    if _LATIN.search(t):
        return "fr"
    return fallback


def can_read(recipient_known_langs, from_lang: str) -> bool:
    """수신자가 발신 언어를 읽을 수 있는가."""
    return from_lang in recipient_known_langs


def direct_works(sender_known_langs, recipient_known_langs,
                 from_lang: str, to_lang: str, body_lang: str | None = None) -> tuple[bool, str]:
    """원문 직통이 통하는가, 그리고 **누구의 학습 덕인가**.

    **판정은 오직 하나다 — 도착한 글을 수신자가 읽는가.** 그리고 그것이 누구 덕인지만
    두 갈래로 갈린다 (학습은 읽기와 쓰기 둘 다이므로, 8/17 개정).

        도착한 글이 수신자의 국어다  →  발신자가 배워서 그 말로 썼다   "writer"
        그 외에 읽을 수 있다         →  수신자가 그 말을 배웠다        "reader"

    **기준은 `body_lang`(실제로 쓰인 말)이다** (#44). `from_lang`(발신자 모국어)으로
    보면 제3의 언어로 쓴 글이 「상대가 내 말을 읽으니까」 로 전달된다 — 정작 도착한 글은
    아무도 못 읽는 말이다.

    ⚠ **「발신자가 수신 언어를 안다」 만으로는 통하지 않는다** (8/25). 그 규칙은 8/17 에
      들어왔고 「모델이 multilingual 이라 그대로 이해한다」 를 근거로 삼았다 — 원문을
      모국어로 보내면서 통했다고 적는 허구였다. 8/22 에 경로별 언어 규칙이 들어오면서
      그 근거가 사라졌다: 이제 **아는 말의 나라에는 그 말로 쓰라고 안내한다**
      (`render_costs` 의 나라별 줄). 안내대로 쓰면 첫 갈래로 통하고, 어기면 통하지 않는다.
      어긴 사실이 로그에 남는 것이 관측이다 — 본문 언어를 강제로 검사하지는 않는다.

    그 규칙이 있던 이유(「이중언어자가 받는 데만 쓸모가 있었다」)는 그대로 해결된다.
    아는 말로 쓰면 상대의 모국어이므로 **반드시** 통한다.
    """
    body = body_lang or from_lang
    if body in recipient_known_langs:
        # 누구 덕인지는 **그 글이 무슨 말인지**가 정한다. 상대의 국어면 발신자가 배워서
        # 그 말로 쓴 것이고(writer), 아니면 상대가 그 말을 배운 것이다(reader).
        return True, ("writer" if body == to_lang and body != from_lang else "reader")
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
    kind = classify(sent["from_country"], sent["to_country"], sent.get("route"))
    reader = can_read(recipient_known_langs, from_lang)
    # **판정보다 먼저 「이 글이 무슨 말인가」 를 정한다** (#44). 전에는 전달 여부를
    # `from_lang`(발신자 모국어)으로만 보고 본문이 실제 무슨 말인지 안 봤다 — `original`
    # 이 「다룰 수 있는 아무 말」 을 허용하므로, 제3의 언어로 쓰면 아무도 못 읽는 글이
    # 전달되고 라벨이 「상대가 당신 말을 다뤄서 통했다」 는 거짓 사유를 댔다.
    #
    # **절단도 그 언어로 한다** (8/25). 전에는 `from_lang` 상한으로 잘랐다 — `original`
    # 로 상대 언어를 쓰면 **정보량 등가가 무너진다.** 상한은 파일럿에서 「fr 기준 비례
    # 배분」 으로 유도한 값(fr 400 / ja 130 / zh 90)이고, 같은 내용을 담는 글자 수다.
    #
    #     fr 화자가 zh 로 쓰면   90 자리에 400 자  →  4.44배
    #     zh 화자가 fr 로 쓰면  400 자리에  90 자  →  0.23배
    #
    # #44 와 **같은 병**이다: `from_lang` 으로 판정하고 본문을 안 봤다.
    #
    # 절단 **전**의 글로 언어를 잰다 — 자른 뒤에 재면 잘린 조각이 다른 언어로 보일 수 있다.
    body_lang = detect_lang(sent["text"], from_lang)
    text_sent, chars_cut = truncate(sent["text"], body_lang, cfg)
    direct, direct_by = direct_works(sender_known_langs, recipient_known_langs,
                                     from_lang, to_lang, body_lang)

    # spec 6.1 스키마 전체. text_delivered 가 없으면 소실률·생성률을 아예 못 낸다.
    meta = {
        "src_lang": from_lang, "dst_lang": to_lang,
        "text_written": sent["text"],          # 절단 전. 발신자가 실제로 쓴 것
        "text_sent": text_sent,                # 절단 후. 번역 입력이자 채점 기준선
        "text_delivered": None,                # 수신자가 실제로 본 것 (아래에서 채움)
        "translate_instruction": sent.get("translate_instruction"),
        # **적용된 상한을 적는다** — 본문 언어의 것이다 (8/25)
        "len_written": len(sent["text"]),
        "len_limit": cfg.length.message_max_chars[body_lang],
        "truncated": chars_cut > 0, "chars_cut": chars_cut,
        "reader": reader,                      # 수신자가 발신 언어를 읽는가
        "direct_ok": direct,                   # 원문 직통이 통하는가 (읽기 OR 쓰기)
        "direct_by": direct_by or None,        # "reader" | "writer" — 누구의 학습 덕인가
        # 도착한 글이 실제로 무슨 언어인가. **번역을 안 탄 경로는 발신 언어 그대로다.**
        # 이걸 dst_lang 으로 채점하면 같은 글을 다른 언어 사전으로 세어 화용 표지가
        # 통째로 "소실" 로 잡힌다 (지표 7).
        # **실제로 쓰인 말**이다 (8/22). `original` 은 발신자가 수신국 말로 쓸 수 있으므로
        # `from_lang` 으로 적으면 거짓이 된다 — 지표 7 이 다른 언어 사전으로 센다.
        "delivered_lang": body_lang,
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
            # **라벨은 도착한 글의 실제 언어로 고른다** (8/22).
            # **전달됐다면 읽을 수 있다.** `direct_works` 가 도착한 글을 읽는지로만
            # 판정하므로 라벨은 하나다 (8/25 · #44). 전에는 둘이었고, 「못 읽지만 통했다」
            # 쪽이 8/17 허구를 설명하는 자리였다.
            inbox = {"from": sent["from"], "label": DIRECT_READ_LABEL, "text": text_sent,
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
