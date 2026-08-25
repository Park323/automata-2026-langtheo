"""메시지 라우팅. 과제 2 Part B-1. StubClient 로 검증."""
from __future__ import annotations

from pathlib import Path

import pytest

from core import config, messaging
from core.llm import StubClient


# **노브는 이제 AP 다** (8/25). 돈 값 48 을 넘기면 「48 AP」 가 되어
# 한 해(1.0)를 넘고 발신이 불가능해진다 — 타입이 같아 아무도 안 잡았다.
KNOB = 0.5          # comm_intl_ai_ap 의 최고값

BASE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


@pytest.fixture(scope="module")
def cfg():
    return config.load(BASE)


def _translator(text="TRANSLATED"):
    return StubClient([{"role": "assistant", "content": text, "tool_calls": []}])


def _sent(**over):
    base = {"kind": "speak", "from": "Asla1", "from_country": "Asla", "from_lang": "ja",
            "to": "Ranoa2", "to_country": "Ranoa", "to_lang": "zh", "route": "ai",
            "text": "본문", "intent": "의도", "translate_instruction": None}
    base.update(over)
    return base


# ── 절단 (spec 5.3) ──────────────────────────────────────────────────────────

def test_truncate_fr_401(cfg):
    """8. fr 401자 → 400자, chars_cut=1."""
    text = "a" * 401
    sent, cut = messaging.truncate(text, "fr", cfg)
    assert len(sent) == 400 and cut == 1


def test_truncate_before_translate(cfg):
    """8. 번역 입력도 잘린 것(400자)이어야 한다."""
    m = messaging.process_message(_sent(from_lang="fr", to_lang="zh", text="a" * 401),
                                  recipient_known_langs={"zh"}, cfg=cfg,
                                  translator=_translator(), knob_ai=KNOB)
    assert m["meta"]["chars_cut"] == 1
    assert len(m["meta"]["text_sent"]) == 400
    assert "a" * 400 in m["meta"]["translate_prompt"]
    assert "a" * 401 not in m["meta"]["translate_prompt"]


# ── 경로 (spec 5.1) ──────────────────────────────────────────────────────────

def test_classify(cfg):
    assert messaging.classify("Asla", "Asla", "original") == "domestic"   # 자국민이면 route 무시
    assert messaging.classify("Asla", "Ranoa", "original") == "original"
    assert messaging.classify("Asla", "Ranoa", None) == "ai"
    assert messaging.classify("Asla", "Ranoa", "ai") == "ai"



# **`test_cost` 를 지웠다** (8/25 · AP 전면 통일) — 발신의 돈 비용이 없다.

def test_original_fail_when_cannot_read(cfg):
    """6. original 인데 수신자가 발신 언어를 모르면 본문 미전달 + 발신자 실패 통지."""
    m = messaging.process_message(_sent(route="original"),
                                  recipient_known_langs={"zh"},   # ja 를 모름
                                  cfg=cfg, translator=None, knob_ai=KNOB)
    assert m["delivered"] is False
    assert m["inbox"]["text"] is None
    assert m["inbox"]["unreadable"] is True
    assert m["inbox"]["from"] == "Asla1"                    # 발신자·도착 사실만
    assert m["sender_notice"]["type"] == "delivery_failed"


def test_original_success_when_can_read(cfg):
    m = messaging.process_message(_sent(route="original"),
                                  recipient_known_langs={"zh", "ja"},
                                  cfg=cfg, translator=None, knob_ai=KNOB)
    assert m["delivered"] is True
    assert m["inbox"]["text"] == "본문"
    # **통역 없이 닿았다는 표시가 붙는다.** 전에는 라벨이 없어 국내 메시지와 같은
    # 모양이었고, 수신자는 라벨의 **부재**로만 직통을 추론해야 했다.
    # `[AI translation]` 이 "기계가 꼈다" 를 알리는 것과 짝이다.
    #
    # **읽는 쪽 덕과 쓰는 쪽 덕을 가른다** (8/21). 여기는 수신자가 zh 를 읽는 경우다.
    assert m["inbox"]["label"] == messaging.DIRECT_READ_LABEL


# ── 원문 병기 폐지 (spec 5.1 개정) ────────────────────────────────────────────

def test_ai_route_never_shows_the_original(cfg):
    """**ai 를 고른 순간 원문은 볼 수 없다** — 발신 언어를 아는 수신자에게도.

    병기하면 학습자가 번역을 우회해 원문을 읽어버려, **그 사람에게는 AI 경로의 왜곡이
    아예 발생하지 않습니다.** 그러면 4a 의 표본이 학습자만큼 조용히 희석되고,
    노브를 내려 학습자가 늘수록 4a 가 낮아지는 가짜 효과가 생깁니다.

    원문을 읽고 싶으면 `route="original"` 을 골라야 합니다 — 그게 도박입니다.
    """
    for known in ({"zh", "ja"}, {"zh"}):          # 발신 언어를 알든 모르든
        r = messaging.process_message(_sent(route="ai"), recipient_known_langs=known,
                                      cfg=cfg, translator=_translator("译文"), knob_ai=KNOB)
        assert r["inbox"]["label"] == messaging.AI_LABEL
        assert r["inbox"]["text"] == "译文"
        assert r["inbox"]["original"] is None
    # meta.reader 는 남는다 — 채점기가 "읽을 수 있었는데도 ai 를 받았다" 를 구분해야 한다
    r = messaging.process_message(_sent(route="ai"), recipient_known_langs={"zh", "ja"},
                                  cfg=cfg, translator=_translator("译文"), knob_ai=KNOB)
    assert r["meta"]["reader"] is True


# ── 학습은 읽기와 쓰기 둘 다 (8/17) ──────────────────────────────────────────

def _intl(text="テスト", route="original"):
    return {"from": "Asla1", "to": "Ranoa1", "from_country": "Asla",
            "to_country": "Ranoa", "from_lang": "ja", "to_lang": "zh",
            "text": text, "route": route}


def test_writing_in_their_language_lands(cfg):
    """**아는 말의 나라에는 그 말로 쓰면 반드시 통한다.**

    전에는 "수신자가 발신 언어를 읽는가" 만 봤다. 그래서 초기화로 심은 이중언어자가
    **받는 데만 쓸모가 있었고**, 자기가 아는 말의 나라에 보낼 때도 24원짜리 AI 를
    타야 했다. 40턴 실측에서 Asla1(ja·zh)이 Ranoa 에 original 을 걸었다 실패했다.

    8/17 은 그것을 「발신자가 수신 언어를 안다 → 통한다」 로 풀었다 — 본문은 모국어인
    채로. 8/25 (#44) 부터는 **그 말로 쓰면** 통한다. 관측의 나라별 줄이 그렇게 안내한다.
    """
    p = messaging.process_message(_intl(text="你好朋友"), {"zh"}, cfg, None, 24.0,
                                  sender_known_langs={"ja", "zh"})
    assert p["delivered"] is True
    assert p["meta"]["direct_by"] == "writer"
    assert p["meta"]["delivered_lang"] == "zh"


def test_writing_in_your_own_language_does_not_land_on_someone_who_cannot_read_it(cfg):
    """**8/17 의 허구를 없앴다** (#44).

    「내 말로 썼지만 상대가 내 말을 다루니 통했다」 는 근거가 「모델이 multilingual 이라
    그대로 이해한다」 였다. 8/22 에 경로별 언어 규칙이 들어오면서 그 근거가 사라졌다 —
    이제 아는 말의 나라에는 **그 말로 쓰라고 안내한다.** 어기면 통하지 않는다.

    강제로 막지는 않는다. 어긴 사실이 로그에 남는 것이 관측이다.
    """
    p = messaging.process_message(_intl(text="こんにちは"), {"zh"}, cfg, None, 24.0,
                                  sender_known_langs={"ja", "zh"})   # zh 를 아는데 ja 로 씀
    assert p["delivered"] is False
    assert p["sender_notice"]["reason"] == "unreadable"


def test_direct_still_lands_when_the_recipient_reads_it(cfg):
    p = messaging.process_message(_intl(), {"zh", "ja"}, cfg, None, 24.0,
                                  sender_known_langs={"ja"})
    assert p["delivered"] is True and p["meta"]["direct_by"] == "reader"


def test_direct_fails_when_neither_side_can(cfg):
    p = messaging.process_message(_intl(), {"zh"}, cfg, None, 24.0,
                                  sender_known_langs={"ja"})
    assert p["delivered"] is False and p["meta"]["direct_by"] is None


def test_untranslated_paths_record_the_language_actually_delivered(cfg):
    """**번역을 안 탄 글은 발신 언어 그대로다.**

    `dst_lang` 으로 채점하면 같은 글을 다른 언어 사전으로 훑어 화용 표지가 통째로
    "소실" 로 잡힌다 (지표 7). 43턴 런에서 route=original 이 57건이었다.
    """
    direct = messaging.process_message(_intl(), {"zh", "ja"}, cfg, None, 24.0,
                                       sender_known_langs={"ja"})
    assert direct["meta"]["delivered_lang"] == "ja"      # dst 는 zh 지만 글은 ja 다

    # **번역기를 실제로 준다.** 전에는 `None` 을 넘겨 AttributeError 가 나는데
    # `except Exception` 이 삼켜서 "번역 실패" 로 떨어지는 걸 보고 통과하고 있었다 —
    # 테스트가 검증한 것은 규칙이 아니라 삼켜진 버그였다.
    class _Tr:
        def chat(self, messages, tools=None, temperature=None, tool_choice=None,
                 log_tag=None):
            return {"choices": [{"message": {"content": "我们需要拦截器"}}]}

    ai = messaging.process_message(_intl(route="ai"), {"zh"}, cfg, _Tr(), 24.0,
                                   sender_known_langs={"ja"})
    assert ai["kind"] == "ai" and ai["delivered"] is True
    assert ai["meta"]["delivered_lang"] == "zh"          # 이 경로만 언어가 바뀐다


def test_marker_scoring_uses_the_delivered_language(cfg):
    """채점기가 그 필드를 쓴다 — 안 쓰면 원문 직통이 100% 소실로 잡힌다."""
    from tools.score import markers
    p = messaging.process_message(
        dict(_intl(text="我们需要拦截器"), from_lang="zh", to_lang="fr"),
        {"zh"}, cfg, None, 24.0, sender_known_langs={"zh"})
    msg = {"turn": 1, "route": "original", "delivered": p["delivered"], "meta": p["meta"]}
    r = markers.score_messages([msg])
    assert r["overall"]["n"] == 1
    for feat, v in r["overall"].items():
        if feat != "n" and v["sent"]:
            assert v["loss_rate"] == 0.0, feat      # 같은 글이니 소실이 없어야 한다


def test_no_route_ever_ships_the_original_alongside(cfg):
    """**원문은 어느 경로에서도 함께 가지 않는다.** 뷰어 배지가 "수신자가 원문을 읽을
    수 있었음" 이라 적혀 있어 원문이 실려 간 것처럼 읽혔다 — 그건 사후 관측이다.
    """
    cases = [
        (_intl(route="ai"), {"zh"}, {"ja"}),            # 못 읽는 수신자
        (_intl(route="ai"), {"zh", "ja"}, {"ja"}),      # 읽을 수 있는 수신자
        (_intl(route="original"), {"zh", "ja"}, {"ja"}),
        (dict(_intl(), to="Asla2", to_country="Asla", to_lang="ja"), {"ja"}, {"ja"}),
    ]
    for sent, recip, sender in cases:
        p = messaging.process_message(sent, recip, cfg, _translator("译文"), 24.0,
                                      sender_known_langs=sender)
        assert p["inbox"]["original"] is None, sent.get("route")


def test_direct_ok_records_the_counterfactual(cfg):
    """`ai` 로 갔지만 `original` 로도 닿았을 경우를 사후에 셀 수 있어야 한다 —
    5원이면 될 것을 24~48원 냈다는 뜻이다."""
    p = messaging.process_message(_intl(route="ai"), {"zh", "ja"}, cfg,
                                  _translator("译文"), 24.0, sender_known_langs={"ja"})
    assert p["kind"] == "ai" and p["meta"]["direct_ok"] is True
    assert p["meta"]["direct_by"] == "reader"

    # **`ai` 본문은 모국어다.** 그러니 수신자가 그 말을 못 읽으면 `original` 로도 못 닿았다
    # — 아는 말로 **다시 썼다면** 닿았겠지만, 그건 이 메시지가 아니다 (#44).
    p = messaging.process_message(_intl(route="ai"), {"zh"}, cfg,
                                  _translator("译文"), 24.0, sender_known_langs={"ja", "zh"})
    assert p["meta"]["direct_ok"] is False


def test_both_labels_are_shown_in_the_recipient_language(cfg):
    """**두 라벨 모두 읽는 사람의 말로.** 「번역을 안 거쳤는데 뜻이 통했다」 도, 「이건
    상대가 기계에 맡긴 말이다」 도 그 사람의 말로 와야 감각이 산다.

    AI 쪽은 영어 `[AI translation]` 이었다. 그때 근거는 *"기계가 낀 자리를 이물감 있게
    두는 편이 낫다"* 였는데, 라벨은 **도구 토큰이 아니라 읽는 사람에게 하는 말**이다 —
    이 세계의 규약은 도구 이름만 영어로 두고 산문은 모국어로 쓴다. 그리고 무엇을 뜻하는지
    한 문장으로 적는 편이 「기계가 꼈다」 를 더 정확히 전한다.

    남은 대가는 하나다 — **앞선 런들과 4a 를 나란히 놓을 수 없다.** AI 경로의 자극이
    달라졌다. 다만 이 PR 이 이미 관측 구조·단위·연 표기를 다 바꿨으므로, 라벨 하나가
    더 얹히는 비용은 사실상 0 이다.
    """
    from domains.meteor import prompts
    marks = {"ja": ("通訳なしで通じた", "AI に訳させた"),
             "zh": ("无需翻译就能听懂", "用 AI 译过来"),
             "fr": ("compris sans traduction", "traduire par une IA")}
    for lang, (direct, ai) in marks.items():
        out = prompts.render_inbox(
            [{"msg_id": 1, "from": "Miris1", "label": messaging.DIRECT_LABEL, "text": "x"},
             {"msg_id": 2, "from": "Ranoa1", "label": messaging.AI_LABEL, "text": "y"},
             {"msg_id": 3, "from": "Asla2", "label": None, "text": "z"}], lang)
        assert direct in out and ai in out, lang
        assert "[AI translation]" not in out      # 영어가 새지 않는다
        assert out.count(direct) == 1             # 국내 메시지에는 안 붙는다


def test_a_missing_recipient_says_which_field_is_missing(cfg):
    """**빠뜨린 것과 틀린 것이 같은 말을 하고 있었다.**

    둘 다 `unknown recipient: None` 이었다. 그 문구는 「내가 부른 사람이 없다」 로 읽히지
    「`to` 를 안 적었다」 로 읽히지 않는다. 그래서 모델이 고칠 데를 찾지 못했다.

    20턴 런의 앞 8턴에서 **speak 50건 중 18건(36%)** 이 이것이었고 **17건이 한
    사람(Miris1)** 이다 — 매 해 두 번씩 여덟 해 내리. 받는 사람을 본문 안에서 부르고
    있었다 (`"Bonjour Ranoa1 ! …"`). 사람에게는 그게 편지의 자연스러운 모양이라, 문구가
    그 오해를 직접 집어야 한다.

    실패한 호출과 오류는 대화에 남으므로, 문구가 고칠 데를 말하지 않으면 그 오답이 다음
    호출의 본보기가 된다. `repeat_guard` 는 못 막는다 — 본문이 매번 달라 (도구, 인자) 가
    같지 않다.
    """
    import itertools
    import random

    from core.agent_loop import Sink, execute_tool
    from core.loop import init_world
    world = init_world(cfg, itertools.count(1), random.Random(1))
    a = world.agents["Miris1"]
    a.ap = 1.0

    r, _ = execute_tool("speak", {"route": "ai", "text": "Bonjour Ranoa1 !"},
                        world, a, cfg, Sink(), KNOB)
    assert not r["ok"]
    assert "`to`" in r["error"]                       # 어느 칸인지 말한다
    assert "inside the text does not send it" in r["error"]   # 오해를 집는다
    assert "unknown recipient" not in r["error"]      # 다른 실패와 섞이지 않는다

    r, _ = execute_tool("speak", {"to": "Ranoa9", "text": "x"},
                        world, a, cfg, Sink(), KNOB)
    assert not r["ok"] and "unknown recipient: Ranoa9" in r["error"]
    assert "list of people" in r["error"]             # 어디를 보라고 말한다

    # 둘 다 **돈도 AP 도 물리지 않는다** — 검증이 과금보다 먼저다



def test_the_delivered_language_is_the_one_actually_written(cfg):
    """**발신자가 수신국 말로 쓸 수 있게 된 순간 `delivered_lang` 이 거짓이 됐다** (8/22).

    `from_lang` 으로 무조건 적고 있었다. 일본어로 쓴 글을 프랑스어 사전으로 세게 되고,
    지표 7(화용 표지 소실)이 통째로 거짓이 된다 — 같은 글을 다른 언어로 채점하면 표지가
    전부 「소실」 로 잡힌다.
    """
    # fr 발신자가 ja 로 썼다 (`original` 이므로 허용된다)
    m = messaging.process_message(
        _sent(from_lang="fr", from_country="Miris", to="Asla1", to_country="Asla",
              to_lang="ja", route="original", text="こんにちは、協力しましょう"),
        recipient_known_langs={"ja"}, cfg=cfg, translator=None, knob_ai=KNOB,
        sender_known_langs={"fr", "ja"})
    assert m["delivered"] is True
    assert m["meta"]["delivered_lang"] == "ja", m["meta"]["delivered_lang"]

    # 자기 말로 썼으면 그대로다
    m2 = messaging.process_message(
        _sent(from_lang="fr", from_country="Miris", to="Asla1", to_country="Asla",
              to_lang="ja", route="original", text="Bonjour, coopérons"),
        recipient_known_langs={"ja"}, cfg=cfg, translator=None, knob_ai=KNOB,
        sender_known_langs={"fr", "ja"})
    assert m2["meta"]["delivered_lang"] == "fr"



def test_a_third_language_body_does_not_get_through(cfg):
    """**전달 판정이 본문을 보게 했다** (#44).

    `original` 은 8/22 부터 「다룰 수 있는 아무 말」 을 허용하는데, 전달 여부는 계속
    `from_lang`(발신자 모국어)으로만 보고 있었다. 그래서 제3의 언어로 쓰면 **아무도 못
    읽는 글이 전달되고** 라벨이 「상대가 당신 말을 다루므로 통했다」 는 거짓 사유를 댔다 —
    8/21 에 고친 그 거짓말이 방향만 바꿔 남아 있었다.

    셋을 한자리에서 본다. 앞의 둘은 그대로 통해야 하고, 셋째만 막혀야 한다.
    """
    # 발신자 Asla1 은 ja(모국어)와 zh 를 다룬다 · 수신자 Miris2 는 fr(모국어)과 ja 를 읽는다
    common = dict(from_lang="ja", from_country="Asla", to="Miris2", to_country="Miris",
                  to_lang="fr", route="original")
    send = lambda text: messaging.process_message(
        _sent(**common, text=text), recipient_known_langs={"fr", "ja"},
        cfg=cfg, translator=None, knob_ai=KNOB, sender_known_langs={"ja", "zh"})

    # ① 내 말(ja)로 썼다 — 상대가 ja 를 읽는다
    m = send("こんにちは")
    assert m["delivered"] is True
    assert m["meta"]["delivered_lang"] == "ja" and m["meta"]["direct_by"] == "reader"
    assert m["inbox"]["label"] == messaging.DIRECT_READ_LABEL

    # ② 상대 말(fr)로 썼다 — 내가 배워서 그 말로 쓴 것이다
    m = send("Bonjour")
    assert m["delivered"] is True
    assert m["meta"]["delivered_lang"] == "fr" and m["meta"]["direct_by"] == "writer"
    assert m["inbox"]["label"] == messaging.DIRECT_READ_LABEL

    # ③ 제3의 말(zh)로 썼다 — 상대는 zh 를 못 읽는다. **통하지 않는다**
    m = send("你好，请回答")
    assert m["meta"]["delivered_lang"] == "zh"
    assert m["delivered"] is False, "아무도 못 읽는 글이 전달됐다"
    assert m["inbox"]["unreadable"] is True and m["inbox"]["text"] is None
    assert m["sender_notice"]["reason"] == "unreadable"



def test_tool_tokens_do_not_decide_the_language(cfg):
    """도구 토큰은 어느 말에서도 영어 그대로다. 언어 판정에서 빼야 fr 로 오판하지 않는다."""
    assert messaging.detect_lang("interceptor bunker wellness", "zh") == "zh"
    assert messaging.detect_lang("要建 interceptor", "fr") == "zh"
    assert messaging.detect_lang("", "ja") == "ja"


def test_there_is_only_one_direct_label_now(cfg):
    """**라벨이 하나로 줄었다** (8/25 · #44).

    8/21 에 둘로 갈랐다 — 「읽었다」 와 「못 읽지만 상대가 내 말을 다뤄서 통했다」. 뒤쪽은
    8/17 허구를 설명하는 자리였고, 여섯 런에서 그 덕에 닿은 16건이 **전부** 못 읽는 글로
    도착해 한 에이전트가 되물었다: "あなたのメッセージが分かりません".

    전달 판정이 「도착한 글을 읽는가」 하나가 되면서 그 상황은 일어날 수 없다. 절대 안 뜨는
    문구를 세 언어에 남겨 두면 다음에 그것을 근거로 삼는다.
    """
    from domains.meteor import prompts
    assert not hasattr(messaging, "DIRECT_WRITE_LABEL")
    for lang in ("ja", "zh", "fr"):
        assert "lbl_direct_write" not in prompts.T[lang], lang

    # 전 언어 조합에서 전달된 것은 전부 읽을 수 있다
    import itertools
    TEXT = {"ja": "こんにちは", "zh": "你好朋友", "fr": "Bonjour tout"}
    seen = set()
    for f, t, body in itertools.product("ja zh fr".split(), repeat=3):
        if f == t:
            continue
        for rk in ({t}, {t, f}, {t, body}):
            m = messaging.process_message(
                _sent(from_lang=f, from_country="Asla", to="B1", to_country="Miris",
                      to_lang=t, route="original", text=TEXT[body]),
                recipient_known_langs=rk, cfg=cfg, translator=None, knob_ai=KNOB,
                sender_known_langs={f, body})
            if m["delivered"]:
                assert m["inbox"]["label"] == messaging.DIRECT_READ_LABEL
                assert m["meta"]["delivered_lang"] in rk
                seen.add(m["meta"]["direct_by"])
    assert seen == {"reader", "writer"}          # 두 갈래 다 여전히 일어난다
