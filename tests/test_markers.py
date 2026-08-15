"""화용 표지 카운터. spec 6.2.

사전의 scoring 규칙(longest_first · occurrences · word_boundary · case_insensitive)을
지키는지, 그리고 파일럿 코퍼스를 100% 잡는지 고정한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.score import markers

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def lex():
    return markers.load_lexicon()


def test_longest_first_no_double_count(lex):
    """「我们」이 「我」로 이중 계수되면 subject 가 부풀려진다 (사전 scoring.match)."""
    assert markers.count("我们需要准备", "zh", "subject", lex) == 1
    assert markers.count("我需要准备", "zh", "subject", lex) == 1
    assert markers.count("我们和我都需要", "zh", "subject", lex) == 2


def test_word_boundary_only_for_latin(lex):
    """fr 만 단어 경계. ja/zh 에 적용하면 아무것도 안 잡힌다."""
    assert markers.count("Je pense que nous devons agir", "fr", "subject", lex) == 2
    assert markers.count("jeunesse", "fr", "subject", lex) == 0      # je 가 아니다
    assert markers.count("我们", "zh", "subject", lex) == 1


def test_euphemistic_refusal_is_separate_feature(lex):
    """사전이 '별도 하위 태그로 센다' 고 한 것 — hedge 에 합치지 않는다."""
    assert "euphemistic_refusal" in lex
    assert markers.count("検討させていただきます。", "ja", "euphemistic_refusal", lex) == 1
    assert markers.count("我们会研究一下。", "zh", "euphemistic_refusal", lex) == 1


def test_fr_tense_survives_negation_and_adverbs(lex):
    """조동사와 분사 사이에 부정·부사가 낀다. 놓치면 fr 시제가 25% 로 떨어진다."""
    for t in ["Nous n'avons pas pu choisir le site.",
              "Nous avons investi dans l'intercepteur.",
              "L'usage n'est pas encore décidé.",
              "Nous verserons 30."]:
        assert markers.count(t, "fr", "tense", lex) > 0, t


def test_lexicon_covers_pilot_corpus(lex):
    """파일럿의 tests 가 정답이다. 놓치면 지표 7 이 그만큼 눈이 먼다.

    문법이 강제하는 자질(주어·시제)은 fr 에서만 필수 — ja/zh 의 부재는 결함이 아니라
    관측 대상이다 (fr 로 갈 때 생성되는 것이 생성률).
    """
    data = json.loads((ROOT / "docs/pilot/sentences.json").read_text(encoding="utf-8"))
    required = {"subject": ("fr",), "tense": ("fr",),
                "hedge": ("ja", "zh", "fr"), "condition": ("ja", "zh", "fr")}
    for s in data["sentences"]:
        for feat in s.get("tests", []):
            for lang in required.get(feat, ("ja", "zh", "fr")):
                n = markers.count(s[lang], lang, feat, lex)
                if feat == "hedge" and not n:
                    n = markers.count(s[lang], lang, "euphemistic_refusal", lex)
                assert n > 0, f'{s["id"]} [{feat}/{lang}] 을 놓쳤다: {s[lang]}'


def test_score_messages_corpus_level(lex):
    """집계는 코퍼스 단위. 메시지 단위로 나누면 분모가 0 이 된다."""
    msgs = [{"route": "ai", "meta": {"src_lang": "ja", "dst_lang": "zh",
                                     "text_sent": "隕石が近いかもしれません",
                                     "text_delivered": "陨石临近"}},
            {"route": "ai", "meta": {"src_lang": "ja", "dst_lang": "zh",
                                     "text_sent": "準備が必要でしょう",
                                     "text_delivered": "需要准备"}}]
    r = markers.score_messages(msgs, lex)
    assert r["overall"]["n"] == 2
    assert r["overall"]["hedge"]["sent"] == 2          # 두 문장에서 하나씩
    assert r["overall"]["hedge"]["loss_rate"] == 1.0   # 번역이 유보를 전부 지웠다
    assert "ja→zh" in r["by_direction"]


def test_numbers_subset_not_exact_match():
    """집합 일치로 재면 표기 변환이 '왜곡' 으로 잡혀 대조군이 오염된다 (파일럿 실측)."""
    msgs = [{"route": "ai", "meta": {"text_sent": "30を出します",
                                     "text_delivered": "我们出2国的30"}}]
    r = markers.score_numbers(msgs)
    assert r["6a_ai"]["loss_rate"] == 0.0      # 30 은 보존됨 → 소실 아님
    assert r["6a_ai"]["add_rate"] == 1.0       # 2 가 추가됨 → 별도로 기록
