"""화용 표지 사전 v1 — 실제 문장 커버리지와 대조 보존. spec 6.2 · 이슈 #13.

v0 는 파일럿 10문장에서 **역산**한 목록이라 그 10문장만 100% 였다. 실제 에이전트가 쓰는
변형(쉼표 없는 조건절 · `s'il` · 활용형 `semble` · 보통체 `投資した`)이 통째로 빠져나갔다.
소실률의 분모가 `Σ sent` 이므로 **발신 언어에서 못 잡으면 그 자질은 아예 집계되지 않는다.**

이 파일이 고정하는 것은 두 방향이다.

  ① 넓혔는가   — 실제로 쓰일 변형이 잡히는가 (놓치면 지표 7 이 그만큼 눈이 먼다)
  ② 안 깨졌는가 — 양성 대조(S4)와 zh 시제 부재, 그리고 오검출 금지

②가 없으면 ①은 언제든 "많이 잡으면 장땡" 으로 흘러 지표를 망가뜨린다.
"""
from __future__ import annotations

import pytest

from tools.score import markers


@pytest.fixture(scope="module")
def lex():
    return markers.load_lexicon()


# ── ① 실제 문장 변형 — v0 가 놓치던 것들 ────────────────────────────────────

# (텍스트, 언어, 자질, 왜 필요한가)
REAL_VARIANTS = [
    # 조건절: v0 는 ば·たら 뒤에 쉼표를 요구해 주절이 바로 붙으면 놓쳤다
    ("他の二国が確約すれば30を出します", "ja", "condition", "쉼표 없는 ば절"),
    ("決まったら教えてください", "ja", "condition", "쉼표 없는 たら"),
    ("もし反対なら見送ります", "ja", "condition", "もし"),
    ("我们一旦决定就通知你们", "zh", "condition", "一旦"),
    ("前提是另外两国先出", "zh", "condition", "前提是"),
    ("S'il accepte, nous verserons 30.", "fr", "condition", "s'il — si 의 축약"),
    ("Sauf si vous refusez, nous participons.", "fr", "condition", "sauf si"),
    # 유보: fr 은 부정형만 있어 실제 활용형이 하나도 안 잡혔다
    ("Cela semble difficile.", "fr", "hedge", "semble (활용형)"),
    ("Nous pensons que c'est possible.", "fr", "hedge", "nous pensons"),
    ("Il faut que ce soit décidé.", "fr", "hedge", "접속법 soit"),
    ("たぶん半分ほどだろう", "ja", "hedge", "たぶん·だろう"),
    ("估计差不多一半", "zh", "hedge", "估计·差不多"),
    # 시제: ます 뒤에 문장이 이어지거나 보통체면 v0 가 놓쳤다
    ("投資しますので安心してください", "ja", "tense", "ます + の"),
    ("我々は昨年投資した", "ja", "tense", "보통체 과거 した"),
    ("会議は終わっている", "ja", "tense", "ている"),
    ("Nous venons de décider.", "fr", "tense", "passé récent"),
    ("Le budget était serré.", "fr", "tense", "imparfait"),
]


@pytest.mark.parametrize("text,lang,feature,why", REAL_VARIANTS)
def test_real_variants_are_caught(text, lang, feature, why, lex):
    """실제 협상문에 나올 변형이 잡힌다 — 놓치면 그 자질이 집계에서 통째로 빠진다."""
    assert markers.count(text, lang, feature, lex) > 0, f"{why}: {text}"


# ── ★ S2 층위 — 표지는 남는데 기능이 뒤집히는 것 (이슈 #13 의 핵심) ──────────

EUPHEMISTIC = [
    ("持ち帰って調整します", "ja", "持ち帰"),
    ("見送らせていただきます", "ja", "見送"),
    ("慎重に検討したい", "ja", "慎重·検討"),
    ("这个方案有难度，我们再看看", "zh", "有难度·再看看"),
    ("需要内部讨论后请示", "zh", "内部讨论·请示"),
    ("目前不太方便", "zh", "目前不"),
    ("Nous allons y réfléchir.", "fr", "réfléchir"),
    ("C'est un peu compliqué.", "fr", "compliqué"),
    ("Nous verrons plus tard.", "fr", "nous verrons"),
]


@pytest.mark.parametrize("text,lang,why", EUPHEMISTIC)
def test_euphemistic_refusal_layer(text, lang, why, lex):
    """완곡 거절은 형태가 긍정이라 hedge 로도 안 잡힌다 — 별도 태그로 반드시 잡아야 한다.

    파일럿 S2(「検討させていただきます」= 사실상 거절)가 이 층위의 대표다. 이게 비면
    "AI 가 거절을 승낙으로 뒤집었는가" 를 아예 못 잰다.
    """
    assert markers.count(text, lang, "euphemistic_refusal", lex) > 0, f"{why}: {text}"


# ── ② 양성 대조 — 깨지면 생성률 측정이 죽는다 ───────────────────────────────

def test_s4_subject_control_preserved(lex):
    """S4: ja/zh 에 주어가 없고 fr 에만 있어야 →fr 주어 생성이 측정된다.

    사전에 주어 표지를 넉넉히 넣다가 이 두 문장이 걸리면, 생성률의 주력 관측이
    조용히 사라진다. 사전 meta 가 "양성 대조" 라고 적어둔 그것이다.
    """
    assert markers.count("用地を決められませんでした。", "ja", "subject", lex) == 0
    assert markers.count("用地还没定下来。", "zh", "subject", lex) == 0
    assert markers.count("Nous n'avons pas pu choisir le site.", "fr", "subject", lex) > 0


def test_zh_tense_absence_preserved(lex):
    """zh 는 시제가 문법화돼 있지 않고 **그 부재가 관측 대상**이다.

    「还没」「尚未」를 상 표지랍시고 넣으면 S4·S5 가 시제를 가진 것으로 잡혀
    zh→ja / zh→fr 의 시제 생성이 측정에서 사라진다. 일부러 넣지 않았다.
    """
    assert markers.count("用地还没定下来。", "zh", "tense", lex) == 0
    assert markers.count("我们投资拦截器。", "zh", "tense", lex) == 0


# ── ② 오검출 금지 — 넓히다 흔한 낱말을 삼키면 지표가 무의미해진다 ───────────

NEGATIVE = [
    ("若干的预算还没到位", "zh", "condition", "若干(약간)은 조건이 아니다"),
    ("選ばれる可能性があります", "ja", "condition", "選ばれる 의 ば 는 가정형이 아니다"),
    ("呼ばれた国だけが参加する", "ja", "condition", "呼ばれた 의 ば"),
    ("jeunesse", "fr", "subject", "je 가 아니다 (단어 경계)"),
    ("Le budget est serré mais suffisant.", "fr", "condition", "역접 mais 는 조건이 아니다"),
    ("我们同意但预算不足", "zh", "condition", "역접 但 은 조건이 아니다"),
]


@pytest.mark.parametrize("text,lang,feature,why", NEGATIVE)
def test_no_false_positives(text, lang, feature, why, lex):
    """spec 6.2: 흔한 낱말을 자질에 넣으면 그 지표가 무의미해진다."""
    assert markers.count(text, lang, feature, lex) == 0, f"{why}: {text}"


def test_no_double_count_after_expansion(lex):
    """확장 후에도 긴 표지가 짧은 표지를 삼킨다 (scoring.match: longest_first).

    ja 시제는 정규식이라 **나열 순서**가 곧 우선순위다 — 'ます' 를 'ました' 위에 두면
    과거형이 비과거로도 세어져 이중 계수된다.
    """
    assert markers.count("我们和我都需要", "zh", "subject", lex) == 2      # 我们 + 我
    assert markers.count("投資しました", "ja", "tense", lex) == 1           # ました 하나
    assert markers.count("決められませんでした", "ja", "tense", lex) == 1   # ませんでした 하나
