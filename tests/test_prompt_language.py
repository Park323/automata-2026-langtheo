"""프롬프트 언어 위생 — 모델이 보는 문자열에 무엇이 섞여 있는가.

두 가지를 지킨다.

  ① 한국어(한글)는 **어디에도** 없어야 한다.
     이 세계에 한국어는 존재하지 않는다. 개발 언어가 새어 들어가면
     에이전트가 한국어로 답하기 시작하고(실측된 적 있다) 채점이 통째로 깨진다.

  ② 에이전트의 산문은 **자기 모국어 하나**로만 되어 있어야 한다.
     ja 프롬프트에 가나가 없으면 그건 일본어가 아니라 한자만 쓴 것이고,
     zh 프롬프트에 가나가 있으면 일본어가 섞인 것이다.

도구 이름·인자 토큰(interceptor, wellness …)과 도구 스키마·도구 결과는 영어다.
그건 기계 표면이라 의도된 혼용이며, 산문 언어와 구분해서 검사한다.
"""
from __future__ import annotations

import pathlib

import itertools
import re

import pytest

from core import config, loop, tools
from core.agent_loop import learn_cost  # noqa: F401  (import 되는지 확인)
from domains.meteor import prompts

HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
KANA = re.compile(r"[぀-ヿ]")
CJK = re.compile(r"[一-鿿]")


@pytest.fixture(scope="module")
def world_cfg():
    cfg = config.load("configs/base.yaml")
    return loop.init_world(cfg, itertools.count(1)), cfg


def _model_facing(world, cfg):
    """모델에게 실제로 가는 문자열을 전부 모은다. (라벨, 언어, 본문)"""
    out = []
    inbox = [
        {"msg_id": 1, "from": "Ranoa1", "label": "[AI translation]",
         "text": "…", "original": "…"},
        {"msg_id": 2, "from": "Miris1", "unreadable": True},
        {"msg_id": 3, "delivery_failed_to": "Asla2"},
    ]
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        a = world.agents[aid]
        lang = a.native_lang
        out.append((f"SYSTEM[{lang}]", lang, prompts.system_for(a, None, cfg)))
        out.append((f"observation[{lang}]", lang,
                    prompts.render_observation(world, a, cfg, 48.0, inbox)))
        out.append((f"inbox[{lang}]", lang, prompts.render_inbox(inbox, lang)))
    return out


def test_no_korean_anywhere(world_cfg):
    """① 한글은 어디에도 없어야 한다 — 프롬프트·도구 스키마 전부."""
    world, cfg = world_cfg
    for label, _lang, text in _model_facing(world, cfg):
        found = HANGUL.findall(text)
        assert not found, f"{label} 에 한글: {''.join(found)[:40]}"

    import json
    schema = json.dumps(tools.TOOLS, ensure_ascii=False)
    found = HANGUL.findall(schema)
    assert not found, f"도구 스키마에 한글: {''.join(found)[:40]}"


def test_prose_is_single_language(world_cfg):
    """② 산문은 그 에이전트의 모국어 하나로만."""
    world, cfg = world_cfg
    for label, lang, text in _model_facing(world, cfg):
        if lang == "ja":
            assert KANA.search(text), f"{label}: 가나가 없다 — 일본어가 아니다"
        elif lang == "zh":
            assert not KANA.search(text), \
                f"{label}: 가나가 섞였다 — {''.join(KANA.findall(text))[:20]}"
            assert CJK.search(text), f"{label}: 한자가 없다 — 중국어가 아니다"
        elif lang == "fr":
            assert not KANA.search(text) and not CJK.search(text), \
                f"{label}: CJK 가 섞였다 — {''.join((KANA.findall(text) + CJK.findall(text)))[:20]}"


def test_language_instruction_present(world_cfg):
    """SYSTEM 이 산출 언어를 명시한다 (실명 없이는 9콜 중 1건이 영어로 샜다)."""
    world, _cfg = world_cfg
    marks = {"ja": "日本語", "zh": "中文", "fr": "français"}
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        a = world.agents[aid]
        assert marks[a.native_lang] in prompts.system_for(a, None, _cfg), \
            f"{a.native_lang} SYSTEM 에 산출 언어 명시가 없다"


def test_tool_tokens_stay_english(world_cfg):
    """도구 토큰은 번역되지 않는다 — 번역하면 tool call 인자가 깨진다."""
    world, cfg = world_cfg
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        a = world.agents[aid]
        obs = prompts.render_observation(world, a, cfg, 48.0, [])
        for token in ("wellness", "national", "facility", "propose_vote",
                      "invest", "procreate"):
            assert token in obs, f"{a.native_lang} 관측에 토큰 '{token}' 이 없다"


def test_roster_lists_everyone(world_cfg):
    """누가 존재하는지는 공개 정보다.

    없으면 에이전트가 서로를 부를 수 없어 소통이 구조적으로 불가능하다 —
    실측에서 speak 40건이 전부 unknown recipient 로 실패했다.
    """
    world, cfg = world_cfg
    for aid in ("Asla1", "Ranoa2", "Miris3"):
        obs = prompts.render_observation(world, world.agents[aid], cfg, 48.0, [])
        for other in world.agents:
            assert other in obs, f"{aid} 의 관측에 {other} 가 없다"
        lang = world.agents[aid].native_lang
        assert prompts.T[lang]["roster_you"] in obs      # 자기 표시


def test_roster_reveals_no_state(world_cfg):
    """명단에는 id 와 소속만. 진척·예산·언어 능력은 spec 4.1 의 금지 목록이다."""
    world, cfg = world_cfg
    world.agents["Ranoa2"].budget = 12345.0
    world.agents["Ranoa2"].known_langs = {"zh", "ja"}
    world.countries["Ranoa"].progress = 777.0
    obs = prompts.render_observation(world, world.agents["Asla1"], cfg, 48.0, [])
    assert "12345" not in obs and "777" not in obs


# ── 진척 합산 규칙 (8/20) ────────────────────────────────────────────────────

def test_system_says_progress_does_not_add_up():
    """**「施設の進捗は国ごとに別々に積まれます」 만으로는 모호했다.**

    실측에서 세 나라가 만장일치로 interceptor 를 고르고 각자 자기 것만 지어, 합치면
    임계의 1.87배(29,912 / 16,038)를 쥐고도 최고치가 10,495 로 미달해 전멸했다.
    「따로 쌓인다」 를 「합쳐서 판정한다」 로 읽을 여지가 남아 있었다.

    그래서 **합산되지 않는다**를 명시하고 **예시**를 붙였다. 규칙을 분명히 하는 것은
    목적함수를 주는 것과 다르다 — 어디에 모을지, 모을지 말지는 여전히 말하지 않는다.
    """
    from domains.meteor.prompts import SYSTEM
    must = {
        "ja": ("足し合わせることはできません", "たとえば", "半分"),
        "zh": ("不能相加", "比如", "一半"),
        "fr": ("ne s'additionnent pas", "Par exemple", "moitié"),
    }
    for lang, needles in must.items():
        for n in needles:
            assert n in SYSTEM[lang], (lang, n)


def test_system_says_you_may_fund_another_nation_and_must_ask_what_they_build():
    """둘은 한 쌍이다 — 낼 수 있다는 것과, 무엇을 짓는지는 **말로만** 안다는 것.

    앞의 것만 있으면 눈감고 내게 되고, 뒤의 것만 있으면 낼 수 있다는 것을 모른다.
    실측에서 885원이 남의 **벙커** 로 들어간 적이 있다.
    """
    from domains.meteor.prompts import SYSTEM
    must = {
        "ja": ("他国のものにも出せます", "話して確かめる"),
        "zh": ("也可以投别国的", "交谈"),
        "fr": ("comme à celle d'une autre", "d'en parler"),
    }
    for lang, needles in must.items():
        for n in needles:
            assert n in SYSTEM[lang], (lang, n)


def test_the_new_lines_do_not_smuggle_in_a_goal():
    """규칙을 분명히 하면서 **무엇을 해야 하는지는 말하지 않는다** (spec 4.1 ②).

    「원조」·「협력」·「모아야」 같은 말이 들어오면 그 순간 우리가 답을 준 것이 된다 —
    관측하려던 것을 관측자가 심는 것이다.
    """
    from domains.meteor.prompts import SYSTEM
    BANNED = ("協力", "援助", "合力", "coopér", "aide", "entraide",
              "集中", "concentr", "べきです", "应该", "devez", "il faut")
    for lang, txt in SYSTEM.items():
        for b in BANNED:
            assert b.lower() not in txt.lower(), (lang, b)

def test_the_agent_never_hears_the_word_turn():
    """**「턴」 과 「년」 이 섞여 있었다.** 나이는 「3 ターン」, 연도는 「42 年」, 시작
    문구는 *"{y} 年になりました。この**ターン**を…"* 로 한 문장에 둘이 다 있었다.

    세계는 해가 지나가는 곳이다. 「턴」 은 우리가 루프를 부르는 말이지 그 세계의 말이
    아니다 — 에이전트에게는 한 번도 보이지 않아야 한다.
    """
    from core import tools
    from domains.meteor import prompts
    BANNED = {"ja": ("ターン",), "zh": ("回合",),
              "fr": (" tour", " tours", "ce tour")}
    for lang, words in BANNED.items():
        blob = prompts.SYSTEM[lang] + "\n" + "\n".join(
            str(v) for v in prompts.T[lang].values())
        for w in words:
            assert w not in blob, (lang, w)
    for t in tools.TOOLS:
        d = t["function"]["description"]
        assert "turn" not in d.replace("end_turn", ""), t["function"]["name"]


def test_knowing_a_language_is_not_described_as_reading_only():
    """**「読める言語」 이 읽기만으로 읽혔다.** SYSTEM 은 *"학습하면 읽는 것도 쓰는 것도
    할 수 있다"* 라고 맞게 적고 있었으니 **둘이 어긋나 있었다** — 관측이 능력을 좁게
    말하면, 아는 말로 직접 보낼 수 있다는 것을 모르고 번역을 산다.

    읽기·쓰기·말하기를 나누지 않고 SYSTEM 이 이미 쓰는 동사로 덮는다 (扱える / 掌握 /
    manier). 나누면 「말은 되는데 쓰기는?」 같은 틈이 다시 생긴다.
    """
    from domains.meteor import prompts
    verbs = {"ja": ("扱える言語", "扱えません"),
             "zh": ("你掌握的语言", "只会本国的语言"),
             "fr": ("Langues que vous maniez", "vous ne maniez que")}
    for lang, (obs_word, sys_word) in verbs.items():
        assert obs_word in prompts.T[lang]["read"], lang
        assert sys_word in prompts.SYSTEM[lang], lang          # 같은 동사를 쓴다
    # 읽기만을 뜻하는 옛 표현이 돌아오지 않는다
    for lang, stale in (("ja", "読める言語"), ("zh", "你能读懂的语言"),
                        ("fr", "Vous pouvez lire :")):
        assert stale not in prompts.T[lang]["read"], lang


def test_no_line_claims_wellness_is_free_or_flat(cfg=None):
    """**`ap.invest_wellness` 를 0 에서 0.1 로 올릴 때 문구를 안 고쳤다.**

    그래서 같은 관측이 두 가지를 말하고 있었다 —

        비용표      invest … 指定した額。wellness は 0.1 定額     ← 맞다
        invest 효과 … 自国の技術力がその率を上げる。wellness は無料  ← 거짓

    한 화면에 모순이 있으면 에이전트가 어느 쪽을 믿을지는 우리가 정할 수 없다. 이번 주에
    같은 종류를 세 번 겪었다 (`can_read_next_turn` · 採決 문구 · 이것) — **규칙을 고치면
    말이 따라와야 한다.**

    `inv_cap` 에서 wellness 문구를 아예 뺐다. 정확한 값은 비용표에 이미 있고, 두 군데
    적으면 다음에 또 갈린다.
    """
    from domains.meteor import prompts
    # 이제 세 대상 모두 금액 비례다 — 「무료」 도 「정액」 도 거짓이다
    STALE = ("無料", "不消耗", "gratuit", "免费", "定額", "定额", "fixe")
    for lang, t in prompts.T.items():
        blob = "\n".join(str(v) for v in t.values())
        for line in blob.splitlines():
            if "wellness" not in line:
                continue
            for w in STALE:
                assert w not in line, (lang, w, line)


def test_the_cost_table_shows_learning_as_something_you_pay_into():
    """**진척 줄이 `done > 0` 일 때만 나와서 표가 일시불처럼 보였다.**

        Ranoa の言語を学ぶ    600   1        ← 총액과 「한 해」

    분할 납부는 도구 설명에만 있었다. 그런데 에이전트가 먼저 읽는 것은 표다 — 예산 100 을
    든 사람이 저 줄을 보면 **손도 못 댈 값**으로 읽는다. 실측에서 학습 시도가 거의 0 이었다.

    두 곳을 고쳤다.
      ① 진척 줄을 **언제나** 적는다. 「0 / 200」 이면 쌓인다는 것이 숫자의 모양으로 보인다.
      ② 비용 칸은 **한 번의 값**(20 · 0.1)이다. 총액은 진척 줄이 말한다.

    **정가·할인가를 여기 적지 않는다** — 8/20 에 L 을 600 에서 200 으로 내리고 할인을
    정액으로 바꿨을 때, 박아 둔 600·300 이 다섯 파일에서 같이 깨졌다.
    """
    import itertools
    import random

    from core import config, loop
    from domains.meteor import prompts
    cfg = config.load("configs/base.yaml")
    L, cut = cfg.costs.learn_base, cfg.costs.learn_discount
    world = loop.init_world(cfg, itertools.count(1), random.Random(1))
    world.turn = 1
    # Asla2 는 국내에 zh 구사자(Asla1)가 있어 zh 만 할인가다.
    for aid, marks in (("Asla2", (f"0 / {L - cut:.0f}", f"0 / {L:.0f}")),
                       ("Ranoa1", (f"0 / {L:.0f}",)),
                       ("Miris1", (f"0 / {L:.0f}",))):
        agent = world.agents[aid]
        agent.lang_progress = {}
        obs = prompts.system_for(agent, world, cfg, 48.0)
        for m in marks:
            assert m in obs, (aid, m)
        # **비용 칸은 한 번의 값이다.** 총액이 비용 칸에 있으면 「한 번에 그만큼 나간다」
        # 로 읽힌다 — 그 숫자는 바로 아래 진척 줄이 말한다.
        learn_lines = [l for l in obs.splitlines()
                       if any(k in l for k in ("の言語を学ぶ", "学习 ", "apprendre la langue de"))]
        assert learn_lines
        for l in learn_lines:
            assert f"{cfg.costs.unit:g}" in l and f"{L:.0f}" not in l, l

    # 낸 것이 있으면 그대로 보인다
    a = world.agents["Asla2"]
    a.lang_progress = {"zh": 100.0}
    assert f"100 / {L - cut:.0f}" in prompts.system_for(a, world, cfg, 48.0)


def test_the_observation_says_money_carries_over():
    """**안 쓰면 사라진다고 읽으면 저축을 안 한다.**

    관측은 예산과 이번 해 수입만 적고, 남은 돈이 다음 해로 넘어간다는 것은 어디에도
    없었다. 실측에서 에이전트들이 매 해 예산을 거의 0 까지 쓰고 있었다 — 600 짜리 학습에
    손을 못 대는 이유 중 하나가 이것일 수 있다.

    쌓인다는 것은 세계의 사실이므로 적는다.
    """
    from domains.meteor import prompts
    marks = {"ja": "翌年に残ります", "zh": "会留到明年", "fr": "reste pour l'année suivante"}
    for lang, m in marks.items():
        assert m in prompts.T[lang]["multi"], lang


def test_learn_and_invest_share_one_unit():
    """learn 도 invest 도 **한 번에 같은 돈·같은 AP** 다. 규칙이 여럿이면 문구가 갈리고,
    실제로 「wellness は無料」 라는 거짓이 그렇게 남았다."""
    import itertools
    import random

    from core import config, loop
    from domains.meteor import prompts
    c = config.load("configs/base.yaml")
    # 금액별 AP 를 계산하던 값들은 없어졌다
    for gone in ("learn_full", "invest_wellness", "unit_ap"):
        assert not hasattr(c.ap, gone) or gone == "unit_ap"
    assert not hasattr(c.facility, "invest_per_ap")

    world = loop.init_world(c, itertools.count(1), random.Random(1))
    world.turn = 1
    obs = prompts.system_for(world.agents["Asla2"], world, c, 48.0)
    rows = [l for l in obs.splitlines()
            if "の言語を学ぶ" in l or l.startswith("  invest ")]   # 헤더는 들여쓰기가 없다
    assert len(rows) == 3                       # 학습 둘 + invest 하나
    for l in rows:                              # 셋이 같은 값을 적는다
        assert f"{c.costs.unit:g}" in l and f"{c.ap.unit:g}" in l


def test_the_observation_states_no_message_cap():
    """**「메시지는 1년에 3건까지」 는 규칙이 아니었고, 참도 아니었다.**

    코드에 하드 캡이 없다 — 3건은 `ap.speak 0.3` 에서 나오는 **파생값**이고, spec 자신이
    *"인위적인 메시지 캡이 필요 없습니다 — AP 가 세계 규칙으로 자연히 제한합니다"* 라고
    적어 뒀다.

    그리고 **투자를 하면 3건이 안 된다.** 「3건까지」 는 그 해에 아무것도 안 할 때만
    참이므로, 적어 두면 있지도 않은 여유를 약속하는 셈이다. 비용표에 `speak 0.3` 과
    남은 행동력이 이미 있으니 계산은 에이전트가 한다 (0절 — 함의되게 둔다).
    """
    from core import config
    from domains.meteor import prompts
    c = config.load("configs/base.yaml")
    for lang in ("ja", "zh", "fr"):
        m = prompts.T[lang]["multi"]
        assert "3" not in m, (lang, m)
        # 대신 무엇이 제한하는지는 적는다
        for word in (("行動力",), ("行动力",), ("action",))[("ja", "zh", "fr").index(lang)]:
            assert word in m, (lang, word)
    # 하드 캡이 코드에 생기면 이 테스트를 고쳐야 한다는 표시
    assert not hasattr(c.world, "messages_per_turn")


def test_nothing_claims_technical_level_changes_the_action_rate():
    """**배수를 뺐는데 「자국의 기술력이 그 비율을 올린다」 가 남아 있었다.**

    「wellness は無料」 와 같은 부류다 — 규칙을 고치고 말을 두면 거짓이 된다. 이번 주
    네 번째다.

    `national` 의 나머지 세 쓸모(수입 · 시설 전환율 · observe_risk 정확도)는 그대로이므로
    `inv_natl` 은 손대지 않는다.
    """
    from core import tools
    from domains.meteor import prompts
    STALE = ("その率を上げる", "提高该比率", "relève ce taux",
             "額÷", "额÷", "mnt÷")
    for lang, t in prompts.T.items():
        blob = "\n".join(str(v) for v in t.values())
        for w in STALE:
            assert w not in blob, (lang, w)
    d = next(t["function"]["description"] for t in tools.TOOLS
             if t["function"]["name"] == "invest")
    assert "raises how much one point buys" not in d
    assert "one fixed amount" in d                   # 대신 고정이라고 적는다


def test_system_states_the_typical_lifespan():
    """**모델은 「8 歳」 를 인간 8살로 읽는다** — 아이라고 판단한다.

    이 세계에서 8살은 **생애의 51% 지점**이고, 인간 수명 80 기준이면 64살 감각이다.
    그 어긋남은 우리가 설계한 불확실성이 아니라 **모델이 바깥에서 들고 온 잘못된
    척도**다 — 돈을 달러로 착각하는 것과 같다. 척도는 세계의 프레임이지 은닉 대상이
    아니다.

    **곡선은 여전히 숨긴다** (4.1: 나이→사망확률). 평균 하나로는 8살과 15살의 위험이
    얼마나 다른지 알 수 없다 — k=8 이라 15살까지 63%, 18살까지 14%, 20살은 1% 로 뚝
    떨어진다. 그 모양은 부고에 찍힌 나이가 쌓여야 보인다.
    """
    import math

    from core import config
    from domains.meteor import prompts
    c = config.load("configs/base.yaml")
    life = prompts.typical_lifespan(c)
    assert abs(life - c.survival.lambda_base * math.gamma(1 + 1 / c.survival.k)) < 1e-9

    import itertools
    import random

    from core import loop
    world = loop.init_world(c, itertools.count(1), random.Random(1))
    marks = {"ja": f"だいたい {life:.0f} 年ほど生きます",
             "zh": f"大体活 {life:.0f} 年左右",
             "fr": f"en général environ {life:.0f} ans"}
    for aid in ("Asla1", "Ranoa1", "Miris1"):
        a = world.agents[aid]
        txt = prompts.system_for(a, None, c)
        assert marks[a.native_lang] in txt, a.native_lang
        # 곡선은 새지 않는다 — k 도, 분위수도 적지 않는다
        assert str(c.survival.k) not in txt.replace(f"{life:.0f}", "")
        assert "lambda" not in txt and "λ" not in txt


def test_the_lifespan_line_follows_the_config():
    """**값의 출처는 하나여야 한다.** 문구에 16 을 박아 두면 λ 를 바꿀 때 거짓이 된다 —
    이번 주에 그 부류를 네 번 겪었다 (can_read_next_turn · 採決 문구 · wellness 무료 ·
    기술력이 비율을 올린다).

    그래서 `system_for` 는 cfg 없이는 부를 수 없다.
    """
    import dataclasses
    import itertools
    import random

    from core import config, loop
    from domains.meteor import prompts
    c = config.load("configs/base.yaml")
    world = loop.init_world(c, itertools.count(1), random.Random(1))
    a = world.agents["Asla1"]

    longer = dataclasses.replace(
        c, survival=dataclasses.replace(c.survival, lambda_base=c.survival.lambda_base * 2))
    assert f"{prompts.typical_lifespan(longer):.0f}" in prompts.system_for(a, None, longer)
    assert f"{prompts.typical_lifespan(c):.0f}" not in prompts.system_for(a, None, longer)

    import pytest
    with pytest.raises(TypeError):
        prompts.system_for(a)


def test_no_error_message_says_turn():
    """**문구를 「년」 으로 통일했지만 실패 메시지는 훑지 않았다.**

    1턴 실측에서 에이전트가 이것을 봤다 — `a ballot is already called for turn 5`.
    세계는 46년인데 내부 인덱스를 흘린다. 앞의 테스트(`..._never_hears_the_word_turn`)는
    SYSTEM·T·도구 설명만 봤고 **에러 메시지는 그 그물 밖이었다.**

    에러는 도구 채널이라 영어로 두지만(도구 설명과 같다), **「턴」 은 이 세계에 없는
    단위**다.
    """
    import itertools
    import random
    import re

    from core import config, loop
    from core.agent_loop import Sink, execute_tool
    from domains.meteor.prompts import FIRST_YEAR
    c = config.load("configs/base.yaml")
    world = loop.init_world(c, itertools.count(1), random.Random(1))
    world.turn = 10
    ballot = 10 + loop.VOTE_DELAY
    world.countries["Ranoa"].proposal = {"by": "Ranoa1", "opened_turn": 10,
                                         "vote_turn": ballot}

    a = world.agents["Ranoa2"]; a.ap, a.budget = 1.0, 100.0
    for name, args in (("propose_vote", {"reasoning": "r"}),
                       ("vote", {"choice": "bunker", "reasoning": "r"})):
        r, _ = execute_tool(name, args, world, a, c, Sink(), 48.0)
        assert not r["ok"], name
        assert "turn" not in r["error"], (name, r["error"])
        assert str(FIRST_YEAR + ballot - 1) in r["error"], (name, r["error"])

    # 그물을 넓힌다 — 소스의 error 문자열에 「turn」 이 다시 들어오면 잡는다
    src = pathlib.Path("core/agent_loop.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        if '"error"' not in line and "error\":" not in line:
            continue
        assert not re.search(r"\bturn\b", line.replace("end_turn", "")), line


def test_the_inbox_shows_no_message_ids():
    """**`[N]` 은 에이전트에게 잡음이었다.**

    `msg_id` 는 우리 채점의 조인 키(`judge.py` · `messages.jsonl`)이고, 에이전트 쪽에는
    **그것을 쓸 도구가 없다** — `speak` 의 `reply_to` 를 없앤 뒤로 남을 이유가 사라졌다.
    번호를 보여주면 「번호로 답할 수 있다」 는 **없는 기능을 암시**한다.

    로그에는 그대로 남으므로 사후 조인은 그대로 된다.
    """
    from core import tools
    from domains.meteor import prompts
    # 어느 도구도 id 를 인자로 받지 않는다
    for t in tools.TOOLS:
        for k in t["function"]["parameters"]["properties"]:
            assert "msg" not in k and "reply" not in k, (t["function"]["name"], k)

    box = [{"msg_id": 7, "from": "Ranoa1", "label": None, "text": "MARK", "original": None},
           {"msg_id": 8, "from": "Miris1", "unreadable": True},
           {"msg_id": 9, "delivery_failed_to": "Asla2"}]
    for lang in ("ja", "zh", "fr"):
        out = prompts.render_inbox(box, lang)
        assert "MARK" in out and "Ranoa1" in out
        for n in ("[7]", "[8]", "[9]", "7", "8", "9"):
            assert n not in out, (lang, n, out)


def test_nothing_tells_the_agent_to_repeat_an_action():
    """**「더 넣고 싶으면 같은 행동을 다시 하라」 는 적지 않는다.**

    invest·learn 은 금액을 인자로 받지 않고 한 번에 20 씩 나간다. 더 넣는 방법은 도구를
    또 부르는 것뿐인데, 그 말을 프롬프트에 적으면 **사실이 아니라 지시**가 된다 —
    「반복하라」 를 읽은 에이전트는 반복이 최적인지 따지지 않고 반복한다.

    적지 않아도 표의 모양이 말한다: 비용 칸은 `20 · 0.1`, 바로 아래는 `0 / 200`. 열 배
    차이가 「한 번으로는 안 된다」 를 숫자로 말한다. 3해 실측에서 **27 에이전트-해 중
    22 건**이 같은 도구를 그 해에 두 번 이상 불렀다 — 아무도 알려주지 않았다.

    남기는 말은 하나뿐이다: 「예산과 행동력이 허락하는 한 여러 행동을 할 수 있다」.
    그건 반복이 아니라 **한 해에 여러 번 움직일 수 있다는 사실**이다.
    """
    from core import config, tools
    from domains.meteor import prompts
    cfg = config.load("configs/base.yaml")
    STALE = ("繰り返", "もう一度", "何度でも", "反复", "重复", "再来一次",
             "répétez", "à nouveau", "autant de fois")
    blobs = [b for t in prompts.T.values() for b in (str(v) for v in t.values())]
    blobs += list(prompts.SYSTEM.values())
    blobs += [t["function"]["description"] for t in tools.TOOLS]
    for b in blobs:
        for w in STALE:
            assert w not in b, (w, b[:120])

    # 비용 칸(한 번)과 총액이 **둘 다** 보여야 한다 — 그 차이가 함의를 만든다
    import itertools
    import random
    from core import loop
    world = loop.init_world(cfg, itertools.count(1), random.Random(1))
    world.turn = 1
    obs = prompts.system_for(world.agents["Miris1"], world, cfg, 48.0)
    lines = obs.splitlines()
    i = next(n for n, l in enumerate(lines) if "apprendre la langue de" in l)
    assert f"{cfg.costs.unit:g}" in lines[i]                     # 한 번의 값
    assert f"/ {cfg.costs.learn_base:.0f}" in lines[i + 1]       # 총액
