"""프롬프트 언어 위생 — 모델이 보는 문자열에 무엇이 섞여 있는가.

두 가지를 지킨다.

  ① 한국어(한글)는 **어디에도** 없어야 한다.
     이 세계에 한국어는 존재하지 않는다. 개발 언어가 새어 들어가면
     에이전트가 한국어로 답하기 시작하고(실측된 적 있다) 채점이 통째로 깨진다.

  ② **프롬프트의** 산문은 그 에이전트의 모국어 하나로만 되어 있어야 한다.
     ja 프롬프트에 가나가 없으면 그건 일본어가 아니라 한자만 쓴 것이고,
     zh 프롬프트에 가나가 있으면 일본어가 섞인 것이다.

     **에이전트가 쓰는 말은 경로가 정한다** (8/22). `ai` 는 모국어여야 하고 — 거기가
     번역 손실을 재는 채널이라 입력 언어가 흔들리면 지표 7 이 죽는다 — `original` 은
     아는 말 아무거나 된다. 번역이 없으니 잴 손실도 없고, 「발신자가 수신 언어를 안다 →
     통한다」 가 비로소 사실이 된다.

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


# **노브는 이제 AP 다** (8/25). 돈 값 48 을 넘기면 「48 AP」 가 되어
# 한 해(1.0)를 넘고 발신이 불가능해진다 — 타입이 같아 아무도 안 잡았다.
KNOB = 0.5          # comm_intl_ai_ap 의 최고값

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
                    prompts.render_observation(world, a, cfg, KNOB, inbox)))
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
        obs = prompts.render_observation(world, a, cfg, KNOB, [])
        for token in ("wellness", "national", "facility", "propose_vote",
                      "invest"):
            assert token in obs, f"{a.native_lang} 관측에 토큰 '{token}' 이 없다"


def test_roster_lists_everyone(world_cfg):
    """누가 존재하는지는 공개 정보다.

    없으면 에이전트가 서로를 부를 수 없어 소통이 구조적으로 불가능하다 —
    실측에서 speak 40건이 전부 unknown recipient 로 실패했다.
    """
    world, cfg = world_cfg
    for aid in ("Asla1", "Ranoa2", "Miris3"):
        obs = prompts.render_observation(world, world.agents[aid], cfg, KNOB, [])
        for other in world.agents:
            assert other in obs, f"{aid} 의 관측에 {other} 가 없다"
        lang = world.agents[aid].native_lang
        assert prompts.T[lang]["roster_you"] in obs      # 자기 표시


def test_roster_reveals_no_state(world_cfg):
    """명단에는 id 와 소속만. 진척·예산·언어 능력은 spec 4.1 의 금지 목록이다."""
    world, cfg = world_cfg
    world.agents["Ranoa2"].known_langs = {"zh", "ja"}
    world.countries["Ranoa"].progress = 777.0
    obs = prompts.render_observation(world, world.agents["Asla1"], cfg, KNOB, [])
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


def test_system_says_you_may_fund_another_nation_but_not_how_to_find_out():
    """**낼 수 있다는 사실만 적는다** (8/25 · Eddie).

    전에는 「무엇을 짓는지는 말해서 확인하는 수밖에 없다」 를 짝으로 붙였다. 그 문장은
    사실이지만 **말하라는 지시**다 — 대화가 유일한 길이라는 것은 규칙에서 따라 나와야
    하고, 그것을 스스로 알아내는지가 관측 대상이다.

    낼 수 있다는 것은 남긴다. 그것 없이는 선택지 자체를 모른다 — 실측에서 885 가 남의
    **벙커** 로 들어간 적이 있는데, 그건 선택지를 알고 잘못 고른 것이라 관측이다.
    """
    from domains.meteor.prompts import SYSTEM
    must = {"ja": "他国のものにも出せます", "zh": "也可以投别国的",
            "fr": "comme à celle d'une autre"}
    gone = ("話して確かめる", "交谈去弄清", "le savoir que d'en parler")
    for lang, needle in must.items():
        assert needle in SYSTEM[lang], (lang, needle)
        for g in gone:
            assert g not in SYSTEM[lang], (lang, g)


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
      ① 진척 줄을 **언제나** 적는다. 「0%」 면 쌓인다는 것이 숫자의 모양으로 보인다.
      ② 비용 칸은 **한 번의 값**(AP)이다. 회당 몇 %인지는 주석이 말한다.

    **%로 바꿨다** (8/25 · Eddie). 「0 / 200」 이었는데 200 이라는 절대 수치는 뜻이 없다 —
    목표는 늘 100% 이므로 분모를 적지 않는다.

    **정가·할인가를 여기 적지 않는다** — 8/20 에 L 을 600 에서 200 으로 내리고 할인을
    정액으로 바꿨을 때, 박아 둔 600·300 이 다섯 파일에서 같이 깨졌다.
    """
    import itertools
    import random

    from core import config, loop
    from domains.meteor import prompts
    cfg = config.load("configs/base.yaml")
    L = cfg.costs.learn_base
    world = loop.init_world(cfg, itertools.count(1), random.Random(1))
    world.turn = 1
    # **목표는 어느 말이든 100%다** (8/25) — 다른 것은 회당 몇 %인지다.
    for aid, marks in (("Asla2", ("0%",)), ("Ranoa1", ("0%",)), ("Miris1", ("0%",))):
        agent = world.agents[aid]
        agent.lang_progress = {}
        obs = prompts.system_for(agent, world, cfg, KNOB)
        for m in marks:
            assert m in obs, (aid, m)
        # **비용 칸은 한 번의 값이다.** 총액이 비용 칸에 있으면 「한 번에 그만큼 나간다」
        # 로 읽힌다 — 그 숫자는 바로 아래 진척 줄이 말한다.
        learn_lines = [l for l in obs.splitlines()
                       if any(k in l for k in ("learn（", "learn ("))]
        assert learn_lines
        # 비용 칸은 AP 이고, 목표(learn_base)는 어디에도 안 나온다
        for l in learn_lines:
            assert f"{cfg.ap.unit:.2f}" in l and f"{L:.0f}" not in l, l

    # 낸 것이 있으면 그대로 보인다 — 절반을 냈으면 50%
    a = world.agents["Asla2"]
    a.lang_progress = {"zh": L / 2}
    assert "50%" in prompts.system_for(a, world, cfg, KNOB)



# **`test_the_observation_says_money_carries_over` 를 지웠다** (8/25 · AP 전면 통일) — 이월이 없다.

def test_learn_and_invest_share_one_unit():
    """learn 도 invest 도 **한 번에 같은 AP** 다. 규칙이 여럿이면 문구가 갈리고, 실제로
    「wellness は無料」 라는 거짓이 그렇게 남았다.

    **돈은 8/22 부터 갈린다.** invest 액수가 사람마다 다르고(`invest_mult`), 학습은 그
    배수를 안 탄다 — 학습 눈금은 `x̂` 를 재는 자이므로 사람마다 달라지면 그 자가 흔들린다.
    그래서 「같은 AP · 같은 규칙」 은 그대로고, 액수만 개체에 따라 다르다.
    """
    import itertools
    import random

    from core import config, loop
    from domains.meteor import prompts
    c = config.load("configs/base.yaml")
    for gone in ("learn_full", "invest_wellness", "unit_ap"):
        assert not hasattr(c.ap, gone) or gone == "unit_ap"
    assert not hasattr(c.facility, "invest_per_ap")

    world = loop.init_world(c, itertools.count(1), random.Random(1))
    world.turn = 1
    a = world.agents["Asla2"]
    obs = prompts.system_for(a, world, c, KNOB)
    rows = [l for l in obs.splitlines()
            if "learn（" in l or l.startswith("  invest ")]   # 헤더는 들여쓰기가 없다
    assert len(rows) == 3                       # 학습 둘 + invest 하나
    for l in rows:                              # 셋이 같은 AP 를 쓴다
        assert f"{c.ap.unit:g}" in l, l
    # **학습은 %로 적고, invest 는 아무 수치도 안 적는다** (8/26). 학습의 절대 수치(40)는
    # 목표를 모르면 뜻이 없어 「회당 몇 %」 로 바꿨고, invest 의 「옮기는 양」 은 아예
    # 지웠다 — AP 가 유일한 단위이기로 했고, 그 값이 국가 효율을 나눗셈으로 드러냈다.
    learn_rows = [l for l in rows if "learn（" in l]
    inv_row = next(l for l in rows if l.startswith("  invest "))
    # 사유(국내 구사자·부모)가 붙으면 배속이 올라 %도 오른다 — 셋 중 하나여야 한다
    base = c.costs.unit / c.costs.learn_base * 100
    ok = {f"{base * (1 + c.costs.learn_speedup * r):.0f}%" for r in (0, 1, 2)}
    for l in learn_rows:
        assert any(x in l for x in ok), (l, ok)
    # invest 줄에 남는 숫자는 **AP 하나뿐**이다
    import re as _re
    nums = _re.findall(r"\d+(?:\.\d+)?", inv_row)
    assert nums == [f"{c.ap.unit:.2f}"], (inv_row, nums)

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
    # **「16 년 더」 로 읽히면 안 된다** (8/21). 「だいたい 16 年ほど生きます」 를 모델들이
    # 남은 수명으로 읽었다 — 16세 에이전트가 "私の寿命はあと60年ほど" 라고 계산하고,
    # 「죽을 때가 가까워지면 procreate 하겠다」 는 조건이 영원히 충족되지 않았다.
    # 30해에 procreate 가 1건이었던 이유다. **나이에 붙여 적는다.**
    marks = {"ja": f"{life:.0f} 歳ごろまでに亡くなります",
             "zh": f"{life:.0f} 岁前后离世",
             "fr": f"meurent vers {life:.0f} ans"}
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

    a = world.agents["Ranoa2"]; a.ap = 1.0
    for name, args in (("propose_vote", {"reasoning": "r"}),
                       ("vote", {"choice": "bunker", "reasoning": "r"})):
        r, _ = execute_tool(name, args, world, a, c, Sink(), KNOB)
        assert not r["ok"], name
        assert "turn" not in r["error"], (name, r["error"])
        assert str(FIRST_YEAR + ballot - 1) in r["error"], (name, r["error"])

    # 그물을 넓힌다 — 소스의 error 문자열에 「turn」 이 다시 들어오면 잡는다
    src = pathlib.Path("core/agent_loop.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        if '"error"' not in line and "error\":" not in line:
            continue
        assert not re.search(r"\bturn\b", line.replace("end_turn", "")), line


def test_no_tool_response_says_turn():
    """**성공 응답은 「년」 통일의 그물 밖이었다** (#43).

    바로 위 테스트는 `"error"` 가 든 줄만 봤다. 그래서 `propose_vote` 의 **성공** 응답이
    `{"ballot_turn": 5}` 로 내부 인덱스를 그대로 돌려주는 것을 못 잡았다 — 같은 採決을
    `vote` 의 실패 응답은 「year 46」 이라고 부르고 있었는데도.

    문자열만 보지 않고 **키까지** 본다. 값을 연도로 고쳐도 키가 `_turn` 이면 그 눈금을
    계속 말한다.
    """
    import itertools
    import random

    from core import config, loop
    from core.agent_loop import Sink, execute_tool
    c = config.load("configs/base.yaml")
    world = loop.init_world(c, itertools.count(1), random.Random(1))
    world.turn = 10

    calls = [("propose_vote", {"reasoning": "r"}),
             ("invest", {"target": "facility", "reasoning": "r"}),
             # Asla1 은 초기화로 Ranoa 말을 이미 안다 (`init_world` 가 나라마다 한 명)
             ("learn", {"country": "Miris", "reasoning": "r"}),
             ("observe_risk", {"reasoning": "r"}),
             ("speak", {"to": "Asla2", "text": "こんにちは", "reasoning": "r"})]
    for name, args in calls:
        a = world.agents["Asla1"]; a.ap = 1.0
        res, _ = execute_tool(name, args, world, a, c, Sink(), KNOB)
        assert res["ok"], (name, res)
        for k in res:
            assert "turn" not in k, (name, k, res)


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
    obs = prompts.system_for(world.agents["Miris1"], world, cfg, KNOB)
    lines = obs.splitlines()
    i = next(n for n, l in enumerate(lines) if "learn (langue de" in l)
    assert f"{cfg.ap.unit:.2f}" in lines[i]                      # 한 번의 행동력
    assert "%" in lines[i] and "%" in lines[i + 1]                # 회당 %와 누적 %


def test_the_year_and_the_steps_inside_it_are_explained():
    """**한 해와 그 안의 手番을 모델들이 섞고 있었다.**

    순차 라운드로빈은 **스텝 단위**로 돈다 (`run_turn_roundrobin`): 한 사람이 한 응답을
    내면 다음 사람으로 넘어가고, AP 가 남은 사람끼리 다시 돈다. 그러니

        한 응답에 여러 도구  →  그 전부가 남들보다 먼저 일어난다
        나눠서 부르면        →  그 사이에 남들이 움직이고, 그 결과가 내 다음 차례에 보인다

    이 구조가 프롬프트에 **한 줄도 없었다.** 그 결과 실측에서
      · 매 스텝 AP 산수를 처음부터 다시 했고 (gemma 는 그 재계산이 상한을 먹어 30% 잘림)
      · 採決일과 스텝 사이에 도착한 메시지를 놓쳤다

    **사실만 적는다.** 한 응답에 몰아 넣는 것이 유리한지 나눠 부르는 것이 유리한지는
    적지 않는다 — 그건 전략이고, 적으면 지시가 된다.
    """
    from core import config
    from domains.meteor import prompts
    cfg = config.load("configs/base.yaml")

    for lang in ("ja", "zh", "fr"):
        s = prompts.T[lang]["steps"]
        assert s and "\n" in s, lang                      # 세 사실을 줄로 나눈다
        assert len(s.splitlines()) >= 3, lang

    # **유리·불리를 말하지 않는다.** 이런 말이 들어오면 사실이 아니라 조언이 된다.
    ADVICE = ("有利", "不利", "べきです", "したほうが", "应该", "最好", "建议",
              "vous devriez", "il vaut mieux", "conseill")
    for lang in ("ja", "zh", "fr"):
        s = prompts.T[lang]["steps"]
        for w in ADVICE:
            assert w not in s, (lang, w)

    # 그리고 실제로 관측에 실린다
    import itertools
    import random
    from core import loop
    world = loop.init_world(cfg, itertools.count(1), random.Random(1))
    world.turn = 1
    for aid, mark in (("Asla1", "行動力が残っている間"),
                      ("Ranoa1", "只要还有行动力"),
                      ("Miris1", "l'année n'est pas terminée")):
        obs = prompts.system_for(world.agents[aid], world, cfg, KNOB)
        assert mark in obs, aid


def test_the_route_decides_which_language_the_agent_may_write():
    """**경로가 언어를 정한다** (8/22).

    실측에서 zh 에이전트가 `ai` 로 **일본어**를 보냈다. 번역기에 이미 도착 언어를 넣는
    셈이라, `src_lang → dst_lang` 손실을 재는 지표 7 이 무의미해진다.

        ai        모국어로 써야 한다 — 여기가 측정 채널이다
        original  아는 말 아무거나 — 번역이 없으니 잴 손실이 없고, 그래야
                  `direct_works()` 의 「발신자가 수신 언어를 안다 → 통한다」 가 사실이 된다

    SYSTEM 이 그 둘을 **갈라서** 말해야 한다. 「반드시 모국어」 만 적으면 `original` 에서
    상대국 말을 쓸 수 있다는 것이 전달되지 않고, 아무 말도 안 적으면 `ai` 가 오염된다.
    """
    from domains.meteor import prompts
    for lang, (own, route_ai, route_orig) in {
            "ja": ("日本語", "`ai`", "`original`"),
            "zh": ("中文", "`ai`", "`original`"),
            "fr": ("français", "`ai`", "`original`")}.items():
        # **경로 문장은 SYSTEM 본문에서 빠졌다** (8/25) — AI 유무로 갈리는 유일한 줄이라
        # `{route_lang}` 로 두고 렌더 때 끼운다. 그러니 렌더한 뒤를 봐야 한다.
        sysmsg = prompts.SYSTEM[lang].format(life=10, route_lang=prompts.ROUTE_LANG[lang])
        assert route_ai in sysmsg and route_orig in sysmsg, lang
        assert own in sysmsg, lang
        # **두 경로가 같은 문장 안에서 갈린다** — 한쪽만 적으면 갈렸다고 할 수 없다
        line = next(l for l in sysmsg.splitlines() if "`ai`" in l)
        assert "`original`" in line, lang

    # 도구 설명도 같은 것을 말한다 (관측과 스키마가 어긋나면 무엇을 믿을지 알 수 없다)
    from core import tools
    d = next(t["function"]["parameters"]["properties"]["text"]["description"]
             for t in tools.TOOLS if t["function"]["name"] == "speak")
    assert "`ai`" in d and "`original`" in d
    # **AI 가 없는 세계** (8/25) — 경로가 하나뿐이고 그 사실이 SYS 와 스키마 둘 다에 있다
    from core import tools as _t
    for lang in ("ja", "zh", "fr"):
        no_ai = prompts.SYSTEM[lang].format(life=10, route_lang=prompts.ROUTE_LANG_NO_AI[lang])
        assert "`ai`" not in no_ai, lang
        assert "`original`" in no_ai, lang
    from core import config as _c
    sp = next(x for x in _t.tools_for(_c.load("configs/base.yaml"), ai=False)
              if x["function"]["name"] == "speak")
    assert sp["function"]["parameters"]["properties"]["route"]["enum"] == ["original"]
    # **「아무 언어나」 를 없앴다** (8/25 · #44). 도구 설명은 정적이라 나라별 안내를
    # 담을 수 없으므로(목록이 모듈 상수) 관측을 가리킨다.
    assert "own language" in d and "the observation names for that destination" in d
    assert "any language you can handle" not in d


def test_the_fact_that_people_differ_is_stated_but_not_the_numbers():
    """**안 적으면 이 기제가 죽는다.**

    남의 소득·처리량은 관측에 없고, 물어보려면 「다를 수 있다」 를 먼저 의심해야 한다.
    근거가 없으면 「모두 나와 같겠지」 가 합리적 기본값이고, 그러면 대화가 시작되지 않는다.

    이 프로젝트에서 같은 부류를 여러 번 겪었다 — 진척 합산 불가는 예시를 못 박고 나서야
    붙었고, 경로별 보장 여부는 나라별로 적어야 했고(자기가 아는 말의 나라에 24원짜리
    `ai` 를 6번 썼다), `give` 는 도구가 있는데 0건이었다.

    **사실만 적는다.** 단계값·평균·「그러니 교환하라」 는 적지 않는다 — 그건 전략이고,
    적으면 지시가 된다.
    """
    from core import config
    from domains.meteor import prompts
    cfg = config.load("configs/base.yaml")
    # **「남의 것은 안 보인다」 는 지웠다** (8/25 · Eddie). 그건 물어보라는 지시다 —
    # 남의 값이 관측에 없다는 것은 관측을 보면 알 수 있고, 그것을 알아채는지가 관측
    # 대상이다. 남기는 것은 **다르다는 사실** 하나다.
    marks = {"ja": "人によって違います", "zh": "因人而异",
             "fr": "varie d'une personne à l'autre"}
    gone = ("他人の分は見えません", "别人的数值你看不到", "ne vous sont pas visibles")
    for lang, differ in marks.items():
        sysmsg = prompts.SYSTEM[lang]
        assert differ in sysmsg, lang        # 다르다는 사실
        for g in gone:
            assert g not in sysmsg, (lang, g)

    # **숫자는 적지 않는다** — 단계값도, 평균도
    for lang in ("ja", "zh", "fr"):
        blob = prompts.SYSTEM[lang]
        for v in cfg.facility.throughput_spread:
            if v != 1.0:
                assert f"{v}" not in blob, (lang, v)

    # **「그러니 ~하라」 도 적지 않는다**
    ADVICE = ("交換", "訊いて", "聞いて", "べきです", "交换", "应该", "问一问",
              "échangez", "demandez", "vous devriez")
    for lang in ("ja", "zh", "fr"):
        for w in ADVICE:
            assert w not in prompts.SYSTEM[lang], (lang, w)


def test_every_cost_row_is_titled_by_its_tool_name():
    """**제목은 도구 이름으로 통일한다** (8/26 · Eddie).

    `speak`·`learn` 만 자국어로 번역돼 있고 나머지 다섯(`invest`·`observe_risk`·
    `propose_vote`·`vote`·`memory_write`)은 함수명이었다:

        話す（自国内）        ← 자연스러운 행위로 읽힌다
        observe_risk         ← 기술적 조작으로 읽힌다

    같은 표 안에서 이름의 결이 갈리면 **선택에 편향이 붙는다.** 표는 값만 비교하게 해야
    한다. 괄호 안 구분과 비고는 자국어로 남긴다 — 그건 제목이 아니라 설명이고, 도구
    호출에 그대로 쓰이지도 않는다.
    """
    import itertools
    import random
    import re

    from core import config, loop
    from domains.meteor import prompts

    c = config.load("configs/base.yaml")
    w = loop.init_world(c, itertools.count(1), random.Random(1))
    w.turn = 1
    TOOLS = ("speak", "learn", "invest", "observe_risk",
             "propose_vote", "vote", "memory_write")
    for a in w.agents.values():
        table = prompts.render_costs(w, a, c, 0.10)
        # **정확히 두 칸**이 제목 줄이다. 세 칸은 딸린 설명(학습 진척 `これまで/目前/déjà`),
        # 네 칸은 나라별 안내다 — 들여쓰기가 이미 그 셋을 가르고 있다.
        rows = [l for l in table.splitlines() if re.match(r"^ {2}\S", l)]
        assert rows, a.id
        for l in rows:
            head = l.strip().split("（")[0].split(" (")[0].split()[0]
            assert head in TOOLS, (a.id, l)
        # 세 경로가 **한 도구**라는 것이 이름으로 보인다
        assert sum(1 for l in rows if l.strip().startswith("speak")) == 3, a.id
