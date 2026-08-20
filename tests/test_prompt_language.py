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
        out.append((f"SYSTEM[{lang}]", lang, prompts.system_for(a)))
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
        assert marks[a.native_lang] in prompts.system_for(a), \
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
