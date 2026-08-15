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
