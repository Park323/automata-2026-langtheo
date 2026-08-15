"""메시지 라우팅. 과제 2 Part B-1. StubClient 로 검증."""
from __future__ import annotations

from pathlib import Path

import pytest

from core import config, messaging
from core.llm import StubClient

BASE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


@pytest.fixture(scope="module")
def cfg():
    return config.load(BASE)


def _translator(text="TRANSLATED"):
    return StubClient([{"role": "assistant", "content": text, "tool_calls": []}])


def _sent(**over):
    base = {"kind": "speak", "from": "Asla1", "from_country": "Asla", "from_lang": "ja",
            "to": "Ranoa2", "to_country": "Ranoa", "to_lang": "zh", "route": "ai",
            "text": "본문", "translate_instruction": None, "reply_to": None}
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
                                  translator=_translator(), knob_ai=48)
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


def test_cost(cfg):
    assert messaging.cost("domestic", cfg, 48) == cfg.costs.comm_domestic
    assert messaging.cost("original", cfg, 48) == cfg.costs.comm_intl_learner
    assert messaging.cost("ai", cfg, 48) == 48


def test_original_fail_when_cannot_read(cfg):
    """6. original 인데 수신자가 발신 언어를 모르면 본문 미전달 + 발신자 실패 통지."""
    m = messaging.process_message(_sent(route="original"),
                                  recipient_known_langs={"zh"},   # ja 를 모름
                                  cfg=cfg, translator=None, knob_ai=48)
    assert m["delivered"] is False
    assert m["inbox"]["text"] is None
    assert m["inbox"]["unreadable"] is True
    assert m["inbox"]["from"] == "Asla1"                    # 발신자·도착 사실만
    assert m["sender_notice"]["type"] == "delivery_failed"


def test_original_success_when_can_read(cfg):
    m = messaging.process_message(_sent(route="original"),
                                  recipient_known_langs={"zh", "ja"},
                                  cfg=cfg, translator=None, knob_ai=48)
    assert m["delivered"] is True
    assert m["inbox"]["text"] == "본문"
    assert m["inbox"]["label"] is None                   # 원문 직통엔 라벨 없음


# ── 원문 병기 (spec 5.1) ──────────────────────────────────────────────────────

def test_ai_label_and_original_juxtaposition(cfg):
    """7. ai 경로 + 수신자가 발신 언어를 알면 번역문 + 원문 둘 다."""
    reader = messaging.process_message(_sent(route="ai"),
                                       recipient_known_langs={"zh", "ja"},
                                       cfg=cfg, translator=_translator("译文"), knob_ai=48)
    assert reader["inbox"]["label"] == messaging.AI_LABEL
    assert reader["inbox"]["text"] == "译文"
    assert reader["inbox"]["original"] == "본문"          # 학습자 → 원문 병기

    non_reader = messaging.process_message(_sent(route="ai"),
                                           recipient_known_langs={"zh"},   # ja 모름
                                           cfg=cfg, translator=_translator("译文"), knob_ai=48)
    assert non_reader["inbox"]["original"] is None        # 못 읽으면 원문 없음
