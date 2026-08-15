"""번역 호출. 과제 2 Part B-2. StubClient 로 검증 (API 안 씀)."""
from __future__ import annotations

from core import translate as T
from core.llm import StubClient


def _stub(text):
    return StubClient([{"role": "assistant", "content": text, "tool_calls": []}])


def test_default_directive_no_compression_words():
    """9. translate_instruction=None → 프롬프트에 '간결/정확/자연' 류가 없다."""
    p = T.build_prompt("fr", "본문", None)
    for banned in ["간결", "정확", "자연", "요약", "concise", "brief", "summar"]:
        assert banned not in p
    assert p.splitlines()[0] == "Translate to French."    # 대상 언어 사실만


def test_system_contract_is_io_only():
    """출력 형식 계약에 방식 지시어가 없다 (spec 5.2)."""
    for banned in ["간결", "정확", "자연", "concise", "accurate", "faithful"]:
        assert banned not in T.SYSTEM_CONTRACT


def test_agent_instruction_passthrough():
    """지시를 쓰면 그 문장이 그대로 들어간다."""
    p = T.build_prompt("zh", "hello", "핵심만 짧게")
    assert "핵심만 짧게" in p


def test_translate_returns_content():
    out = T.translate(_stub("TRANSLATED"), "ja", "zh", "원문")
    assert out["text"] == "TRANSLATED"
    assert "Chinese" in out["prompt"]        # 대상 언어(사실)는 붙는다
