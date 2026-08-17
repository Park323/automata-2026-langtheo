"""밤새 도는 배치가 죽지 않게 하는 것들.

50턴 실측 하나가 이 파일을 통째로 만들었습니다. **2.5시간을 돌다 43턴에서 죽었고,
벽시계 149분 중 146분을 전체 호출의 1.4%(20건)가 먹었습니다.** 나머지 1,375건은
전부 5초 미만이었습니다.

여기서 지키는 것은 셋입니다 — 호출 하나가 무한정 끌지 못하게, 번역 실패가 런을
죽이지 못하게, 그리고 그 실패가 **세계의 사건으로 위장되지 못하게.**
"""
from __future__ import annotations

import threading
import time

import pytest

from core import config, llm, messaging


# ── 벽시계 상한 ─────────────────────────────────────────────────────────────────

class _SlowOpener:
    """느리게 응답하는 서버 흉내. `urlopen(timeout=)` 은 이걸 못 막는다 —
    소켓 읽기 하나의 제한이지 호출 전체의 제한이 아니기 때문이다."""

    def __init__(self, delay: float):
        self.delay = delay
        self.started = threading.Event()

    def __call__(self, req, timeout=None):
        self.started.set()
        time.sleep(self.delay)
        raise AssertionError("여기까지 오면 안 된다 — deadline 이 먼저 걸려야 한다")


def test_deadline_abandons_a_hanging_call(monkeypatch):
    """호출 하나가 30분씩 끄는 것을 막는다. 실측에서 최악이 1,857초였다."""
    c = llm.OpenRouterClient("m", api_key="k", retries=1, deadline=0.05)
    monkeypatch.setattr(llm.urllib.request, "urlopen", _SlowOpener(30))
    t0 = time.time()
    with pytest.raises(Exception):
        c.chat([{"role": "user", "content": "x"}])
    assert time.time() - t0 < 5, "deadline 이 안 걸렸다"


def test_deadline_failure_is_recorded_raw(monkeypatch):
    """버려진 호출도 raw 에 남아야 한다 — 안 남으면 왜 느렸는지 사후에 못 본다."""
    seen = []
    c = llm.OpenRouterClient("m", api_key="k", retries=1, deadline=0.05,
                             recorder=seen.append)
    monkeypatch.setattr(llm.urllib.request, "urlopen", _SlowOpener(30))
    with pytest.raises(Exception):
        c.chat([{"role": "user", "content": "x"}])
    assert seen and "deadline" in (seen[0]["error"] or "")


def test_fast_call_is_untouched(monkeypatch):
    """정상 호출에는 아무 영향이 없어야 한다 (1,375/1,395 가 여기 해당)."""
    import io
    import json as _json

    def _fast(req, timeout=None):
        return io.StringIO(_json.dumps({"choices": [{"message": {"content": "ok"}}]}))

    c = llm.OpenRouterClient("m", api_key="k", deadline=5)
    monkeypatch.setattr(llm.urllib.request, "urlopen", _fast)
    assert c.chat([{"role": "user", "content": "x"}])["choices"][0]["message"]["content"] == "ok"


# ── 번역 실패 ───────────────────────────────────────────────────────────────────

class _BrokenTranslator:
    def chat(self, *a, **k):
        raise RuntimeError("재시도 소진")


@pytest.fixture(scope="module")
def cfg():
    return config.load("configs/base.yaml")


def _intl(text="我们需要拦截器"):
    return {"from": "Ranoa1", "to": "Miris1", "from_country": "Ranoa",
            "to_country": "Miris", "from_lang": "zh", "to_lang": "fr",
            "text": text, "route": None}


def test_translation_failure_does_not_kill_the_run(cfg):
    """번역 호출 하나가 2.5시간짜리 런을 통째로 날렸다. 미전달로 떨어뜨린다."""
    p = messaging.process_message(_intl(), set(), cfg, _BrokenTranslator(), 24.0)
    assert p["delivered"] is False
    assert p["inbox"]["unreadable"] is True and p["inbox"]["text"] is None
    assert p["sender_notice"]["type"] == "delivery_failed"


def test_translation_failure_is_tagged_as_an_engine_fault(cfg):
    """지표 9(전달 실패)와 섞이면 '읽을 수 없어서 못 받았다' 로 오독된다."""
    p = messaging.process_message(_intl(), set(), cfg, _BrokenTranslator(), 24.0)
    assert "RuntimeError" in p["meta"]["translate_failed"]
    assert p["kind"] == "ai"           # route 는 그대로 ai — 원문 직통이 아니다


def test_engine_fault_is_counted_apart_from_metric_9(cfg):
    from tools.score import metrics
    p = messaging.process_message(_intl(), set(), cfg, _BrokenTranslator(), 24.0)
    msgs = [{"turn": 1, "msg_id": 1, "from": "Ranoa1", "to": "Miris1",
             "route": p["kind"], "delivered": p["delivered"], "meta": p["meta"]}]
    s = metrics.message_shape(msgs)
    assert s["engine_translate_failed"]["n"] == 1
    assert s["9_delivery_failure"]["n"] == 0        # route=original 이 아니므로 분모 밖


def test_engine_fault_is_not_judged_as_unreadable(cfg):
    """판정에서도 구분한다 — 같은 'skip' 으로 묶으면 원인을 사후에 못 가른다."""
    from tools.score import judge
    p = messaging.process_message(_intl(), set(), cfg, _BrokenTranslator(), 24.0)
    (r,) = judge.link([{"turn": 1, "msg_id": 1, "from": "Ranoa1", "to": "Miris1",
                        "route": p["kind"], "delivered": p["delivered"],
                        "meta": p["meta"]}], [])
    assert r["skip"] == judge.SKIP_TRANSLATE_FAILED


def test_domestic_path_never_touches_the_translator(cfg):
    """자국 내 메시지는 번역기를 안 탄다 — 번역기가 죽어도 국내 소통은 살아 있다.

    4c(번역 없는 기저선)가 엔진 장애에 오염되지 않는다는 뜻이기도 하다.
    """
    m = dict(_intl("Bonjour"), to_country="Ranoa", to_lang="zh", to="Ranoa2")
    p = messaging.process_message(m, set(), cfg, _BrokenTranslator(), 24.0)
    assert p["kind"] == "domestic" and p["delivered"] is True


# ── HTTP 200 인데 error 페이로드 ──────────────────────────────────────────────

def test_error_payload_with_status_200_is_a_failure(monkeypatch):
    """프로바이더가 **200 에 error 를 실어 보낸다.**

    gemma :free 에서 22콜 중 5건이 `{"error":{"code":504,…}}` 였고, choices 를 그대로
    인덱싱하다 KeyError 로 **런 전체가 죽었다.** 예외로 바꿔야 재시도·백오프를 탄다.
    """
    import io
    import json as _json

    def _err(req, timeout=None):
        return io.StringIO(_json.dumps(
            {"error": {"message": "Provider timed out after 11686ms", "code": 504}}))

    seen = []
    c = llm.OpenRouterClient("m", api_key="k", retries=1, recorder=seen.append)
    monkeypatch.setattr(llm.urllib.request, "urlopen", _err)
    with pytest.raises(Exception) as ei:
        c.chat([{"role": "user", "content": "x"}])
    assert "504" in str(ei.value)
    assert seen and seen[0]["response"]["error"]["code"] == 504   # 원본은 raw 에 남는다


def test_malformed_response_kills_only_that_agent(cfg):
    """모양이 다른 응답에 인덱싱하다 터지면 스레드 풀을 타고 런 전체가 죽는다."""
    import itertools
    from core import loop
    from core.agent_loop import Sink, run_agent_turn
    from domains.meteor import prompts

    class _Weird:
        def chat(self, *a, **k):
            return {"unexpected": True}

    w = loop.init_world(cfg, itertools.count(1))
    a = w.agents["Asla1"]; a.ap, a.budget = 1.0, 500.0
    lg = run_agent_turn(w, a, cfg, _Weird(), Sink(), 48.0,
                        prompts.system_for(a), prompts.render_observation(w, a, cfg, 48.0))
    assert lg["ended_by"] == "error" and "malformed response" in lg["error"]


# ── 재현 (8/17) ──────────────────────────────────────────────────────────────

def test_seed_and_temperature_reach_the_request(monkeypatch):
    """**temperature 0.7 에 시드만 걸면 절반만 잡힌다.** 같은 프롬프트 4회 실측:

        temp 0.7 · seed 없음  고유 4/4      temp 0.7 · seed 42  고유 2/4
        temp 0.0 · seed 없음  고유 2/4      temp 0.0 · seed 42  고유 1/4  ← 고정

    온도를 0 으로 내려야 시드가 일한다. 둘 다 요청에 실려야 그 조합이 가능하다.
    """
    import io
    import json as _json

    seen = {}

    def _fake(req, timeout=None):
        seen.update(_json.loads(req.data.decode()))
        return io.StringIO(_json.dumps({"choices": [{"message": {"content": "ok"}}]}))

    monkeypatch.setattr(llm.urllib.request, "urlopen", _fake)
    llm.OpenRouterClient("m", api_key="k", temperature=0.0, seed=42).chat(
        [{"role": "user", "content": "x"}])
    assert seen["temperature"] == 0.0 and seen["seed"] == 42

    seen.clear()
    llm.OpenRouterClient("m", api_key="k", temperature=0.7).chat(
        [{"role": "user", "content": "x"}])
    assert "seed" not in seen          # 안 주면 안 보낸다
