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
from core.llm import LLMCallError


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
    """**경계가 선언한 실패**를 낸다 (LLMCallError). 전에는 맨 RuntimeError 였는데,
    상류가 `except Exception` 이라 아무 예외나 통했다 — 무엇을 흉내내는지가 흐렸다."""

    def chat(self, *a, **k):
        raise LLMCallError("재시도 소진")


class _BuggyTranslator:
    """**우리 코드의 버그**를 흉내낸다. 이건 삼켜져선 안 된다."""

    def chat(self, *a, **k):
        raise KeyError("dst_lang")


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
    assert p["sender_notice"]["type"] == "delivery_failed"


def test_engine_failure_never_says_it_was_the_language(cfg):
    """**엔진 장애를 「상대가 그 언어를 읽지 못한다」 로 통지하고 있었다.**

    상대의 언어 능력과 아무 상관 없는 일인데 그것을 언어 사실로 심는다 — 이 실험의
    핵심 변수(누가 무엇을 읽는가)를 에이전트의 머릿속에서 오염시킨다. 원문 직통의
    실패는 세계의 사실이라 원인을 붙이지만, 엔진 장애는 「닿지 않았다」 만 사실이다.
    """
    from domains.meteor import prompts
    eng = messaging.process_message(_intl(), set(), cfg, _BrokenTranslator(), 24.0)
    assert eng["sender_notice"]["reason"] == "engine"
    # 수신자에게는 흔적을 남기지 않는다 — ai 경로였으므로 엔진이 살아 있었다면
    # 읽을 수 있게 도착했다. 「읽을 수 없는 메시지가 왔다」 도 같은 거짓이다.
    assert eng["inbox"] is None

    lang = messaging.process_message({**_intl(), "route": "original"}, set(), cfg,
                                     _BrokenTranslator(), 24.0)
    assert lang["sender_notice"]["reason"] == "unreadable"
    assert lang["inbox"]["unreadable"] is True and lang["inbox"]["text"] is None

    # 통지 문구가 실제로 갈린다
    def notice(reason):
        return prompts.render_inbox(
            [{"msg_id": 1, "from": None, "text": None, "label": None, "original": None,
              "delivery_failed_to": "Miris1", "delivery_failed_reason": reason}], "ja")
    assert "読めません" in notice("unreadable")
    assert "読めません" not in notice("engine")
    assert "届きませんでした" in notice("engine")      # 사실은 그대로 전한다


def test_translation_failure_is_tagged_as_an_engine_fault(cfg):
    """지표 9(전달 실패)와 섞이면 '읽을 수 없어서 못 받았다' 로 오독된다."""
    p = messaging.process_message(_intl(), set(), cfg, _BrokenTranslator(), 24.0)
    assert "LLMCallError" in p["meta"]["translate_failed"]
    assert p["kind"] == "ai"           # route 는 그대로 ai — 원문 직통이 아니다


def test_a_code_bug_is_not_swallowed_as_a_translation_failure(cfg):
    """**런이 터지더라도 버그는 잡아야 한다.**

    상류가 `except Exception` 이라 우리 코드의 버그가 "번역 실패" 통계 한 줄로 묻혔다.
    실제로 이 테스트 파일 옆에서 `translator=None` 을 넘기는 테스트가 **삼켜진
    AttributeError 를 보고 통과하고 있었다** — 검증한 것이 규칙이 아니라 버그였다.
    """
    with pytest.raises(KeyError):
        messaging.process_message(_intl(), set(), cfg, _BuggyTranslator(), 24.0)


def test_the_agent_turn_also_only_swallows_declared_failures(cfg):
    """에이전트 쪽도 같다. API 실패로 50턴 런이 죽으면 안 되지만, 프롬프트 렌더링·도구
    실행의 버그까지 삼키면 **그 에이전트가 매 턴 조용히 아무것도 못 한다** — 로그에
    `error` 한 줄만 남고 원인을 찾을 방법이 없다."""
    import itertools
    import random

    from core import loop
    from core.agent_loop import Sink, run_agent_turn
    from domains.meteor import prompts

    world = loop.init_world(cfg, itertools.count(1), random.Random(1))
    world.turn = 1
    a = world.agents["Asla1"]
    obs = prompts.render_observation(world, a, cfg, 48.0)

    class _Api:
        def chat(self, *ar, **kw):
            raise LLMCallError("HTTP 503 Service Unavailable")

    lg = run_agent_turn(world, a, cfg, _Api(), Sink(), 48.0, prompts.system_for(a, None, cfg), obs)
    assert lg["ended_by"] == "error" and "LLMCallError" in lg["error"]

    class _Bug:
        def chat(self, *ar, **kw):
            raise AttributeError("NoneType has no attribute 'get'")

    with pytest.raises(AttributeError):
        run_agent_turn(world, a, cfg, _Bug(), Sink(), 48.0, prompts.system_for(a, None, cfg), obs)


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
                        prompts.system_for(a, None, cfg), prompts.render_observation(w, a, cfg, 48.0))
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


# ── 터진 자리가 디스크에 남는가 (8/18) ──────────────────────────────────────

def test_a_crash_leaves_enough_to_debug_it(tmp_path, cfg):
    """**예외를 좁혀 버그를 드러내기로 했으니, 드러난 자리가 남아야 한다.**

    전에는 `summary.json` 에 `"KeyError: dst_lang"` 300자뿐이었고 트레이스백은 stderr
    로만 갔다 — 야간 배치나 nohup 이면 스택이 그대로 사라진다. 런이 터지는 것은
    괜찮지만, 터진 자리를 못 찾으면 좁힌 뜻이 절반만 이뤄진다.
    """
    import json

    from core.run_io import RunWriter

    w = RunWriter("crash", cfg_raw={"x": 1}, root=tmp_path)
    w.last_turn = 7
    try:
        raise KeyError("dst_lang")
    except KeyError as e:
        e.add_note("[agent Ranoa2 · turn 7 · age 4]")
        w.crash(e, where="run")
    w.close({"final": {"outcome": "aborted"}})

    (row,) = [json.loads(l) for l in (w.dir / "events.jsonl").read_text().splitlines()]
    # 7턴까지 끝났고 8턴이 돌던 중이었다. 한 값만 적으면 "시작도 못 했다" 로 읽힌다.
    assert row["type"] == "crash"
    assert (row["turn"], row["last_completed_turn"]) == (8, 7)
    assert row["exc"] == "KeyError"
    assert "Ranoa2" in row["notes"][0]          # 어느 에이전트였는지
    assert "test_robustness" in row["traceback"]   # 스택이 통째로 남는다


def test_the_failing_agent_is_named_in_the_traceback(cfg):
    """병렬이면 ThreadPoolExecutor 가 예외를 주 스레드로 옮기는데, 그때 **어느
    에이전트였는지가 사라진다** — 프레임에 run_agent_turn 만 남고 aid 값은 안 보인다."""
    import random

    from core import loop
    from domains.meteor import prompts

    class _Bug:
        def chat(self, *a, **k):
            raise AttributeError("boom")

    bug = _Bug()
    with pytest.raises(AttributeError) as ei:
        loop.run_agentic(cfg, random.Random(1), lambda aid: bug, bug, 48.0,
                         render_obs=prompts.render_turn_open,
                         system_prompt=prompts.system_for,
                         parallel=True, sim_turns=1)
    notes = " ".join(getattr(ei.value, "__notes__", []) or [])
    assert "agent" in notes and "turn 1" in notes
