"""공개 등급 — 무엇을 누가 알 수 있는가. spec 4.1 을 코드로 옮긴 표의 시험.

**청중이 호출부마다 흩어져 있던 것이 이 파일의 이유다.** 부고는 같은 나라를 훑는 루프로,
출자 결과는 출자자 한 명에게, 메시지는 수신자에게 — 각각 다른 곳에서 각각의 방식으로
정했다. 그래서 두 번 어긋났다.

  · 타국 출자의 진척 증가분 — 액수를 쌓으면 상대국 생산배수가 복원됐다 (실측 1.13)
  · 「wellness 는 수명을 늘린다」 — λ 곡선을 말로 알려주고 있었다

그리고 **부고의 청중은 테스트가 아예 없었다.** 그 테스트는 기록만 확인하고
*"배선은 run_turn_agentic 에 있으므로"* 라고 적어 두고 넘어갔다 — PUBLIC 에서 GLOBAL 로
바꿔도 아무것도 실패하지 않았다.
"""
from __future__ import annotations

import itertools
import random

import pytest

from core import config, loop, visibility
from core.visibility import Vis

BASE = "configs/base.yaml"


@pytest.fixture()
def cfg():
    return config.load(BASE)


@pytest.fixture()
def world(cfg):
    w = loop.init_world(cfg, itertools.count(1), random.Random(1))
    w.turn = 5
    return w


# ── 표 자체 ─────────────────────────────────────────────────────────────────

def test_every_fact_declares_a_level_and_a_reason():
    """**숨기는 것에는 근거가 있어야 한다.**

    `budget` 은 「내 예산」 이 곧 이유다 — 자명한 것에 산문을 요구하면 잡음이 된다.
    그런데 SECRET 은 다르다. **무엇을 숨기는지는 우리가 고른 것**이고, 그 선택이 실험의
    결과를 정한다 — 고를 때마다 왜 그랬는지가 남아야 다음 사람이 다시 검토할 수 있다.
    """
    for fact, (vis, why) in visibility.FACTS.items():
        assert isinstance(vis, Vis), fact
        assert why.strip(), fact
        if vis is Vis.SECRET:
            assert len(why) > 15, f"{fact}: 숨기는 근거가 너무 짧다"


def test_an_undeclared_fact_is_refused():
    """표에 없는 것을 보내려 하면 막는다 — **청중을 즉흥으로 정하지 않게.**"""
    with pytest.raises(KeyError) as e:
        visibility.level("some_new_thing")
    assert "FACTS" in str(e.value)                 # 어디에 적어야 하는지 알려준다


def test_the_hidden_list_of_spec_4_1_is_all_secret():
    """spec 4.1 은닉 목록이 표와 어긋나면 안 된다. **어긋난 것을 두 번 겪었다.**"""
    must_be_secret = ("success_prob", "lifespan_lambda", "hazard_curve", "wellness_gain",
                      "threshold_truth", "impact_turn_truth", "growth_fn",
                      "inner_reasoning", "other_nation_state")
    for fact in must_be_secret:
        assert visibility.level(fact) is Vis.SECRET, fact


# ── 라우터 ──────────────────────────────────────────────────────────────────

def test_secret_reaches_nobody_not_even_the_actor(world):
    """**행위자조차 모른다.** wellness 가 자기 수명을 얼마 늘렸는지 본인도 모르는 것이
    이 등급의 뜻이다 — 알면 수명이 계산 가능한 투자가 된다."""
    assert visibility.audience(world, "wellness_gain", actor="Asla1") == []
    assert visibility.audience(world, "lifespan_lambda", actor="Asla1") == []


def test_private_reaches_only_the_actor(world):
    assert visibility.audience(world, "fac_gain", actor="Asla1") == ["Asla1"]
    assert visibility.audience(world, "memory", actor="Ranoa2") == ["Ranoa2"]
    # 없는 사람에게는 아무것도 가지 않는다 (사망·교체)
    assert visibility.audience(world, "fac_gain", actor="Asla99") == []


def test_public_stays_inside_the_nation(world):
    """국토·진척·採決은 그 나라 사람들의 것이다. **타국은 소통으로만** 안다 (4.1)."""
    got = visibility.audience(world, "ballot_result", nation="Ranoa")
    assert got == ["Ranoa1", "Ranoa2", "Ranoa3"]
    assert not any(a.startswith(("Asla", "Miris")) for a in got)
    # nation 을 안 주면 actor 의 나라로 본다
    assert visibility.audience(world, "progress", actor="Miris2") == \
        ["Miris1", "Miris2", "Miris3"]


def test_global_reaches_everyone_alive(world):
    got = visibility.audience(world, "obituary", actor="Asla1", nation="Asla")
    assert len(got) == 9 and got == sorted(got)
    world.agents["Miris3"].alive = False
    assert "Miris3" not in visibility.audience(world, "obituary", actor="Asla1")


# ── 부고가 GLOBAL 이 된 것 ──────────────────────────────────────────────────

def test_an_obituary_now_reaches_every_nation(cfg, world):
    """**8/20 에 PUBLIC → GLOBAL.**

    `roster` 가 이미 교체를 드러낸다 — 명단에서 누가 사라지고 누가 왔는지 보인다. 그래서
    등급을 올려서 새로 새는 것은 **나이**뿐이고, 그것이 수명을 배우는 유일한 경로다
    (곡선은 여전히 SECRET, 평균만 SYSTEM).

    자기 부고는 받지 않는다. 그 자리에 태어난 아이는 **다른 사람**이라(3.2) uid 로 걸러진다.
    """
    result = loop.RunResult(world=world)
    result.deaths_log.append({"turn": 5, "who": "Ranoa1", "born": "Ranoa4",
                              "age": 14, "country": "Ranoa", "by": "natural"})
    loop._queue_obituaries(world, result, itertools.count(900))

    told = sorted(e["to"] for e in world.inbox_queue)
    assert "Asla1" in told and "Miris1" in told          # 타국도 안다
    assert "Ranoa1" not in told                          # 자기 부고는 아니다
    assert len(told) == 8                                # 9명 중 죽은 이만 빠진다
    for e in world.inbox_queue:
        assert e["msg"]["died"] == "Ranoa1" and e["msg"]["age"] == 14


def test_the_obituary_is_the_only_way_age_gets_out(cfg, world):
    """부고에 나이가 실리는 것이 **의도**다. 그 밖에 남의 나이를 아는 길은 없다 —
    관측은 자기 나이도 안 적고(해 시작 문구가 적는다), 명단은 이름만이다."""
    from domains.meteor import prompts
    obs = prompts.render_observation(world, world.agents["Asla1"], cfg, 48.0)
    for aid, a in world.agents.items():
        if aid == "Asla1":
            continue
        assert f"{a.age} 歳" not in obs, aid
    assert visibility.level("obituary") is Vis.GLOBAL
    assert visibility.level("hazard_curve") is Vis.SECRET     # 곡선은 여전히 숨는다


# ── 우회로가 없는가 ─────────────────────────────────────────────────────────

def test_nothing_reaches_an_agent_except_through_the_table():
    """**청중을 즉흥으로 정하는 길이 남아 있으면 표가 장식이 된다.**

    `world.inbox_queue.append` 는 두 곳에만 있어야 한다 — `_notify`(등급이 청중을 정한다)
    와 `_queue_obituaries`(그 안에서 `visibility.audience` 를 부른다). 그 밖에서 큐에
    직접 쓰면 등급을 우회한 것이다.
    """
    import pathlib
    import re
    src = pathlib.Path("core/loop.py").read_text(encoding="utf-8")
    # 함수 단위로 잘라 어느 함수가 큐에 직접 쓰는지 본다
    blocks = re.split(r"\n(?=def |\nasync def )", src)
    writers = {b.split("(")[0].replace("def ", "").strip()
               for b in blocks if "inbox_queue.append" in b}
    assert writers == {"_notify", "_queue_obituaries"}, writers


def test_a_message_reaches_only_its_recipient(cfg, world):
    """말은 보낸 이와 받는 이만 안다. **같은 나라 사람도 엿듣지 못한다** — 그러면
    국내 소통이 사실상 공개 방송이 되고, 「말해야만 안다」 가 무너진다."""
    from core.agent_loop import Sink
    from core.llm import StubClient
    sink = Sink()
    sink.messages.append({
        "kind": "speak", "from": "Asla1", "from_country": "Asla", "from_lang": "ja",
        "to": "Asla2", "to_country": "Asla", "to_lang": "ja", "route": None,
        "text": "OVERHEARD?", "translate_instruction": None})
    loop._settle_step(world, cfg, random.Random(0), sink,
                      StubClient([{"role": "assistant", "content": "x", "tool_calls": []}] * 3),
                      48.0, itertools.count(900), loop.RunResult(world=world), {}, [], [])
    told = [e["to"] for e in world.inbox_queue]
    assert told == ["Asla2"]                       # Asla3 도 못 듣는다


def test_a_delivery_failure_reaches_only_the_sender(cfg, world):
    """닿지 않았다는 사실은 **보낸 사람만** 안다. 받을 사람은 온 적이 없으므로 알 것이
    없고, 남이 알면 「누가 누구에게 말을 걸었나」 가 새어 나간다."""
    from core.agent_loop import Sink
    from core.llm import StubClient
    sink = Sink()
    # **씨앗 이중언어자를 피한다.** 각 나라 1번은 이웃 나라 말을 하나 알고 태어난다 —
    # Miris1 은 ja 를 읽으므로 원문이 그대로 닿아 실패가 나지 않는다.
    sink.messages.append({          # Asla2 는 fr 을 모르고 Miris2 는 ja 를 모른다
        "kind": "speak", "from": "Asla2", "from_country": "Asla", "from_lang": "ja",
        "to": "Miris2", "to_country": "Miris", "to_lang": "fr", "route": "original",
        "text": "NOT_DELIVERED", "translate_instruction": None})
    loop._settle_step(world, cfg, random.Random(0), sink,
                      StubClient([{"role": "assistant", "content": "x", "tool_calls": []}] * 3),
                      48.0, itertools.count(900), loop.RunResult(world=world), {}, [], [])
    fails = [e for e in world.inbox_queue if "delivery_failed_to" in e["msg"]]
    assert [e["to"] for e in fails] == ["Asla2"]


# ── SECRET 이 프롬프트에 새지 않는가 ────────────────────────────────────────

def test_no_secret_value_reaches_any_rendered_string(cfg):
    """**표를 기준으로 관측 전체를 훑는다.**

    지금까지 은닉 검사가 사례별로 흩어져 있었다 — 「배수가 없나」, 「임계가 없나」,
    「success_prob 이 없나」 를 각각 다른 테스트가 봤다. 그래서 새 값이 새면 그것을 보는
    테스트가 없었다.

    여기서는 SECRET 값들에 **눈에 띄는 숫자**를 박고, 에이전트가 볼 수 있는 문자열
    전부(SYSTEM+관측 · 해 시작 문구 · 사건 · 도구 응답)에 그 숫자가 나오는지 본다.
    """
    import dataclasses
    import json

    from core.agent_loop import Sink, execute_tool
    from domains.meteor import prompts

    # 우연히 겹치지 않는 값들
    probe = dataclasses.replace(
        cfg,
        world=dataclasses.replace(cfg.world, success_prob=0.4321),
        survival=dataclasses.replace(cfg.survival, lambda_base=1234.5, k=77),
        thresholds=dataclasses.replace(cfg.thresholds, interceptor=987654,
                                       bunker_scale=876543),
        growth=dataclasses.replace(cfg.growth, growth_coef=0.9753),
    )
    world = loop.init_world(probe, itertools.count(1), random.Random(1))
    world.turn = 3
    world.countries["Asla"].land = "interceptor"
    world.countries["Asla"].national_capital = 5000.0
    a = world.agents["Asla1"]
    a.ap, a.budget = 1.0, 500.0
    a.lam = 1234.5                                  # 개인의 λ

    seen = [prompts.system_for(a, world, probe, 48.0),
            prompts.render_turn_open(world, a, probe, 48.0, [])]
    sink = Sink()
    for name, args in (("observe_risk", {"reasoning": "r"}),
                       ("invest", {"target": "wellness", "reasoning": "r"}),
                       ("invest", {"target": "national", "reasoning": "r"}),
                       ("invest", {"target": "facility", "reasoning": "r"}),
                       ("learn", {"country": "Miris", "reasoning": "r"}),
                       ("speak", {"to": "Asla2", "text": "x", "reasoning": "r"})):
        a.ap = 1.0
        r, _ = execute_tool(name, args, world, a, probe, sink, 48.0)
        seen.append(json.dumps(r, ensure_ascii=False))
    # 사건도 — 이 판에서 생긴 것 전부
    loop._settle_step(world, probe, random.Random(0), sink, None, 48.0,
                      itertools.count(900), loop.RunResult(world=world), {}, [], [])
    for e in world.inbox_queue:
        seen.append(prompts.render_events(world.agents[e["to"]], [e["msg"]]))

    blob = "\n".join(seen)
    for secret in ("0.4321", "1234.5", "1234", "987654", "876543", "0.9753", "77"):
        assert secret not in blob, f"SECRET 이 새어 나갔다: {secret}"


def test_the_audit_would_actually_catch_a_leak(cfg):
    """**감사가 감사를 하는지** 본다. 위 테스트가 늘 통과하기만 하면 그물이 없는 것과
    같으므로, 일부러 새게 만들어 잡히는지 확인한다."""
    import dataclasses

    from domains.meteor import prompts
    probe = dataclasses.replace(
        cfg, thresholds=dataclasses.replace(cfg.thresholds, interceptor=987654))
    world = loop.init_world(probe, itertools.count(1), random.Random(1))
    world.turn = 3
    # 임계를 관측에 실으면 (옛 설계가 그랬다) 그 숫자가 문자열에 나타난다
    leaked = prompts.render_observation(world, world.agents["Asla1"], probe, 48.0) \
        + f"\n  interceptor に要る進捗: {probe.thresholds.interceptor}"
    assert "987654" in leaked                        # 그물이 잡을 수 있는 모양이다
    clean = prompts.render_observation(world, world.agents["Asla1"], probe, 48.0)
    assert "987654" not in clean                     # 지금은 안 새고 있다
