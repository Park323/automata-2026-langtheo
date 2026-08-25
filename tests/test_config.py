"""설정 로더와 assert 검증. 과제 1 Part A."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from core import asserts, config
from core.config import Config, ConfigError

BASE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def _raw() -> dict:
    with open(BASE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _with(**overrides) -> Config:
    """base.yaml 을 얕게 변형해 assert 없이 Config 로 만든다. overrides 는 'a.b.c': v 형식."""
    d = copy.deepcopy(_raw())
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node = d
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return config.from_dict(d)


# ── 정상 로드 ────────────────────────────────────────────────────────────────

def test_valid_config_loads():
    cfg = config.load(BASE)
    assert cfg.thresholds.interceptor == 16048     # 행동력 용량 창의 0.30 지점 (8/25)
    assert cfg.k == pytest.approx(0.3)          # eff 1.0 × success_prob 0.3


def test_window_values():
    """A-4 자가검증. **60턴** 기준: A 5400 · B 5400 · E 6480 < 임계 9558 < C×0.6 9720.

    창은 `total_turns` 에 선형이다 (A 만 `epoch_turns` 에 걸린다). 100턴에서 60턴으로
    내리면서 임계도 같은 **상대 위치(0.95)** 로 옮겼다 — 100턴의 [12960, 16200] 안 16038
    이 60턴의 [6480, 9720] 안 9558 이다.

    **숫자를 여기 두 번 적지 않는다** — 공식에서 유도하고, 순서만 못으로 박는다.
    """
    cfg = config.load(BASE)
    a, b, c, e = asserts.window(cfg)
    n, total, epoch = (cfg.world.agents_per_country, cfg.world.total_turns,
                       cfg.world.epoch_turns)
    # **실효 소득으로 잡는다** (8/22) — 소득이 나이와 함께 오르므로 「전 기간 총소득」 이
    # `per_turn × n × total` 보다 크다. 그 값을 그대로 쓰면 창이 좁아지고 임계가
    # 「도달 가능」 쪽에 붙는다.
    per = asserts.capacity_per_year(cfg)
    # **가장 잘 짓는 나라 기준** (8/23). 나라마다 요격기 진척 속도가 다르므로 네 조건이
    # 모두 최선의 나라에 몰아줬을 때로 걸린다 — ★B 가 그 기준을 강제한다.
    k = cfg.k * max(cfg.facility.build_spread)
    assert a == pytest.approx(3 * per * n * epoch * k)
    assert b == pytest.approx(per * n * total * k)
    assert e == pytest.approx(3 * per * n * (total - epoch) * k * 0.6)
    assert c == pytest.approx(3 * per * n * total * k)
    # **순서가 곧 설계다** — A·B·E 아래면 미루기·독주·휴식이 통하고, C×0.6 위면 아무도 못 짓는다
    assert max(a, b, e) < cfg.thresholds.interceptor < c * 0.6
    assert not asserts.check_all(cfg)


# ── 일부러 깨뜨리기 (A-4 표) ──────────────────────────────────────────────────

def test_break_interceptor_4000():
    fails = asserts.check_all(_with(**{"thresholds.interceptor": 4000}))
    joined = " ".join(fails)
    assert "★B" in joined
    assert "★E" in joined


def test_break_interceptor_above_window():
    """**창 밖 값을 손으로 적지 않는다** (8/22). 16400 이 창 안으로 들어와 버렸다 —
    `adult_age` 를 내리며 실효 소득이 커지자 상한이 11900 → 16780 으로 올라갔다."""
    cfg = config.load(BASE)
    _, _, c, _ = asserts.window(cfg)
    fails = asserts.check_all(_with(**{"thresholds.interceptor": c * 0.6 + 1}))
    assert any("★C" in f for f in fails)


def test_break_bunker_shallow():
    """벙커가 한 주기 진척(=to_progress(국가_한주기)=1800) 미만이면 벙커↓ 가 걸린다.

    ⚠ 과제 A-4 표는 `bunker: 1000` 을 예로 드는데, 스펙 공식상 하한은
       (50턴 시절) 900 이라 1000 은 통과했다. 100턴에서는 하한이 1800 이다.
       하한 미만인 1600 으로 벙커↓ 를 시연한다 → A-4 예시값이 스펙과 어긋나는 지점.
    """
    fails = asserts.check_all(_with(**{"thresholds.bunker": 1600}))
    assert any("벙커↓" in f for f in fails)


def test_bunker_just_above_floor():
    """하한 바로 위는 통과한다 — 경계가 어디인지를 코드로 고정.

    **하한을 손으로 적지 않는다** (8/22). 하한은 「한 주기 전력 진척」 이고, 그 값이
    실효 소득(나이 배수 포함)에서 나오므로 1800 → 2204 로 움직였다.
    """
    cfg = config.load(BASE)
    eff = asserts.capacity_per_year(cfg)
    floor = eff * cfg.world.agents_per_country * cfg.world.epoch_turns * cfg.k
    fails = asserts.check_all(_with(**{"thresholds.bunker": floor + 1}))
    assert not any("벙커↓" in f for f in fails)
    # 그리고 하한 **아래**는 걸린다
    fails = asserts.check_all(_with(**{"thresholds.bunker": floor - 1}))
    assert any("벙커↓" in f for f in fails)


def test_break_success_prob_half():
    """임계는 그대로 두고 success_prob 만 0.5 → 임계가 진척 단위임이 드러나 검사가 걸린다.

    ⚠ 과제 A-4 표는 ★C 라 적었지만, 스펙 공식으로 계산하면 창이 위로 이동(E=10800)해
       interceptor(8019)가 하한 아래로 내려가 실제로는 ★E 가 걸린다. 요점(임계가 진척
       단위라 재계산을 잊으면 검사가 잡는다)은 동일. → A-4 라벨이 스펙과 어긋나는 지점.
    """
    fails = asserts.check_all(_with(**{"world.success_prob": 0.5}))
    assert fails                              # 어떤 검사든 반드시 걸려야 한다
    assert any("★E" in f for f in fails)      # 스펙 공식상 실제로 걸리는 것은 ★E


def test_break_knob_below_speak():
    """**AI 발신이 원문보다 싸지면 아무도 배우지 않는다** (8/25 · AP 전면 통일).

    돈으로 매기던 시절에는 `comm_intl_learner`(5) 와 겨뤘다. 이제 비교 대상은
    `ap.speak` 다 — 자국·original 발신의 행동력.
    """
    fails = asserts.check_all(_with(**{"knob.comm_intl_ai_ap": [0.1, 0.3]}))
    assert any("노브" in f for f in fails)


def test_break_knob_above_a_whole_year():
    """**한 해 AP 를 넘으면 노브가 아니라 금지다.** 비싼 것과 없는 것이 구분되지 않는다."""
    fails = asserts.check_all(_with(**{"knob.comm_intl_ai_ap": [0.2, 1.5]}))
    assert any("노브" in f for f in fails)


def test_break_knob_out_of_order():
    """오름차순이어야 「i 번째로 비싼 노브」 라는 뜻을 갖는다."""
    fails = asserts.check_all(_with(**{"knob.comm_intl_ai_ap": [0.5, 0.2]}))
    assert any("노브" in f for f in fails)


def test_load_raises_on_broken():
    d = copy.deepcopy(_raw())
    d["thresholds"]["interceptor"] = 4000
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        yaml.safe_dump(d, tf)
        path = tf.name
    with pytest.raises(ConfigError):
        config.load(path)


# ── 백엔드 ─────────────────────────────────────────────────────────────────────

def test_gemini_backend_drops_openrouter_only_fields():
    """**OpenRouter 전용 필드는 저쪽에 없는 이름이다** — 실으면 400 이다.

    `provider`(프로바이더 라우팅)와 `reasoning`(OpenRouter 통합 사고 파라미터)은
    Google 의 OpenAI 호환 엔드포인트에 없다. config 에는 둘 다 들어 있으므로, 백엔드를
    바꿀 때 **몸통에서 빼는 것을 잊으면 전 호출이 400** 이다.
    """
    from core.llm import BACKENDS, OpenRouterClient

    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        raise RuntimeError("여기까지 보면 된다")

    # **클라이언트는 별 스레드에서 urlopen 을 부른다** (`_call_with_deadline`). 그래서
    # 예외가 밖으로 안 나오고 `box["exc"]` 로 옮겨져 다시 던져진다 — 어느 쪽이든
    # `sent` 는 채워지므로 그것만 본다.
    import urllib.request
    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        for backend, expect in (("openrouter", True), ("gemini", False)):
            c = OpenRouterClient("m/x", api_key="k", backend=backend,
                                 reasoning={"enabled": False},
                                 provider={"order": ["a"]}, retries=1)
            try:
                c.chat([{"role": "user", "content": "x"}])
            except Exception:
                pass
            assert sent["url"] == BACKENDS[backend], backend
            assert ("provider" in sent["body"]) is expect, backend
            assert ("reasoning" in sent["body"]) is expect, backend
            # 몸통의 공통부는 그대로다
            assert sent["body"]["model"] == "m/x"
    finally:
        urllib.request.urlopen = orig


def test_each_backend_has_its_own_key_name():
    """열쇠가 백엔드마다 다르다. 이름을 잘못 집으면 남의 열쇠로 부른다."""
    from core.llm import BACKENDS, KEY_ENV, key_for
    assert set(KEY_ENV) == set(BACKENDS)
    assert KEY_ENV["gemini"] == "GEMINI_API_KEY"
    with pytest.raises(RuntimeError, match="모르는 백엔드"):
        key_for("nope")


def test_the_world_section_rejects_keys_it_cannot_use():
    """**조용한 무시가 가장 나쁜 실패다.**

    `World` 를 다섯 필드만 골라 넘기고 있었다. 그래서 `adult_age`·`init_age_spread`·
    `init_age_max` 가 **yaml 에서 읽히지 않았고**, 기본값과 우연히 같아서 드러나지 않았다 —
    config 를 고쳐도 아무 일이 안 일어나는 상태다.

    이제 `world` 절을 통째로 넘기고 dataclass 필드와 대조한다. 새 키를 yaml 에만 넣고
    배선을 잊으면 로드가 **실패**한다.
    """
    import dataclasses

    import yaml

    from core.config import World
    d = yaml.safe_load(Path(BASE).read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(World)}
    assert set(d["world"]) <= known, set(d["world"]) - known

    # 그리고 yaml 의 값이 **실제로 읽힌다**
    # `adult_age` 를 지운 뒤로 (8/25) 이 자리에 쓸 필드를 바꿨다
    cfg = _with(**{"world.init_age_max": 6})
    assert cfg.world.init_age_max == 6

    with pytest.raises(ConfigError, match="모르는 키"):
        _with(**{"world.zzz_unwired": 1})
