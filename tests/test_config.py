"""설정 로더와 assert 검증. 과제 1 Part A."""
from __future__ import annotations

import copy
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
    assert cfg.thresholds.interceptor == 8019
    assert cfg.k == pytest.approx(0.3)          # eff 1.0 × success_prob 0.3


def test_window_values():
    """A-4 자가검증: A 2700  B 4500  E 6480  <  임계 8019  <  C×0.6 8100."""
    cfg = config.load(BASE)
    a, b, c, e = asserts.window(cfg)
    assert a == pytest.approx(2700)
    assert b == pytest.approx(4500)
    assert e == pytest.approx(6480)
    assert c * 0.6 == pytest.approx(8100)
    assert a < b < e < cfg.thresholds.interceptor < c * 0.6


# ── 일부러 깨뜨리기 (A-4 표) ──────────────────────────────────────────────────

def test_break_interceptor_4000():
    fails = asserts.check_all(_with(**{"thresholds.interceptor": 4000}))
    joined = " ".join(fails)
    assert "★B" in joined
    assert "★E" in joined


def test_break_interceptor_8200():
    fails = asserts.check_all(_with(**{"thresholds.interceptor": 8200}))
    assert any("★C" in f for f in fails)


def test_break_bunker_shallow():
    """벙커가 한 주기 진척(=to_progress(국가_한주기)=900) 미만이면 벙커↓ 가 걸린다.

    ⚠ 과제 A-4 표는 `bunker_scale: 1000` 을 예로 드는데, 스펙 공식상 하한은 900 이라
       1000 은 통과한다(아래 test_bunker_1000_note 로 명시). 하한 미만인 800 으로
       벙커↓ 를 시연한다. → A-4 예시값(1000)이 스펙과 어긋나는 지점. 형에게 확인 필요.
    """
    fails = asserts.check_all(_with(**{"thresholds.bunker_scale": 800}))
    assert any("벙커↓" in f for f in fails)


def test_bunker_1000_note():
    """스펙 공식(하한 900)상 1000 은 통과한다 — A-4 표와의 불일치를 코드로 고정."""
    fails = asserts.check_all(_with(**{"thresholds.bunker_scale": 1000}))
    assert not any("벙커↓" in f for f in fails)


def test_break_success_prob_half():
    """임계는 그대로 두고 success_prob 만 0.5 → 임계가 진척 단위임이 드러나 검사가 걸린다.

    ⚠ 과제 A-4 표는 ★C 라 적었지만, 스펙 공식으로 계산하면 창이 위로 이동(E=10800)해
       interceptor(8019)가 하한 아래로 내려가 실제로는 ★E 가 걸린다. 요점(임계가 진척
       단위라 재계산을 잊으면 검사가 잡는다)은 동일. → A-4 라벨이 스펙과 어긋나는 지점.
    """
    fails = asserts.check_all(_with(**{"world.success_prob": 0.5}))
    assert fails                              # 어떤 검사든 반드시 걸려야 한다
    assert any("★E" in f for f in fails)      # 스펙 공식상 실제로 걸리는 것은 ★E


def test_break_knob_low():
    """comm_intl_ai 최저값이 learner(5) 이하면 노브가 무의미."""
    fails = asserts.check_all(_with(**{"knob.comm_intl_ai": [4, 12]}))
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
