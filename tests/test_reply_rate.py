"""답장률 — **로그에서 센다.** spec 8.2 보강.

`speak` 에 `reply_to` 인자를 두고 그것을 세려 했는데, 그 인자는 **도구 스키마에 없었다.**
배관만 여섯 곳에 깔려 있고 모델에게 준 적이 없어서 값이 늘 `None` 이었다. 그런데 그
「답장률 0%」 가 *"대화가 구조적으로 죽었다"* 의 근거로 쓰였다 — 실제로는 근거에
*"Responding to Asla1's inquiry"* 라고 적으면서 답장을 하고 있었다.

**측정하지 못한 것을 없다고 읽은 것이다.** 그래서 필드를 없애고 방향과 시점으로 센다.
"""
from __future__ import annotations

from tools.score.metrics import reply_metrics


def m(turn, frm, to, route="domestic"):
    return {"turn": turn, "from": frm, "to": to, "route": route}


def test_a_reply_in_the_same_turn_counts():
    """순차 라운드로빈은 **같은 턴에 배달**되므로 그 턴에 답이 올 수 있다.

    같은 턴이면 **서로가 서로의 답**이다 — 창 `[t, t+1]` 이 양쪽을 다 덮는다.
    """
    r = reply_metrics([m(3, "A1", "B1"), m(3, "B1", "A1")])
    assert r["overall"]["answered"] == 2
    assert r["overall"]["rate"] == 1.0


def test_a_reply_in_the_next_turn_counts():
    """병렬 경로는 다음 턴이 가장 이른 답이다. **둘을 한 지표로 비교하려면 창이 둘을
    다 덮어야 한다.**

    창은 **앞으로만** 본다. 그래서 t3 의 A→B 는 답을 받았지만, t4 의 B→A 는 (아직) 답을
    못 받은 것이다 — 답장이 그 자신을 답받은 것으로 만들지 않는다.
    """
    r = reply_metrics([m(3, "A1", "B1"), m(4, "B1", "A1")])
    assert r["overall"]["answered"] == 1 and r["overall"]["n"] == 2


def test_two_turns_later_is_not_a_reply():
    """창을 넓히면 무엇이든 답장이 된다 — 우연한 왕래와 갈리지 않는다."""
    r = reply_metrics([m(3, "A1", "B1"), m(5, "B1", "A1")])
    assert r["overall"]["answered"] == 0


def test_a_reply_must_come_back_to_the_sender():
    """B 가 딴 사람에게 말한 것은 A 에 대한 답이 아니다."""
    r = reply_metrics([m(3, "A1", "B1"), m(3, "B1", "C1")])
    assert r["overall"]["answered"] == 0


def test_one_message_counts_once_even_if_two_come_back():
    """짝이 아니라 **발신 기준**이다. 안 그러면 한 번 말 걸고 두 번 답 받으면 200% 가 된다."""
    r = reply_metrics([m(3, "A1", "B1"), m(3, "B1", "A1"), m(4, "B1", "A1")])
    assert r["overall"]["n"] == 3 and r["overall"]["answered"] == 2
    assert r["overall"]["rate"] == 0.6667


def test_domestic_and_international_are_split():
    """국제 답장률이 **언어 채널의 상태**를 직접 재고, 그것이 노브에 반응하는 값이다."""
    rows = [m(1, "A1", "A2"), m(1, "A2", "A1"),            # 국내 왕복
            m(1, "A1", "B1", "ai"),                        # 국제 편도
            m(1, "A1", "C1", "original"), m(2, "C1", "A1", "ai")]
    r = reply_metrics(rows)
    assert r["domestic"]["rate"] == 1.0
    # A1→B1 은 답이 없고, A1→C1 은 t2 에 돌아왔고, 그 답 자체는 아직 답을 못 받았다
    assert r["international"]["n"] == 3 and r["international"]["answered"] == 1
    assert r["by_route"]["ai"]["n"] == 2 and r["by_route"]["original"]["n"] == 1


def test_empty_is_not_a_crash():
    r = reply_metrics([])
    assert r["overall"] == {"n": 0, "answered": 0, "rate": None}
