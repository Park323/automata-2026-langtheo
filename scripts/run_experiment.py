"""このリポジトリを動かす方法。

## 1) 鍵

    cp .env.example .env.local        # OPENROUTER_API_KEY を入れる

## 2) 動作確認（無料）

    python -m scripts.run_experiment --check

## 3) お試し実行（数十秒、数十コール）

    python -m scripts.run_experiment --turns 3

## 4) 本実験（config の total_turns 分 — 現在30ターン）

    python -m scripts.run_experiment --knob 0.06 --seed 1
    python -m scripts.run_experiment --no-ai --seed 1        # 対照群（AI翻訳なし）

全オプション:

    python -m scripts.run_experiment --help

## 出力

`runs/<run-id>/` に生成される。

    state.jsonl            ターンごとのエージェント状態
    metrics.jsonl           ターンごとの集計指標
    events.jsonl             行動・通信・環境イベントのログ
    messages.jsonl       送受信メッセージ
    raw_calls.jsonl         LLM 呼び出しの生ログ
    config_snapshot.yaml  実行時の設定
    summary.json           最終結果

## 見る（ビューア）

リポジトリのルートで

    python -m http.server 8000 -d viewer

を立てて http://localhost:8000/board.html を開き、📁 ボタンで
「runs/<run-id>」フォルダを選ぶ（file:// で直接開いても同じボタンで動く）。

URL に直接 `?run=../runs/<run-id>` を付けても開く
（例 http://localhost:8000/board.html?run=../runs/<run-id>）。
"""
from scripts.smoke_3turns import main

if __name__ == "__main__":
    main()
