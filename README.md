# DeepSeek-V4 Flash on Mac Studio

**284B パラメータ MoE モデル「DeepSeek-V4 Flash」** を Apple Silicon Mac でローカル推論する環境。
mxfp8 量子化（~155GB）、MLX ネイティブ、OpenAI 互換 API + Gradio チャット UI 付き。

実測 **22 tok/s**、実使用メモリ **~164GB**（256K context 込み）、最大コンテキスト **1M**。

> 💡 **動作要件:** Apple Silicon Mac で **ユニファイドメモリ 192GB 以上**
> - M2 Ultra 192GB ✅（256K context まで現実的）
> - M3 Ultra 256GB ✅
> - M3 Ultra 512GB ✅（1M context まで余裕）
> - 128GB 以下 ❌

---

## なぜカスタムビルドが必要なのか

2026年4月時点、`mlx-lm` 公式版は **`deepseek_v4` モデルタイプに未対応**です：

```
ValueError: Model type deepseek_v4 not supported.
```

`transformers` も同様に未対応のため、`pip install mlx-lm` だけでは動きません。
このリポジトリは以下を組み合わせて動作させています：

1. **mlx_lm の Blaizzy fork** — DeepSeek-V4 対応 PR ブランチ
2. **トークナイザーの手動構成** — `transformers` の `AutoTokenizer` を回避
3. **シングルスレッド HTTP サーバー** — MLX の GPU ストリーム制約を回避

詳細は [SETUP_GUIDE.md](SETUP_GUIDE.md) を参照。

---

## クイックスタート

### 1. リポジトリ取得

```bash
git clone https://github.com/Shinka-Man/DeepSeek-V4-Flash-MacStudio.git
cd DeepSeek-V4-Flash-MacStudio
```

### 2. モデルダウンロード（~155GB）

```bash
pip install huggingface-hub[hf_xet]
huggingface-cli download mlx-community/DeepSeek-V4-Flash-mxfp8 \
  --local-dir ~/.lmstudio/models/mlx-community/DeepSeek-V4-Flash-mxfp8
```

### 3. Python 環境構築

```bash
python3.11 -m venv venv
source venv/bin/activate

# DeepSeek-V4 対応 mlx_lm（Blaizzy fork）
pip install --force-reinstall \
  'git+https://github.com/Blaizzy/mlx-lm@pc/add-deepseekv4flash-model'

# 最新 transformers（V4 model_type 認識用）
pip install --upgrade 'git+https://github.com/huggingface/transformers.git'

# UI / API サーバー依存
pip install gradio httpx huggingface-hub
```

### 4. tokenizer.json の検証（LMStudio 経由ダウンロードの場合）

```bash
file ~/.lmstudio/models/mlx-community/DeepSeek-V4-Flash-mxfp8/tokenizer.json
# "Unicode text, UTF-8 text" であれば OK
# "data" と出たら破損 → 以下で修復
```

破損時：
```bash
python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('mlx-community/DeepSeek-V4-Flash-mxfp8', 'tokenizer.json'))"
# 出力されたパスのファイルを LMStudio のディレクトリにコピー
```

### 5. 起動

**ワンクリック起動（推奨）:**
- `launch.command` をダブルクリック、もしくはデスクトップにシンボリックリンク：
  ```bash
  ln -s "$PWD/launch.command" ~/Desktop/"DeepSeek-V4 Flash.command"
  ```

**手動起動:**
```bash
# ターミナル1: API サーバー
python server.py

# ターミナル2: Gradio UI
python ui.py
```

ブラウザで http://127.0.0.1:7860 を開く。

---

## 構成

```
.
├── launch.command       ← ダブルクリック起動
├── server.py            ← OpenAI 互換 API サーバー (port 7777)
├── ui.py                ← Gradio チャット UI (port 7860)
├── model_utils.py       ← モデル/トークナイザー読み込みユーティリティ
├── run.py               ← 一発推論スクリプト
├── chat.py              ← ターミナルチャット
├── README.md            ← このファイル
└── SETUP_GUIDE.md       ← 詳細セットアップ＆ハマりポイント解説
```

| ポート | 役割 | アクセス |
|---|---|---|
| **7777** | OpenAI 互換 API | `0.0.0.0`（LAN 内から接続可） |
| **7860** | Gradio チャット UI | `127.0.0.1`（ローカル） |

---

## OpenAI 互換 API の使い方

### 基本リクエスト

```bash
curl http://0.0.0.0:7777/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-mxfp8",
    "messages": [{"role": "user", "content": "こんにちは"}],
    "max_tokens": 512,
    "stream": false
  }'
```

### ストリーミング

```bash
curl -N http://0.0.0.0:7777/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-mxfp8",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256,
    "stream": true
  }'
```

### サードパーティアプリ設定例

| アプリ | 設定箇所 | 設定値 |
|---|---|---|
| Open WebUI | Connections → OpenAI API | URL: `http://<Mac IP>:7777/v1` / Model: `deepseek-v4-flash-mxfp8` |
| Continue.dev (VSCode) | `models[].apiBase` | `http://<Mac IP>:7777/v1` |
| LangChain `ChatOpenAI` | `base_url`, `model` | 同上 |

`<Mac IP>` は Mac Studio の LAN IP。`ifconfig | grep "inet "` で確認。

### モデル ID の確認

```bash
curl http://127.0.0.1:7777/v1/models
```

---

## UI の使い方

| 操作 | 方法 |
|---|---|
| メッセージ送信 | テキスト入力後 **Enter** または **送信 ▶** |
| 履歴クリア | **履歴クリア** ボタン |
| System Prompt | 右パネル（デフォルト: "You are a helpful assistant."） |
| Max Tokens | スライダー（256〜131072） |
| Temperature | スライダー（0.0〜2.0） |
| Context Length | スライダー（128K〜1M、128K刻み） |
| 思考過程 | `<think>` タグの内容は折りたたみ表示 |

---

## パフォーマンス（Mac Studio M3 Ultra 512GB 実測）

| 指標 | 値 |
|---|---|
| 推論速度 | **22.4 tok/s** |
| 実使用メモリ（256K context） | **~164 GB** |
| TTFT（短プロンプト） | ~数秒 |
| モデルロード時間 | 3〜5分（初回） |

### メモリ目安（context 別）

| コンテキスト | 実使用メモリ目安 |
|---|---|
| 256K（デフォルト） | ~164 GB |
| 512K | ~180 GB |
| 1M（最大） | ~210 GB |

NSA（Native Sparse Attention）の compress_ratios 効果込み。
**M2 Ultra 192GB でも 256K context まで動作します。**

---

## アーキテクチャ

| パラメータ | 値 |
|---|---|
| model_type | deepseek_v4 |
| パラメータ数 | 284B |
| num_hidden_layers | 43 |
| hidden_size | 4,096 |
| num_attention_heads | 64 |
| num_key_value_heads | 1（MLA） |
| head_dim | 512 |
| qk_rope_head_dim | 64 |
| n_routed_experts | 256 |
| num_experts_per_tok | 6 |
| max_position_embeddings | 1,048,576（1M） |
| Attention | Native Sparse Attention（NSA）|
| RoPE scaling | YaRN（factor=16） |

---

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `Model type deepseek_v4 not supported` | mlx_lm 公式版を使っている → Blaizzy fork を再インストール |
| `KeyError: 'deepseek_v4'` (transformers) | `model_utils.py` 経由でロードする（`mlx_lm.load()` を直接使わない） |
| `JSONDecodeError: tokenizer.json` | tokenizer.json 破損 → HuggingFace から再取得（手順4参照） |
| `There is no Stream(gpu, 1)` | サーバーをマルチスレッドで動かしている → `server.py` を使う（シングルスレッド） |
| 文字欠落（「DeepSe」「バージョ」等） | 古い fork を使っている → `pip install --force-reinstall` で最新版に更新 |
| `python: command not found` | venv をアクティベート: `source venv/bin/activate` |

---

## 関連ドキュメント

- [SETUP_GUIDE.md](SETUP_GUIDE.md) — ゼロベースの構築手順とハマりポイント詳細解説
- [Blaizzy/mlx-lm PR #1192](https://github.com/ml-explore/mlx-lm/pull/1192) — DeepSeek-V4 対応 PR（マージ待ち）
- [mlx-community/DeepSeek-V4-Flash-mxfp8](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-mxfp8) — モデル本体

---

## ライセンス

このリポジトリのコード: MIT

モデルウェイト本体は [DeepSeek 公式ライセンス](https://huggingface.co/deepseek-ai) に従ってください。

---

## 謝辞

- **Prince Canuma (@Blaizzy)** — DeepSeek-V4 mlx-lm 対応の中心人物。バグ報告に数時間で応答してくれました
- **Lambda.ai** — Mac Studio を寄贈し、mlx-community のコミュニティ貢献を支えています
- **mlx-community** — MLX 向けモデル変換と公開
