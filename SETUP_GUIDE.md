# DeepSeek-V4 Flash を Apple Silicon Mac でローカル推論する完全ガイド

> **対象読者：** Apple Silicon Mac（M1 Ultra以上推奨）で DeepSeek-V4 Flash を動かしたい人  
> **検証環境：** Mac Studio M3 Ultra 512GB / macOS 15.7.2 / 2026年4月時点  
> **所要時間：** ダウンロード除きで約30分

---

## 目次

1. [前提条件](#1-前提条件)
2. [モデルの選び方（mxfp8 vs nvfp4）](#2-モデルの選び方mxfp8-vs-nvfp4)
3. [モデルのダウンロード](#3-モデルのダウンロード)
4. [Python 環境構築](#4-python-環境構築)
5. [最重要：mlx_lm の正しいバージョン](#5-最重要mlx_lm-の正しいバージョン)
6. [tokenizer の罠と回避策](#6-tokenizer-の罠と回避策)
7. [推論テスト](#7-推論テスト)
8. [Gradio チャット UI の構築](#8-gradio-チャット-ui-の構築)
9. [ワンクリック起動スクリプト](#9-ワンクリック起動スクリプト)
10. [実測パフォーマンス](#10-実測パフォーマンス)
11. [LMStudio は使えるか？](#11-lmstudio-は使えるか)
12. [よくあるエラーと対処](#12-よくあるエラーと対処)

---

## 1. 前提条件

### 必要なハードウェア

| 要件 | 最低 | 推奨 |
|---|---|---|
| チップ | M1 Ultra 以上 | M3 Ultra |
| ユニファイドメモリ | 192 GB（nvfp4, コンテキスト短め） | **512 GB** |
| ストレージ空き | 160 GB | 200 GB |

DeepSeek-V4 Flash は **284B パラメータの MoE モデル**。量子化後でも 141〜155 GB のメモリが必要です。128GB Mac では動きません。

### 必要なソフトウェア

- **Python 3.11**（Homebrew 推奨。3.12以上は mlx の互換性に注意）
- **Git**
- **Xcode Command Line Tools**（`xcode-select --install`）

---

## 2. モデルの選び方（mxfp8 vs nvfp4）

HuggingFace の mlx-community から2種類の量子化が公開されています。

| | mxfp8（Mixed FP8） | nvfp4（NF4 4-bit） |
|---|---|---|
| ディスク容量 | 155 GB | 151 GB |
| メモリ使用量 | ~155 GB | ~152 GB |
| 精度 | ◎ FP8は高精度 | ○ 4-bitなりの劣化あり |
| 推論速度 | ~22 tok/s | ~22 tok/s |
| **結論** | **✅ こちらを推奨** | サイズ差3GBで精度が落ちる |

**サイズがほぼ同じなので、mxfp8 一択。** nvfp4 を選ぶ積極的理由がありません。

---

## 3. モデルのダウンロード

### 方法A：huggingface-hub（推奨）

```bash
pip install huggingface-hub[hf_xet]

# ~155GB、回線速度次第で数時間
huggingface-cli download mlx-community/DeepSeek-V4-Flash-mxfp8 --local-dir ./DeepSeek-V4-Flash-mxfp8
```

### 方法B：LMStudio でダウンロード

LMStudio の GUI からモデルを検索してダウンロードできますが、**tokenizer.json が壊れる場合がある**ことを確認しています（全バイトがゼロになる現象）。LMStudio経由でダウンロードした場合は、後述の「tokenizer の罠」を必ず確認してください。

ダウンロード先：`~/.lmstudio/models/mlx-community/DeepSeek-V4-Flash-mxfp8/`

---

## 4. Python 環境構築

```bash
# プロジェクトディレクトリ作成
mkdir -p ~/Documents/Deepseek-v4
cd ~/Documents/Deepseek-v4

# venv 作成（Python 3.11）
python3.11 -m venv venv
source venv/bin/activate

# 基本パッケージ
pip install gradio httpx
```

**ここで `pip install mlx-lm` をやってはいけません。** 次のセクションを読んでください。

---

## 5. 最重要：mlx_lm の正しいバージョン

### なぜ普通にインストールしてはダメか

2026年4月時点で、**mlx_lm の公式版（PyPI: 0.31.3）は `deepseek_v4` モデルタイプに対応していません。**

```
ValueError: Model type deepseek_v4 not supported.
```

このエラーが出ます。GitHub の main ブランチにもまだマージされていません。

### 正しいインストール方法

mlx-community の Blaizzy 氏が DeepSeek-V4 対応の PR ブランチを公開しています：

```bash
pip install --force-reinstall 'git+https://github.com/Blaizzy/mlx-lm@pc/add-deepseekv4flash-model'
```

これにより `mlx_lm/models/deepseek_v4.py` がインストールされ、モデルのロードが可能になります。

### 確認方法

```bash
python -c "import mlx_lm; print('OK')"
ls venv/lib/python3.11/site-packages/mlx_lm/models/deepseek_v4.py
# ファイルが存在すれば OK
```

### 今後について

mlx_lm 公式にマージされれば `pip install mlx-lm` だけで済むようになります。
PR の状況：https://github.com/ml-explore/mlx-lm/pull/1192

---

## 6. tokenizer の罠と回避策

### 罠①：tokenizer.json の破損（LMStudio経由の場合）

LMStudio でダウンロードしたモデルの `tokenizer.json` が全バイト0x00になっている場合があります。

```bash
# 確認方法
file /path/to/model/tokenizer.json
# "data" と出たらNG。"Unicode text, UTF-8 text" ならOK
```

**修復方法：**

```python
from huggingface_hub import hf_hub_download
hf_hub_download("mlx-community/DeepSeek-V4-Flash-mxfp8", "tokenizer.json")
```

ダウンロードされたファイルを元のディレクトリにコピーしてください。

### 罠②：transformers が deepseek_v4 を認識しない

mlx_lm の `load()` 関数は内部で `transformers.AutoTokenizer.from_pretrained()` を呼びます。
しかし **transformers ライブラリも `deepseek_v4` model_type を認識しない**（2026年4月時点）ため、以下のエラーになります：

```
KeyError: 'deepseek_v4'
ValueError: The checkpoint you are trying to load has model type `deepseek_v4`
but Transformers does not recognize this architecture.
```

`pip install git+https://github.com/huggingface/transformers.git` で最新を入れても解決しません。

### 回避策：tokenizer を手動構成する

`mlx_lm.load()` は使えないので、**モデルとトークナイザーを別々にロード**します。

```python
# model_utils.py
from pathlib import Path
from mlx_lm.utils import load_model
from transformers import PreTrainedTokenizerFast

MODEL_PATH = Path("/path/to/DeepSeek-V4-Flash-mxfp8")

def load(model_path: Path = MODEL_PATH):
    # モデル（mlx_lm は deepseek_v4 を認識できる）
    model, _config = load_model(model_path, lazy=False)

    # トークナイザー（AutoTokenizer を迂回して直接構成）
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json"),
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<｜end▁of▁sentence｜>",
        pad_token="<｜end▁of▁sentence｜>",
    )
    with open(model_path / "chat_template.jinja") as f:
        tokenizer.chat_template = f.read()

    return model, tokenizer
```

**ポイント：**
- `load_model()` は mlx_lm 内部関数で、transformers を経由せずモデルだけをロードする
- `PreTrainedTokenizerFast` は config.json を読まないので `deepseek_v4` 問題を回避できる
- `chat_template.jinja` を手動で読み込んで `apply_chat_template()` を使えるようにする
- 特殊トークン（bos/eos/pad）は `tokenizer_config.json` の値をそのまま指定する

---

## 7. 推論テスト

### 最小テストコード

```python
from mlx_lm import generate
from model_utils import load  # 上で作ったユーティリティ

model, tokenizer = load()

messages = [{"role": "user", "content": "こんにちは！"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)

response = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=True)
```

### 期待される出力

```
Prompt: 14 tokens, 8.988 tokens-per-sec
Generation: 243 tokens, 22.419 tokens-per-sec
Peak memory: 154.930 GB
```

- **22 tok/s 前後**が出れば正常
- ピークメモリが 155 GB 程度であることを確認
- 日本語で応答が返ること

---

## 8. Gradio チャット UI の構築

### 設計のポイント

1. **サーバーレス：** mlx_lm.server や LMStudio を介さず、`stream_generate()` で直接推論
2. **`<think>` タグ対応：** DeepSeek-V4 は `<think>`...`</think>` で思考過程を出力する。Gradio の `reasoning_tags` で折りたたみ表示
3. **メモリ見積もり表示：** MLA アーキテクチャの KV キャッシュサイズをリアルタイム計算

### KV キャッシュの計算式

```
KV/token = num_hidden_layers × (head_dim + qk_rope_head_dim) × 2bytes
         = 43 × (512 + 64) × 2
         = 49,536 bytes ≈ 48.4 KB/token
```

| コンテキスト | KV追加 | モデル込み合計 |
|---|---|---|
| 256K | ~12 GB | ~167 GB |
| 512K | ~24 GB | ~179 GB |
| 1M | ~50 GB | ~205 GB |

512GB Mac なら 1M コンテキストでも余裕です。
※ 実際は NSA（Native Sparse Attention）の compress_ratios（4x/128x）により、これより小さくなります。

---

## 9. ワンクリック起動スクリプト

`.command` ファイルを作るとダブルクリックで Terminal.app が開いて実行されます。

```bash
#!/bin/zsh
PROJECT_DIR="$HOME/Documents/Deepseek-v4"
UI_PORT=7860

cd "$PROJECT_DIR"
source venv/bin/activate
lsof -ti :$UI_PORT | xargs kill -9 2>/dev/null

venv/bin/python ui.py &
UI_PID=$!

# UI がリッスンするまで待つ
while ! curl -s http://127.0.0.1:$UI_PORT > /dev/null 2>&1; do
    sleep 2; printf "."
done
open "http://127.0.0.1:$UI_PORT"

trap "kill $UI_PID 2>/dev/null; wait $UI_PID 2>/dev/null" EXIT INT TERM
wait $UI_PID
```

```bash
chmod +x launch.command
# デスクトップにシンボリックリンク
ln -sf "$PWD/launch.command" ~/Desktop/"DeepSeek-V4 Flash.command"
```

---

## 10. 実測パフォーマンス

Mac Studio M3 Ultra 512GB での実測値：

| 指標 | 値 |
|---|---|
| 推論速度 | **22.4 tok/s** |
| TTFT（短プロンプト） | **~9 tok/s**（プロンプト処理） |
| ピークメモリ | **154.9 GB** |
| モデルロード時間 | 3〜5分（初回） |

M3 Ultra の帯域幅 800 GB/s に対して、MoE のアクティブ重み（6/256エキスパート＋共有エキスパート）が
比較的小さいため、Kimi K2.6（14 tok/s）より大幅に高速です。

---

## 11. LMStudio は使えるか？

**2026年4月時点：使えません。**

LMStudio も内部で mlx_lm を使用しており、同じ `Model type deepseek_v4 not supported` エラーが発生します。

```bash
lms load deepseek-v4-flash-mxfp8 -c 262144 -y
# → Error: ValueError: Model type deepseek_v4 not supported.
```

`lms runtime update` を実行しても解消されません。
LMStudio のランタイムが更新されるのを待つ必要があります。

---

## 12. よくあるエラーと対処

### `Model type deepseek_v4 not supported`

**原因：** mlx_lm 公式版を使っている  
**対処：** Blaizzy fork をインストール
```bash
pip install --force-reinstall 'git+https://github.com/Blaizzy/mlx-lm@pc/add-deepseekv4flash-model'
```

### `KeyError: 'deepseek_v4'` / `AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'`

**原因：** transformers が deepseek_v4 を認識しない → `AutoTokenizer` が失敗  
**対処：** `model_utils.py` で `PreTrainedTokenizerFast` を直接使う（セクション6参照）

### `JSONDecodeError: Failed to parse tokenizer.json`

**原因：** tokenizer.json が壊れている（LMStudio ダウンロードで発生することがある）  
**対処：** HuggingFace から再取得
```python
from huggingface_hub import hf_hub_download
hf_hub_download("mlx-community/DeepSeek-V4-Flash-mxfp8", "tokenizer.json")
```
取得先：`~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-mxfp8/snapshots/*/tokenizer.json`

### `TypeError: unsupported operand type(s) for /: 'str' and 'str'`

**原因：** `load_model()` に文字列パスを渡している  
**対処：** `pathlib.Path` オブジェクトを渡す
```python
from pathlib import Path
model, config = load_model(Path("/path/to/model"), lazy=False)
```

---

## まとめ：ハマりポイント一覧

| # | 落とし穴 | 影響 | 回避策 |
|---|---|---|---|
| 1 | mlx_lm 公式版が V4 未対応 | モデルロード不可 | Blaizzy fork を使う |
| 2 | transformers が V4 未対応 | tokenizer ロード不可 | PreTrainedTokenizerFast で直接構成 |
| 3 | LMStudio が V4 未対応 | lms load 失敗 | mlx_lm 直接推論に切り替え |
| 4 | LMStudio の tokenizer.json 破損 | JSON パースエラー | HuggingFace から再取得 |
| 5 | load_model に str を渡す | TypeError | Path オブジェクトを使う |

**公式対応が進めばこれらの問題は解消されます。** それまではこのガイドの回避策で動作します。

---

*最終更新: 2026-04-25 / 検証環境: Mac Studio M3 Ultra 512GB, macOS 15.7.2*
