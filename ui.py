"""DeepSeek-V4 Flash — Gradio チャット UI
server.py (port 7777) に接続して推論する。モデルはUI側に持たない。
"""
import os
import re
import time
import json
import threading
import httpx
import gradio as gr

SERVER_URL = "http://127.0.0.1:7777/v1/chat/completions"
MODEL_NAME = "deepseek-v4-flash-mxfp8"

DARK_CSS = """
body, .gradio-container { background: #0d0d0d !important; color: #e0e0e0 !important; }
.block, .panel, .form { background: #141414 !important; border-color: #2a2a2a !important; }
.chatbot { background: #0d0d0d !important; }
.message.user { background: #1e3a5f !important; }
.message.bot  { background: #1a1a1a !important; }
textarea, input[type=text] { background: #1a1a1a !important; color: #e0e0e0 !important; border-color: #333 !important; }
.gr-button-primary { background: #1e6fbf !important; border-color: #1e6fbf !important; }
.gr-button-secondary { background: #2a2a2a !important; border-color: #444 !important; color: #ccc !important; }
label, .label-wrap { color: #aaa !important; }
"""

THINKING_OPEN  = "<think>"
THINKING_CLOSE = "</think>"
_MIN_CTX = 131072
_MAX_CTX = 1048576
_DEFAULT_CTX = 262144


def _ctx_label(n: int) -> str:
    return f"**Context: {n // 1024}K tokens**"

def _strip_for_history(text) -> str:
    if isinstance(text, list):
        parts = []
        for item in text:
            if isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
            else:
                parts.append(str(item))
        text = "".join(parts)
    elif isinstance(text, dict):
        text = text.get("text") or text.get("content") or ""
    elif not isinstance(text, str):
        text = str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.split("\n\n---\n")[0]
    return text.strip()


def chat_stream(message, history, system_prompt, max_tokens, temperature):
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": _strip_for_history(assistant_msg)})
    messages.append({"role": "user", "content": message})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": True,
    }

    full_text = ""
    reasoning_text = ""
    response_text = ""
    token_count = 0
    ttft = 0.0
    request_start = time.perf_counter()
    first_token_at = None
    in_think = False
    think_done = False

    yield "⏳ 生成開始..."

    try:
        with httpx.stream("POST", SERVER_URL, json=payload, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                tok = delta.get("content", "")
                if not tok:
                    continue

                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                    ttft = now - request_start
                token_count += 1
                full_text += tok

                if not think_done:
                    if THINKING_OPEN in full_text and not in_think:
                        in_think = True
                    if in_think and THINKING_CLOSE in full_text:
                        think_done = True
                        parts = full_text.split(THINKING_CLOSE, 1)
                        reasoning_text = parts[0].replace(THINKING_OPEN, "")
                        response_text = parts[1] if len(parts) > 1 else ""
                    elif in_think:
                        reasoning_text = full_text.replace(THINKING_OPEN, "")
                        response_text = ""
                    else:
                        response_text = full_text
                else:
                    idx = full_text.find(THINKING_CLOSE)
                    response_text = full_text[idx + len(THINKING_CLOSE):] if idx >= 0 else full_text

                elapsed_gen = now - first_token_at
                tps = token_count / elapsed_gen if elapsed_gen > 0 else 0.0
                display_parts = []
                if reasoning_text:
                    tag = THINKING_CLOSE if think_done else ""
                    display_parts.append(f"{THINKING_OPEN}{reasoning_text}{tag}")
                if response_text:
                    display_parts.append(response_text)
                stats = (f"\n\n---\n`TTFT {ttft*1000:.0f}ms | {tps:.1f} tok/s | "
                         f"{token_count} tokens | {now - request_start:.1f}s`")
                yield "\n\n".join(display_parts) + stats

    except httpx.ConnectError:
        yield "**エラー:** サーバーに接続できません。`python server.py` を先に起動してください。"
        return
    except Exception as e:
        yield f"**エラー:** {e}"
        return

    if token_count > 0:
        total = time.perf_counter() - request_start
        tps = token_count / (total - ttft) if total > ttft else 0.0
        stats = (f"\n\n---\n`TTFT {ttft*1000:.0f}ms | {tps:.1f} tok/s | "
                 f"{token_count} tokens | {total:.1f}s`")
        display_parts = []
        if reasoning_text:
            display_parts.append(f"{THINKING_OPEN}{reasoning_text}{THINKING_CLOSE}")
        if response_text:
            display_parts.append(response_text)
        yield "\n\n".join(display_parts) + stats


with gr.Blocks(title="DeepSeek-V4 Flash Chat") as demo:
    gr.Markdown("## DeepSeek-V4 Flash · mxfp8 MLX")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=500, render_markdown=True, label="",
                                 reasoning_tags=[(THINKING_OPEN, THINKING_CLOSE)])
            with gr.Row():
                msg_box = gr.Textbox(placeholder="メッセージを入力...  (Enter で送信)",
                                     show_label=False, scale=6, lines=1, max_lines=1)
                send_btn = gr.Button("送信 ▶", variant="primary", scale=1, min_width=80)

        with gr.Column(scale=1, min_width=220):
            system_prompt = gr.Textbox(label="System Prompt", value="You are a helpful assistant.", lines=3)
            max_tokens = gr.Slider(256, 131072, value=131072, step=256, label="Max Tokens")
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            gr.Markdown("---")
            ctx_slider = gr.Slider(_MIN_CTX, _MAX_CTX, value=_DEFAULT_CTX, step=131072, label="Context Length")
            ctx_info = gr.Markdown(_ctx_label(_DEFAULT_CTX))
            gr.Markdown("---")
            clear_btn = gr.Button("履歴クリア", variant="secondary")
            gr.Markdown("API: `0.0.0.0:7777/v1`")

    def _get_content(msg):
        return msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

    def respond(message, history, sys_p, max_t, temp):
        if not message.strip():
            yield message, history
            return
        history = list(history or [])
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        yield "", history
        past_pairs = []
        for i in range(0, len(history) - 3, 2):
            past_pairs.append([_get_content(history[i]), _get_content(history[i + 1])])
        for partial in chat_stream(message, past_pairs, sys_p, max_t, temp):
            history[-1]["content"] = partial
            yield gr.update(), history

    inputs = [msg_box, chatbot, system_prompt, max_tokens, temperature]
    outputs = [msg_box, chatbot]
    msg_box.submit(respond, inputs, outputs)
    send_btn.click(respond, inputs, outputs)
    clear_btn.click(lambda: [], outputs=chatbot)
    ctx_slider.change(_ctx_label, inputs=ctx_slider, outputs=ctx_info)

    # ============================================================
    # ブラウザを閉じたら 30 秒後にプロセス終了
    # → launch.command が server.py も殺してメモリ完全解放
    # （30秒のグレースは「リロード」誤判定回避）
    # ============================================================
    _active_sessions = set()
    _shutdown_timer = {"t": None}
    _lock = threading.Lock()

    def _on_load(request: gr.Request):
        with _lock:
            _active_sessions.add(request.session_hash)
            if _shutdown_timer["t"]:
                _shutdown_timer["t"].cancel()
                _shutdown_timer["t"] = None
                print(f"🟢 セッション復帰: {len(_active_sessions)} 個アクティブ")

    def _on_unload(request: gr.Request):
        with _lock:
            _active_sessions.discard(request.session_hash)
            print(f"🟡 セッション終了: {len(_active_sessions)} 個残存")
            if not _active_sessions:
                print("⏳ 30秒後にシャットダウン予定（再接続あればキャンセル）")
                _shutdown_timer["t"] = threading.Timer(30.0, _shutdown)
                _shutdown_timer["t"].daemon = True
                _shutdown_timer["t"].start()

    def _shutdown():
        print("🛑 ブラウザ未接続 — UI 終了 → server.py もパージ")
        os._exit(0)

    demo.load(_on_load)
    demo.unload(_on_unload)


demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    show_error=True,
    css=DARK_CSS,
    js="() => document.documentElement.classList.add('dark')",
)
