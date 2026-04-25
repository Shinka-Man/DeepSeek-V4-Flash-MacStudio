"""DeepSeek-V4 Flash — OpenAI互換サーバー (シングルスレッド)
MLX は推論をモデルをロードしたスレッドでしか実行できないため、
http.server (シングルスレッド) で動かす。
"""
import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from model_utils import load
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

HOST = "0.0.0.0"
PORT = 7777
MODEL_ID = "deepseek-v4-flash-mxfp8"

print("Loading DeepSeek-V4 Flash model...")
model, tokenizer = load()
print(f"Model loaded! Server starting on {HOST}:{PORT}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}")

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))

        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 4096)
        temperature = req.get("temperature", 0.7)
        stream = req.get("stream", False)

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        ts = int(time.time())

        if not stream:
            full = ""
            for chunk in stream_generate(model, tokenizer, prompt=prompt,
                                         max_tokens=max_tokens,
                                         sampler=make_sampler(temp=temperature)):
                if chunk.text:
                    full += chunk.text
            self._json(200, {
                "id": rid, "object": "chat.completion", "created": ts,
                "model": req.get("model", MODEL_ID),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": full}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
            return

        # SSE streaming
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        mdl = req.get("model", MODEL_ID)
        for chunk in stream_generate(model, tokenizer, prompt=prompt,
                                     max_tokens=max_tokens,
                                     sampler=make_sampler(temp=temperature)):
            if not chunk.text:
                continue
            d = {"id": rid, "object": "chat.completion.chunk", "created": ts, "model": mdl,
                 "choices": [{"index": 0, "delta": {"content": chunk.text}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(d)}\n\n".encode())
            self.wfile.flush()

        d = {"id": rid, "object": "chat.completion.chunk", "created": ts, "model": mdl,
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(f"data: {json.dumps(d)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()


httpd = HTTPServer((HOST, PORT), Handler)
print(f"OpenAI API ready: http://{HOST}:{PORT}/v1")
httpd.serve_forever()
