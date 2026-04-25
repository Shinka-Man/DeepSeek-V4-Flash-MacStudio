#!/bin/zsh
# DeepSeek-V4 Flash — ダブルクリックで起動
# OpenAI互換APIサーバー (0.0.0.0:7777) + Gradio UI (127.0.0.1:7860)

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
UI_PORT=7860
API_PORT=7777

cd "$PROJECT_DIR"
source venv/bin/activate

lsof -ti :$UI_PORT  | xargs kill -9 2>/dev/null
lsof -ti :$API_PORT | xargs kill -9 2>/dev/null

echo "=========================================="
echo "  DeepSeek-V4 Flash · mxfp8 MLX"
echo "=========================================="
echo ""

# ① OpenAI 互換 API サーバー (シングルスレッド)
echo "⏳ API サーバー起動中 (port $API_PORT, モデルロード数分)..."
venv/bin/python server.py > /tmp/dsv4_server.log 2>&1 &
API_PID=$!

while ! curl -s http://127.0.0.1:$API_PORT/v1/models 2>/dev/null | grep -q model; do
    sleep 2
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "❌ API サーバー起動失敗  (ログ: /tmp/dsv4_server.log)"
        read -k "?Enter で閉じる..."
        exit 1
    fi
    printf "."
done
echo ""
echo "✅ API サーバー準備完了"

# ② Gradio UI
echo "⏳ Gradio UI 起動中 (port $UI_PORT)..."
venv/bin/python ui.py > /tmp/dsv4_ui.log 2>&1 &
UI_PID=$!

while ! curl -s http://127.0.0.1:$UI_PORT > /dev/null 2>&1; do
    sleep 1; printf "."
done

echo ""
open "http://127.0.0.1:$UI_PORT"

echo ""
echo "=========================================="
echo "  🟢 起動完了"
echo "  UI:  http://127.0.0.1:$UI_PORT"
echo "  API: http://0.0.0.0:$API_PORT/v1"
echo "  終了: Ctrl+C またはこのウィンドウを閉じる"
echo "=========================================="

cleanup() {
    echo ""
    echo "🛑 シャットダウン中..."
    kill $UI_PID $API_PID 2>/dev/null
    wait $UI_PID $API_PID 2>/dev/null
    echo "👋 終了しました"
}
trap cleanup EXIT INT TERM

wait $API_PID $UI_PID
