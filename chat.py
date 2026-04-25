"""DeepSeek-V4 Flash — ターミナルチャット（mlx_lm直接）"""
from mlx_lm import generate
from model_utils import load

print("Loading model...")
model, tokenizer = load()
print("Model loaded. Type 'quit' to exit.\n")

history = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        break
    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Assistant: ", end="", flush=True)
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024,
        verbose=True,
    )

    history.append({"role": "assistant", "content": response})
    print()
