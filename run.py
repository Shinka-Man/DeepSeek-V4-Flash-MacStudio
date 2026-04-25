"""DeepSeek-V4 Flash — 一発推論（mlx_lm直接）"""
from mlx_lm import generate
from model_utils import load

print("Loading model...")
model, tokenizer = load()
print("Model loaded.\n")

messages = [
    {"role": "user", "content": "こんにちは！自己紹介してください。"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=512,
    verbose=True,
)
