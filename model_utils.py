"""DeepSeek-V4 Flash model/tokenizer loading utilities.

transformers が deepseek_v4 model_type を未認識のため、
tokenizer を手動で構成し TokenizerWrapper で包んでロードする。
"""
from pathlib import Path
from mlx_lm.utils import load_model
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import PreTrainedTokenizerFast

DEFAULT_MODEL_PATH = Path(
    "/Users/m3ultra/.lmstudio/models/mlx-community/DeepSeek-V4-Flash-mxfp8"
)


def load(model_path: Path = DEFAULT_MODEL_PATH):
    """Load model and tokenizer, returns (model, tokenizer)."""
    model, _config = load_model(model_path, lazy=False)

    base_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json"),
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<｜end▁of▁sentence｜>",
        pad_token="<｜end▁of▁sentence｜>",
    )
    with open(model_path / "chat_template.jinja") as f:
        base_tokenizer.chat_template = f.read()

    tokenizer = TokenizerWrapper(base_tokenizer)

    return model, tokenizer
