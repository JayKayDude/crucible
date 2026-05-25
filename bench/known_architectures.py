"""Lookup tables mapping model name fragments to architecture types and HF tokenizers."""

KNOWN_ARCHITECTURES: dict[str, str] = {
    "qwen":     "gated-delta",
    "gemma":    "sliding-window",
    "llama":    "standard",
    "mistral":  "standard",
    "phi":      "standard",
    "deepseek": "moe",
    "mixtral":  "moe",
    "falcon":   "standard",
}

# Maps model name fragments to a compatible HuggingFace tokenizer ID.
# Used by lm-eval when lmeval_tokenizer is not explicitly set in the model config.
# Picks the smallest publicly available model in each family (tokenizer files only
# are downloaded — not weights). Llama requires HF login; see README.
KNOWN_TOKENIZERS: dict[str, str] = {
    "qwen":     "Qwen/Qwen2.5-0.5B",
    "gemma":    "google/gemma-3-1b-it",
    "llama":    "meta-llama/Meta-Llama-3.1-8B",    # requires HF login
    "mistral":  "mistralai/Mistral-7B-v0.1",
    "phi":      "microsoft/Phi-3-mini-4k-instruct",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "mixtral":  "mistralai/Mistral-7B-v0.1",
    "falcon":   "tiiuae/falcon-7b",
}


def lookup_architecture(model_id: str) -> str | None:
    """Return architecture type for a model ID, or None if unknown."""
    lower = model_id.lower()
    for fragment, arch in KNOWN_ARCHITECTURES.items():
        if fragment in lower:
            return arch
    return None


def lookup_tokenizer(model_id: str) -> str | None:
    """Return a compatible HuggingFace tokenizer ID for a model, or None if unknown."""
    lower = model_id.lower()
    for fragment, tokenizer in KNOWN_TOKENIZERS.items():
        if fragment in lower:
            return tokenizer
    return None
