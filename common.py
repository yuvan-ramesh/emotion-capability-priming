import re
import numpy as np
import torch
from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"
SEED = 42


def load_model(model_path):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, processor


def decoder_layers(model):
    return model.model.language_model.layers


def load_mmlu(n_questions):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=SEED)
    return [dict(x) for x in ds.select(range(n_questions))]


def mmlu_prompt(row, prefix=""):
    options = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    return (
        f"{prefix}"
        "Answer the following multiple-choice question.\n\n"
        f"{row['question']}\n\n{options}\n\n"
        "Use no more than 150 words for your reasoning. "
        "Then finish with exactly:\nAnswer: X\n\n"
        "where X is the letter of the correct answer."
    )


ANSWER_RE = re.compile(r"Answer\s*:\s*\(?\s*([A-J])\b", re.IGNORECASE)
REFUSAL_RE = re.compile(
    r"\bi (can'?t|cannot|won'?t|will not) (help|answer|provide|assist)\b"
    r"|\bi don'?t know\b|\bi refuse\b|\bunable to determine\b",
    re.IGNORECASE,
)


def parse_answer(text):
    found = ANSWER_RE.findall(text or "")
    return found[-1].upper() if found else None


def outcome(text, gold, truncated):
    pred = parse_answer(text)
    refusal = pred is None and bool(REFUSAL_RE.search(text or ""))
    return {
        "correct": pred == gold,
        "refusal": refusal,
        "parse_failure": pred is None and not refusal,
        "truncated": bool(truncated),
    }


def make_add_hook(vec, alpha):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + alpha * vec.view(1, 1, -1)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook


def make_remove_hook(vec):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        v = vec.view(1, 1, -1)
        h = h - (h * v).sum(-1, keepdim=True) * v
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook


def generate(model, processor, prompt, layer, vec=None, alpha=0.0, remove=False, max_new_tokens=2048):
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inp = processor.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    handle = None
    if vec is not None and remove:
        handle = decoder_layers(model)[layer].register_forward_hook(make_remove_hook(vec))
    elif vec is not None and alpha != 0:
        handle = decoder_layers(model)[layer].register_forward_hook(make_add_hook(vec, alpha))
    try:
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        if handle is not None:
            handle.remove()
    ids = out[0, inp["input_ids"].shape[1]:]
    eos = model.config.get_text_config().eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    truncated = len(ids) >= max_new_tokens and not bool(set(ids.tolist()) & eos)
    return processor.decode(ids, skip_special_tokens=True), len(ids), truncated


def bootstrap_mean_diff(a, b, n_boot=2000, seed=SEED):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (a[idx] - b[idx]).mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
