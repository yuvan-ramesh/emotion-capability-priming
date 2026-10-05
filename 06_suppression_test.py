import argparse
import json
import os
import torch

from common import generate, load_mmlu, load_model, mmlu_prompt

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--directions", required=True)
p.add_argument("--frozen", required=True)
p.add_argument("--out", default="out/suppression")
p.add_argument("--n_questions", type=int, default=1000)
p.add_argument("--max_new_tokens", type=int, default=2048)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
with open(args.frozen) as f:
    frozen = json.load(f)
layer = frozen["layer"]
vec = torch.load(args.directions, weights_only=False)["directions"][layer].float()
rows = load_mmlu(args.n_questions)
model, processor = load_model(args.model)
vec = vec.to(model.device, next(model.parameters()).dtype)

cells = [
    ("neutral_off", "", False),
    ("neutral_on", "", True),
    ("sadness_off", "You are feeling sad right now. Keep that feeling in mind while answering.\n\n", False),
    ("sadness_on", "You are feeling sad right now. Keep that feeling in mind while answering.\n\n", True),
]

with open(f"{args.out}/records.jsonl", "w") as f:
    for cell, prefix, remove in cells:
        for row in rows:
            text, n_tokens, truncated = generate(
                model,
                processor,
                mmlu_prompt(row, prefix=prefix),
                layer,
                vec=vec,
                remove=remove,
                max_new_tokens=args.max_new_tokens,
            )
            f.write(
                json.dumps(
                    {
                        "cell": cell,
                        "question_id": str(row["question_id"]),
                        "category": row["category"],
                        "gold": row["answer"],
                        "response": text,
                        "n_new_tokens": n_tokens,
                        "truncated": truncated,
                    }
                )
                + "\n"
            )
