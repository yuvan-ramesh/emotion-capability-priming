import argparse
import json
import os
import numpy as np
import torch

from common import SEED, generate, load_mmlu, load_model, mmlu_prompt

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--directions", required=True)
p.add_argument("--frozen", required=True)
p.add_argument("--out", default="out/random_control")
p.add_argument("--n_random", type=int, default=20)
p.add_argument("--n_questions", type=int, default=200)
p.add_argument("--multipliers", default="-2,-1,1,2")
p.add_argument("--max_new_tokens", type=int, default=2048)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
with open(args.frozen) as f:
    frozen = json.load(f)
layer = frozen["layer"]
delta = frozen["delta"]
base_vec = torch.load(args.directions, weights_only=False)["directions"][layer].float()
rows = load_mmlu(args.n_questions)
model, processor = load_model(args.model)
dtype = next(model.parameters()).dtype
rng = np.random.default_rng(SEED)
multipliers = [float(x) for x in args.multipliers.split(",")]

with open(f"{args.out}/records.jsonl", "w") as f:
    for random_id in range(args.n_random):
        arr = rng.normal(size=base_vec.numel()).astype(np.float32)
        arr /= np.linalg.norm(arr) + 1e-8
        vec = torch.from_numpy(arr).to(model.device, dtype)
        for multiplier in multipliers:
            alpha = multiplier * delta
            for row in rows:
                text, n_tokens, truncated = generate(
                    model,
                    processor,
                    mmlu_prompt(row),
                    layer,
                    vec=vec,
                    alpha=alpha,
                    max_new_tokens=args.max_new_tokens,
                )
                f.write(
                    json.dumps(
                        {
                            "random_id": random_id,
                            "question_id": str(row["question_id"]),
                            "category": row["category"],
                            "alpha_multiplier": multiplier,
                            "alpha": alpha,
                            "gold": row["answer"],
                            "response": text,
                            "n_new_tokens": n_tokens,
                            "truncated": truncated,
                        }
                    )
                    + "\n"
                )
