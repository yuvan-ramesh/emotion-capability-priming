import argparse
import json
import os
import torch

from common import generate, load_mmlu, load_model, mmlu_prompt

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--directions", required=True)
p.add_argument("--frozen", required=True)
p.add_argument("--out", default="out/mmlu")
p.add_argument("--name", default="sad")
p.add_argument("--multipliers", default="-4,-2,-1,-0.5,0,0.5,1,2,4")
p.add_argument("--n_questions", type=int, default=1000)
p.add_argument("--max_new_tokens", type=int, default=2048)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
with open(args.frozen) as f:
    frozen = json.load(f)
layer = frozen["layer"]
delta = frozen["delta"]
vec = torch.load(args.directions, weights_only=False)["directions"][layer].float()
multipliers = [float(x) for x in args.multipliers.split(",")]
rows = load_mmlu(args.n_questions)
model, processor = load_model(args.model)
vec = vec.to(model.device, next(model.parameters()).dtype)

with open(f"{args.out}/run_meta.json", "w") as f:
    json.dump(
        {
            "model": args.model,
            "direction": args.name,
            "layer": layer,
            "delta": delta,
            "multipliers": multipliers,
            "n_questions": args.n_questions,
            "seed": 42,
            "max_new_tokens": args.max_new_tokens,
        },
        f,
        indent=2,
    )

with open(f"{args.out}/records.jsonl", "w") as f:
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
                        "question_id": str(row["question_id"]),
                        "category": row["category"],
                        "direction": args.name,
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
