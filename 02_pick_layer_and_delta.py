import argparse
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--out", default="out")
args = p.parse_args()

directions = torch.load(
    f"{args.out}/directions.pt",
    weights_only=False,
)["directions"].numpy()
val = np.load(f"{args.out}/activations_val.npz")


def auc_at(layer):
    pos = val["target"][:, layer] @ directions[layer]
    neg = val["other"][:, layer] @ directions[layer]
    return roc_auc_score([1] * len(pos) + [0] * len(neg), np.r_[pos, neg])


n_layers = directions.shape[0]
layer = max(range(int(0.1 * n_layers), n_layers), key=auc_at)
pos = val["target"][:, layer] @ directions[layer]
neg = val["other"][:, layer] @ directions[layer]
delta = float(pos.mean() - neg.mean())

with open(f"{args.out}/frozen.json", "w") as f:
    json.dump({"layer": int(layer), "delta": delta}, f, indent=2)

print(f"layer {layer}, delta {delta:.4f}")
