import argparse
import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from common import SEED, load_model

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--emotion", default="sad")
p.add_argument("--n_other", type=int, default=1200)
p.add_argument("--pca", type=float, default=0.50)
p.add_argument("--holdout_topics", type=int, default=20)
p.add_argument("--out", default="out")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(SEED)

stories = load_dataset(
    "ryancodrai/emotion-probes",
    data_files="expression/stories.parquet",
    split="train",
).to_pandas()
neutral = load_dataset(
    "ryancodrai/emotion-probes",
    data_files="expression/neutral_stories.parquet",
    split="train",
).to_pandas()

stories["emotion_key"] = stories.emotion.str.lower()
topics = np.array(sorted(stories.topic.unique()))
holdout = set(rng.choice(topics, size=min(args.holdout_topics, len(topics)), replace=False))
main = stories[~stories.topic.isin(holdout)].copy()
new_topics = stories[stories.topic.isin(holdout)].copy()

target = main[main.emotion_key == args.emotion.lower()]
other = main[main.emotion_key != args.emotion.lower()]

per_topic = max(1, args.n_other // max(1, other.topic.nunique()))
other = other.groupby("topic", group_keys=False).apply(
    lambda g: g.sample(min(len(g), per_topic), random_state=SEED)
)


def split(df):
    idx = rng.permutation(len(df))
    a = int(0.6 * len(df))
    b = int(0.8 * len(df))
    return df.iloc[idx[:a]], df.iloc[idx[a:b]], df.iloc[idx[b:]]


tar_tr, tar_va, tar_te = split(target)
oth_tr, oth_va, oth_te = split(other)
neu_tr, _, _ = split(neutral)

model, processor = load_model(args.model)


def activations(texts, start_token=50):
    out = []
    for text in texts:
        inp = processor(
            text=text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)
        with torch.inference_mode():
            hs = model(**inp, output_hidden_states=True, use_cache=False).hidden_states
        n = inp["input_ids"].shape[1]
        s = min(start_token, max(0, n // 2))
        out.append(
            torch.stack([h[0, s:n].float().mean(0).cpu() for h in hs]).numpy()
        )
    return np.stack(out)


A = {
    "tar_tr": activations(tar_tr.story.tolist()),
    "tar_va": activations(tar_va.story.tolist()),
    "tar_te": activations(tar_te.story.tolist()),
    "oth_tr": activations(oth_tr.story.tolist()),
    "oth_va": activations(oth_va.story.tolist()),
    "oth_te": activations(oth_te.story.tolist()),
    "neu_tr": activations(neu_tr.story.tolist()),
}

raw = A["tar_tr"].mean(0) - A["oth_tr"].mean(0)
directions = np.zeros_like(raw)
for layer in range(raw.shape[0]):
    v = raw[layer].copy()
    if args.pca > 0:
        pca = PCA().fit(A["neu_tr"][:, layer])
        keep = int(
            np.searchsorted(np.cumsum(pca.explained_variance_ratio_), args.pca) + 1
        )
        pcs = pca.components_[:keep]
        v = v - pcs.T @ (pcs @ v)
    directions[layer] = v / (np.linalg.norm(v) + 1e-8)

rows = []
for name, pos_arr, neg_arr in [
    ("validation", A["tar_va"], A["oth_va"]),
    ("test", A["tar_te"], A["oth_te"]),
]:
    for layer in range(directions.shape[0]):
        pos = pos_arr[:, layer] @ directions[layer]
        neg = neg_arr[:, layer] @ directions[layer]
        auc = roc_auc_score([1] * len(pos) + [0] * len(neg), np.r_[pos, neg])
        rows.append({"comparison": name, "layer": layer, "auc": auc})

if len(new_topics):
    new_target = new_topics[new_topics.emotion_key == args.emotion.lower()]
    new_other = new_topics[new_topics.emotion_key != args.emotion.lower()]
    per_topic_new = max(1, args.n_other // max(1, new_other.topic.nunique()))
    new_other = new_other.groupby("topic", group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_topic_new), random_state=SEED)
    )
    new_tar_a = activations(new_target.story.tolist())
    new_oth_a = activations(new_other.story.tolist())
    for layer in range(directions.shape[0]):
        pos = new_tar_a[:, layer] @ directions[layer]
        neg = new_oth_a[:, layer] @ directions[layer]
        auc = roc_auc_score([1] * len(pos) + [0] * len(neg), np.r_[pos, neg])
        rows.append({"comparison": "new_topics", "layer": layer, "auc": auc})

if args.emotion.lower() == "sad":
    similar_names = ["gloomy", "lonely", "miserable", "melancholy"]
    similar = main[main.emotion_key.isin(similar_names)]
    if len(similar):
        _, _, sim_te = split(similar)
        sim_a = activations(sim_te.story.tolist())
        for layer in range(directions.shape[0]):
            pos = A["tar_te"][:, layer] @ directions[layer]
            neg = sim_a[:, layer] @ directions[layer]
            auc = roc_auc_score([1] * len(pos) + [0] * len(neg), np.r_[pos, neg])
            rows.append({"comparison": "similar_emotions", "layer": layer, "auc": auc})

pd.DataFrame(rows).to_csv(f"{args.out}/auc_by_layer.csv", index=False)
torch.save(
    {"directions": torch.from_numpy(directions), "emotion": args.emotion},
    f"{args.out}/directions.pt",
)
np.savez(
    f"{args.out}/activations_val.npz",
    target=A["tar_va"],
    other=A["oth_va"],
)
