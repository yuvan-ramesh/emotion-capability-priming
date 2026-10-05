import argparse
import json
from math import comb
import numpy as np
import pandas as pd

from common import bootstrap_mean_diff, outcome

p = argparse.ArgumentParser()
p.add_argument("--sad", required=True)
p.add_argument("--anxiety", required=True)
p.add_argument("--random", default=None)
p.add_argument("--suppression", default=None)
args = p.parse_args()


def scored(path):
    rows = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(rows)
    s = pd.DataFrame([outcome(r.response, r.gold, r.truncated) for r in df.itertuples()])
    for col in s.columns:
        if col != "truncated":
            df[col] = s[col].values
    return df


def paired_p(a, b):
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    n10 = int((x & ~y).sum())
    n01 = int((~x & y).sum())
    n = n10 + n01
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(n10, n01) + 1))
    return min(1.0, 2 * tail / 2 ** n)


sad = scored(args.sad)
anx = scored(args.anxiety)

print("sadness vs anxiety")
for strength in [-4.0, -2.0, 2.0, 4.0]:
    a = sad[sad.alpha_multiplier == strength].set_index("question_id")
    b = anx[anx.alpha_multiplier == strength].set_index("question_id")
    ids = a.index.intersection(b.index)
    av = a.loc[ids, "correct"].values
    bv = b.loc[ids, "correct"].values
    lo, hi = bootstrap_mean_diff(av, bv)
    print(
        strength,
        round(av.mean(), 3),
        round(bv.mean(), 3),
        round(av.mean() - bv.mean(), 3),
        (round(lo, 3), round(hi, 3)),
        round(paired_p(av, bv), 4),
    )

print("answer length")
for name, df in [("sadness", sad), ("anxiety", anx)]:
    neg = df[df.alpha_multiplier == -2.0].set_index("question_id")
    pos = df[df.alpha_multiplier == 2.0].set_index("question_id")
    ids = neg.index.intersection(pos.index)
    change = pos.loc[ids, "n_new_tokens"].values - neg.loc[ids, "n_new_tokens"].values
    rng = np.random.default_rng(42)
    means = []
    for _ in range(2000):
        idx = rng.integers(0, len(change), len(change))
        means.append(change[idx].mean())
    print(name, round(change.mean(), 1), tuple(round(x, 1) for x in np.quantile(means, [0.025, 0.975])))

if args.random:
    rnd = scored(args.random)
    print("random control")
    table = rnd.groupby(["random_id", "alpha_multiplier"]).correct.mean().groupby("alpha_multiplier").mean()
    print(table.round(3).to_string())

if args.suppression:
    sup = scored(args.suppression)
    print("suppression")
    cell_acc = sup.groupby("cell").correct.mean()
    print(cell_acc.round(3).to_string())
    did = (
        cell_acc["sadness_on"]
        - cell_acc["sadness_off"]
        - cell_acc["neutral_on"]
        + cell_acc["neutral_off"]
    )
    print("difference_in_differences", round(float(did), 4))
