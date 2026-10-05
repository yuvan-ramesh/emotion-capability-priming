import argparse
import json
from math import comb
import numpy as np
import pandas as pd

from common import outcome

p = argparse.ArgumentParser()
p.add_argument("records")
p.add_argument("--direction", default=None)
args = p.parse_args()

records = [json.loads(line) for line in open(args.records)]
df = pd.DataFrame(records)
if args.direction is not None:
    df = df[df.direction == args.direction].copy()

scores = pd.DataFrame(
    [outcome(r.response, r.gold, r.truncated) for r in df.itertuples()]
)
for col in scores.columns:
    if col != "truncated":
        df[col] = scores[col].values


def mcnemar(base, cond):
    b = int((base & ~cond).sum())
    c = int((~base & cond).sum())
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def bh(values):
    values = np.asarray(values)
    order = np.argsort(values)
    ranked = values[order] * len(values) / (np.arange(len(values)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(values))
    out[order] = np.minimum(ranked, 1)
    return out


wide = df.pivot(index="question_id", columns="alpha_multiplier", values="correct").astype(bool)
base = wide[0.0].values
mults = [x for x in wide.columns if x != 0.0]
pvals = [mcnemar(base, wide[x].values) for x in mults]
qvals = dict(zip(mults, bh(pvals)))
pvals = dict(zip(mults, pvals))

rows = []
for strength, group in df.groupby("alpha_multiplier"):
    if strength == 0:
        right_to_wrong = 0
        wrong_to_right = 0
    else:
        cond = wide[strength].values
        right_to_wrong = int((base & ~cond).sum())
        wrong_to_right = int((~base & cond).sum())
    rows.append(
        {
            "strength": strength,
            "n": len(group),
            "accuracy": group.correct.mean(),
            "change_pts": 100 * (group.correct.mean() - base.mean()) if strength != 0 else np.nan,
            "mcnemar_p": pvals.get(strength, np.nan),
            "bh_q": qvals.get(strength, np.nan),
            "right_to_wrong": right_to_wrong,
            "wrong_to_right": wrong_to_right,
            "parse_fail_pct": 100 * group.parse_failure.mean(),
            "truncated_pct": 100 * group.truncated.mean(),
            "mean_tokens": group.n_new_tokens.mean(),
            "passes_gate": bool(group.parse_failure.mean() < 0.10 and group.truncated.mean() < 0.10),
        }
    )

result = pd.DataFrame(rows).sort_values("strength")
print(result.round(3).to_string(index=False))
result.to_csv(args.records.rsplit("/", 1)[0] + "/scores.csv", index=False)

x = result.strength.values
y = result.accuracy.values
c2, c1, c0 = np.polyfit(x, y, 2)
r2 = 1 - ((y - np.polyval([c2, c1, c0], x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"quadratic term {c2:.4f}, R^2 {r2:.2f}, peak {-c1 / (2 * c2):.2f}")
