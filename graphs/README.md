# Graphs

These are the main graphs from my sadness steering experiments. They summarize the representation tests, MMLU-Pro steering results, control experiments, and suppression test.

## 01 - Representation AUC by layer

**Purpose:** Check how well the sadness direction can separate sad stories from other stories at each layer.

**Result:** The sadness direction worked pretty well on held-out examples and also on topics it was not trained on. The main held-out AUC was about **0.877**, and the new-topic AUC was about **0.904**. Layer 10 was chosen for the steering experiments.

```mermaid
xychart-beta
    title "Sadness representation AUC by layer"
    x-axis "Layer" 0 --> 35
    y-axis "AUC" 0.5 --> 1.0
    line [0.860,0.751,0.770,0.774,0.782,0.846,0.839,0.865,0.861,0.877,0.875,0.859,0.816,0.834,0.853,0.780,0.821,0.854,0.875,0.866,0.857,0.834,0.836,0.828,0.845,0.863,0.863,0.840,0.842,0.854,0.845,0.840,0.838,0.838,0.845,0.801]
    line [0.898,0.785,0.804,0.818,0.826,0.883,0.873,0.892,0.892,0.897,0.904,0.885,0.843,0.868,0.880,0.834,0.854,0.866,0.883,0.861,0.876,0.860,0.860,0.848,0.871,0.887,0.883,0.866,0.863,0.878,0.871,0.867,0.867,0.872,0.882,0.865]
```

First line = held-out sad vs other emotions. Second line = new topics.

## 02 - MMLU-Pro accuracy by steering

**Purpose:** See how changing the sadness steering strength affects reasoning accuracy on 1,000 MMLU-Pro questions.

**Result:** The normal model got **54.6%**. The -1 and -0.5 levels were a little higher, but the differences were not statistically significant. The biggest change was at **+4**, where accuracy dropped to **50.3%**. This was the main statistically significant accuracy result.

```mermaid
xychart-beta
    title "MMLU-Pro accuracy by sadness steering"
    x-axis ["-4","-2","-1","-0.5","0","+0.5","+1","+2","+4"]
    y-axis "Accuracy (%)" 45 --> 60
    line [51.4,53.7,56.2,56.4,54.6,55.2,54.6,55.1,50.3]
```

## 03 - Sadness vs anxiety accuracy

**Purpose:** Check whether the accuracy change happens with another emotion direction too, or if it is more specific to sadness.

**Result:** At +4, sadness steering dropped to **50.3%**, while anxiety steering was **54.3%**, which was close to the normal model. This makes the +4 effect look more specific to the sadness direction, although more emotion controls would still be useful.

```mermaid
xychart-beta
    title "Sadness vs anxiety accuracy"
    x-axis ["-4","-2","0","+2","+4"]
    y-axis "Accuracy (%)" 45 --> 60
    line [51.4,53.7,54.6,55.1,50.3]
    line [49.9,52.4,54.6,56.1,54.3]
```

First line = sadness. Second line = anxiety.

## 04 - Answer length by steering

**Purpose:** Check whether steering changes how long the model's answers are.

**Result:** Positive sadness steering generally made the answers shorter. The average went from about **388 tokens at -2** to about **306 tokens at +2**. Anxiety did not have nearly as large of a change, so this is something I want to investigate more.

```mermaid
xychart-beta
    title "Average answer length"
    x-axis ["-4","-2","0","+2","+4"]
    y-axis "Tokens" 250 --> 460
    line [441.0,387.9,339.2,305.9,317.7]
    line [339.3,345.2,339.2,339.7,387.8]
```

First line = sadness. Second line = anxiety.

## 05 - Random control accuracy

**Purpose:** Use random directions as a control to see whether any direction of the same size would cause similar changes.

**Result:** The random directions caused the model outputs to break down badly, even at fairly small steering strengths. The mean random-direction accuracy was about **0.08%**, and the best single random direction was only about **5.5%**. Because of this, I do **not** think this is a fair control, and I am planning to replace it with a better control method.

```mermaid
xychart-beta
    title "Random-direction control"
    x-axis ["Mean random accuracy","Best random direction"]
    y-axis "Accuracy (%)" 0 --> 6
    bar [0.08,5.5]
```

## 06 - Answer flips by strength

**Purpose:** Look at how many individual questions changed from right to wrong or wrong to right, instead of only looking at total accuracy.

**Result:** Even when the total accuracy did not change much, a lot of individual answers still changed. At +4, **115** questions changed from right to wrong and **72** changed from wrong to right, which gives the 43-question net drop in accuracy.

```mermaid
xychart-beta
    title "Answer flips compared with baseline"
    x-axis ["-4","-2","-1","-0.5","+0.5","+1","+2","+4"]
    y-axis "Questions" 0 --> 120
    bar [111,85,56,47,58,70,74,115]
    bar [79,76,72,65,64,70,79,72]
```

First bars = right to wrong. Second bars = wrong to right.

## 07 - Output quality

**Purpose:** Make sure the accuracy results were not mainly caused by answers failing to parse or getting cut off.

**Result:** The parse failure and truncation rates stayed below about **3%**, which is well under the 10% cutoff I set. This means the main accuracy results are probably not just from broken or unfinished outputs.

```mermaid
xychart-beta
    title "Output quality by steering strength"
    x-axis ["-4","-2","-1","-0.5","0","+0.5","+1","+2","+4"]
    y-axis "Rate (%)" 0 --> 3
    line [2.7,1.5,1.3,1.0,0.7,0.8,0.5,0.8,0.7]
    line [2.7,1.4,1.2,0.9,0.6,0.6,0.4,0.5,0.7]
```

First line = parse failures. Second line = truncated answers.

## 08 - Suppression test

**Purpose:** Test whether giving the model a sad prompt changes its reasoning, and whether removing the sadness direction could undo that effect.

**Result:** The sad prompt itself did not cause a clear accuracy change. Because there was not much of an effect to reverse, I am treating the suppression test as **inconclusive** instead of saying it worked or failed.

```mermaid
xychart-beta
    title "Suppression test accuracy"
    x-axis ["Neutral off","Neutral on","Sad prompt off","Sad prompt on"]
    y-axis "Accuracy (%)" 50 --> 57
    bar [54.2,55.2,54.5,54.9]
```
