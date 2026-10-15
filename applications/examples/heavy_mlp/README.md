# heavy_mlp: a deep MLP on a problem no linear model can solve

Three JSON files and no C++ at all. This directory used to hold a pair of
programs; it holds configuration now, and the two drivers that read it:
`applications/programs/source/train.cpp` and `applications/programs/source/inference.cpp`: name no layer type, no loss, no
dataset and no optimizer. That is the point: they were written once and do not
change as the library grows.

```bash
sa build
sushiai train      applications/examples/heavy_mlp/train.json
sushiai inference  applications/examples/heavy_mlp/model.json \
                   applications/examples/heavy_mlp/heavy_mlp.sushi \
                   applications/examples/heavy_mlp/eval.json
```

The checkpoint lands **beside `train.json`**, not in your working directory:
`"checkpoint": "heavy_mlp.sushi"` is relative, and a relative path in a
configuration file resolves against that file's own directory.

## What it does, measured

Training takes about 30 seconds on a CPU/OpenCL box (AMD Ryzen 5 7600X,
50 epochs over 64 batches of 64), and reports:

```
  epoch   1/50  loss 0.402302   accuracy  85.94%
  epoch  10/50  loss 0.123111   accuracy  95.31%
  epoch  25/50  loss 0.064925   accuracy  96.88%
  epoch  50/50  loss 0.012367   accuracy 100.00%

compiles: 1
wrote applications/examples/heavy_mlp/heavy_mlp.sushi (108996 scalars)
```

Inference over the 512 held-out points then reports **86.33% top-1
agreement**. That is the honest number: the 100% above is one training batch of
64, and this is 512 samples the model has never seen, drawn from a different
seed. The gap between the two is what an evaluation set is for.

The `compiles: 1` line is the compile-once contract: the step is compiled
exactly once no matter how many batches run, and two CTest cases pin it.

---

## The problem, and why it is this one

Four concentric rings in a plane, one class per ring. A point's class is a
function of `x² + y²` (a degree-2 polynomial in the inputs), so the decision
boundary is not a half-space and **no linear classifier can separate it**. A
single `linear` layer tops out near chance accuracy on this data no matter how
it is trained.

That is the whole reason this model is *heavy*: four hidden layers of GELU are
what turn a radius into something the final linear layer can cut with
hyperplanes. The synthetic dataset the test suite uses elsewhere
(`SyntheticClassification`) is linearly separable by construction, so any
correct implementation passes it; this one is a gate a shallow model actually
fails.

Four purely Gaussian **distractor** features ride along with `(x, y)`, so the
network also has to learn which of its inputs carry signal. `Data::Rings`
generates them, and `input.features` is `6` for that reason: two coordinates
plus four distractors.

## The three files

| File | Describes |
|---|---|
| `model.json` | the architecture alone: `6 → 256 → 256 → 128 → 64 → 4`, GELU between |
| `train.json` | the objective, the data, the optimizer, the epochs, the checkpoint |
| `eval.json` | the data to evaluate on |

`train.json` **references** `model.json` rather than embedding it, and
`sushiai inference` is handed the same file. Two copies of one architecture
are two things that drift apart silently.

**`eval.json`'s seed differs from `train.json`'s**, so the evaluation set is
genuinely unseen. That is not incidental: an evaluation on the training
distribution's own samples measures memorisation, and the seed is the only
thing separating the two here.

## Editing it

Change a width, add a layer, switch the optimizer: nothing is rebuilt:

```json
{ "type": "linear", "units": 512 }
```

Two rules are worth knowing, because each closes a way of being quietly wrong:

- **`units` is an output width only.** Each layer's input is its predecessor's
  output and `input.features` seeds the chain, so a mismatch between adjacent
  layers cannot be written down at all.
- **An unknown key is an error.** A typed `"unitss": 512` is refused, with the
  message listing what the layer *does* accept, rather than training silently
  against a default.

`sushiai train --help` and `sushiai inference --help` print their usage;
[`docs/guides/CLI_GUIDE.md`](../../../docs/guides/CLI_GUIDE.md) documents the schema in full.
