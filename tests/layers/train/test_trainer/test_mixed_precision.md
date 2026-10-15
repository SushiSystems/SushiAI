# tests/layers/train/test_trainer/test_mixed_precision.cpp


Suites: `Integration_MixedPrecision`.

This is the device half of AI-4. `tests/core/autograd/test_backward.cpp` pins the graph
shape with no hardware. This file holds what only a real Engine can answer:

- `sycl::half` executes here, so the fp16 path is a computed answer and not a graph nobody ran.
- The tensors behind the graph carry the types the IR claims. A master weight is a 4-byte
  buffer, and the copy a GEMM reads is a 2-byte one.
- The trainer refuses, before recording anything, a graph whose dtypes disagree with the policy
  it was handed.

No test asserts speed. This machine has no low-precision matrix unit, and a timing gate would
pin an artifact of emulation.

## optimizer_options

The helper returns AdamW at 3e-3, the rate `sa demo mlp` uses, and not the library default.
1e-3 is the reference AdamW default and is right as a default. It does not move a hidden layer
this small far in the couple of hundred steps a test can afford, and a test whose "did it
learn" threshold has to be relaxed until it passes asserts nothing. At the demo's rate the test
and the demo answer the same question at the same setting.

## AModelWithFP16ProductsStillLearns

Correctness is the one claim the mixed-precision change can make on this hardware. It is not
faster here and cannot be.

A zeroed classifier starts at a loss of ln(4). A last loss below half of that shows a model
that learned something and did not merely run.

## TheMastersAreFourByteBuffersAndTheCopiesAreTwo

The IR says FLOAT32 and HALF, and the test checks that the tensors underneath say the same.
The two could disagree. The test reads each parameter and gradient slot and requires FLOAT32, and
it reads each cast destination the pool holds and requires HALF. A planner that sized a slot from
the wrong dtype would hand out a view that overruns its neighbour.

That the step ran at all is the loud half of the check. SushiBLAS refuses a mixed-dtype gemm
by name, so a graph with a missing conversion could not get this far.

## TheTrainerRefusesAGraphItsPolicyDisowns

A graph built without the cast pass is handed to a trainer that was told it is mixed. A caller
makes this mistake by setting one of the two knobs and forgetting the other. Without the
refusal it would not be an error; it would be an fp32 run reported as a mixed one.

## AccuracyRefusesLogitsItCannotRead

The classifier objective's accuracy reads the logits on the host. A fallback to
`data_as<float>()` for anything that is not FLOAT64 would read two elements of a half buffer
as one and report an accuracy that is fiction. Nothing throws, nothing is NaN, and the run says 100%.

`build_classifier` lists the logits as observed, so the shipped path cannot reach this. A
caller running the pass themselves can, and gets an error.

Suites: `Regression_MixedPrecision`.

Every guarantee in this file fails silently, which is what puts these tests in the regression tier
and not beside the integration tests. A run that reports success cannot catch any of them.

- A master weight retyped to fp16 does not throw. The optimizer keeps working, the loss keeps
  falling, and updates smaller than the weight's own fp16 spacing are discarded. That is the
  failure master weights exist to prevent, and it looks like a learning rate that wants tuning.
- A conversion placed on the wrong side of a reduction does not throw either; the model learns
  worse.
- The control matters as much as the case. If the fp32 policy also inserted casts, every "mixed
  inserted N casts" assertion in this repository would measure the pass's unconditional behaviour
  and not the policy's decision.

## optimizer_options

1e-3 is the reference AdamW default. It is not enough to move a model this small far in the couple
of hundred steps a test can afford, and a comparison between two runs that both barely moved would
agree for the wrong reason. 3e-3 is the demo's rate, so the numbers in this file and the ones in
docs/README.md come from the same setting.

## train

Everything except the policy is fixed, so a difference between two runs is the policy and nothing
else. The initialiser and the dataset are both seeded, and the seeds are the library defaults and
not values this file picked. A change to either default therefore shows up as a changed trajectory
in both arms, not as a silent divergence in one.

The helper reads the parameter dtypes after the run and not before. Fusion and the cast pass both
rewrite nodes, and a master weight that survived the passes could still have been swapped for a
narrow copy by the optimizer wiring.

## TheFP32PolicyIsIndistinguishableFromNoPolicyAtAll

This test is the control, in the strongest form available: zero casts, the same recorded operation
count, and the same losses to the bit. If the pass ever starts inserting a conversion
unconditionally, this test fails before any mixed-precision test can look correct for the wrong
reason.

## NarrowingCostsConversionsAndNothingElseStructural

Casts are nodes, so they show up as recorded operations, and the count has to be exactly the casts.
A higher count would mean a conversion landed inside a fusable chain and cost the fusion as well.
graph/cast_insertion.hpp's ordering argument exists to avoid that failure, and no other test in
this file would notice it.

## MixedAndSinglePrecisionTrainToTheSamePlace

This test makes the numeric claim, and its tolerance is derived, not tuned until green. fp16
carries 11 bits of significand, so one rounded value is accurate to about 2^-11 = 4.9e-4 relative.
A step's loss passes through a handful of rounded values, so a few times that figure is the
expected deviation.

The measured figure on `sa demo mlp --epochs 8 --batches 16` (a wider model than this one, 128
steps) was 9.5e-4 relative, at the end of the first epoch. That is the largest loss whose six
printed decimals still carry enough digits to divide. The test's bound is 2e-2, about twenty times
the measured figure, and it is a bound and not a measurement. It was chosen so the test survives a
change of seed or schedule. Passing it says the deviation is under 2e-2 and nothing more precise.

The test first asserts that both runs learned. Two runs that both diverged would agree and the
agreement would mean nothing.

## AGELULayerIsNotFoldedUnderANarrowingPolicy

graph/precision.hpp's rules 2 and 5 meet in this test, and it asserts that they meet consistently.

Rule 2 says a fused epilogue runs inside the GEMM's kernel in the GEMM's element type, so an
activation the catalog folds has to compute in the GEMM's type or fusion would change the numbers.
Rule 5 says GELU stays wide: the tanh approximation computes `x*x*x`, which overflows fp16 at |x|
about 40.3. Since FusedForm::GEMMBiasGELU exists, the two rules would contradict each other if
nothing kept them apart.

The pass order keeps them apart, and it does so by structure. The cast pass runs before fusion, so
an activation the policy keeps wide gets a K_CAST between it and the narrowed K_BIAS_ADD, and a chain
with a conversion in the middle matches no catalog entry. Under fp32 the layer folds whole. Under a
narrowing policy it falls back to gemm+bias and the GELU runs on its own, as it did before the
entry existed.

If this test ever sees a gemm+bias+gelu node, an fp16 `x*x*x` is being computed, and nothing else
in the suite would say so.

The test also asserts that fusion did not stop altogether. The bias still folds on both layers, so
the fallback is the shorter catalog entry and not an unfused graph.

The last assertion is that the K_GELU is still in the graph and still in the master type. That is the
other half of rule 5 and the reason the fold is refused.
