# tests/layers/train/test_trainer/test_determinism.cpp


Suites: `Medium_DeterminismRun`, `Medium_DeterminismInitialisation` and `Medium_DeterminismReplay`.

This file tests bit-for-bit reproducibility from end to end.

Reproducibility is the guarantee the stack is arranged around, and no test proved it before this
file. SushiRuntime promises sushiruntime_deterministic_fp. autograd/tape.cpp sums a fan-out through
an ordered K_ACCUMULATE chain and not through float atomics, and its own header says an atomic sum
"would break that quietly, showing up as a training run that cannot be reproduced rather than as a
test failure". A claim whose failure mode is "not a test failure" needs a test, and the test has to
compare bits. EXPECT_NEAR passes on a run that reorders a reduction, and reordering a reduction is
the change the claim forbids.

The file sits in the regression tier because every failure here is silent. A model that trains to
a slightly different place each run still converges and still reports good accuracy. The defect
shows on the day someone needs to reproduce a result.

Every test in this file builds a fresh Engine per run (the helper `Rig`), and that is part of the contract. SushiBLAS's
set_seed() leaves the engine's monotonic RNG offset counter alone, so two generators built on one
engine draw different streams from one seed. dataset.hpp documents this.

## Rig

One fresh Engine, its backend, the pool over it and the classifier bound on them. A run, an
initialisation and a replay each build their own, so two of them share nothing.

## train_once

Everything the run depends on is either fixed in this helper or seeded: the model shape, the
initialisation seed (InitPolicy's default), the data seed and the optimizer. Two calls therefore
differ in nothing, so a difference in their output is a finding.

## expect_identical

The loss is read out of device memory as a float. Two identical runs must produce the same float,
so `==` is the right comparison. EXPECT_NEAR would admit that the two runs might differ.

## TwoIdenticalRunsAgreeByteForByte

One row per optimizer, `SgdWithMomentum` and `AdamW`. The run is two epochs. With momentum, step N
depends on every step before it, so a single divergent bit cannot cancel out. AdamW divides by the
square root of an accumulated second moment, so after that division a one-ULP difference in the
gradient is more than one ULP. The two rows replace two cases that differed in the optimizer alone.

## InitialisationFromOneSeedIsBitwiseIdentical

One row per parameter of the model (four). The starting point has to be identical before anything
downstream can be. Initialisation is also the only part of the run that goes through the device RNG
and not through arithmetic. The dataset's reproducibility is held by `test_dataset.cpp` and by
`TwoDatasetsOfOneSeedHoldTheSameInputs` in `test_convergence.cpp`, so it is not repeated here.

## ReplayingOneCompiledStepOnFixedInputsIsBitwiseStable

One row per parameter. This is the narrowest form of the claim, with the optimizer and the RNG
removed: the same compiled plan and the same buffers, run once and then five times more. Each
replay leaves the row's gradient with the bytes the first run gave it. Anything the runtime does
non-deterministically shows up here with nothing else to blame it on. A work-stealing reduction, an
atomic accumulation and a scheduler-dependent order are the candidates.

The optimizer's learning rate is zero. That keeps the parameters fixed, so every replay sees the
state the one before it saw.
