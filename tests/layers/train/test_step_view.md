# tests/layers/train/test_step_view.cpp


Suites: `Small_StepView`.

Tests what a `Train::StepOutput` holds by default and what a `Train::StepView` carries.

This file needs no device; the outputs point at default tensors. `Train::only_output` is pinned
in `test_metrics.cpp` and is not repeated here.

## ADefaultOutputIsUnbound

A default `StepOutput` has an empty name and a null tensor.

## AViewCarriesTheBatchTheLossAndTheOutputsInOrder

A view built over a batch, a loss of 0.75 and the outputs `p3` and `p4` refers to that batch by
address, returns the loss exactly and lists the two outputs in the order given, each with its
own tensor.
