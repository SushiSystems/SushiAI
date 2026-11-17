# tests/layers/train/test_step_view.cpp


Suites: `Small_StepView`.

Tests what a `Train::StepView` carries.

This file needs no device; the outputs and values point at default tensors. `Train::only_output`
is pinned in `test_metrics.cpp` and is not repeated here.

## AViewCarriesTheBatchTheLossAndTheOutputsInOrder

A view built over a batch, a loss of 0.75 and the outputs `p3` and `p4` refers to that batch by
address, returns the loss 0.75 by equality and lists the two outputs in the order given, each with its
own tensor.

## ANamedValueIsFoundByName

A view that holds the values `classes` and `scores` returns, for `named_value(view, "scores")`,
the tensor given under `scores`, by address.

## AnUnknownValueNameIsRefusedListingTheNamesHeld

`named_value(view, "missing")` over a view that holds `classes` fails with `InvalidDescription`
naming "no objective value 'missing'" and `'classes'`.

## AViewBuiltWithoutValuesHoldsNone

A view built with a batch, a loss and no outputs or values has an empty `values`, and
`named_value` fails with `InvalidDescription` naming "no objective value 'classes'" and "(none)".

## AValueWithNoTensorIsRefused

A value published under `classes` with a null tensor fails with `InvalidDescription` naming
`'classes'` and "no tensor".
