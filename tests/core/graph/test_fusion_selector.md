# tests/core/graph/test_fusion_selector.cpp


Suites: `Small_FusionSelector`.

Tests `DeviceFusionSelector::choose` on device profiles written by hand, so no device is needed.
`fp64_device` is a default profile with native doubles, and `query_of` is a float32 match over a
[4, 5] result. Running a selector inside a compiled step is covered in
`tests/layers/train/test_trainer/test_fusion_numerics.cpp`.

## AMatchIsTakenWholeWhenNothingArguesAgainstIt

A parameterised case, one row for each of a GEMM with bias, a GEMM with bias and GELU and the
elementwise exp, sub and scale form. On a device with doubles in the automatic mode each is
answered with the matched form and a reason.

## ADoubleMatchOnADeviceWithoutDoublesIsDeclined

A float64 match on the default profile, which has no native doubles, is answered with no form
and a reason naming double precision.

## TheReferenceModeDeclinesAGEMMEpilogue

In `ForceReference` a GEMM with bias and tanh is declined.

## TheReferenceModeKeepsTheElementwiseForm

In `ForceReference` the elementwise exp, sub and scale form is still taken.

## AStridedEpilogueOperandIsDeclined

A GEMM with bias whose bias is not contiguous is answered with no form and a reason naming
contiguity.
