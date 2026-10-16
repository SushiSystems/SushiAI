# tests/core/graph/test_precision.cpp


Suites: `Small_PrecisionPolicy`.

Tests the `PrecisionPolicy` interface and the names and defaults of its two implementations. What
each operation computes in under the two policies is pinned by the `Small_Precision` cases of
`tests/core/autograd/test_backward.cpp`. The file-local `EverythingWide` answers float64 to
every operation.

## ThePoliciesNameThemselvesAndTheMixedOneDefaultsToHalf

`UniformPrecisionPolicy` is named "fp32" and `MixedPrecisionPolicy` "mixed". A mixed policy
built without an argument narrows to half, and one built with float32 keeps float32 as its low
type.

## TheDefaultQueryAndChoiceAreFloat32

A default `PrecisionQuery` declares float32, and a default `PrecisionChoice` is float32 with no
reason.

## APolicyOfItsOwnIsConsultedThroughTheInterface

`EverythingWide`, reached through a `PrecisionPolicy` reference, gives its own name and answers
float64 for a float32 RELU. `find_dtype_mismatches` over a float32 RELU graph and that policy
reports mismatches, so the finder decides through the interface and not through a fixed policy.
