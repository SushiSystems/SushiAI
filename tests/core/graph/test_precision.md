# tests/core/graph/test_precision.cpp


Suites: `Small_PrecisionPolicy`.

Tests the `PrecisionPolicy` interface and its two shipped implementations. What
each operation computes in under the two policies is pinned by the `Small_Precision` cases of
`tests/core/autograd/test_backward.cpp`.

## ThePoliciesNameThemselves

`UniformPrecisionPolicy` is named "fp32" and `MixedPrecisionPolicy` "mixed".

## MakeStoresARealFloatLowType

`MixedPrecisionPolicy::make(Float32)` returns a policy whose low type is float32.

## TheMixedPolicyDefaultsToHalf

A mixed policy built without an argument narrows to half.

## MakeRefusesALowTypeThatIsNotARealFloat

`MixedPrecisionPolicy::make` returns `InvalidDescription` naming "narrows to a real floating
type" for int32 and for complex32.

## MakeAcceptsHalfAsTheLowType

`MixedPrecisionPolicy::make(Half)` succeeds.

## APolicyIsConsultedThroughTheInterface

`find_dtype_mismatches` over a float32 matmul graph reports none for the uniform policy and some
for the mixed one, so the finder decides through the policy it is given.
