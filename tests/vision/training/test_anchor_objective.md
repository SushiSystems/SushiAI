# tests/vision/training/test_anchor_objective.cpp


Suites: `Small_AnchorObjective`, `Small_AnchorObjectiveHeadCount` and
`Small_AnchorObjectiveOptions`.

Tests `AnchorObjective` of `anchor_objective.hpp`. The cases build the objective over a 64 by 64
pixel network, four anchors in two heads (`head.0` on a 4 by 4 grid, `head.1` on a 2 by 2 grid),
two classes and a batch of 2, so a head has 14 channels per cell. None needs a device.

### DeclaresOneLeafPerHeadInTheShapeOfItsOutput

`declare` adds `delta.head.0` of `[2, 4, 4, 14]` and `delta.head.1` of `[2, 2, 2, 14]`, in head
order, float32, neither differentiated. `leaf_role(1)` is `delta.head.1`.

### TheRootIsTheDeltaLossSummedOverTheHeads

The model publishes `head.1` before `head.0`. The root has one element. On `HostEvaluator`, a delta
of 1 over a head of 2 on 448 elements and a delta of 2 over a head of 3 on 112 elements give
`-(448 * 1 * 2 + 112 * 2 * 3) / 2 = -784`, exactly. A head that met the leaf of the other would
change the value, so the case shows that each head meets the leaf of its own name. The operation
names, node indices and stored scales that an earlier version asserted are not.

### AModelWithoutAHeadIsRefusedNamingTheHead

`loss` over outputs that hold `head.0` alone fails with `InvalidDescription` naming `head.1`.

### AHeadOfAnotherShapeIsRefusedNamingBothShapes

A `head.0` of `[2, 4, 4, 21]` fails with `InvalidDescription` naming `[2, 4, 4, 14]` and
`[2, 4, 4, 21]`.

### TheCostIsTheSumOverTheHeads

Two heads of one anchor and one class, the second with an object weight of 0.5, and deltas set by
hand. `measure_cost` returns a total of 7 + 2.5, a non-box share of 6.5 + 0.25, a box share of
0.5 + 2.25 and a stored cost of 14 + 5. Every value is a sum of halves and quarters, so the
comparison is equality.

### NoBatchAloneFeedsIt

`is_fed_by_batch()` is false. The getters for the name, the heads, the layout and the network
width are plain accessors and have no case.

### AHeadCountOtherThanTheLayoutsIsRefused

Rows: no head and one head over a two-head layout. Each fails with `InvalidDescription`; the
messages hold "0 heads" and "1 heads", each with "holds 2".

### HeadsOfDifferentBatchesAreRefused

A second head with a batch of 3 fails with `InvalidDescription`, and the message holds `head.1`,
"batch of 3" and `head.0`.

### AnEmptyOutputNameIsRefused and ARepeatedOutputNameIsRefused

A head with an empty output name fails with "head 1" and "names no model output". Two heads that
both name `head.0` fail with "both read" and `head.0`. Both are `InvalidDescription`.

### AnExtentBelowOneIsRefused and ANetworkSideBelowOneIsRefused

A head of height 0 fails naming `head.0` and `[2, 0, 4]`. A network of width 0 fails naming "0 by
64". Both are `InvalidDescription`.

### OptionsTheValidatorsRefuseAreRefused

Rows: a head of no classes ("at least one class") and a head with a box weight of -1 ("box weight
of an anchor head"). Each fails with `InvalidDescription` and names its reason; the earlier
version of the second row asserted the code alone.
