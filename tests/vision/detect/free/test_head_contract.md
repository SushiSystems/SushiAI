# tests/vision/detect/free/test_head_contract.cpp


Suites: `Small_DetectFreeHeadContract`.

Tests `detect_port_name` and `detect_class_bias` of `free/head_contract.hpp`, through
`Vision::Detect::Free`. The cases that compared the name and bias constants with literals are
deleted: the names are checked through built ports in `tests/vision/blocks/test_detect_head.cpp`,
and the bias constants through `detect_class_bias` below. The re-export of the same names in
`Vision::Blocks` is pinned by `tests/vision/blocks/test_detect_head.cpp`. The file needs no device.

## K_BIAS_TOLERANCE

`K_BIAS_TOLERANCE` is 16384 u, where u is the double unit roundoff, the bound on a class bias against the value computed apart.

## ThePortNameJoinsCopyLevelAndRole

The name is "<copy>.level<level>.<role>": "one2many.level0.boxes", "one2many.level2.scores" and
"one2one.level12.scores".

## ThePortNameKeepsWhatItIsGiven

A copy "copy" and a role "role" at level 3 give "copy.level3.role"; an empty copy and role at
level 1 give ".level1.".

## TheClassBiasFollowsTheStrideAndTheClassCount

`detect_class_bias` equals log(5 / classes / (640 / stride)^2), as computed outside the library:
-11.536641991587544 for 80 classes at stride 8, -10.150347630467653 at 16, -8.764053269347762 at 32,
-4.382026634673881 for 1 class at 32 and -5.991464547107982 for 20 classes at 64.

## TheClassBiasRisesWithTheStrideAndFallsWithTheClasses

For 80 classes the bias at stride 8 is below the one at 16, which is below the one at 32. At stride
32 the bias for 80 classes is below the one for 1 class, and the latter is below 0.
