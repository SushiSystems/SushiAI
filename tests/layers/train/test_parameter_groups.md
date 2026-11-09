# tests/layers/train/test_parameter_groups.cpp


Suites: `Small_ParameterGroups`.

Tests `Train::takes_weight_decay`, which decides by a parameter's role whether weight decay
applies.

This file needs no device. A `static_assert` at the top of the file pins that the decision is a
constant expression, which an optimizer may rely on when it groups slots. The same promise at run
time was also tested in `test_steps.cpp`; that copy is deleted.

## RoleRow

`RoleRow` holds a parameter role and whether it takes decay.

## TheWeightIsTheOnlyRoleThatDecays

One row per role: the weight takes decay, and the bias, the scale and the embedding table do not.
