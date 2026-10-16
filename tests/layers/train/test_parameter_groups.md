# tests/layers/train/test_parameter_groups.cpp


Suites: `Small_ParameterGroups`.

Tests `Train::takes_weight_decay`, which decides by a parameter's role whether weight decay
applies.

This file needs no device. `test_steps.cpp` checks each role at run time; this file pins that
the decision is a constant expression, which an optimizer may rely on when it groups slots.

## K_ROLES

`K_ROLES` lists the four values of `NN::ParameterRole`, the weight first.

## decaying_role_count

`decaying_role_count` counts the roles of `K_ROLES` that take decay, and is itself `constexpr`.

## TheWeightIsTheOnlyRoleThatDecays

`static_assert` lines pin that exactly one role decays, that it is the weight and that the
embedding table does not. The run-time expectation repeats the count so the case reports a
result.
