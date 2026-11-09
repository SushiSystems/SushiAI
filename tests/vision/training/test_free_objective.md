# tests/vision/training/test_free_objective.cpp


Suites: `Small_FreeObjective` and `Medium_FreeObjectiveDeathTest`.

Tests `FreeObjective` of `free_objective.hpp`, the anchor-free objective, over the two-level model
of `free_objective_ports.hpp`: two levels, `reg_max` 2, a batch of 2, 3 classes. No case needs a
device. `test_free_objective_factory.cpp` tests how it is built from a section.

### DeclaresTheLeavesTheLossReads

`declare` adds the 21 stage leaves, less the 6 that a gain of 0 for L1 leaves out (the anchor
strides, the target distances and the L1 scale), and the projection: 16 leaves. The anchor strides
and the L1 scale of level 0 are absent.

### DeclaresTheProjection

The projection leaf is `[reg_max, 1]`.

### DeclaresTheTargetScoresOfALevelInItsGrid

The target scores of level 1 are `[2, 2, 2, 3]`: the batch, the grid of the level and the classes.

### AnL1GainDeclaresTheL1Leaves

With an L1 gain of 1 the objective declares 20 leaves, among them the target distances of level 1
and the L1 scale of level 0.

### TheLossIsAOneElementRoot

Over the ports of both levels as parameters, `loss` returns a root of one element.

### AMissingPortIsFatalNamingThePort

`loss` over outputs without the ports stops the process; the report names `one2many.level0.boxes`.
The case is a death test, so it is `Medium_FreeObjectiveDeathTest`.

### TheMetricsNameEachTerm

`add_free_loss_metrics` reports `box`, `class`, `distribution` and `l1` with the values 1, 2, 3 and
4.
