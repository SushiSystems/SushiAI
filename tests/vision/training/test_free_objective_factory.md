# tests/vision/training/test_free_objective_factory.cpp

Suites: `Small_FreeObjectiveFactory`.

Tests `build_free_objective` and `make_free_objective` of `free_objective_factory.hpp`. `free_objective_ports.hpp` holds the two-level model of ports that this file and `test_free_objective.cpp` share.

### TheFactoryDerivesTheFactsAndReadsTheGains

A section with a box gain of 5 builds an objective of two levels, `reg_max` 2, a stride of 16 at the second level and the default class gain.

### TheFactoryRefusesAKeyThatIsNoGain

The key "reg_max" fails with `InvalidDescription` naming it.

### TheBranchWeightKeysAreKeysOfTheSection

The keys `one_to_many_weight_start` and `one_to_many_weight_end` are accepted.

### ASectionOfAnotherTypeIsRefused

A section of the type "anchor" fails with `InvalidDescription`.

### AModelWithoutATaskIsRefused

A context without a task fails with `InvalidDescription` naming "'task'".

### ADataClassCountOtherThanThePortsIsRefused

A context that states 5 classes against score ports of 2 fails with `Incompatible`; the message holds "5 classes" and "hold 2".

### DataThatStatesNoClassCountIsAccepted

A context that states 0 classes builds an objective of the 2 classes of the ports.

### TheRegistryEntryBuildsTheObjective, TheRegistryEntryPrefixesARefusalWithItsContext

`make_free_objective` returns an objective named `K_FREE_OBJECTIVE_TYPE`; for a context without a task the failure holds the context "the objective block" and "'task'".
