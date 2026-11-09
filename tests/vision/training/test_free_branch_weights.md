# tests/vision/training/test_free_branch_weights.cpp


Suites: `Small_FreeBranchWeights`.

Tests `read_free_branch_weights` and `one_to_many_weight` of `free_branch_weights.hpp`: the weight
of the one-to-many branch, which falls from a start to an end over the epochs of a run. The section
is built in memory; no case needs a device or a file.

### AnAbsentKeyKeepsItsDefault

A section of the type alone gives a start of 0.8 and an end of 0.1.

### BothStatedWeightsAreRead

A start of 0.5 and an end of the whole number 0 are read as stated.

### AWeightAboveOneIsRefusedNamingItsKey

An end of 1.5 fails with `InvalidDescription` naming `one_to_many_weight_end` and "0..1".

### AWeightBelowZeroIsRefusedNamingItsKey

A start of -0.25 fails with `InvalidDescription` naming `one_to_many_weight_start` and "0..1".

### TheWeightFallsInAStraightLineAndStaysAtTheEnd

Over 8 epochs the weight is 0.8 at epoch 0, 0.7 at epoch 1 and 0.1 at epoch 7, and it stays at
0.1 at epoch 12.

### ARunOfOneEpochUsesTheStart

A run of one epoch has no last epoch to reach and uses the start, 0.8.
