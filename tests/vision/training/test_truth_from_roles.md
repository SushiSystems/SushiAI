# tests/vision/training/test_truth_from_roles.cpp


Suites: `Small_TruthRoles`, `Medium_TruthFromRoles`, `Medium_TruthFromRolesMissing`,
`Medium_TruthFromRolesBadRole` and `Medium_TruthFromRolesFlag`.

Tests `describe_truth_roles` and `truth_from_roles` of `truth_from_roles.hpp`: the three roles of a
detection ground truth (`boxes`, `classes`, `valid`) and the conversion of a batch that carries
them into a `GroundTruthBatch`. The batch is two images of three object slots over five classes.
The `Small_` cases describe the roles. The `Medium_` cases allocate a batch on the CPU SYCL device
through `Data::allocate_batch`, so they run on an engine; a host-built batch would make the
refusals decided on roles and shapes `Small_`, and that seam is on the list of RW-155.

### DescribesBoxesClassesAndValidOverTheSlots

The roles are `boxes` of `[3, 4]` float32, `classes` of `[3]` int32 over 5 classes, and `valid` of
`[3]` float32.

### DescribesAClassCountOfZero

A class count of 0 is described, for a dataset of any count.

### RefusesNoSlot and RefusesANegativeClassCount

No slot fails with `InvalidDescription` naming "object slot" and "got 0". A class count of -1 fails
with `InvalidDescription` naming "classes" and "got -1".

### CopiesTheCorners, WidensTheClasses and MakesTheFlagsBytes

A batch filled with the corners, the classes 4, 0, 2, 0, 0, 1 and the flags 1, 0, 1, 0, 0, 1 gives
a truth of 2 images and 3 slots whose boxes equal the corners, whose labels are the classes as
64-bit values, and whose valid flags are the bytes 1, 0, 1, 0, 0, 1. Every value is a copy or an
exact conversion.

### FindsTheRolesInAnotherOrder

With the boxes and the flags swapped in the batch, the truth is the same.

### KeepsTheStorageOfATruthBetweenBatchesOfOneSize

A second call with a batch of the same size writes into the arrays of the first: the three data
pointers do not change and label 0 reads 3.

### RefusesABatchThatLacksARole

Rows: the batch without `boxes`, without `classes` and without `valid`. Each fails with
`InvalidDescription` naming the missing role and "it carries".

### RefusesABatchWithNoRole

A batch with no role fails with `InvalidDescription` naming `'boxes'` and "(none)".

### RefusesARoleTheTruthCannotRead

Rows, each failing with `Incompatible`: `ClassesOfFloat32`, `ValidOfInt32`, `BoxesOfRankThree`,
`BoxesOfWidthThree`, `ClassesOfAnotherSlotCount` and `ValidOfAnotherSlotCount`. The message names
the role and what it holds.

### RefusesAFlagThatIsNeitherZeroNorOne

Rows: a flag of 0.5 and a flag that is not a number, at slot 1 of image 1. Each fails with
`InvalidDescription` naming `'valid'` and "slot 1 of image 1".

### ARefusedBatchLeavesTheWholeTruthUnchanged

After a refused call the boxes, the labels and the valid flags of the truth equal what an accepted
call wrote before. The header promises that a failure leaves the truth untouched.
