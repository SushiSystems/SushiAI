# tests/vision/training/test_truth_from_roles.cpp


Suites: `Small_TruthRoles`, `Small_TruthFromHostRoles`, `Small_TruthFromHostRolesMissing`,
`Small_TruthFromHostRolesBadRole`, `Small_TruthFromHostRolesFlag` and `Medium_TruthFromRoles`.

Tests `describe_truth_roles`, `truth_from_host_roles` and `truth_from_roles` of
`truth_from_roles.hpp`: the three roles of a detection ground truth (`boxes`, `classes`, `valid`)
and the conversion of the roles of a batch into a `GroundTruthBatch`. The roles are two images of
three object slots over five classes. The `Small_` cases describe the roles and run the copy over
`HostRole` views of vectors the case holds (`HostRoles`), so every value is host-produced and no
engine runs. The one `Medium_` case allocates a batch on the CPU SYCL device through
`Data::allocate_batch` and checks that its tensors reach the same copy.

### DescribesBoxesClassesAndValidOverTheSlots

The roles are `boxes` of `[3, 4]` float32, `classes` of `[3]` int32 over 5 classes, and `valid` of
`[3]` float32.

### DescribesAClassCountOfZero

A class count of 0 is described, for a dataset of any count.

### RefusesNoSlot and RefusesANegativeClassCount

No slot fails with `InvalidDescription` naming "object slot" and "got 0". A class count of -1 fails
with `InvalidDescription` naming "classes" and "got -1".

### CopiesTheCorners, WidensTheClasses and MakesTheFlagsBytes

Host roles filled with the corners, the classes 4, 0, 2, 0, 0, 1 and the flags 1, 0, 1, 0, 0, 1 gives
a truth of 2 images and 3 slots whose boxes equal the corners, whose labels are the classes as
64-bit values, and whose valid flags are the bytes 1, 0, 1, 0, 0, 1. Every value is a copy or an
exact conversion.

### FindsTheRolesInAnotherOrder

With the boxes and the flags swapped in the batch, the truth is the same.

### KeepsTheStorageOfATruthBetweenBatchesOfOneSize

A second call with a batch of the same size writes into the arrays of the first: the three data
pointers do not change and label 0 reads 3.

### RefusesABatchThatLacksARole

Rows: the roles without `boxes`, without `classes` and without `valid`. Each fails with
`InvalidDescription` naming the missing role and "it carries".

### RefusesABatchWithNoRole

A batch with no role fails with `InvalidDescription` naming `'boxes'` and "(none)".

### RefusesARoleTheTruthCannotRead

Rows, each failing with `Incompatible`: `ClassesOfFloat32`, `ValidOfInt32`, `BoxesOfRankTwo`,
`BoxesOfWidthThree`, `ClassesOfAnotherSlotCount` and `ValidOfAnotherSlotCount`. The message names
the role and what it holds.

### RefusesARoleTheHostCannotReadInOrder

A classes role whose values are null fails with `Incompatible` naming `'classes'`, `[2, 3]` and
"cannot read in order".

### RefusesAFlagThatIsNeitherZeroNorOne

Rows: a flag of 0.5 and a flag that is not a number, at slot 1 of image 1. Each fails with
`InvalidDescription` naming `'valid'` and "slot 1 of image 1".

### ARefusedBatchLeavesTheWholeTruthUnchanged

After a refused call the boxes, the labels and the valid flags of the truth equal what an accepted
call wrote before. The header promises that a failure leaves the truth untouched.

### TheTensorsOfABatchReachTheTruth

A batch allocated on the engine and filled with the same values gives the same boxes, labels and
valid bytes. This is the one case that needs a device: it checks `truth_from_roles` reads the
tensors of a batch into the views.
