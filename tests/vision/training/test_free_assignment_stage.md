# tests/vision/training/test_free_assignment_stage.cpp


Suites: `Small_VisionTrainingFreeAssignmentStage`, `Small_VisionTrainingFreeAssignmentStagePorts`
and `Small_VisionTrainingFreeAssignmentStageMake`.

Tests `FreeAssignmentStage` of `free_assignment_stage.hpp`, which runs the task-aligned assigner
over the ports of an anchor-free model and fills the leaves the loss reads. The fixture is two
images over two levels (4 by 4 cells at stride 8, 2 by 2 at stride 16: 20 anchors per image), two
classes, and a truth of three boxes (two in image 0, one in image 1, one padding row). Three cells
match their box; every other cell decodes to a small distance. No case needs a device.

### MakeRefusesABatchBelowOne

A batch of 0 fails with `InvalidDescription`.

### MakeRefusesWhatItCannotAssign

Rows, each failing with `InvalidDescription`: a `reg_max` of 0 ("got 0 and 2"), a class count of 0
("got 1 and 0") and a `topk` of 21 against 20 anchors ("topk 21").

### LeavesListEachKindPerLevelThenTheSum

A raw-distance stage lists 19 leaves: the target scores of level 0 first, the box mask of level 1
in position 3, and the sum last with the shape `[1]`. It lists no distribution weights.

### AOneToOneStageNamesItsLeavesApart

A stage for the one-to-one branch lists 19 leaves, the first `free.one_to_one.target_scores.level0`
and the last `free.one_to_one.target_scores_sum`, and no leaf outside the one-to-one branch.

### AOneToOneStageOfAModelWithoutTheCopyIsRefused

A one-to-one stage over facts without the copy fails with `InvalidDescription` naming "one-to-one
copy".

### RawDistancesGiveHandWorkedTargets

The foreground anchors are the one of truth A on level 0, the one of truth B on level 1, the one of
truth C on level 1 and the four anchors of level 0 inside truth C that the top-k keeps at metric 0.
The target scores, the normaliser of 3, the normalised weight of 1 / 3, the target boxes, the target
distances of 0.5 and the L1 scale of 1 / 48 are the values worked by hand, each to 2048 float32
unit roundoffs. There are no distribution weights.

### RawDistancesFillTheBoxMaskLeaf

The box mask leaf of level 1 is 1 in the four values of cells 3 and 4 and 0 elsewhere.

### RawDistancesFillTheSumLeaf

The sum leaf holds 3, to 3 times the tolerance.

### BinnedDistancesGiveTheSameTargetsAndBinWeights

At `reg_max` 4 the targets are the same, the bin weights of the foreground anchor of truth A are
1 / 24 on bins 0 and 1 and 0 on bins 2 and 3 for each side, a background anchor has none, and the
distribution leaf of level 0 is `[2, 4, 4, 16]`.

### RefusesANonFinitePortNamingLevelAndIndex

Rows: a NaN box output at index 7 of level 1 ("level 1 box output 7") and an infinite score at
index 3 of level 0 ("level 0 score 3"). Each `assign` fails with `InvalidDescription`.

### AcceptsANonFinitePaddingRow

A NaN in the padding row of the truth is accepted.

### RefusesANonFiniteValidTruthBoxNamingTheSlot

A NaN on a valid box fails with `InvalidDescription` naming "truth slot 1" and "at 1".
