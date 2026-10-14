# tests/vision/detect

The tests of the module `vision/detect`, one file per header of `modules/vision/detect/include/`. The suite prefix of a case gives its ctest label.

## anchor/test_anchor_clustering.cpp

Suites: `Unit_DetectAnchorClustering`.

Tests `SeededSequence` and `cluster_anchors`. The clustering cases use a ladder of six box
sizes with 400 boxes per rung, 2400 boxes in all, each jittered by 2 %. A rung counts as
recovered when an anchor lies within 5 % of it.

### SequenceOfSeedZeroMatchesThePublishedNumbers

The first eight values of `SeededSequence(0).next()` equal the eight published SplitMix64
numbers the case lists, from 0xE220A8397B1DCDAF on.

### UnitDrawIsTheUpperBitsOfTheNextNumber

The first `next_unit()` of seed 0 has the bits of the upper 53 bits of 0xE220A8397B1DCDAF
divided by 2^53, and lies in [0, 1).

### LadderIsRecoveredWithinFivePercent

With the default seed and a count of 6, every rung is recovered. The mean best IoU is above
0.95 and not below its value before refinement, 2400 boxes are used and none rejected, and the
iteration and sweep counts stay within the options' limits.

### AnchorsAreInAscendingArea

Each anchor's area is larger than the area of the anchor before it.

### AnotherSeedStillRecoversTheLadder

With the seed 0x1234 every rung is recovered and the mean best IoU is above 0.95.

### OutliersLeaveEveryCentreWithinFivePercent

With 24 extra boxes of 0.9 x 0.9, 1 % of the ladder, 2424 boxes are used, every rung is still
recovered, and the mean best IoU is not below its value before refinement.

### TwoRunsAreBitIdentical

Two runs over the same boxes and options return anchors whose widths and heights have the same
bits, the same bits in both mean IoU numbers, and equal iteration and sweep counts.

### ZeroSweepsKeepTheFitBeforeRefinement

With `max_refinement_sweeps` 0 the result reports zero sweeps, and its mean best IoU equals the
value before refinement to 1e-9.

### RefusesAnEmptyBoxList

An empty box list fails with `Incompatible`.

### RefusesFewerDistinctBoxesThanAnchors

Two distinct sizes, each repeated ten times, with a count of 3 fail with `Incompatible`.

### LeavesOutAZeroAndANaNSide

Three sizes repeated five times, plus one box with a zero side and one with a NaN side, give
2 rejected and 15 used boxes. The three anchors equal the three sizes to 1e-6 and the mean best
IoU is 1 to 1e-9.

### LeavesOutABoxAtTheSmallestSide

With `min_side` 0.125, the boxes 0.125 x 0.5 and 0.5 x 0.125 are rejected and 0.25 x 0.5 is the
one box used and the one anchor.

### RefusesOptionsOutsideTheirRange

A count of 0, a `max_iterations` of 0, a `max_refinement_sweeps` of -1, a `min_side` of -0.5
and a NaN `min_side` each fail with `InvalidDescription`.

## anchor/test_anchor_count.cpp

Suites: `Unit_DetectAnchorCount`.

Tests `choose_anchor_count`, which clusters to one more anchor per head until the gain in mean
best IoU falls below `min_gain`. The file uses three box sets: a ladder of six sizes with 400
boxes per rung, one blob of 2000 boxes, and a set of two sizes.

### LadderWithTwoHeadsChoosesThreePerHead

The ladder with two heads gives three anchors per head and six anchors. There are four trials,
for 1 to 4 per head. The gain of the second and of the third trial is at least `min_gain`, and
the gain of the fourth is below it.

### TrialsHoldTheFitOfEachCountClusteredDirectly

Each trial's mean best IoU equals, to 1e-9, that of `cluster_anchors` called with the trial's
count. The chosen clustering's mean equals `measure_mean_best_iou` over its anchors and the
mean of the chosen trial.

### BlobChoosesOnePerHead

The blob with three heads gives one anchor per head and three anchors. There are two trials,
and the gain of the second is below `min_gain`.

### MaxPerHeadOfTwoStopsAtTwo

With `max_per_head` 2 the ladder with two heads gives two per head, four anchors and two
trials, and the clustering's mean equals the second trial's.

### EqualBoundsMakeOneTrial

With `min_per_head` and `max_per_head` both 2 and three heads, the choice is two per head, six
anchors and one trial.

### TwoSizesTakeTheSecondAnchorAtTheDefaultGain

Two sizes with one head and `max_per_head` 2 give two per head. The first trial's mean is
11.625 / 21 and the second's is 1, both to 1e-6.

### TwoSizesKeepOneAnchorWhenTheGainIsBelowMinGain

With `min_gain` 0.5 the same boxes give one anchor per head and one anchor whose mean is
11.625 / 21. Both trials are returned, and the second's mean is 1.

### IgnoresTheCountOfTheClusteringOptions

With `clustering.count` set to 17, the two sizes still give two per head and two anchors.

### RefusesAMinPerHeadAboveMaxPerHead

`min_per_head` 3 with `max_per_head` 2 fails with `InvalidDescription`.

### RefusesOptionsOutsideTheirRange

A `min_per_head` of 0, a `min_gain` of -0.01, a NaN `min_gain` and a clustering
`max_iterations` of 0 each fail with `InvalidDescription`.

### RefusesAHeadCountOutsideItsRange

A head count of 0, of -1 and of the largest `int64_t` each fail with `InvalidDescription`.

### RefusesFewerDistinctBoxesThanTheFirstCount

Two sizes with three heads fail with `Incompatible`.

### RefusesAnEmptyBoxList

An empty box list fails with `Incompatible`.

## anchor/test_anchor_fit.cpp

Suites: `Unit_DetectAnchorFit`.

Tests `measure_anchor_fit`, which measures how well the anchors of an `AnchorLayout` fit a list
of box sizes. Every number is compared to 1e-9.

### OneBoxPerAnchorGivesOneEverywhere

One box at each of three anchors' sizes gives 3 boxes, a mean best IoU, a best possible recall
and a tenth percentile of 1, an unowned share of 0 and a share of 1/3 per anchor.

### HalfSizedBoxesGiveAQuarterAndAreRecalledAtAQuarter

Boxes at half each anchor's sides give a mean best IoU and a tenth percentile of 0.25. The best
possible recall is 1 with `recall_iou` 0.25 and 0 with `recall_iou` 0.3.

### V4TinyMasksLeaveASixthOfTheBoxesUnowned

For a six-anchor layout whose two heads list five of the anchors, 100 boxes at each anchor's
size give 600 boxes, an unowned share of 1/6, a best possible recall of 5/6, a mean best IoU of
1 and a share of 1/6 per anchor.

### MasksOverEveryAnchorLeaveNoBoxUnowned

With the heads listing all six anchors, the same boxes give an unowned share of 0 and a best
possible recall of 1.

### FourNumbersEqualTheBoxByBoxSums

For four anchors, of which no head lists anchor 0, and twelve boxes, the case finds each box's
best anchor with `shape_iou` and sums by hand. The mean best IoU, the recall at 0.5, the unowned
share, the tenth percentile, which is the second smallest best IoU, and the four anchor shares
equal those sums. The case also asserts that some boxes are unowned, some recalled and some
neither.

### TenthPercentileOfTenBoxesIsTheSmallest

Nine boxes at the anchor's size and one at half its sides give a tenth percentile of 0.25 and a
mean best IoU of 0.925.

### EqualAnchorsGiveTheBoxToTheLowerIndex

With two equal anchors, of which the one head lists anchor 1, one box at that size gives the
shares 1 and 0, an unowned share of 1 and a best possible recall of 0.

### RefusesAnEmptyBoxList

An empty box list fails with `Incompatible`.

### RefusesALayoutThatDoesNotValidate

A layout whose head names an anchor past the list fails with `InvalidDescription`.

### RefusesARecallIoUOutsideZeroToOne

A `recall_iou` of -0.1, of 1.5 and of NaN each fail with `InvalidDescription`.

## anchor/test_anchor_layout.cpp

Suites: `Unit_DetectAnchorLayout`.

Tests `shape_iou` over two `AnchorSize` values and `validate_anchor_layout` over a layout of
six anchors split between two heads.

### ShapeIoUOfASizeWithItselfIsOne

`shape_iou` of 0.3 x 0.7 with itself is 1 to 1e-6.

### ShapeIoUOfAContainedSizeIsTheRatioOfTheAreas

`shape_iou` of 0.2 x 0.4 and 0.1 x 0.2 is 0.25 in either order.

### ShapeIoUOfCrossedSizesCountsTheSharedRectangle

`shape_iou` of 0.4 x 0.1 and 0.1 x 0.4 is 1/7.

### ShapeIoUWithAZeroAreaIsZero

`shape_iou` is exactly 0 when either size has a zero side, and for two default sizes.

### AcceptsOneHeadPerGroupOfAnchors

`validate_anchor_layout` is ok for the two-head layout.

### AcceptsASharedAnchorAndAnAnchorNoHeadUses

With the masks {3, 4, 5} and {1, 2, 3}, where anchor 3 is shared and anchor 0 unused, the
layout is accepted.

### RefusesAnIndexPastTheAnchors

A mask entry of 6 with six anchors fails with `InvalidDescription`.

### RefusesANegativeIndex

A mask entry of -1 fails with `InvalidDescription`.

### RefusesAnEmptyHead

A head with an empty mask fails with `InvalidDescription`.

### RefusesALayoutWithoutAnchorsOrWithoutHeads

A layout with no anchor and a layout with no head each fail with `InvalidDescription`.

### RefusesAnAnchorWithoutAFinitePositiveSide

An anchor with a zero width, one with a NaN height and one with an infinite width each fail
with `InvalidDescription`.

## anchor/test_anchor_plan.cpp

Suites: `Unit_DetectAnchorPlan`.

Tests `plan_anchors`, which chooses or takes a count per head, clusters the boxes, assigns the
anchors to the heads by stride, measures the fit and derives the assignment thresholds. The
file uses a ladder of six sizes with 400 boxes per rung, one blob of 2000 boxes and a set of
two sizes.

### LadderWithTwoStridesPlansTwoHeadsOfThreeAnchors

The ladder under the strides 16 and 32 plans six anchors with the masks {0, 1, 2} and
{3, 4, 5}. The fit counts every box, has a best possible recall of 1, an unowned share of 0 and
a mean best IoU above 0.95. The thresholds equal `derive_assignment_thresholds` of the fit: an
ignore threshold of 0.7 and a match threshold at the fit's tenth percentile. There are four
trials, the last for 4 per head.

### ChosenCountEqualsTheStepsMadeByHand

The plan equals, to 1e-9, a plan the case builds from `choose_anchor_count`,
`assign_anchors_to_heads`, `measure_anchor_fit` and `derive_assignment_thresholds` called in
that order.

### GivenCountTriesNoCount

With `per_head` 3 the plan has no trial, and its layout is the one `cluster_anchors` to six
anchors and `assign_anchors_to_heads` give. Each head has three anchors and the best possible
recall is 1.

### GivenCountIgnoresTheBoundsOfTheCountChoice

With `per_head` 3, a plan made with a `min_per_head` of 0, a `max_per_head` of -1, a NaN
`min_gain` and a clustering count of 17 is identical to one made with the default count
options.

### StridesInAnyOrderGiveEachHeadTheAnchorsOfItsStride

Under the strides 32 and 16 the masks are {3, 4, 5} and {0, 1, 2}, and with the two masks
swapped the plan is identical to the plan for 16 and 32. Under 16, 64 and 32 with two per head,
the masks are {0, 1}, {4, 5} and {2, 3}.

### TwoCallsGiveTheSamePlan

Two calls with the seed 0x5EED and the same arguments return identical plans.

### BlobPlansOneAnchorPerHead

The blob under the strides 8, 16 and 32 plans the masks {0}, {1} and {2}, with two trials and a
best possible recall of 1.

### RefusesNoStrideAndAStrideOfZero

An empty stride list and the strides 16 and 0 each fail with `InvalidDescription`, with a
chosen count and with `per_head` 1.

### RefusesOptionsOutsideTheirRange

Six option sets each fail with `InvalidDescription`: a `per_head` of -1 and of the largest
`int64_t`, a `min_per_head` of 0, a clustering `max_iterations` of 0 with and without a given
`per_head`, and a `recall_iou` of 2.

### RefusesBoxesThatDisagreeWithTheCount

An empty box list, with a chosen and with a given count, fails with `Incompatible`. So do two
sizes under three strides with a chosen count, and two sizes under two strides with two per
head.

## anchor/test_anchor_review.cpp

Suites: `Unit_DetectAnchorReview`.

Tests `review_anchors`, which measures a current `AnchorLayout` on a box list, clusters a
proposal with the same counts per head and returns `AnchorVerdict::Keep` or
`AnchorVerdict::Propose`. The boxes are a ladder of six sizes with 400 boxes per rung, and the
shifted ladder is the same one scaled by 0.75.

### FittedLayoutIsKeptOnItsOwnBoxes

A layout fitted to the ladder and reviewed on the ladder gets `Keep`. Its best possible recall
is 1, and the proposal's mean best IoU equals the current one to 1e-12.

### ShiftedBoxesGetAProposalWithBothFits

Reviewed on the shifted ladder, the layout gets `Propose`. The reported current fit and
proposed fit equal `measure_anchor_fit` called directly on each layout, and their mean best IoU
and recall equal sums the case makes by hand to 1e-6. The proposal's gain in mean best IoU is
above `min_gain`.

### ProposalRecoversTheShiftedRungs

The proposal has six anchors, and each lies within 5 % of the width and the height of its
shifted rung.

### ProposalKeepsTheCountsPerHead

For a current layout with 2 and 4 anchors under the strides 32 and 16, the proposal has the
same anchor count and the masks {4, 5} and {0, 1, 2, 3}.

### ProposalIsTheClusteringAssignedToTheHeads

With `clustering.count` set to 99 in the options, the proposal still equals the layout fitted
directly to the shifted boxes with three anchors per head.

### InputLayoutIsUnchanged

After a review that returns `Propose`, the reviewed layout holds the anchors and masks it held
before.

### RecallAndGainEachDecideTheVerdict

With `min_gain` 10 the shifted review has a current recall of 1 and returns `Keep`. With
`recall_iou` raised to 0.6 as well, the current recall is 0, which equals the hand sum, and the
verdict is `Propose`.

### RefusesOptionsOutsideTheirRange

A `min_gain` of -0.01 and of NaN, a `min_recall` of 1.5 and of -0.1, and a `recall_iou` of 2
each fail with `InvalidDescription`.

### RefusesALayoutWithoutHeads

A layout with one anchor and no head fails with `InvalidDescription`.

### RefusesBoxesStridesAndCountsThatDisagree

Three sizes with a layout of masks {0} and {1, 2} under two strides are accepted. An empty box
list, four copies of one box, one stride for two heads, and a layout whose masks {0, 1} and
{1, 2} hold four slots for three anchors each fail with `Incompatible`.

## anchor/test_anchor_task.cpp

Suites: `Unit_DetectAnchorTask`.

Tests `read_anchor_task` and `write_anchor_task` in
`modules/vision/detect/include/SushiAI/vision/detect/anchor/anchor_task.hpp`. A task is
compared field by field, every float bit for bit. A section is compared by the JSON text
`write_json` gives it. The refusal cases start from a valid section of three classes, four
anchors, two heads of two anchors each and the criterion "iou", and replace one member.

### AThreeHeadTaskRoundTripsThroughItsSection

A task of 80 classes, nine anchors and three heads, with `DistanceIoU` and an exponent of
0.6f, comes back unchanged after `write_anchor_task`, `write_json`, `parse_json` and
`read_anchor_task`.

### ATwoHeadTaskRoundTripsThroughItsSection

A task of 3 classes, six anchors and two heads under the plain IoU criterion comes back
unchanged after the same four steps.

### AThreeHeadSectionRoundTripsThroughItsTask

A section of three heads over nine anchors with the criterion "distance_iou" is written back
with the same JSON text after `read_anchor_task` and `write_anchor_task`.

### ATwoHeadSectionRoundTripsThroughItsTask

A section of two heads over six anchors with the criterion "iou" is written back with the
same JSON text.

### ReadsEveryFieldOfASection

The three-head section reads as 80 classes, its nine anchors, the masks {6, 7, 8}, {3, 4, 5},
{0, 1, 2}, and the heads head_0 and head_1 with `Logit` and the scales 1.25 and 1.125 and
head_2 with `Activated` and 2.0. The criterion is `DistanceIoU` with the exponent 0.5.

### WritesTheTypeTheKeysInOrderAndEachFloatAsItsDouble

The written section has the keys `type`, `classes`, `anchors`, `heads`, `suppression` in that
order, the type `K_ANCHOR_TASK_TYPE` and 3 classes. A head has the keys `output`, `anchors`,
`coding`, `scale_xy`. The second head's scale equals `double(1.05f)` and the third anchor's
sides equal `double(0.09f)` and `double(0.14f)`.

### ReadsAndWritesEachCodingString

"logit" reads as `HeadCoding::Logit` and "activated" as `HeadCoding::Activated`, and each is
written back as the same string.

### ReadsAndWritesEachCriterionString

"iou" reads as `SuppressionCriterion::IoU` and "distance_iou", with an exponent of 0.75, as
`SuppressionCriterion::DistanceIoU`. Each is written back as the same string.

### TheExponentBelongsToTheCriterionThatReadsIt

A task under IoU with the exponent 0.25 writes a `suppression` object with the one key
`criterion`, and that section reads back with `K_DEFAULT_DISTANCE_EXPONENT`. Under
`DistanceIoU` the object has the keys `criterion`, `distance_exponent` and the exponent 0.25.

### ReadsAWholeNumberAsAScale

A `scale_xy` of 2 reads as 2.0f.

### AcceptsAnAnchorTwoHeadsShare

Heads with the anchors [1, 2] and [0, 1] read with those two masks.

### RefusesASectionThatIsNotAnObject

`read_anchor_task` fails with `InvalidDescription` and "task" for an array ("an array") and
for a null value ("null").

### RefusesAnotherTypeOrNone

The read fails with `InvalidDescription` for the type "anchor_free_detection" ("'type'",
"'anchor_free_detection'", "'anchor_detection'"), for the type 3 ("'type'",
"a whole number") and for a section without a type ("task has no 'type'").

### RefusesAnUnknownKeyAtEachLevel

The read fails with `InvalidDescription` for the key `clases` in the section ("'clases'",
"'classes'"), the key `scale` in a head ("'heads' entry 0", "'scale'", "'scale_xy'") and the
key `threshold` in `suppression` ("'suppression'", "'threshold'").

### RefusesASectionWithoutOneOfItsMembers

A section with all five members reads. Without `classes`, `anchors`, `heads` or `suppression`
it fails with `InvalidDescription` and "task has no" followed by the quoted key.

### RefusesAClassCountBelowOneOrOfAnotherKind

`classes` 0 fails with `InvalidDescription`, "'classes'", "at least 1" and "0". `classes` 3.5
fails with "'classes'" and "a whole number".

### RefusesAnAnchorThatIsNotAPairOfNumbers

The read fails with `InvalidDescription` for `anchors` given as a number ("'anchors'",
"an array"), an anchor of three values ("'anchors' entry 1", "[width, height]", "3 values"),
an anchor given as a number ("'anchors' entry 2", "[width, height]") and a height given as a
string ("'anchors' entry 0 height", "a string").

### RefusesAnAnchorSideThatIsNotPositiveAndFinite

The read fails with `InvalidDescription` for a width of 0.0 ("'anchors' entry 2",
"above zero"), a height of -0.5 ("'anchors' entry 1", "above zero"), a width of 1e-60
("'anchors' entry 0", "above zero"), a height of 1e60 ("'anchors' entry 3 height", "float32")
and an empty list ("'anchors'", "no anchor").

### RefusesNoHeadAndAHeadThatIsNotAnObject

The read fails with `InvalidDescription` for an empty `heads` ("'heads'", "no head"), for
`heads` given as an object ("'heads'", "an object") and for a second head given as the number
7 ("'heads' entry 1", "a whole number").

### RefusesAMaskIndexThatNamesNoAnchor

Over four anchors the read fails with `InvalidDescription` for the index 4
("'heads' entry 1 'anchors' entry 2", "anchor 4", "4 anchors"), the index -1
("'heads' entry 0 'anchors' entry 0", "anchor -1"), the index 1.5
("'heads' entry 0 'anchors' entry 1", "a whole number") and an empty mask
("'heads' entry 0 'anchors'", "no anchor").

### RefusesAHeadWithABadOutputCodingOrScale

The read fails with `InvalidDescription` for an empty `output` ("'heads' entry 0 'output'",
"empty string"), a head without `output` ("'heads' entry 0", "no 'output'"), the coding
"linear" ("'heads' entry 1 'coding'", "'linear'", "'logit', 'activated'"), a `scale_xy` of 0.0
("'heads' entry 0 'scale_xy'", "above zero") and a `scale_xy` given as a string
("'heads' entry 0 'scale_xy'", "a string").

### RefusesABadCriterionOrExponent

The read fails with `InvalidDescription` for the criterion "giou"
("'suppression' 'criterion'", "'giou'", "'iou', 'distance_iou'"), an exponent stated under
"iou" ("'suppression' 'distance_exponent'", "'iou'", "does not read"), "distance_iou" without
an exponent ("'suppression'", "no 'distance_exponent'") and `suppression` given as an array
("'suppression'", "an array").

### WriterRefusesAHeadOfAnotherClassCount

A task of 3 classes whose second head has 4 fails in `write_anchor_task` with
`InvalidDescription`, "'heads' entry 1", "4 classes", "'classes'" and "3".

### WriterRefusesTheValuesTheReaderRefuses

`write_anchor_task` fails with `InvalidDescription` for 0 classes ("'classes'",
"at least 1"), an anchor height of NaN ("'anchors' entry 4", "above zero"), a mask index of 6
over six anchors ("'heads' entry 0 'anchors' entry 1", "anchor 6", "6 anchors"), no head
("'heads'", "no head"), one head for two masks ("'heads'", "1 heads", "2 masks"), an empty
output ("'heads' entry 0 'output'", "empty string") and an infinite scale
("'heads' entry 1 'scale_xy'", "finite").

### WriterRefusesAnEnumeratorWithoutAString

`write_anchor_task` fails with `InvalidDescription` for a `HeadCoding` of 7
("'heads' entry 1 'coding'", "7", "'logit', 'activated'") and for a `SuppressionCriterion` of
9 ("'suppression' 'criterion'", "9", "'iou', 'distance_iou'").

### WriterRefusesAnExponentThatIsNotFiniteUnderDistanceIoU

With an infinite exponent the task writes under IoU. Under `DistanceIoU` the write fails with
`InvalidDescription`, "'suppression' 'distance_exponent'" and "finite".

## anchor/test_head_assignment.cpp

Suites: `Unit_DetectAnchorHeadAssignment`.

Tests `assign_anchors_to_heads`, which gives each head, in the order of its stride, a run of
anchor indices in ascending area and returns an `AnchorLayout`.

### NineAnchorsOverThreeHeadsGiveThreeRunsOfThree

Nine anchors over the strides 32, 16 and 8 give the masks {6, 7, 8}, {3, 4, 5} and {0, 1, 2}.

### SixAnchorsOverTwoHeadsGiveTwoRunsOfThree

Six anchors of rising area over the strides 32 and 16 give {3, 4, 5} and {0, 1, 2}.

### ReproducesTheMasksOfFiveCfgs

For five named cases the file holds, each an anchor list in pixels, an input side, the strides
and the expected masks, the computed masks equal the expected ones.

### DiffersFromTheSecondMaskOfYolov4TinyByOneAnchor

For six rising anchors over the strides 32 and 16, the first computed mask is {3, 4, 5}. The
second is {0, 1, 2}, which is not {1, 2, 3}.

### TheRemainderGoesToTheHeadOfSmallestStride

Seven anchors over the strides 32 and 16 give {4, 5, 6} and {0, 1, 2, 3}. Over the strides 16
and 32 they give {0, 1, 2, 3} and {4, 5, 6}.

### ARemainderOfTwoGoesToTheTwoSmallestStrides

Eleven anchors over the strides 16, 32 and 8 give {4, 5, 6, 7}, {8, 9, 10} and {0, 1, 2, 3}.

### UnsortedAnchorsKeepTheirOrderAndAreNamedByArea

Four anchors whose ascending area is 3, 1, 2, 0 give the masks {2, 0} and {3, 1} over the
strides 32 and 16. The layout's anchors are the inputs in their given order, compared exactly.

### AnchorsOfEqualAreaGoInIndexOrder

Two anchors of one area over the strides 8 and 16 give {0} and {1}.

### TheResultIsAValidLayout

`validate_anchor_layout` is ok for the layout of a nine-anchor, three-head assignment.

### PerHeadSetsTheCountOfEachHead

With `per_head` {2, 4} over the strides 32 and 16, six anchors give {4, 5} and {0, 1, 2, 3}.

### RefusesAPerHeadThatDoesNotSumToTheCount

For six anchors, `per_head` {2, 3}, {3, 4} and two counts of the largest `int64_t` each fail
with `Incompatible`.

### RefusesAPerHeadOfAnotherLengthThanTheStrides

For two strides, `per_head` {6} and {2, 2, 2} each fail with `Incompatible`.

### RefusesAPerHeadWithAnEmptyHead

`per_head` {6, 0} and {7, -1} each fail with `InvalidDescription`.

### RefusesFewerAnchorsThanHeads

Two anchors over three strides fail with `Incompatible`.

### RefusesMissingAnchorsAndBadStrides

No anchor, no stride, a stride of 0 and a stride of -8 each fail with `InvalidDescription`.

### RefusesAnAnchorWithoutAFinitePositiveSide

An anchor with a zero height and an anchor with a NaN width each fail with
`InvalidDescription`.

## anchor/test_head_truth.cpp

Suites: `Unit_DetectAnchorHeadTruth`.

Tests `validate_head_truth` over a batch of one image with two truth rows, three classes and a
64 by 64 network. The row under test is row 1, given by its pixel corners and its class. Every
refusal has the code `InvalidDescription`.

### AcceptsRowsWhoseCentreLiesInsideTheInput

A row with the corners -8 and 8 on both axes, whose centre is exactly 0, and the class 2 is
accepted.

### RefusesACentreAtTheRightEdge

A centre of exactly 1 along the columns is refused; the message holds "truth row 1 of image 0",
"centre" and "(1, 0.5)".

### RefusesACentreLeftOfTheInput

A centre of -0.25 along the columns is refused, and the message holds "(-0.25, 0.5)".

### RefusesACentreBelowTheInput

A centre of 1.25 along the rows is refused, and the message holds "(0.5, 1.25)".

### RefusesACentreAboveTheInput

A centre of -0.25 along the rows is refused, and the message holds "(0.5, -0.25)".

### RefusesACentreThatIsNotANumber

A NaN corner is refused with a message holding "truth row 1 of image 0" and "centre".

### RefusesAWidthThatIsNotPositive

A row of zero width is refused, and the message holds "size 0 by 0.5".

### RefusesAHeightThatIsNotPositive

A row whose bottom lies above its top is refused, and the message holds "size 0.5 by -0.25".

### RefusesAClassAtTheClassCount

The class 3 with three classes is refused, and the message holds "class 3" and "3 classes".

### RefusesANegativeClass

The class -1 is refused, and the message holds "class -1".

### LeavesAPaddingRowUnread

A row with an out-of-range centre and class is accepted when its `valid` flag is 0.

### RefusesArraysThatDoNotHoldTheirShape

A batch count of 2 against a truth of one image is refused with "2 images" and "batch of 1". A
truth one box value short is refused with "8 box values" and "7".

### RefusesANetworkSideBelowOne

A network size of 64 by 0 is refused with "network size" and "64 by 0".

### TheDeltaRefusesATruthOutsideTheGrid

`compute_head_delta` given the row with a centre of exactly 1 fails with the same message
parts, and leaves `delta` and `assigned_row` of its result empty.

## free/test_anchors.cpp

Suites: `Unit_DetectAnchors`, `Unit_DetectAnchorLeaves`, `Unit_DetectLevelRows`.

Tests the anchor grid, the per-image anchor leaves and the copies between a level and the joined
anchor axis.

This file needs no device. Its three suites, `Unit_DetectAnchors`, `Unit_DetectAnchorLeaves` and
`Unit_DetectLevelRows`, cover `modules/vision/detect/include/SushiAI/vision/detect/anchors.hpp`,
`modules/vision/detect/include/SushiAI/vision/detect/anchor_leaves.hpp` and `modules/vision/detect/include/SushiAI/vision/detect/level_rows.hpp`.
`modules/vision/detect/README.md` quotes the reference lines each unit ports. One case reads
`loss.sacp` through `tests/common/reference_fixture.hpp` and reports itself skipped when the
file is absent.

### K_EXACT

`K_EXACT` is the tolerance 0. Every expected value in the file is a multiple of 0.5, which
float32 holds exactly.

### K_REFERENCE_LEVELS

`K_REFERENCE_LEVELS` holds the three levels `tests/fixtures/yolo26/MANIFEST.json` records for a
640 px input: 80 by 80 at stride 8, 40 by 40 at stride 16 and 20 by 20 at stride 32.

### K_SMALL_LEVELS

`K_SMALL_LEVELS` is a grid of 2 by 3 cells at stride 8 followed by 1 by 2 cells at stride 16,
eight anchors in all, so a case can write every expected value out.

### CountsOneAnchorPerCell

`count_anchors` returns 8 for `K_SMALL_LEVELS` and 8400 for `K_REFERENCE_LEVELS`.

### OrdersPointsByLevelThenRowThenColumn

`make_anchors` over `K_SMALL_LEVELS` writes the sixteen point values and the eight strides the
case lists: the first level row by row, x before y, then the second level.

### ScalesEachPointByItsStride

`scale_points` multiplies both coordinates of a point by its anchor's stride. The centres 0.5,
1.5 and 2.5 become 4, 12 and 20 on the first level, and 0.5 and 1.5 become 8 and 24 on the
second.

### ScaleRejectsStridesThatDoNotMatchThePoints

Seven strides for eight points throw with a message containing "[A] strides" and "got 16, 7 and
16 values".

### RejectsALevelWithoutCellsOrStride

`count_anchors` throws on a level of height 0 with a message containing "height 0" and
"width 3", and on a level of stride 0 with a message containing "stride 0".

### RejectsOutputsOfTheWrongSize

`make_anchors` given a point array of 15 values throws with a message containing "16 point
values" and "got 15".

### MatchesTheReferenceAnchorPoints

The points of `K_REFERENCE_LEVELS`, scaled to pixels, equal the tensor `anchors.points` of
`loss.sacp` at tolerance 0, and that tensor has shape [8400, 2]. The case is skipped when
`loss.sacp` is absent.

### GivesEachLevelItsStrideOverItsAnchorRange

The strides of `K_REFERENCE_LEVELS` are 8 over anchors 0 to 6399, 16 over 6400 to 7999 and 32
over 8000 to 8399. The offsets 0, 6400 and 8000 are the `anchor_offset` entries of the manifest.

### RepeatsThePointsAndTheStridesForEveryImage

`build_anchor_leaves` for a batch of two writes the eight points of `K_SMALL_LEVELS` twice. It
writes the strides with four values per anchor: 24 values of 8 and 8 values of 16, also twice.

### SplitsIntoALevelsPointsAndStrides

`split_level` cuts the second level, anchors 6 and 7, out of both leaves. The points come out as
[2, 1, 2, 2] and the strides as sixteen values of 16.

### RejectsABatchWithoutImagesAndAModelWithoutLevels

A batch of 0 throws with a message containing "positive batch" and "batch 0 and 2 levels". An
empty level list throws with a message containing "at least one level" and "batch 1 and 0
levels".

### JoinPlacesALevelAtItsAnchorOffset

`join_level` with two images, five anchors of two values and a level at anchors 3 and 4 writes
the level's rows into values 6 to 9 of each image and leaves the other values at 0.

### SplitReturnsTheRowsJoinPlaced

The joined array holds its own index in every value. `split_level` for anchors 1 to 3 returns
values 2 to 7 of image 0 and 12 to 17 of image 1.

### RejectsALevelPastTheLastAnchor

A level of two anchors at offset 4 of five throws with a message containing "2 anchors",
"offset 4" and "5 anchors".

### SplitRejectsALevelArrayOfTheWrongSize

`split_level` into an array of 10 values, where the placement needs 12, throws with a message
containing "12 level values", "20 joined values" and "got 10 and 20".

## free/test_assigner.cpp

Suites: `Unit_DetectAssigner`, `Unit_DetectGroundTruth`, `Unit_DetectTargetLeaves`.

Tests the task-aligned assigner, its presets, the padded ground truth and the target leaves.

This file needs no device. Its three suites are `Unit_DetectAssigner`, `Unit_DetectGroundTruth`
and `Unit_DetectTargetLeaves`, over `modules/vision/detect/include/SushiAI/vision/detect/assigner.hpp`,
`modules/vision/detect/include/SushiAI/vision/detect/assigner_config.hpp`, `modules/vision/detect/include/SushiAI/vision/detect/ground_truth.hpp` and
`modules/vision/detect/include/SushiAI/vision/detect/target_leaves.hpp`. Most cases build one image on a 4 by 4 grid by
hand, with `alpha` and `beta` both 1, so the metric is the class score times the overlap and
each expected score is a fraction a comment derives. Three cases read `loss.sacp` and report
themselves skipped when the file is absent. `modules/vision/detect/README.md`, under
"Reference tensors the assigner tests read", lists the tensors.

### Tolerances

`K_EXACT` is 0, for copied values and multiples of 0.5. `K_SCORE_TOLERANCE` is 1e-6, eight
float32 steps at 1, for a target score computed by hand. `K_REFERENCE_SCORE_TOLERANCE` is 1e-5
for a reference target score, where `std::pow` and `std::atan` may round apart from torch.
`K_REFERENCE_BOX_TOLERANCE` is 1.3e-4, two float32 steps at 640, for a reference box in pixels.

### The hand-built grid

`K_GRID_SIDE` is 4 cells, `K_GRID_ANCHORS` is 16, `K_GRID_STRIDE` is 8 px and `K_CLASSES` is 2.
`K_FAR_BOX` is the box (100, 100, 101, 101), which no hand-built ground truth touches.

### Scene

`Scene` holds one image on that grid: anchor `a` sits in column `a % 4` and row `a / 4`, with
pixel centres 4, 12, 20 and 28. Its constructor gives every anchor one score for both classes
and the prediction `K_FAR_BOX`. `add_truth` appends a valid ground-truth box, `predict` sets one
anchor's predicted box, and `inputs` returns the scene as an `AssignerInputs`.

### make_hand_config

`make_hand_config` returns an `AssignerConfig` with the given `topk` and `topk2`, two classes,
`alpha` 1, `beta` 1 and `stride_val` 8.

### foreground_anchors

`foreground_anchors` returns the rows an `Assignment` marks as foreground, in ascending order.

### make_crowded_scene

`make_crowded_scene` returns a scene with one box of class 1 over the whole 32 px image. Every
anchor predicts that box and scores 0.1, except anchors 5, 10 and 2, which score 0.9, 0.6 and
0.3 in class 1.

### make_contested_scene

`make_contested_scene` returns a scene with two boxes, of classes 0 and 1, that both contain
anchor 5. Anchors 0, 1 and 4 predict the first box, anchors 6, 9 and 10 the second, and anchor 5
predicts the box the caller passes.

### box_row

`box_row` returns the four values of one row of a `[rows, 4]` array.

### make_idle_assignment

`make_idle_assignment` returns an `Assignment` of the given batch and anchor count with no
foreground and every array at zero.

### set_foreground

`set_foreground` marks one row of an `Assignment` as foreground and gives it a pixel box and two
class scores.

### Reference constants

`K_REFERENCE_LEVELS` holds the manifest's three levels for a 640 px input,
`K_REFERENCE_IMAGE_SIDE` is 640, and `K_SMALL_BOX_ROW` is 3, the ground-truth row of the
fixture that is smaller than a stride.

### pad_reference_truth

`pad_reference_truth` reads `batch.batch_idx`, `batch.cls` and `batch.bboxes` from `loss.sacp`
and pads them with `pad_ground_truth` for one 640 px image. It returns an empty string, or the
reason a tensor could not be read.

### count_holders

`count_holders` counts the foreground anchors of a mask whose target index is a given
ground-truth row. It is a template over the mask's element type, so it takes the assigner's
mask and the reference's.

### expect_reference_assignment

`expect_reference_assignment` runs one branch's assigner on the reference's scores, boxes and
anchor points and compares all five outputs: the mask, the indices and the labels for equality,
the boxes within `K_REFERENCE_BOX_TOLERANCE` and the scores within
`K_REFERENCE_SCORE_TOLERANCE`. It then compares the foreground count and the count of anchors
that hold box 3 with the two numbers the caller passes, and the second with the same count
taken from the reference. Equality of the mask and the indices relies on the fixture having no
tied metrics. The helper skips when `loss.sacp` is absent.

### OneToManyPresetMatchesTheReferenceSettings

`one_to_many_assigner_config(80, {8, 16, 32})` has `topk` 10, `topk2` 10, 80 classes, `alpha`
0.5, `beta` 6, `stride_val` 16 and `eps` 1e-9. These are the `assigner.one2many` values of
`loss.sacp` in `tests/fixtures/yolo26/MANIFEST.json`.

### OneToOnePresetMatchesTheReferenceSettings

`one_to_one_assigner_config(80, {8, 16, 32})` has `topk` 7, `topk2` 1, `alpha` 0.5, `beta` 6 and
`stride_val` 16, the `assigner.one2one` values of the manifest.

### PresetTakesStrideValFromTheSecondLevel

`stride_val` is 12 for the strides 4 and 12, 4 for the one stride 4, and 16 for an empty stride
list.

### ConfigDefaultsMatchTheReferenceConstructor

A default `AssignerConfig` has `topk` 13, `topk2` 13, 80 classes, `alpha` 1, `beta` 6,
`stride_val` 16 and `eps` 1e-9, the arguments of the reference's `TaskAlignedAssigner.__init__`.

### RejectsAConfigWithoutCandidatesOrWithANegativeMargin

Constructing a `TaskAlignedAssigner` throws for `topk` 0, with a message containing "positive
topk" and "got 0, 3 and 2", for `stride_val` -1, with "stride_val", "at least 0" and "-1", and
for `eps` -1, with "eps of at least 0" and "-1".

### TheMetricRaisesTheScoreToAlphaAndTheOverlapToBeta

The case uses the one-to-many preset, so `alpha` is 0.5 and `beta` is 6. Anchor 5 predicts the
box exactly with score 0.25, so its metric is sqrt(0.25) times 1, which is 0.5. Anchor 10
predicts a 24 px box with score 0.81; its overlap is 576 / 1024 - 32 / 2048, which is 35 / 64,
and its metric is 0.9 times (35 / 64)^6, about 0.024. With `topk` 1 the foreground is anchor 5
alone, and with `topk` 2 it is anchors 5 and 10. The target score of anchor 5 is 1 and that of
anchor 10 is its metric over 0.5. An assigner that used the scores or the overlaps unraised
gives another second score.

### NormalisesByTheLargestOverlapOfTheBox

Anchors 0 and 1 both predict a box offset by 8 px from a 16 px truth, with scores 0.8 and 0.4.
The overlap is 64 / 448 - 128 / 1152, which is 2 / 63. The target scores are that overlap and
half of it: each metric over the largest metric, times the largest overlap of the box.

### APredictionApartFromItsBoxHasNoOverlap

Anchors 0 and 1 score 0.9 and predict `K_FAR_BOX`, whose complete IoU with the box is negative
and is clamped to 0. With `topk` 2 the foreground is anchors 4 and 5, which predict the box.
With `topk` 3 the third place goes to anchor 0, the first candidate of metric zero, and its
target score is 0.

### AContestedAnchorOfEqualOverlapsGoesToTheFirstBox

Two ground-truth rows hold the same box with classes 0 and 1, and anchor 5 predicts it. Anchor 5
is the one foreground anchor, its target index and label are 0, its score in class 0 is 1 and
its score in class 1 is 0.

### AContestedAnchorCanGoToABoxThatDidNotKeepIt

With `topk` 1, boxes 0 and 1 both keep anchor 5 and box 2 keeps anchor 2. Anchor 5 predicts box
2 exactly, so its largest overlap is with box 2 and it goes there, with target index 2 and
label 1. Its score in class 1 is 0.1 / 0.9 and its score in class 0 is 0. The case pins that
the maximum overlap is taken over every box, including one that did not keep the anchor.

### AnImageWithoutGroundTruthGetsNoForeground

A scene with no ground truth has no foreground anchor. Every target label is 2, the class
count, and every index, box and score is 0.

### APaddingRowAssignsNothing

The crowded scene's one box is marked invalid. No anchor is foreground and every target score
is 0, although the box contains all sixteen anchors.

### KeepsTheTopKCandidatesWhenABoxHoldsMore

In the crowded scene every overlap is 1, so the metric is the class score. With `topk` 3 the
foreground is anchors 2, 5 and 10, with target scores 0.3 / 0.9, 1 and 0.6 / 0.9 in class 1 and
0 in class 0. Every target label is 1, the label of ground-truth row 0, on and off the
foreground.

### TheSecondTopKKeepsTheBestAnchorOfABox

With `topk` 3 and `topk2` 1 the crowded scene keeps anchor 5 alone, with target score 1.

### AContestedAnchorGoesToTheHigherOverlap

Anchor 5 predicts the second box of the contested scene. The foreground is anchors 0, 1, 4, 5,
6, 9 and 10, and the target index and label of anchor 5 are 1. Its score is 1 in class 1 and 0
in class 0, and its target box starts at x = 8.

### AContestedAnchorCanGoToTheFirstBox

The same scene with anchor 5 predicting the first box gives anchor 5 the target index 0, score
1 in class 0 and 0 in class 1.

### ASmallBoxIsEnlargedToStrideVal

A 2 px box centred at 14 contains no anchor centre. With `stride_val` 16 it is tested as the
span 6 to 22, which holds the centres 12 and 20 on both axes, so anchors 5, 6, 9 and 10 are
foreground. With `stride_val` 1 there is no foreground. The target box of anchor 5 still starts
at 13: the enlargement applies to the candidate test and not to the target.

### AssignsEachImageFromItsOwnRows

A batch of two holds an image with one invalid row and the crowded scene. The foreground rows
are 18, 21 and 26, which are anchors 2, 5 and 10 of image 1. The target label of row 0 is 0 and
that of row 16 is 1.

### RejectsTooFewAnchorsAndAnUnknownClass

`assign` with `topk` 17 on sixteen anchors throws with a message containing "16 anchors" and
"topk 17". A valid box of class 2 with two classes throws with "class 2 of 2".

### PadsLabelRowsPerImage

Three label rows, two for image 0 and one for image 2, are padded to a batch of three 64 by 32
images with `max_boxes` 2. The boxes are converted from normalised centre and size to pixel
corners, the labels are 3, 1, 0, 0, 7, 0 and the valid flags are 1, 1, 0, 0, 1, 0.

### MarksARowWithoutExtentInvalid

A row whose four values are 0 is padded in with its label 4 and the valid flag 0. The row after
it is valid.

### PadsNoRowsToABatchWithoutBoxes

With no label rows and a batch of 2, `max_boxes` is 0 and the labels, boxes and flags are empty.

### RejectsARowOutsideTheBatch

A row naming image 3 in a batch of 3 throws with a message containing "row 0 names image 3" and
"batch of 3".

### BuildsTheLossLeavesOfAnAssignment

The crowded scene is assigned with `topk` 3 and its leaves are built for an image 64 px wide and
32 px high. The three weights are 1, 2/3 and 1/3, so `target_scores_sum` is 2 and the
normalised weights of anchors 5 and 10 are 1/2 and 1/3. Row 5 has a box mask of four ones, the
target box (0, 0, 4, 4) in grid units, the target distances (1.5, 1.5, 2.5, 2.5) and an
`l1_scale` of 8 / 64 * 0.25 * 0.5 across and 8 / 32 * 0.25 * 0.5 down. The unequal width and
height separate the two `l1_scale` values.

### ReadsEachRowsAnchorAcrossImagesAndLevels

Two levels give five anchors, and a hand-built assignment for two images marks rows 1 and 9.
Row 1 is anchor 1 of image 0, at point (1.5, 0.5) on stride 8. Row 9 is anchor 4 of image 1, at
point (0.5, 0.5) on stride 16. Each row's target box, distances and `l1_scale` are computed
from its own point and stride, and `target_scores_sum` is 1.5. Row 4, the same anchor as row 9
in image 0, has a box mask of zeros.

### AnUnassignedRowCarriesTheSafeBoxAndNoWeight

Row 0 of the crowded scene is off the foreground. Its target box is (0, 0, 1, 1), and its box
mask, `l1_scale`, target distances, weight and normalised weight are 0. `safe_boxes` holds
(0, 0, 1, 1) in row 5 as well, which is a foreground row.

### TheScoreSumIsAtLeastOne

With no ground truth, `target_scores_sum` is 1.

### PadsTheReferenceLabelsToItsTargetBoxes

The reference labels pad to four valid rows. Gathering the padded boxes and labels by the
reference's `one2many.assignment.target_indices` gives its `target_boxes` within
`K_REFERENCE_BOX_TOLERANCE` and its `target_labels` exactly. The case is skipped when
`loss.sacp` is absent.

### ReproducesTheReferenceOneToManyAssignment

`expect_reference_assignment` runs the one-to-many preset and expects 31 foreground anchors,
the manifest's `foreground_anchor_count.one2many`, of which 4 hold box 3. The case is skipped
when `loss.sacp` is absent.

### ReproducesTheReferenceOneToOneAssignment

The same helper runs the one-to-one preset and expects 4 foreground anchors, the manifest's
`foreground_anchor_count.one2one`, of which 1 holds box 3. The case is skipped when `loss.sacp`
is absent.

## test_average_precision.cpp

Suites: `Unit_AveragePrecision`.

Tests detection matching, the 101-point average precision and the validation table.

This file needs no device and reads no fixture. Every case in `Unit_AveragePrecision` builds
detection rows and ground-truth boxes by hand and calls `match_detections`,
`integrate_precision` or the `AveragePrecision` accumulator of
`modules/vision/detect/include/SushiAI/vision/detect/average_precision.hpp`. A comment in each case derives the expected
number from the recall and precision the detections produce.

### K_TOLERANCE

`K_TOLERANCE` is 1e-12, the absolute tolerance of a metric, about 4500 float64 steps at 1.

### K_FULL_AP

`K_FULL_AP` is 0.995, the area under an envelope that is 1 up to recall 1. The integration
samples 101 recall points and the envelope is 0 at the last one, so the area is
0.99 + 0.01 / 2 and not 1.

### The boxes

`K_BOX_A` is the 8 by 8 box at the origin. `K_BOX_B` and `K_BOX_FAR` are 8 by 8 boxes that
touch neither it nor each other. `K_BOX_LOOSE` is the 8 by 5 box whose IoU with `K_BOX_A` is
40 / 64, which is 0.625, and `K_BOX_NEAR` is the 8 by 7 box whose IoU with it is 56 / 64, which
is 0.875. `K_NO_VALUES` and `K_NO_CLASSES` are empty arrays.

### make_row

`make_row` returns one detection row: a box, then a score and a class.

### join

`join` lays several arrays end to end.

### match

`match` calls `match_detections` for one image and returns the flags. It sizes the array from
`K_DETECTION_VALUES` and fills it with 7 first, so a flag the function does not write is
neither 0 nor 1.

### add_image

The file-local `add_image` passes three vectors to `AveragePrecision::add_image`.

### summarize_image

`summarize_image` returns the table of an accumulator that holds one image.

### HoldsTheTenReferenceThresholds

`K_IOU_THRESHOLDS` has ten entries, and each equals a hexadecimal float literal that carries
the float32 bits of `torch.linspace(0.5, 0.95, 10)`. The comparison is exact, so a threshold
that rounds to a neighbouring float32 fails it.

### IntegratesAnEmptyCurveToOneHalf

`integrate_precision` of two empty curves is 0.5. With no points the envelope is the line from
1 at recall 0 to 0 at recall 1.

### IntegratesUnderThePrecisionEnvelope

Recall 0.5, 0.5, 1 with precision 1, 0.5, 2/3 integrates to
0.49 + 0.05 / 6 + 0.98 / 3 + 0.01 / 3. The envelope lifts the dip to 0.5 up to the 2/3 that
follows it at higher recall.

### RejectsCurvesOfUnequalLength

Two recall values with one precision value throw with a message containing "1 and 2".

### PerfectDetectorScoresTheInterpolationCeiling

Two images each hold one box, of classes 0 and 1, and one detection that equals it. Both rows
have precision 1, recall 1 and `K_FULL_AP` in both AP columns, and the four means are the same
numbers.

### NoDetectionsScoreZero

One box of class 0 and no detection give one row, with label 0 and one instance. Precision,
recall and both AP columns are 0 in the row and in the means.

### FalsePositiveAboveTheTruePositiveHalvesPrecision

A miss at score 0.9 ranks above a hit at 0.6. Recall is 0 then 1 and precision 0 then 0.5, so
the AP is 0.99 * 0.5 + 0.01 * 0.25, which is 0.4975, at every threshold. The mean F1 is flat
below confidence 0.6, so the first confidence point is read: precision 0.5 and recall 1.

### ReadsPrecisionAndRecallAtTheBestF1Confidence

A hit at 0.9 ranks above a miss at 0.3. F1 rises from 2/3 at confidence 0.3 to 1 at 0.9 and is 0
above it, so the smoothed curve is largest at point 849 of 1000. Precision there is
1 - 0.5 * (0.9 - 849 / 999) / (0.9 - 0.3), and the case also compares it with the literal
0.9582082255854392. Recall is 1. The row and the means hold the same values. Reading the first
confidence point gives precision 0.5.

### TwoDetectionsOnOneBoxCountOneMatch

Two detections equal `K_BOX_A` and a third equals `K_BOX_B`, all of class 0. The flags are 1
for the first, 0 for the second and 1 for the third at all ten thresholds. The AP at IoU 0.5 is
0.49 + 0.05 / 6 + 0.98 / 3 + 0.01 / 3, precision is 2/3 and recall is 1. Three matches give
0.995.

### ClassWithoutGroundTruthIsLeftOutOfTheMean

A detection of class 3 has no ground truth of its class. The table has one row, for class 0,
and the means are `K_FULL_AP`, 1 and 1. A row for class 3 halves every mean.

### ClassWithoutDetectionsCountsAsZeroInTheMean

Classes 0 and 1 both have a box and only class 0 has a detection. The row of class 1 has AP 0,
and the means are 0.4975 for both AP columns and 0.5 for precision and recall.

### EqualScoresKeepTheOrderTheyWereAddedIn

A miss and a hit share the score 0.5. Added miss first, the AP at IoU 0.5 is 0.4975; added hit
first, it is `K_FULL_AP`. The ranking is therefore stable.

### LooseDetectionCountsAtTheLowThresholdsOnly

A detection at IoU 0.625 has the flags 1, 1, 1 and then seven zeros. Its AP at IoU 0.5 is
`K_FULL_AP` and its mean over the ten thresholds is 3 * 0.995 / 10, which is 0.2985.

### ImageWithoutGroundTruthAddsFalsePositives

The second image has no ground truth and one detection at score 0.9, which ranks above the
first image's hit at 0.6. The summary counts 2 images and 1 instance, the class row counts 1
image, the AP at IoU 0.5 is 0.4975, precision is 0.5 and recall is 1.

### ImageWithoutDetectionsLowersRecall

The second image has a box and no detection. `image_count()` is 2, the class row counts 2
images and 2 instances, recall is 0.5, precision is 1, and both AP columns are 0.49 + 0.005,
which is 0.495.

### DetectionWhoseBestBoxIsTakenStaysUnmatched

The first detection takes `K_BOX_A`. The second overlaps `K_BOX_A` by 0.875 and `K_BOX_LOOSE`
by 40 / 56, about 0.714. Its flags are ten zeros: a detection is matched to its box of largest
IoU or to none.

### DetectionTakesTheBoxOfLargestIoU

The boxes are listed `K_BOX_LOOSE` then `K_BOX_A`, and one detection equals `K_BOX_NEAR`. Its
flags are eight ones and two zeros, which is the IoU 0.875 it has with the box listed second. A
matcher that took the first box passing 0.5 gives five ones.

### DetectionWithNaNBoxDoesNotMatch

A detection whose x1 is NaN has a NaN IoU with `K_BOX_A`, and its flags are ten zeros. The row
is built by hand because `make_row` takes a box constant.

### BoxGoesToItsFirstDetectionAtEachThreshold

A detection at IoU 0.625 ranks above one at IoU 1. The first holds the box at 0.5, 0.55 and
0.6; the second gets it from 0.65 up. The flags are 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 and
0, 0, 0, 1, 1, 1, 1, 1, 1, 1.

### DetectionOfAnotherClassDoesNotMatch

A detection of class 1 that equals a box of class 0 has ten zero flags.

### NoGroundTruthGivesNoRows

An empty accumulator has no rows. After an image with one detection and no ground truth it
counts 1 image, has no rows, and its four means are 0.

### ClearDropsEveryImage

An image with a miss is added and `clear()` is called. `image_count()` is 0. An image with a
hit added afterwards gives 1 image, 1 instance and AP `K_FULL_AP`, so nothing of the first
image remains.

### RejectsMalformedImages

`add_image` throws for a detection array of 5 values ("[N, 6]", "got 5"), for one box with two
classes ("4 values", "2 classes"), for a ground-truth class of -1 ("box 0", "negative class
-1"), for a detection class of 0.5 ("detection 0", "not a class index") and for a NaN score
("detection 0", "score"). The accumulator still holds no image and no rows afterwards.
`match_detections` throws for a box without a class ("0 classes") and for a flag array of 9
values ("need 10 values", "got 9").

## anchor/test_head_delta.cpp

Suites: `Unit_DetectAnchorDeltaIou`.

Tests the delta of one anchor head when more than one anchor may take a truth and the box
deltas come from the IoU gradient. Truths are given in pixels of a 64-pixel network side. A box
delta is compared with a value the file computes in double, within 1e-5 absolute plus 1e-5
relative.

### TwoAnchorsAboveTheMatchThresholdAreBothAssigned

With a match threshold of 0.5, the best slot and a second slot of the truth's cell both get the
box deltas of their own decoded box against the truth, an objectness delta of 0.5 and a class
delta of 1. The other ten slots get an objectness delta of -0.5 and no box delta. The result
counts two assignments, and only the best slot records the truth in `assigned_row` and
`assigned_class`.

### AFurtherAssignmentAddsItsIoUToTheSum

With zero logits in both assigned slots, the IoU sum is 1 + 0.2 / 0.29 to 1e-6.

### AMatchThresholdOfOneAssignsTheBestAnchorAlone

At the default threshold the best slot gets its box deltas, the second slot gets none and an
objectness delta of -0.5, and the result counts one assignment.

### AnAnchorBelowTheMatchThresholdIsNotAssigned

With a match threshold of 0.75, above the second anchor's shape IoU of 0.2 / 0.29, the result
counts one assignment and the second slot has no box delta.

### AHeadWithoutTheBestAnchorStillTakesAMatchingTruth

Head 1, which does not own the truth's best anchor, gets box deltas, an objectness delta of 0.5
and a class delta of 1 at a match threshold of 0.5. The result counts one assignment, and
`assigned_row` and `assigned_class` stay -1.

### TwoPositiveClassDeltasHalveTheBoxDeltasOfTheSlot

Two truths of different classes in one slot, at a match threshold of 0.5 and a box weight of
0.07, leave the slot with the sum of the two box deltas divided by 2. Both class deltas are
0.5, and the result counts two assignments.

### AMatchThresholdOfOneDividesNothing

At the default threshold the same slot holds the undivided sum of the two box deltas.

### TheDivisionCoversSquaredErrorBoxes

With `BoxLoss::SquaredError`, a match threshold of 0.5 and two truths that are the anchor at
the cell centre, the width delta is 2 * k * (0 - 0.5) divided by 2, with
k = (2 - 0.5 * 0.5) * 0.75, and the other three box deltas are 0.

### ABoxDeltaBeyondTheBoundIsClipped

With `max_delta` 0.5, each box delta whose magnitude exceeds 0.5 is 0.5 with its sign, and the
others are unchanged. The case asserts that at least one component is clipped and at least one
is not.

### ANonFiniteBoxComponentAddsZero

With a width logit of 200, which decodes to infinity, the width delta is exactly 0, all four
box deltas are finite and the objectness delta is 0.5.

Suites: `Unit_DetectAnchorDeltaActivated`.

Tests the delta of one anchor head whose entries arrive activated. The case holds two slots
and two classes over a 64-pixel network side. Slot 0 is assigned the truth. Slot 1 overlaps the
truth with an IoU of 0.8 and is not assigned. Both publish an objectness of 0.3.

### AnIgnoredSlotIsPulledTowardItsBestIoU

Slot 1's objectness delta is 0.8 - 0.3, to 1e-6, and its `assigned_row` is -1.

### TheObjectWeightScalesAnIgnoredSlot

With an object weight of 0.5, slot 1's objectness delta is 0.5 * (0.8 - 0.3).

### TheTargetOneLeavesAnIgnoredSlotAlone

With `ObjectnessTarget::One`, slot 1's objectness delta is exactly 0.

### ASlotWithoutAGatedClassIsNotIgnored

With both class entries of slot 1 at 0.2, its objectness delta is 0 - 0.3.

### AnAssignedSlotKeepsADeltaThatIsNotZero

Slot 0's `assigned_row` is 0 and its objectness delta is 0 - 0.3.

### AnAssignedSlotWithAZeroDeltaIsPulledTowardOne

With slot 0's objectness set to exactly 0, its objectness delta is 1.

### TheTargetOnePullsAnAssignedSlotTowardOne

With `ObjectnessTarget::One`, slot 0's objectness delta is 1 - 0.3.

### TheBoxDeltaIsTheWeightedGradientWithNoSizeFactor

With `scale_xy` 2 and a box written into slot 0, the four box deltas equal the gradient at the
decoded box against the truth, weighted by 0.75, within 1e-5 absolute plus 1e-5 relative. The
expected width entry is above 0.01 in magnitude. Slot 1's four box deltas are exactly 0.

### TheClassDeltasReadTheEntriesAsProbabilities

Slot 0's class deltas are 1 - 0.6 and 0 - 0.2. Slot 1's class deltas are exactly 0.

Suites: `Unit_DetectAnchorDeltaGradient`.

Tests the two forms of the box gradient a head's training options select. Each case decodes
one slot to a chosen box against one truth on a 20-pixel network side and reads the four box
deltas. Hand values are compared to 1e-5.

### TheDefaultFormIsAnalytic

With `box_gradient` taken from a default `HeadTrainingOptions`, a 0.2 square centred in a 0.4
square gets a width delta and a height delta of 1.25.

### TheCompatibleFormDoublesTheSizeSlopeOfTheOverlap

For the same boxes the analytic form writes 1.25 into width and height and the compatible form
writes 2.5. Both write 0 into x.

### TheCompatibleFormFlipsTheEnclosingTermOfTheCentre

For two disjoint equal squares the x delta is 0.18 / 0.2809 in the analytic form and
0.88 / 0.2809 in the compatible form. Both width deltas are 0.7 * 0.25 / 0.2809, and the
compatible form's y and height deltas are 0.

### TheCompatibleFormSimplifiesTheAspectTerm

For a 0.2 by 0.1 box centred in a 0.4 square, the case computes the aspect term and its
trade-off in double and expects the width and height deltas of each form from them. The
compatible form's are 1.25 and 2.5 moved by the trade-off times the slope. The analytic form's
scale the overlap part by one minus the squared trade-off.

### TheCompatibleFormReproducesTheDumpedDeltas

For each of the twelve cells in `K_DUMPED_CELLS`, with a box weight of 0.05, a `max_delta` of 2,
a match threshold of 0.2 and `ObjectnessTarget::BestIoU`, the compatible form's IoU sum equals
the cell's IoU to 1e-6 and each box delta equals the cell's to 1e-4 relative.

### TheAnalyticFormDepartsFromTheDumpedSizeDeltas

For the first dumped cell, the analytic form's width and height deltas each differ from the
cell's by more than 10 % of the cell's value.

## test_iou_gradient.cpp

Suites: `Unit_DetectIouGradient`.

Tests the box gradient declared in
`modules/vision/detect/include/SushiAI/vision/detect/iou_gradient.hpp` in its two forms,
`BoxGradient::Analytic` and `BoxGradient::DarknetCompatible`. A box is a centre and a size.
The analytic form is compared with a central difference of step 1e-6, within 1e-6 relative
plus 1e-7 absolute. Hand values are compared to 1e-5.

### AnalyticMatchesTheDifferenceOnOverlappingBoxes

The analytic gradient equals the central difference for two pairs of boxes that overlap
without one holding the other.

### AnalyticMatchesTheDifferenceOnDisjointBoxes

The same holds for a pair apart along one axis and a pair apart along both.

### AnalyticMatchesTheDifferenceOnNestedBoxes

The same holds for a box inside its target and for a box around its target.

### AnalyticMatchesTheDifferenceOnSeededPairs

The same holds for 100 pairs drawn from a seeded sequence, with centres in [0.2, 0.8] and sides
in [0.1, 0.5]. A pair that lies near a kink of the measure is drawn again and not counted.

### DarknetCompatibleMatchesTheDumpedDeltas

For each of the twelve cells in `K_DUMPED_CELLS`, the compatible gradient times the box weight
0.05 equals each of the cell's deltas to 1e-4 relative.

### DarknetCompatibleDoublesTheSizeSlopeOfTheOverlap

For a 0.2 square centred in a 0.4 square, the analytic width and height components are 1.25
and the compatible ones 2.5. Both x components are 0.

### DarknetCompatibleFlipsTheEnclosingTermOfTheCentre

For two disjoint equal squares, the x component is 0.18 / 0.2809 in the analytic form and
0.88 / 0.2809 in the compatible form. Both width components are 0.7 * 0.25 / 0.2809, and the
compatible y and height components are 0.

### DarknetCompatibleSimplifiesTheAspectTerm

For a 0.2 by 0.1 box centred in a 0.4 square, the case computes the aspect term and its
trade-off in double, with `K_ASPECT_TRADE_OFF_EPS`, and expects the width and height components
of each form from them.

### BothFormsAreZeroForTwoBoxesAtOnePoint

For two boxes of zero size at one point, every component of both forms is exactly 0.

## test_suppression_criterion.cpp

Suites: `Unit_DetectSuppressionCriterion`, `Unit_DetectSuppressionCriterionDeathTest`.

Tests `Detect::measure_suppression`, the six-argument `Detect::suppress_overlaps`, and the
criterion fields of `Detect::NMSConfig`. Boxes are corner pairs, and most cases measure against
one 10 by 10 square. Measures are compared to 1e-6.

### IoUOfOverlappingBoxesIsTheRatioOfAreas

The square and a copy shifted by 1 along x have an IoU of 90 / 110.

### IoUIsZeroForDisjointAndTouchingBoxes

A box 10 units away and a box that shares only an edge with the square each have an IoU of
exactly 0.

### IoUOfTwoEmptyBoxesIsZeroNotNaN

Two boxes of zero area at one point have an IoU that is not NaN and is exactly 0.

### DistanceIoUSubtractsTheNormalisedCentreDistance

For the square and a copy shifted down by 6, with the exponent 0.6, the measure is
0.25 - (36 / 356)^0.6.

### DistanceIoUOfDisjointBoxesIsNegative

For the square and a box 20 units to its right, the measure is -(0.4^0.6).

### ZeroExponentSubtractsOne

With the exponent 0 the measure of the square and its shifted copy is 0.25 - 1.

### DistanceIoUOfIdenticalBoxesIsOne

The square measured against itself gives 1.

### DistanceIoUOfCoincidentPointsIsTheIoU

Two boxes of zero area at one point measure exactly 0, and the value is not NaN.

### DistanceIoUIsSymmetric

The measure of the square and a diagonally shifted box is the same in both orders.

### DistanceIoUKeepsTheFartherOfTwoEqualIoUBoxes

Of three boxes scored 0.9, 0.8 and 0.7, at a threshold of 0, the DistanceIoU criterion keeps
rows 0 and 1. The IoU criterion keeps row 0 alone.

### IoUCriterionKeepsABoxAtTheThreshold

Two boxes with an IoU of exactly 0.5 are both kept at a threshold of 0.5. At a threshold of
0.25 only the first is kept.

### ZeroScoreRowsAreNeverKept

Of three boxes that do not overlap, scored 0.9, 0 and 0.5, the rows 0 and 2 are kept.

### EmptyInputKeepsNothing

With no box and no score, a `kept` vector that held one entry comes back empty.

### FourArgumentFormEqualsTheIoUCriterion

For four boxes that do not overlap, scored 0.1, 0.9, 0.5 and 0.5, the four-argument
`suppress_overlaps` keeps the rows 1, 2, 3, 0 in that order, and the six-argument form with the
IoU criterion keeps the same.

### SelectionAppliesTheConfiguredCriterion

`select_nms_detections` with the DistanceIoU criterion, a confidence threshold of 0.1 and an
IoU threshold of 0 over the three boxes of the equal-IoU case, given as centre and size,
returns one image with two detections: the corners (0, 0, 10, 10) scored 0.9 and
(0, 6, 10, 16) scored 0.8, both of class 0.

### NMSConfigDefaultsToTheIoUCriterion

A default `NMSConfig` has the criterion `SuppressionCriterion::IoU` and a `distance_exponent`
of 0.6.

### RejectsBoxesOfTheWrongSizeFatally

Five box values with two scores stop the process with a message holding "a suppression needs"
and "got 5 and 2 values".

### RejectsAnUnknownCriterionFatally

A criterion cast from 7 stops the process with a message holding "must be IoU or DistanceIoU"
and "got 7".

## anchor/test_head_cost.cpp

Suites: `Unit_DetectAnchorCost`.

Tests `measure_head_cost`
over a head with two classes whose delta the file builds by hand. Its box part sums to 5.25 and
its non-box part to 26, over a batch of 2. Every number is compared to 1e-12.

### SquaredErrorSplitsBoxFromNonBoxAndDividesByBatch

With the squared-error box loss and an objectness weight of 1, the total is 15.625, the non-box
share 13, the box share 2.625 and the stored cost 31.25.

### ObjectWeightScalesTheNonBoxShare

With an objectness weight of 2 the non-box share is 26 and the box share -10.375. The total
stays 15.625 and the stored cost 31.25.

### CompleteIoUStoredAddsTheIoUTermAndWeightedNonBoxSum

With the CompleteIoU box loss, a box weight of 0.75, an objectness weight of 2 and two
assignments whose IoU sums to 1, the stored cost is 52.375. The total is 15.625, the non-box
share 26 and the box share -10.375.

### CompleteIoUWithoutAssignmentsOmitsTheIoUTerm

A CompleteIoU head with no assignment and an objectness weight of 1 stores 26.

### ZeroDeltaReportsZeroCost

A delta of 28 zeros reports 0 in the total, the box share, the non-box share and the stored
cost.

## anchor/test_assignment_thresholds.cpp

Suites: `Unit_DetectAnchorThresholds`.

Tests `AssignmentThresholds` and `derive_assignment_thresholds`. Each fit in the file is given
by its tenth percentile and its mean best IoU, and thresholds are compared to 1e-6.

### DefaultsAreOneAndSevenTenths

A default `AssignmentThresholds` has a match threshold of exactly 1 and an ignore threshold of
exactly 0.7.

### FitInsideBothRangesGivesItsOwnNumbers

A tenth percentile of 0.35 and a mean of 0.62 give a match threshold of 0.35 and an ignore
threshold of 0.62.

### FourFitsFollowTheFormula

For the fits (0.35, 0.62), (0.21, 0.51), (0.5, 0.69) and (0.25, 0.55), the match threshold is
the tenth percentile and the ignore threshold is the mean.

### MatchThresholdIsClampedAtBothEnds

Tenth percentiles of 0.05, 0 and 0.2 give a match threshold of 0.2. Tenth percentiles of 1 and
1.3 give 1.

### IgnoreThresholdIsClampedAtBothEnds

Means of 0.3, 0 and 0.5 give an ignore threshold of 0.5. Means of 0.7 and 0.95 give 0.7.

### PerfectFitGivesOneAndSevenTenths

A fit of 1 in both numbers gives a match threshold of 1 and an ignore threshold of 0.7.

## anchor/test_head_training_section.cpp

Suites: `Unit_DetectAnchorTrainingSection`.

Tests `read_head_training_section` and `write_head_training_section` in
`modules/vision/detect/include/SushiAI/vision/detect/anchor/head_training_section.hpp`.
Options are compared field by field, every float bit for bit. The full section states the
type "anchor" and all nine option keys.

### TheDefaultsRoundTripThroughTheirSection

A default `HeadTrainingOptions` comes back unchanged after `write_head_training_section`,
`write_json`, `parse_json` and `read_head_training_section`.

### OptionsWithNoDefaultRoundTripThroughTheirSection

Options with the thresholds 0.213f and 0.7f, `CompleteIoU`, `DarknetCompatible`, `BestIoU`,
the weights 0.07f and 0.4f, a `max_delta` of 5.0f and a class gate of 0.3f come back unchanged
after the same four steps.

### AFullSectionRoundTripsThroughItsOptions

The full section, read and written back, gives the same JSON text.

### ReadsEveryFieldOfASection

The full section reads as the thresholds 0.25 and 0.5, `BoxLoss::CompleteIoU`,
`BoxGradient::DarknetCompatible`, `ObjectnessTarget::BestIoU`, a box weight of 0.0625, an
object weight of 1.5, a `max_delta` of 5.0 and a class gate of 0.125.

### AKeyLeftOutKeepsTheDefaultOfItsField

An empty section and a section with the type alone both read as the default options. A
section with `max_delta` 2.5 alone reads as the defaults with that one field changed.

### WritesTheTypeEveryKeyInOrderAndAFloatAsItsDouble

The written section has the keys `type`, `match_threshold`, `ignore_threshold`, `box_loss`,
`box_gradient`, `objectness_target`, `box_weight`, `object_weight`, `max_delta`, `class_gate`
in that order. Its type is `K_ANCHOR_OBJECTIVE_TYPE` and its `box_loss` "complete_iou". The
match threshold equals `double(0.213f)` and the box weight `double(0.07f)`.

### ReadsAndWritesEachEnumeratorString

The writer gives "squared_error" and "complete_iou" for `BoxLoss`, "analytic" and
"darknet_compatible" for `BoxGradient`, and "one" and "best_iou" for `ObjectnessTarget`. A
section of "squared_error", "analytic" and "one" reads as `SquaredError`, `Analytic` and
`One`.

### ReadsAWholeNumberAsAWeight

An `object_weight` of 2 reads as 2.0f.

### RefusesASectionThatIsNotAnObject

An array and a null value both fail with `InvalidDescription`, "objective" and "an object".

### RefusesAnotherType

The type "cross_entropy" fails with `InvalidDescription`, "objective 'type'",
"'cross_entropy'" and "'anchor'". The type 3 fails with "objective 'type'" and "a string".

### RefusesAnUnknownKey

The key `iou_normalizer` fails with `InvalidDescription`, "objective", "'iou_normalizer'",
"'type', 'match_threshold'" and "'class_gate'". The key `thresholds` fails with "objective"
and "'thresholds'".

### RefusesAValueOfAnotherKind

The read fails with `InvalidDescription` for a `box_weight` given as a string
("objective 'box_weight'", "a number"), an `ignore_threshold` given as an array
("objective 'ignore_threshold'", "a number") and a `box_loss` of 1 ("objective 'box_loss'",
"a string").

### RefusesAStringNoEnumeratorHas

The read fails with `InvalidDescription` for the `box_loss` "ciou" ("objective 'box_loss'",
"'ciou'", "'squared_error', 'complete_iou'"), the `box_gradient` "exact"
("objective 'box_gradient'", "'exact'", "'analytic', 'darknet_compatible'") and the
`objectness_target` "iou" ("objective 'objectness_target'", "'iou'", "'one', 'best_iou'").

### RefusesANumberFloat32CannotHold

A `max_delta` of 1e39 fails with `InvalidDescription`, "objective 'max_delta'" and "float32".

### RefusesTheValuesTheOptionsCheckRefuses

The read fails with `InvalidDescription` and "objective" for a `match_threshold` of 0.0
("match threshold", "(0, 1]"), an `ignore_threshold` of 1.5 ("ignore threshold", "1.5"), a
`class_gate` of -0.25 ("class gate", "-0.25") and a `box_weight` of -1.0 ("box weight",
"-1").

### WriterRefusesTheValuesTheOptionsCheckRefuses

`write_head_training_section` fails with `InvalidDescription` and "objective" for an object
weight of NaN ("object weight"), an infinite `max_delta` ("max_delta") and a `BoxLoss` of 7
("BoxLoss", "7").
