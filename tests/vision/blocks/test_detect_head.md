# tests/vision/blocks/test_detect_head.cpp


Suites: `Medium_BlockDetectHead`.

Tests `DetectHead` and its factory `Vision::Blocks::make_detect_node` at row 23 of YOLO26n:
three feature levels of 64, 128 and 256 channels with the sides 80, 40 and 20, and 80 classes.
The traced cases feed a batch of 2 with the sides 8, 4 and 2.

## ThePortsCarryTheFixtureNamesAndShapes

The head takes three inputs and has twelve ports. They are named
`<copy>.level<n>.<role>`, with the copy "one2many" before "one2one", the role "boxes" before
"scores", and the levels in order inside a role. A boxes port has 4 channels and a scores port
80, at its level's side.

## StateMatchesManifestRecords390To593

Walked under the prefix "model.23" after a training trace, the head yields 204 records in
order: the box branches under `cv2`, the class branches under `cv3`, then the same two groups
under `one2one_cv2` and `one2one_cv3`.

## ATrainingTraceDetachesEachFeatureOnce

A training trace holds three STOP_GRADIENT nodes, whose inputs are the three features in level
order. `forward` appends twelve values, each of its port's channel count at the traced side of
its level, with the batch extent 2 in front.

## AnEvaluationTraceEmitsNoStopGradient

An evaluation trace holds no STOP_GRADIENT and no BATCHNORM2D node, 36 BATCHNORM_FOLD nodes and
12 DEPTHWISE_CONV2D nodes, and appends twelve values.

## TheFirstLevelSetsTheBranchWidths

A head over one [4, 4, 128] level and 3 classes has 68 state records after a trace. Among them
are `cv2.0.0.conv.weight` of [32, 3, 3, 128], `cv2.0.2.weight` of [4, 1, 1, 32],
`cv3.0.0.1.conv.weight` of [128, 1, 1, 128] and `cv3.0.2.weight` of [3, 1, 1, 128]. Port 1 has
the shape [4, 4, 3].

## ForwardRefusesAWrongInputCount

`forward` given two values for three levels fails with `InvalidDescription`, "input count is 3"
and "given is 2", and appends nothing.

## CreateRefusesLevelsOrClassesThatDoNotFit

`create` fails with `InvalidDescription` for no level ("at least one feature level"), for a
rank-2 second level ("level 1", "[40, 128]", "rank-3") and for a class count of zero
("at least one class", "0").

## TheFactoryBuildsRow23

`make_detect_node` over the three levels, with "classes" 80 and the norm constants, returns a
node with three inputs and twelve ports. Port 0 is "one2many.level0.boxes" of [80, 80, 4] and
port 11 is "one2one.level2.scores" of [20, 20, 80]. A training trace holds 36 BATCHNORM2D nodes
that carry the row's epsilon, every update carries the row's momentum, and the state walk
yields 204 records.

## TheFactoryRefusesARowThatDoesNotFit

An unknown key "filters" fails with a message holding "layer 23 'detect'" and "'filters'". A
row without "classes" fails with "'classes'" and "absent". An empty level list fails with
"layer 23 'detect'" and "at least one feature level".

## AWideBoxPlainSingleCopyHeadCarriesItsRecords

With `reg_max` 16, `end2end` false and `class_branch` "plain", the 64, 128 and 256 channel head
walked under "model.22" yields 72 records: per level a 64-wide box branch ending in a 1x1 of 64
channels under `cv2`, and a plain class branch of two 3x3 `Conv` of width 80 and a 1x1 of 80
channels under `cv3`. No STOP_GRADIENT and no DEPTHWISE_CONV2D node is traced, and `forward`
appends six values.

## ThePortCountFollowsTheCopiesAndTheBoxWidthFollowsRegMax

The defaults give twelve ports with 4 box channels. `reg_max` 16 makes every boxes port 64
channels wide. `end2end` false leaves six ports, all "one2many".

## TheFactoryBuildsAWideBoxPlainSingleCopyRow

The row `{"classes": 80, "reg_max": 16, "end2end": false, "class_branch": "plain"}` builds the
same head as the options do: six ports, 72 records.

## CreateRefusesOptionsThatDoNotFit

`create` refuses a `reg_max` of 0, two strides for three levels and a stride of 0.

## TheFactoryRefusesABadValueForEachNewKey

The factory refuses `reg_max` of 0 and of "16", `end2end` of "yes", and `class_branch` of
"wide" and of 3, naming the key each time.

## WithoutStridesEveryLastBiasStartsFromZero

With no strides, the twelve `.2.bias` parameters of either class branch form start from the
constant 0.

## StridesSetTheLastBiasesOfBothCopies

With strides 8, 16 and 32, each last box bias starts from 1 and each last class bias from
log(5 / 80 / (640 / stride)^2), on the one-to-many and the one-to-one copy, in both forms.

## ThePublishedNamesAndConstantsEqualTheSchemeLiterals

`detect_port_name` gives "one2many.level0.boxes" and "one2one.level12.scores". The branch names
are "cv2", "cv3", "one2one_cv2" and "one2one_cv3", the last convolution's index is 2, the box
bias is 1, and the class bias constants are 5 and 640.
