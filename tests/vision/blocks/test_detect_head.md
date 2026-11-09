# tests/vision/blocks/test_detect_head.cpp


Suites: `Small_BlockDetectHead`, `Small_BlockDetectHeadPorts`, `Small_BlockDetectHeadOptions`, `Small_BlockDetectHeadRegMax`, `Small_BlockDetectHeadClassBranch` and `Small_BlockDetectHeadBiases`. Every case traces a head into a graph on the host, so no case waits on a device, a file, a thread or a clock.

Tests `DetectHead` and its factory `Vision::Blocks::make_detect_node` at row 23 of YOLO26n:
three feature levels of 64, 128 and 256 channels with the sides 80, 40 and 20, and 80 classes.
The traced cases feed a batch of 2 with the sides 8, 4 and 2.


## Row23ReadsThreeLevelsAndPublishesTwelvePorts

The head built over the three levels takes three inputs and has twelve ports.

## APortCarriesItsFixtureNameAndShape

Each of the twelve ports is a row of `Small_BlockDetectHeadPorts`, which builds the head once in `SetUpTestSuite` and frees it in `TearDownTestSuite`. Ports are named `<copy>.level<n>.<role>`, with the copy "one2many" before "one2one", the role "boxes" before "scores", and the levels in order inside a role. A boxes port has 4 channels and a scores port 80, at its level's side. The names and shapes are typed from `forward.sacp` in `MANIFEST.json`.

## StateMatchesManifestRecords390To593

Walked under the prefix "model.23" after a training trace, the head yields 204 records in
order: the box branches under `cv2`, the class branches under `cv3`, then the same two groups
under `one2one_cv2` and `one2one_cv3`.

## ATrainingTraceDetachesEachFeatureOnce

A training trace holds STOP_GRADIENT nodes whose inputs are the three features, once each, in level order.

## ATrainingTraceAppendsOneValueOfEachPortsShape

`forward` appends twelve values, each of its port's channel count at the traced side of its level, with the batch extent 2 in front.

## AnEvaluationTraceEmitsNoStopGradient

An evaluation trace holds no STOP_GRADIENT and no BATCHNORM2D node, and appends twelve values.

## TheFirstLevelSetsTheBranchWidths

A head over one [4, 4, 128] level and 3 classes has 68 state records after a trace. The walk holds `cv2.0.0.conv.weight` of [32, 3, 3, 128], `cv2.0.2.weight` of [4, 1, 1, 32], `cv3.0.0.1.conv.weight` of [128, 1, 1, 128] and `cv3.0.2.weight` of [3, 1, 1, 128], found by name. Port 1 has the shape [4, 4, 3].

## ForwardRefusesAWrongInputCount

`forward` given two values for three levels fails with `InvalidDescription`, "input count is 3"
and "given is 2", and appends nothing.

## CreateRefusesNoLevel

`create` over no level fails with `InvalidDescription` and "at least one feature level".

## CreateRefusesALevelOfRankTwo

`create` over a second level of rank 2 fails with "level 1", "[40, 128]" and "rank-3".

## CreateRefusesZeroClasses

`create` with a class count of zero fails with "at least one class" and "0".

## TheFactoryBuildsRow23

`make_detect_node` over the three levels, with "classes" 80 and the norm constants, returns a node with three inputs and twelve ports. Port 0 is "one2many.level0.boxes" of [80, 80, 4] and port 11 is "one2one.level2.scores" of [20, 20, 80]. In a training trace every norm carries the row's epsilon and every update the row's momentum, and the state walk yields 204 records.

## TheFactoryRefusesAnUnknownKey

An unknown key "filters" fails with a message holding "layer 23 'detect'" and "'filters'".

## TheFactoryRefusesAnAbsentClassCount

A row without "classes" fails with "'classes'" and "absent".

## TheFactoryRefusesNoLevel

An empty level list fails with "layer 23 'detect'" and "at least one feature level".

## AWideBoxPlainSingleCopyHeadCarriesItsRecords

With `reg_max` 16, `end2end` false and `class_branch` "plain", the 64, 128 and 256 channel head walked under "model.22" yields 72 records: per level a 64-wide box branch ending in a 1x1 of 64 channels under `cv2`, and a plain class branch of two 3x3 `Conv` of width 80 and a 1x1 of 80 channels under `cv3`.

## ThePortCountFollowsTheCopiesAndTheBoxWidthFollowsRegMax

Four option sets are rows of `Small_BlockDetectHeadOptions`. The defaults give twelve ports with 4 box channels. `reg_max` 16 makes every boxes port 64 channels wide. `end2end` false leaves six ports, the first "one2many.level0.boxes" and the last "one2many.level2.scores". Wide boxes with one copy and plain class branches leave six ports of 64 box channels and 80 score channels.

## TheFactoryBuildsAWideBoxPlainSingleCopyRow

The row `{"classes": 80, "reg_max": 16, "end2end": false, "class_branch": "plain"}` builds six ports and a first port of 64 box channels, as the options do. The state is pinned by `AWideBoxPlainSingleCopyHeadCarriesItsRecords`.

## CreateRefusesARegMaxBelowOne

`create` refuses a `reg_max` of 0, naming "reg_max of at least one" and "0".

## CreateRefusesStridesOfTheWrongCount

`create` refuses two strides for three levels, naming "one stride per level", "2 strides" and "3 levels".

## CreateRefusesAStrideThatIsNotPositive

`create` refuses a stride of 0, naming "positive finite strides".

## ARegMaxThatIsNotAPositiveWholeNumberIsRefused

The factory refuses `reg_max` of 0 ("'reg_max'", "positive") and of the string "16" ("'reg_max'", "whole number"). The suite is `Small_BlockDetectHeadRegMax`, one row each.

## TheFactoryRefusesAnEnd2EndFlagThatIsText

The factory refuses `end2end` of "yes", naming "'end2end'" and "true or false".

## AClassBranchThatNamesNoStyleIsRefused

The factory refuses `class_branch` of "wide" and of 3, naming "'class_branch'", "depthwise" and "plain". The suite is `Small_BlockDetectHeadClassBranch`, one row each.

## WithoutStridesEveryLastBiasStartsFromZero

With no strides, the twelve `.2.bias` parameters of either class branch form start from the constant 0, compared exactly because a constant initialiser stores the value it is given. The suite is `Small_BlockDetectHeadBiases`, one row per form.

## StridesSetTheLastBiasesOfBothCopies

With strides 8, 16 and 32, each last box bias starts from 1 and each last class bias from log(5 / 80 / (640 / stride)^2), on the one-to-many and the one-to-one copy. Each constant is compared to its closed form with `TestSupport::measure_error` and the bound (4 + |reference|) times the double unit roundoff: four roundings inside the logarithm's argument and one in the logarithm.
