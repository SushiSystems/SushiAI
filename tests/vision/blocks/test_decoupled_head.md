# tests/vision/blocks/test_decoupled_head.cpp

Suites: `Small_BlockDecoupledHead`. Every case traces a head into a graph on the host, so no
case waits on a device, a file, a thread or a clock.

Tests `DecoupledHead` and its factory `make_decoupled_head_node` over three levels of 16, 32
and 64 channels with the sides 8, 4 and 2, 3 classes and a hidden width of 8. The traced cases
feed a batch of 2.

## ThreeLevelsPublishNinePortsByLevelAndRole

The head takes three inputs and has nine ports, in the order classes, boxes, objectness per level.
Each port is named "level<l>.<role>" and has the shape [side, side, 3], [side, side, 4] or
[side, side, 1] of its level.

## PortNamesAreTheContractsNames

The names of ports 0, 1, 2 and 8 equal `yolox_port_name` of the level and the role constant.

## ATraceAppendsOneValueOfEachPortsShape

`forward` appends nine values, each with its port's shape behind the batch extent 2.

## TheFirstLevelYieldsClassesBoxesAndObjectness

The 8 by 8 level yields [2, 8, 8, 3], [2, 8, 8, 4] and [2, 8, 8, 1].

## StateCarriesTheYoloxRecordNames

Walked under "head", the head yields 93 records in the order of YOLOX's state dict:
"head.cls_convs.<l>.<j>", "head.reg_convs.<l>.<j>", "head.cls_preds.<l>", "head.reg_preds.<l>",
"head.obj_preds.<l>", then "head.stems.<l>". A tower convolution has the weight and four norm
records, a prediction a weight and a bias, a stem the weight and four norm records. The
prediction weights are [3, 1, 1, 8], [4, 1, 1, 8] and [1, 1, 1, 8].

## TheBiasesStartFromThePriorAndZero

Each class and objectness bias starts from `yolox_prior_bias()` and each box bias from 0. A
constant initialiser stores the value it is given, so the comparison is exact.

## ForwardRefusesAWrongInputCount

Two values for three levels fail with `InvalidDescription`, "input count is 3" and "given is 2",
and append nothing.

## CreateRefusesNoLevel

`create` over no level fails with "at least one feature level".

## CreateRefusesALevelOfRankTwo

A second level of rank 2 fails with "level 1", "[4, 32]" and "rank-3".

## CreateRefusesZeroClassesAndZeroHidden

A class count of 0 fails with "at least one class"; a hidden width of 0 fails with "hidden width".

## TheFactoryBuildsTheHead

The row `{"classes": 3, "hidden": 8, "bn_epsilon": 0.001, "bn_momentum": 0.03}` builds a node
with three inputs and nine ports; port 0 is "level0.classes" and port 8 has the shape [2, 2, 1].

## TheFactoryRefusesAnUnknownKey

An unknown key "width" fails with "layer 30 'decoupled_head'" and "'width'".

## TheFactoryRefusesAnAbsentKey

A row without "classes" fails with "'classes'" and "absent"; a row without "hidden" fails with
"'hidden'" and "absent".

## TheFactoryRefusesNoLevel

An empty level list fails with "layer 30 'decoupled_head'" and "at least one feature level".
