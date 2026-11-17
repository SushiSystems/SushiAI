# tests/vision/registration/test_registration/test_yolox_x.cpp

Suites: `Medium_YoloxXFile`, `Medium_YoloxXLevel`, `Medium_YoloxXMap`.

Tests `applications/examples/yolox_x/model.json` and `map.json`. No weights are read and no
tensor is allocated: the model is built and traced over a batch of one at 800 by 1440, and its
state is listed by a walk of the built model, which gives the name and shape of every parameter
and buffer. The expected tensors are written in the test from the structure of the YOLOX source
(CSPDarknet, YOLOPAFPN and YOLOXHead at depth 1.33 and width 1.25), not read from the model file.
The unit width 80 is 64 times 1.25, the repeats 4 are round(3 times 1.33), and the head width 320
is 256 times 1.25.

## HoldsItsRows

The file holds 25 rows: the Focus stem, the four stages of the backbone with the spatial pyramid
pooling in the last, the nine rows of the neck up to its three outputs, and the head.

## TakesAnInputOf800By1440By3

The input shape is [800, 1440, 3], the size ByteTrack runs a MOT17 frame at.

## PublishesNineOutputsNamedByLevelAndRole

The nine outputs are named `level0.classes`, `level0.boxes`, `level0.objectness`, then the same for
levels 1 and 2, as `yolox_port_name` builds them, so the inference row finds them by the names of
`yolox_head_contract.hpp`.

## PublishesItsThreePortsOnTheGridOfItsStride

For each level 0, 1 and 2 (strides 8, 16 and 32), the classes port is [1, 800 / s, 1440 / s, 1], the
boxes port [1, 800 / s, 1440 / s, 4] and the objectness port [1, 800 / s, 1440 / s, 1]: grids of
100 by 180, 50 by 90 and 25 by 45.

## TheTaskSectionReadsBackItsThresholds

`read_yolox_task` reads the section as score threshold 0.1 and an IoU suppression at 0.7.

## EveryRuleNamesARecordTheModelHas

Each rename rule of `map.json` has a target prefix that at least one record of the built model
starts with.

## TheStateDictionaryFillsTheRecordsExactly

The 748 tensors of the YOLOX-X state dictionary (146 BaseConv of five tensors each, and the weight and bias of nine predictors), listed by the test with their PyTorch
shapes, go through the rename rules. No two reach one name, every result is a record of the model
with the shape the record states (a convolution weight [out, in, kh, kw] becomes [out, kh, kw,
in]), and no record is left unfilled, so the counts of renamed tensors and of records are equal.

## TheBatchCountersAreDropped

The `num_batches_tracked` entry of every normalisation of the dictionary is on the map's drop list.

## NamesTheModelKeyOfTheCheckpoint

The map reads the state dictionary under the key `model`, which is where YOLOX checkpoints keep it.
