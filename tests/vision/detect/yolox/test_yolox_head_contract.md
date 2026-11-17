# tests/vision/detect/yolox/test_yolox_head_contract.cpp

Suites: `Small_DetectYoloxHeadContract`.

Tests the names and the bias that `DecoupledHead` and `decode_yolox` share.

## APortNameIsTheLevelAndTheRole

`yolox_port_name` gives "level0.classes", "level2.boxes" and "level11.objectness".

## ThePriorBiasIsTheLogitOfOneHundredth

`yolox_prior_bias` is negative and equals -log((1 - 0.01) / 0.01) within two double roundings
of its magnitude plus two.
