# tests/vision/detect/yolox

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_yolox_decode.cpp` | Tests the decoding of a YOLOX head's outputs against boxes and scores worked by hand. |
| `test_yolox_head_contract.cpp` | Tests the port names and the objectness prior of the YOLOX head contract. |
| `test_yolox_task.cpp` | Tests that a YOLOX task and its task section convert both ways and name a bad key. |
