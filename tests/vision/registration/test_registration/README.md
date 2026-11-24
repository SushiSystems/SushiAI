# tests/vision/registration/test_registration

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `registration_test_support.hpp` | Declares the helpers the tests of the registration module share. |
| `test_registration.cpp` | Tests what the vision registration adds: its rows, the MNIST loader and the YOLO26n model file against the reference. |
| `test_ultralytics_files.cpp` | Tests that the yolov8n and yolo11n model files build, and what their task sections state. |
| `test_yolov5n.cpp` | Tests that the yolov5n model file builds, and what its task and biases state. |
| `test_yolox_x.cpp` | Tests that the yolox_x model file builds, what it publishes, and that its name map fills every record of the model. |
| `test_yolox_x_import.cpp` | Tests that ByteTrack's mot17x checkpoint imports into the yolox_x model with nothing refused. |
