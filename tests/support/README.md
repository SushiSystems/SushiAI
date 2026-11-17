# tests/support

Helpers that the tests of several modules share. A test includes one as
`<support/<name>.hpp>`, and only tests do: the folder is on the include path of the test programs and
of nothing else. Each helper sits in the namespace `SushiAI::Test`. A helper with a `.cpp` beside its
header, such as `free_detection_loss_host`, is listed in the `SOURCES` of each test program that uses it. The oracle's own check, `test_free_detection_loss_host.cpp` ([`test_free_detection_loss_host.md`](test_free_detection_loss_host.md)), sits beside it and is listed in the `SOURCES` of `tests/vision/training`.

| Header | Holds |
| --- | --- |
| `free_detection_loss_host.hpp` | `FreeLevelLossArrays` and `evaluate_free_detection_loss`, the double-precision host oracle of the anchor-free detection loss; its definitions are in `free_detection_loss_host.cpp` |
| `gradient_probe.hpp` | `GradientProbe`, a FLOAT64 graph whose loss and gradient a test evaluates at a point, and `expect_gradient_matches`, which compares the gradient with central differences; the build callback names the variable and returns the log-probabilities the loss reads, and `GradientProbeOptions` runs the fusion pass or fills every other leaf with noise |
| `host_row_values.hpp` | `HostLeaf` and `evaluate_on_host`, which evaluate a traced graph on the host in double, write the host tables a trace declared, and return the elements of one value |
| `imaging_images.hpp` | `make_constant` and `make_flat_image`, which build an RGB image of one colour and an image of one level, and `expect_same_image` and `expect_bytes`, which compare an image with another image and with a list of bytes |
| `in_memory_late_increment_recorder.hpp` | `InMemoryLateIncrementRecorder`, a step recorder that records one late-count task; it launches a kernel, so only device-compiled tests include it |
| `padded_text_comparison.hpp` | `leading`, `padded_batch` and `expect_close`, which cut rows out of a batch result, join a short text, its padding and a full text into one batch, and compare a computed row with the same text run alone within the bound of a float or a double path |
| `reference_detections.hpp` | `Detection`, `read_expected`, `comes_first` and `expect_same_detections`, which read the detections of a reference file, order them by class and falling score, and compare a runtime's detections with them within `K_ROW_BOUND`, 2048 u of the input side; the Ultralytics and the YOLOX detection tests share them |
| `scratch_names.hpp` | `unique_scratch_name`, which puts a random hex tag before the extension of a scratch file or folder name, so no two cases or shard processes share one |
| `test_names.hpp` | `parameter_name`, which spells a row name as a gtest parameter name by turning every character that is not a letter or a digit into `_` |
