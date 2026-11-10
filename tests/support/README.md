# tests/support

Header-only helpers that the tests of several modules share. A test includes one as
`<support/<name>.hpp>`, and only tests do: the folder is on the include path of the test programs and
of nothing else. Each helper sits in the namespace `SushiAI::Test`.

| Header | Holds |
| --- | --- |
| `imaging_images.hpp` | `make_constant` and `make_flat_image`, which build an RGB image of one colour and an image of one level, and `expect_same_image` and `expect_bytes`, which compare an image with another image and with a list of bytes |
| `in_memory_late_increment_recorder.hpp` | `InMemoryLateIncrementRecorder`, a step recorder that records one late-count task; it launches a kernel, so only device-compiled tests include it |
| `scratch_names.hpp` | `unique_scratch_name`, which puts a random hex tag before the extension of a scratch file or folder name, so no two cases or shard processes share one |
| `test_names.hpp` | `parameter_name`, which spells a row name as a gtest parameter name by turning every character that is not a letter or a digit into `_` |
