# tests/layers/train/compile_training_graph.cpp


A compile-only unit: it holds no test and the runner never lists it. It checks at build time that
`TrainingGraph` in `modules/layers/train/include/SushiAI/train/training_graph.hpp` is move
constructible, since `build_training_graph` returns it by moving it into a `Result`.

`StepSpec`, `ParameterBinding` and `TrainingGraph` are plain data with no behaviour of their own.
The three cases that stated their default values were deleted: a default value is not a promise.
The builders fill these structs, and `test_steps.cpp` tests what they hold.

The file needs no device and runs nothing.
