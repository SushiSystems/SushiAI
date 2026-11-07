# tests/core/graph/compile_step_recording.cpp


A compile-only unit: it holds no test and the runner never lists it. It checks at build time that
`SushiAI/graph/step_recording.hpp` compiles on its own.

The file needs no device and runs nothing. It keeps `step_recording.hpp` under a file of its own in
this folder, which holds one test file per header. `Graph::StepRecording` holds two pointers and
returns them; the compiled-step and step-recorder tests drive it, and the detached recording is
tested in `test_step_recorder.cpp`.
