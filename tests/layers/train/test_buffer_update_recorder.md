# tests/layers/train/test_buffer_update_recorder.cpp


Suites: `Small_BufferUpdateRecorder`.

`bind_buffer_updates` pairs each update's target buffer in the model state with its source value in
the pool. The cases check the two refusals that binding makes. The blend the recorder replays is
tested in `tests/core/graph/test_buffer_update_recorder.cpp`, since the recorder lives in
`core/graph`. Every case runs on the engine of a `HostExecutionBackend`.

## BindRefusesATargetTheStateLacks

The state holds one entry, `bn.running_mean`, under identity 1. The update targets identity 7.
Binding is refused as `Incompatible`, and the message contains "no entry".

## BindRefusesADTypeMismatch

The target entry is float32 and the source tensor is float64. Binding is refused as
`Incompatible`, and the message names both dtypes by their `dtype_name`.
