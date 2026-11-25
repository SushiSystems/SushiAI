# Distributed residency: plan

**Status:** Open — implemented and syntax-checked on 2026-11-25; the orchestrator builds and runs the tests.

`docs/agent/2026_11_25_DEVICE_RESIDENCY/PLAN.md` moved every tensor a kernel addresses into
device memory. Four host copies in `modules/core/distribution` and the loss read in
`applications/programs/source/cluster_run.cpp` still dereference such a tensor, and
`require_host_visible` stops the process at each of them. This work routes each one through
`SushiAI/allocation/host_copy.hpp`, takes a pack's device from its tensors, and corrects the
pages the residency change made false.

## Decisions

| Id | Decision | Reason |
| --- | --- | --- |
| D1 | A pack a worker reads stays pinned host memory (`TensorPack` allocates `Residency::Host`), as the staging behind the gather and the scatter. No change to `Cluster::share()` or `ClusterBinding::share()`. | `Cluster::share_handle` in SushiRuntime stops on a handle the host cannot address, and the `share(handle, bytes)` form states the same precondition, because the master's transport reads and writes the shared allocation on the host. `TensorPack` already allocates `Residency::Host`, and `record_gather`/`record_scatter` are queue copies, which hold for a device tensor on one side and pinned memory on the other. Teaching the sharing path to accept device memory would put the same staging copy inside the transport, in a second repository, and change nothing the packs do. |
| D2 | Every host copy between a model tensor and host bytes goes through `host_copy.hpp`: `ReplicaStep::begin` uses `copy_from_host`, `ReplicaAccumulator::add` and `ParameterPublish::make` use `copy_to_host`, `cluster_run.cpp` reads the loss with `read_reals`. | The copies hold for every residency, so no call site branches on one and none names one. |
| D3 | `ReplicaAccumulator::add` returns `Status`; it copies each result into a host staging buffer of the layout's size, then sums from it. | A failed device copy is a `Status`, which `ReplicaStep::run_micro_batch` already returns. The staging buffer is allocated once, in the constructor. |
| D4 | `cluster_run.cpp`'s progress line reads the loss with `read_reals` and stops the process on a failed copy. | `EpochSteps::report` returns nothing; a failed device copy there is a broken device, as `Trainer::current_loss` treats it (D7 of the device residency plan). |
| D5 | `GradientExchange` allocates its position tensor with `TensorAccess::Host`; `before_replay` keeps writing its three words in place. | The host writes the tensor before every replay and the gather reads it on the queue, the case D4 of the device residency plan gives to pinned host memory. The class that allocates the tensor is the class that dereferences it. |
| D6 | `host_access.hpp` (`is_host_visible`, `require_host_visible`) is deleted with its test, its umbrella include and its `noexcept` assertions. | After D2 to D5 nothing calls it. |
| D7 | A pack is allocated on the device its tensors' storage names. `storage_device.hpp` becomes `require_pack_device(tensor, what, index)`, which returns that device or refuses a storage that names none (runtime-placed) as `InvalidDescription`. `pack_device`, `describe_device` and `K_ENGINE_DEFAULT_DEVICE` are deleted. | SushiBLAS's `record_gather` and `record_scatter` call `PackLayout::require_tensors`, which calls `Storage::require_device_index` and throws for a runtime-placed storage, and which refuses a tensor on another device than the pack. A pack on device 0 beside a runtime-placed tensor therefore fails at the first recording; the fallback only moved the failure from `make()` to the step. No other source of the device exists: `SushiBLAS::Engine` holds none, and `allocate_tensor` and `allocate_arena` call `TensorFactory::create` with its default device, which the storage records. SushiAI allocates no runtime-placed tensor today. |
| D8 | `ParameterPublish::make` takes the pack's device from parameter 0 and needs no device for an empty list. | `PackLayout::of` refuses an empty list and `PackLayout`'s default constructor is private, so a layout holds at least one entry, and the count check already refuses an empty parameter list. |

No change in SushiBLAS or SushiRuntime.

## Host access sites and what each becomes

| Site | Was | Becomes |
| --- | --- | --- |
| `modules/core/distribution/source/replica_step.cpp` `ReplicaStep::begin` | `require_host_visible`, `memcpy` into `target.data()` | `copy_from_host(target, 0, elements, width, source.data)` |
| `modules/core/distribution/source/replica_accumulator.cpp` `ReplicaAccumulator::add` | `require_host_visible`, sums read from `result.data()` | `copy_to_host` into `staging_`, sums read from `staging_` |
| `modules/core/distribution/source/parameter_publish.cpp` `ParameterPublish::make` | `require_host_visible`, `memcpy` from `parameters[index].data()` into the pack | `copy_to_host(parameters[index], ...)` into the pack's pinned bytes |
| `modules/core/distribution/source/gradient_exchange.cpp` `GradientExchange::before_replay` | `data_as<std::int64_t>()` on a device tensor | unchanged write; the tensor is allocated `TensorAccess::Host` |
| `applications/programs/source/cluster_run.cpp` `read_loss`, `run_on_cluster` | `require_host_visible`, `data_as` on the loss | `read_reals(loss)`, fatal on a failed copy |
| `tests/core/distribution/replica_test_support.hpp` `LinearModel::feed`, `fill_parameters`, `SingleMachineStep` | `data_as`, `memcpy` from `result.data()` | `write_elements`, `copy_to_host`; `fill_parameters` returns `Status` |
| `tests/core/distribution/test_gradient_exchange.cpp` `bytes_of` | `tensor.data()` | `copy_to_host` |
| `tests/core/distribution/test_parameter_publish.cpp` `fill`, `expect_pack_holds_parameters` | `data_as`, `memcmp` against `parameters_[index].data()` | `write_elements`, `read_elements` then `memcmp` |
| `tests/applications/programs/source/test_cluster_run.cpp` `bytes_of`, `values_of` | `data()`, `data_as<float>()` on the parameters | `copy_to_host`, `read_elements<float>` |
| `modules/vision/darknet_import_cases/source/darknet_import.cpp` `fill_imported_state`, `expect_same_state` | `memcpy` into and `memcmp` over state tensors | `copy_from_host`; `copy_to_host` then compare |
| `tests/layers/io/test_import_weights.cpp` `write_checkpoint`, `BytesInMemoryLoad...` | `writer.write(record, tensor.data())` | `read_elements<float>` then `writer.write` |
| `tests/vision/registration/test_registration/test_registration.cpp` weight loader | `read_payload(record, tensor->data())` | `read_payload` into staging bytes, then `copy_from_host` |

Host dereferences kept, each of pinned host memory: the packs' bytes (`TensorPack::data()`,
D1), a worker's bound views (`BoundPack`, `ReplicaPosition::read`), and the position tensor (D5).
`training_replica_model.cpp` and `sushiai_worker.cpp` dereference no tensor: the replica's feed
reads rows into a batch the dataset allocated `TensorAccess::Host`, and copies them into the
pool's input with a recorded `blas.copy`.

## Documentation

Module READMEs and design lines this work corrects, the ones the device residency plan lists as
false first:

- `modules/core/allocation/README.md`: Owns, the public entry points of `allocation.hpp` and
  `host_copy.hpp`.
- `modules/core/tensor_values/README.md`: the copies of `write_tensor` and `read_tensor`, and
  Depends on.
- `modules/core/data/README.md`: the residency of a batch.
- `modules/core/optim/README.md`: where `StepScalars` lives.
- `modules/layers/train/README.md`: `host_values.hpp`, `count_confusion`, the leaf fill.
- `modules/vision/training/README.md`: the host stage of the heads.
- `modules/core/distribution/README.md`: the entry-point table, `ParameterPublish`,
  `ReplicaStep`, "Host copies".
- `docs/design/GRAPH_CAPABILITIES/RUNTIME.md`, `BLAS_BROADCAST_AND_OPERATIONS.md`,
  `docs/design/LAYERED_EXECUTION/README.md`: the lines that state the residency.
- `tests/core/distribution/README.md`: the `test_host_access.cpp` row.
- `modules/core/distribution/include/SushiAI/distribution/parameter_publish.hpp` and
  `replica_model.hpp`: the doc blocks of `make`, `parameters()` and `results()` no longer require
  host-addressable tensors.

## Tests

| Task | Done when |
| --- | --- |
| `test_parameter_publish.cpp`: `MakeRefusesARuntimePlacedParameter` | `make` over a `TensorFactory::planned` tensor fails `InvalidDescription` naming the parameter |
| The distribution suites that read or wrote model tensors in place | They copy through `host_copy.hpp` and pass on device memory |
| `test_host_access.cpp` | Deleted with the header it tests |

## Follow-up

Recorded in the report for the orchestrator to file; see "Backlog findings" there.
