# Device residency: plan

**Status:** Open — implemented and syntax-checked on 2026-11-25; the orchestrator builds and runs the tests.

A training step that reads a tensor the host wrote in Shared memory pays about 0.7 ms per
`cuLaunchKernel` on WDDM. Every tensor SushiAI allocates was `Residency::Shared`. After this
change a tensor that only kernels address lives in device memory, a tensor the host writes every
step lives in pinned host memory, and every other host access is an explicit copy.

## Decisions

| Id | Decision | Reason |
| --- | --- | --- |
| D1 | `enum class TensorAccess : std::uint8_t { Kernels, Host }` in `SushiAI/allocation/allocation.hpp`; `allocate_tensor` takes it last, default `Kernels`. | A caller states who addresses the tensor, not where it lives. The default is the common case, so the hundreds of callers that only feed kernels change nothing. |
| D2 | `residency_of(TensorAccess)` in `modules/core/allocation/source/allocation.cpp` is the one place a residency is chosen: `Kernels` gives `Device`, `Host` gives `Host`. Arenas are always `Kernels`. | One function owns the choice, so no caller branches on residency and a new backend changes one switch. |
| D3 | New public header `SushiAI/allocation/host_copy.hpp`: `copy_to_host`, `copy_from_host`, `read_elements` (range and whole-vector forms), `write_elements` (range and vector forms), `read_reals`, `write_reals`. | Every copy goes through `SushiBLAS::Storage::copy_to_host` and `copy_from_host`, which work for every residency, so one module checks range, width and contiguity once. |
| D4 | Library code dereferences a tensor address only for a tensor allocated `Host`: the batches of `EngineBatchAllocator`, `Optim::StepScalars` and the dropout position scratch of `lowering/random.cpp`. | These are written by the host on every step; pinned host memory costs the kernel a PCIe read, not a managed-memory fault. |
| D5 | Bulk reads replace per-element reads: `Train::host_real` becomes `read_host_reals(tensor, elements, what)`; `count_confusion` takes `span<const double>` logits and truth; vision `Detail::write_reals`/`read_reals` become `Detail::fill_reals(tensor, fill)`. | One copy per tensor instead of one host dereference per element, and the confusion count no longer knows where its numbers came from. |
| D6 | `TensorPool::create`, `TensorPool::scratch` and `LoweringContext::scratch` take `TensorAccess access = Kernels`. `scratch` refuses a `Host` request for a slot the plan placed in an arena, as `InvalidDescription`. `TensorSource::create` is unchanged; a source pool ignores the access. | Arenas are device memory; a host-written buffer inside one would be read by the host from device memory. Leaving `TensorSource` alone keeps `core/distribution` untouched. |
| D7 | Readbacks inside `const noexcept` interfaces: `EvaluationOnePositionStep` keeps `mutable std::vector<float> scores_`; `EvaluationWindowLogits` copies its logits in `run()`; `GradientClip::applied_scale`, `Trainer::current_loss` and `LossScaler::before_replay`/`after_replay` are fatal on a failed copy; `FreeTrainingStep::read_terms` returns `Result<FreeLossTerms>`. | The interfaces promise a value without a `Status`; a failed device copy there is a broken device, which the error contract makes fatal. |
| D8 | `Train::mark_leaves_unwritten` returns `Status`; it fills the leaves with one `copy_from_host` each, and `require_leaves_written` reads them back with `copy_to_host`. | The objective leaves are pool tensors, now in device memory. |
| D9 | `Apps::HostReadback` in `applications/programs/source/host_readback.{hpp,cpp}` holds the host copies of the outputs a detection report decodes; `refresh()` copies them once per batch. | The three reports bound spans over device addresses once and read them every batch; a readback object keeps the bind-once shape. |

No change in SushiBLAS or SushiRuntime: `Storage::copy_to_host`/`copy_from_host` and
`Residency::Host` already exist.

## Host access sites and what each became

Converted to explicit copies (all under `modules/` unless noted):

| Site | Was | Now |
| --- | --- | --- |
| `core/data/source/dataset.cpp:209` | means read through `data_as` | `read_elements<float>` |
| `core/eager/source/step.cpp:296` | `read_scalar` dereferenced | `read_reals` |
| `core/graph/source/lowering/mask.cpp:94` | triangle written in place | staged vector, `write_elements` |
| `core/optim/source/gradient_clip.cpp:99, 253` | arena cleared and scale read in place | `write_reals`, `read_reals` |
| `core/optim/source/loss_scaler.cpp:73, 219, 283` | factor, counts written and read in place | `write_elements`, `read_elements` over a whole-arena view |
| `layers/io/source/model_records.cpp:75, 87`, `run_checkpoint.cpp` | payloads streamed into `tensor.data()` | `write_tensor_record`, `read_tensor_payload` staging bytes |
| `layers/train/source/*_objective.cpp`, `objective.cpp:198` | `host_real` per element | `read_host_reals` |
| `layers/train/source/dense_targets.cpp:79`, `one_hot.cpp:103`, `row_mask.cpp:237`, `steps.cpp:170`, `in_memory_objective.cpp:61` | leaves written in place | `write_reals` / `write_elements` |
| `layers/train/source/leaf_writes.cpp:108, 129` | pattern written and checked in place | `copy_from_host`, `copy_to_host` |
| `layers/train/source/trainer.cpp:353` | loss read in place | `read_reals` |
| `language/decision_models/source/decision_session.cpp:55-101` | outputs per element | `read_host_reals` |
| `language/generation/source/evaluation_one_position_step.cpp:85`, `evaluation_window_logits.cpp:103` | logits span over the tensor | copy into `scores_` |
| `language/training/source/expert_balance_objective.cpp:168`, `next_token_targets.cpp:94` | read and write in place | `read_host_reals`, `write_elements` |
| `vision/importers/source/darknet_store.cpp:150` | weights permuted into the tensor | staged, `write_elements` |
| `vision/training/source/anchor_bias_init.cpp:184-194`, `free_bias_init.cpp:138` | bias written in place | read, modify, `write_elements` |
| `vision/training/source/anchor_training_step.cpp:230, 254`, `free_training_step.cpp:90-387` | heads read, leaves written in place | `read_elements`, `write_elements` |
| `vision/training/source/dice_targets.cpp`, `pixel_targets.cpp`, `segmentation_metrics.cpp:79` | leaves and classes in place | `write_elements`, `fill_reals`, `read_elements` |
| `vision/darknet_forward_cases`, `vision/darknet_training_cases` | inputs and outputs in place | `write_elements`, `read_elements` |
| `applications/programs/source/detection_report_{anchor,free,yolox}.cpp` | spans over outputs | `HostReadback::bind`, `refresh()` per batch |
| `applications/programs/source/inference_program.cpp:119` | scores read in place | `read_elements<float>` |
| `applications/programs/source/training_replica_model.cpp:276` | `mark_leaves_unwritten` call | `SA_TRY(...)`; not syntax-checked, distributed build only |

Allocated `TensorAccess::Host` and kept as host dereferences: `materialised_dataset.cpp:141`
(every batch, so `dataset.cpp:226-323`, `decision_dataset.cpp:129`, `next_token_collator.cpp:135`,
`prompt_dataset.cpp:165`, the generation feeds in `cached_generator.cpp:158`,
`ring_position_feed.cpp:89`, `table_position_feed.cpp:66`, `evaluation_window_logits.cpp:100`,
`in_memory_one_position_step.cpp:104`, and the label reads in `objective.cpp:209`,
`one_hot.cpp:88`, `dense_targets.cpp:72-95`, `next_token_inputs.cpp:62`,
`next_token_targets.cpp:90`, `decision_session.cpp:107`, `truth_from_roles.cpp:215`,
`applications/programs/source/detection_report.cpp:90`); `step_scalars.cpp:61`;
`lowering/random.cpp:135`. `loss_scaler.cpp:241` takes a device address for `gate()` and
dereferences nothing.

Left as they are: `modules/core/distribution` and `applications/programs/source/cluster_run.cpp`
(see "Follow-up").

## Documentation made false

DISPATCH lists no module README as writable, so these are left for the orchestrator with their
new text.

- `modules/core/allocation/README.md`, Owns: "Tensor and arena allocation on a
  `SushiBLAS::Engine`, views into an arena, and the host copies of a tensor's elements.
  `allocate_tensor` takes a `TensorAccess`: `Kernels`, the default, allocates device memory;
  `Host` allocates pinned host memory for a tensor the host writes every step.
  `residency_of` in `allocation.cpp` is the only place a residency is chosen."
  Public entry points: add `TensorAccess` to the `allocation.hpp` row and a row
  `SushiAI/allocation/host_copy.hpp` | `copy_to_host`, `copy_from_host`, `read_elements`,
  `write_elements`, `read_reals`, `write_reals`.
- `modules/core/tensor_values/README.md:8-12`: "`write_tensor` and `read_tensor` copy through
  `host_copy.hpp`, so they hold for a tensor in device memory as well as in host memory, and
  wait for the copy." Depends on: "`allocation`, for the host copies, `dtype`, and GTest."
- `modules/core/data/README.md:207-219`: "`allocate_tensor` ... builds a `SushiBLAS::Storage`
  with `TensorAccess::Host`, which is `Residency::Host`, pinned host USM ... Its `allocate`
  calls `sycl::aligned_alloc_host` ... Pinned host USM is host memory, so a worker writes it as
  it writes any buffer." Line 230: "79 MB of pinned host USM".
- `modules/core/optim/README.md:9`: "`StepScalars` holds the scalars the update kernels read
  from pinned host memory, which the host writes before every replay."
- `modules/layers/train/README.md:102`: `host_values.hpp` | `require_host_reals`,
  `read_host_reals`. Line 113: `count_confusion` (over spans of host doubles). Line 245: "The
  fill is one copy to the device of the bytes of the leaves, the targets of one batch."
- `modules/vision/training/README.md:157-158`: "The host stage copies the heads to the host and
  its leaves back with `read_elements` and `write_elements`, so both must be float32, as the
  loss and the logits `Train::Trainer` reads are."
- `modules/core/distribution/README.md:300-308`: false once KI-26 lets the module compile;
  see "Follow-up".

Design pages (not writable; reported): `docs/design/GRAPH_CAPABILITIES/RUNTIME.md:7` and `:17-19`
(the pool allocates Shared), `docs/design/GRAPH_CAPABILITIES/BLAS_BROADCAST_AND_OPERATIONS.md:340`
(SushiAI reads Shared tensors through `data_as`), `docs/design/LAYERED_EXECUTION/README.md:74`
(`reserve_storage()` always allocates `Residency::Shared`).

## Tests

| Task | Done when |
| --- | --- |
| `tests/core/allocation/test_allocation.cpp`: `AKernelsTensorLivesInDeviceMemory` replaces `AnAllocatedTensorIsAddressableByTheHost` | A default tensor's storage reports `Residency::Device` |
| `test_allocation.cpp`: `AHostTensorLivesInPinnedHostMemory` | A `TensorAccess::Host` tensor's storage reports `Residency::Host` |
| `tests/core/allocation/test_host_copy.cpp` (new): five cases | Round trip, range copy, real copies of both float types, wrong width and out-of-range refusals |
| `tests/core/graph/test_arena.cpp`: `AHostScratchBufferThePlanPutInAnArenaIsRefused` | `scratch` with `Host` on a planned slot fails `InvalidDescription` |
| `tests/layers/umbrella/test_SushiAI/compile_noexcept_surface.cpp` | `count_confusion`, `read_host_reals`, `allocate_tensor` with access and the six copies assert `noexcept` |
| `modules/core/tensor_values`: `write_tensor`, `read_tensor`, `read_tensor_bytes` copy | Every test that reads a device tensor through them compiles unchanged |
| Tests that dereferenced a pool, state or output tensor (about 25 files) | Each reads or writes through `read_elements`/`write_elements`; batch tensors built by hand use `TensorAccess::Host` |

## Follow-up

- `modules/core/distribution/source/host_access.hpp` `require_host_visible` stops the process for
  a tensor in device memory. Once KI-26 lets the module compile, `ReplicaStep::begin`, the
  micro-batch sum, `ParameterPublish::make` and `cluster_run.cpp:151` stop on every model tensor.
  They need copies through `host_copy.hpp`, or their tensors allocated `TensorAccess::Host`.
  The same holds for `Cluster::share()`, which refuses a device-resident handle.
- The dropout mask (RW-94) still fills a host position scratch; it now lives in pinned host
  memory, not Shared.
