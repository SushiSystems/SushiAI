# Heterogeneous step: plan

**Status:** Open — implemented and syntax-checked on 2026-11-25; the orchestrator builds, runs the
tests and the profile.

The owner's requirement of 2026-11-25: one training step can run on one machine split across its
CPU and its GPU, as an option that is off by default. This plan records why the CPU takes no work
today, the option chosen, and the tasks that build it.

## Why the CPU takes no work

The YOLOv4 profile record opens device 0, the RTX 3080 Ti (CUDA), and device 1, the Ryzen 7600X
(OpenCL CPU), and none of the 5302 tasks of a step ran on device 1. Five blockers hold together;
removing one leaves the others.

| Id | Blocker | Where |
| --- | --- | --- |
| B1 | The partitioner runs only with a device profile model or a journal replay. SushiAI sets no `RuntimeConfig::device_profile_file`, so `is_active()` is false and the graph gets no partition problem. | `sushiruntime/modules/surface/api/source/graph_partition.cpp:52`, `modules/engine/partition_session/source/partition_sessions.cpp:52-58` |
| B2 | The partitioner places only buffers with no owning device (`Residency::Planned`). SushiAI allocates `Device` or `Host` tensors, never `Planned`, so `planned_buffers` is 0. | `graph_partition.cpp:59`; `sushiai/modules/core/allocation/source/allocation.cpp:65-73` (`residency_of`) |
| B3 | A node is placed on the owner of its first write region (rule 2) unless it states a device. Every SushiAI tensor is created on device index 0: `reserve_storage` calls `tensor_factory().create` with no index, whose default is 0. The owner is device 0 before and after the residency change. | `allocation.cpp:92`; `sushiblas/modules/data/tensor/include/SushiBLAS/tensor/tensor_factory.hpp:115`; `sushiruntime/modules/engine/task/README.md:169` |
| B4 | Most SushiBLAS operations record raw addresses, so their tasks are `FixedToHome` and not movable. Only a task with bound handles is `Rebindable`. A GEMM states a split only when no operand is device-only, which after the residency change is no GEMM. `KernelVariant::GemmDepthSliced` (branch `perf/gemm-yolov4`, its plan's Decision 3) records addresses, takes no binding and has no split; several YOLOv4 dW GEMMs with deep K select it. | `sushiblas/modules/engine/engine/source/graph_task_sink.cpp:132`; `sushiblas/modules/operations/blas/include/SushiBLAS/blas/dispatch.hpp:277` |
| B5 | The rebalancer moves tasks between worker threads, not between devices, and it is off by default. No `set_work_migration` exists in SushiRuntime. | `sushiruntime/modules/engine/execution/README.md:438` |

Before the residency change every tensor was `Shared` on device 0, so B1, B2, B3 and B5 held
then too: the CPU never took work. `git log -S` over SushiAI finds no `device_profile_file`,
`Residency::Planned`, `place_on` or `rebalancer` ever written.

## The option

Each batch is split by rows across device classes, and each class runs the whole step over its
rows. This is data parallelism inside one machine. Task placement was not chosen: it needs B1 to
B4 removed, B4 is SushiBLAS-wide (binding by handle for every operation, a split for device-only
GEMMs and for `GemmDepthSliced`), and the list-scheduling partitioner gives no promise that both
devices take work.

The training file states it:

```json
"sushiai_train": 5,
"device_split": [{"device": "gpu", "rows": 12}, {"device": "cpu", "rows": 4}]
```

Without the key no new code runs: the runtime configuration, the trace, the run table and every
residency are as before, so the 0.47 s YOLOv4 step does not move.

## Decisions

| Id | Decision | Reason |
| --- | --- | --- |
| D1 | The option is the training-file key `device_split`, version 5 (`K_TRAIN_DEVICE_SPLIT_VERSION`, `K_TRAIN_VERSION` = 5). No command-line flag. | The training file describes the run, as `cluster` does; `train.cpp` keeps its one flag, which says how to measure, not what to run. |
| D2 | `Config::DeviceClass { GPU, CPU }`, `Config::DeviceShare { device, rows }`, `TrainSpec::device_split`, `Config::device_class_name`. Refused: fewer than two shares, a class twice, a class other than `gpu` and `cpu`, rows below 1, the key beside `cluster` (`Unsupported`). | A class, not a device index: the index of a device changes with the runtime's filter and the machine, the class does not. Two classes exist, so the set is closed and an enum fits (default 1, "not when"). A split inside a cluster has no defined order of sums, so it is refused rather than guessed. |
| D3 | Each share runs on a runtime of its own, narrowed to its class (`Apps::narrow_to_class`). The first share runs on the program's runtime, which `train.cpp` narrows with `narrow_to_split`; every other share is an `Apps::ShareStep`. | With one class per runtime the share's device is device 0, where `allocate_tensor` already puts every tensor (B3). No SushiBLAS or SushiRuntime change is needed, and no allocation call site learns a device index. |
| D4 | The first share's step is one `Train::TrainingStep` with a host stage that ends at the last gradient and the loss. The host stage waits for the other shares and copies their gradients in; the rest blends them with `Graph::BufferUpdateRecorder` and applies the optimizer in the plan. | The other shares run while the first replays its forward and backward. One compiled step keeps the `compiles: 1` contract. The blend is the recorder that already computes `(1 - m) * target + m * source`; with m = rows of the share over rows blended so far, the result is the mean gradient over the batch, so no new recorder is written. |
| D5 | Gradients and parameters cross through host memory (`copy_to_host`, `copy_from_host`), once per update each. | CUDA and OpenCL share no context and SushiRuntime's peer path is DPC++ only, so a copy between them passes through the host whatever does it. An OpenCL CPU device's memory is host RAM, so the cost is P floats each way, what `HostStagedPath` would move. |
| D6 | `prepare_run` traces the first share's setup over its rows when the file splits; its signature does not change. | The training file is already its input, and a second parameter would let a caller trace rows the file does not state. |
| D7 | `RowFeed` reads rows of a batch into a setup's leaves; `TrainingReplicaModel::feed` and both kinds of share call it. | The replica's feed was the only copy of that sequence; a second copy would drift. |
| D8 | An objective with a run of its own (`anchor`, `anchor_free`) is refused as `Unsupported`. YOLOv4 trains through `anchor_free`, so it cannot split yet. | Those runs own their step shape (the anchor run's forward-only step, the free run's loss terms); splitting them is their own task. |
| D9 | Running statistics are blended from the first share's rows; the other shares' buffers are not written. | Batch-norm statistics of a share are statistics of fewer rows; the mean of two shares' running statistics is not the full batch's either. The checkpoint holds the first share's state. |
| D10 | `RunSummary::shares` (`Apps::ShareRun`: class, device name, rows, replays) and one printed `share` line per share. | The end-to-end test and the profile read which device stepped; a run without a split reports none and prints no line. |

## Tasks

Hierarchy order: the configuration first, the runtime narrowing and the feed next, the shares and
the run over them last.

| Task | Files | Done when |
| --- | --- | --- |
| T1 Configuration key | `modules/core/config/include/SushiAI/config/train_config.hpp`, `source/train_config.cpp`, `tests/core/config/test_train_config.cpp` | The eight `Small_TrainConfig` split cases pass and `AFutureTrainVersionIsAnError` refuses version 6 |
| T2 Runtime narrowing | `applications/programs/source/share_runtime.{hpp,cpp}`, `train.cpp`, `tests/applications/programs/source/test_share_runtime.cpp` | The three `Small_ShareRuntime` cases pass, the first showing a file without a split leaves the configuration unchanged |
| T3 Row feed | `applications/programs/source/row_feed.{hpp,cpp}`, `training_replica_model.{hpp,cpp}`, `tests/applications/programs/source/test_row_feed.cpp` | `Medium_RowFeed.AFeedCopiesItsRowsIntoTheInputs` passes, and the existing `Medium_TrainingReplicaModel` cases pass unchanged |
| T4 Share step | `applications/programs/source/share_step.{hpp,cpp}`, `tests/applications/programs/source/test_share_step.cpp` | `Medium_ShareStep.AShareReplaysItsStepOncePerBatch` passes: a CPU share reports 2 replays after two batches |
| T5 Split run | `applications/programs/source/device_split_run.{hpp,cpp}`, `run_preparation.{hpp,cpp}`, `training_run.hpp`, `training_program.cpp`, `applications/programs/CMakeLists.txt` | `Medium_DeviceSplitRun.TheSplitStepRunsOnTheGpuAndOnTheCpu` passes: two shares, GPU then CPU, 4 replays each |
| T6 Same loss | `tests/applications/programs/source/test_device_split_run.cpp` | `TheSplitStepReachesTheGpuOnlyLoss` passes: the last loss of a 6 + 2 split is within 1e-3 of the GPU-only run's |
| T7 Off is as today | same test file | `WithoutASplitTheRunReportsNoShare` passes, and every existing programs test passes unchanged |

The build and the test run come once, after T7. The `Medium_DeviceSplitRun` cases skip on a
machine without both a GPU and a CPU device.

## Expected numbers

- `sa test --type release -j 7`: the eight new `Small_TrainConfig` cases, three
  `Small_ShareRuntime`, two `Medium_RowFeed`, three `Medium_ShareStep` and four
  `Medium_DeviceSplitRun` cases pass on the owner's machine
  (RTX 3080 Ti and Ryzen 7600X both visible); no existing case changes result.
- The YOLOv4 profile with the option off: the step stays at 0.47 s, since no line it runs
  changed. YOLOv4 cannot run with the option on (D8).
- The tiny run with a 6 + 2 split: the loss of the last epoch within 1e-3 of the GPU-only run, two
  `share` lines with 4 steps each.

## Requests to other repositories

None is needed for this option. A placement-based split, where one runtime holds both devices and
tasks of one graph run on each, needs:

- SushiBLAS: binding by handle for every operation, a split for a GEMM with a device-only operand,
  and a split or a binding for `GemmDepthSliced` (B4).
- SushiBLAS: an engine default device index, or a device index on `allocate_tensor`'s path, so a
  tensor can be created on a device other than 0 (B3).
- SushiRuntime: partitioning without a device profile file, from a cost model it builds itself
  (B1), and placement of `Device` buffers, not only `Planned` ones (B2).

## Documentation

| Page | Lines |
| --- | --- |
| `modules/core/config/README.md` | the header table row of `train_config.hpp`; "Training file versions": versions 1 to 5, the `device_split` paragraph |
| `applications/programs/README.md` | four rows of the source table after `run_preparation`; new section "Splitting a step across devices" before "The compile-count line" |
| `tests/core/config/README.md` | eight new cases under `test_train_config.cpp` |
| `tests/applications/programs/README.md` | sections `source/test_share_runtime.cpp`, `source/test_device_split_run.cpp`, `source/test_row_feed.cpp` and `source/test_share_step.cpp` |

`docs/reference/CHANGELOG.md` and the design topic `LAYERED_EXECUTION` are the orchestrator's.

## Follow-up

- Split an objective with a run of its own, `anchor_free` first, so YOLOv4 can split (D8).
- Measure a split step of a real model against the GPU-only step; the copies of D5 cost two
  transfers of every parameter per update, which a small CPU share may not repay.
- Pick a share's rows from measured throughput instead of the file, once a measurement exists.
