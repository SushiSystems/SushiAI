# Layered execution

**Status:** Open — designed and ruled; nothing built
**About:** SushiAI's share of the stack's layered execution: it stops using SushiBLAS's engine graph, and it trains one model over the owner's five machines through SushiRuntime's distributed...
**Modules:** applications/programs, modules/core/execution

## Problem

SushiAI's share of the stack's layered execution: it stops using SushiBLAS's engine graph, and
it trains one model over the owner's five machines through SushiRuntime's distributed layer,
with SushiBLAS carrying the tensor side. This document also holds the programme's backlog entry,
since SushiAI is the repository whose run needs every other share.

## Decision

Made on 2026-10-19.

| Subject | Ruling |
| --- | --- |
| Route | The run goes through SushiRuntime's distributed API, and SushiBLAS is the library that carries it |
| Grain | SushiBLAS does not offload its operations one at a time |
| Machines | Five, with devices of different kinds holding 6 to 12 GB each, running Linux and Windows |
| Network | 1 Gbps |
| The layer on the machines | SushiRuntime's distributed layer has run on them |
| Dataset | Every machine holds its own copy, for now |
| Repeatability | A run repeats bit for bit on the same hardware |
| Gradient types | The exchange works for every data type |
| New parts | SushiBLAS's `modules/engine/cluster/` and `ordered_sum()`, SushiAI's `modules/core/distribution/`, the program `sushiai_worker`, the options `SB_ENABLE_DISTRIBUTED` and `SA_ENABLE_DISTRIBUTED` |
| Placement | Written as placement rules the runtime evaluates, never as a policy class in SushiAI |
| SushiBLAS's engine | Keeps no graph; SushiAI records only on graphs it owns |
| First run | YOLOv3-tiny on the five machines |
| Order | SushiRuntime's share first, SushiBLAS's second, SushiAI's third |

## Design

### The shares of the stack

Each repository of the stack holds its own share in a document of this name:

| Repository | Its share |
| --- | --- |
| SushiRuntime | Placement class and key, scopes, topology, placement rules, the distributed layer's needs (R1 to R14) |
| SushiBLAS | An engine without its own graph, the per-index seam, two general operations, the cluster module (B1 to B13) |
| SushiAI | This document (A0 to A7) |
| SushiEngine | Two lanes behind one contract: SushiRuntime and SushiBLAS, or the engine's native runtime and BLAS; placement scopes; physics islands on several machines in the runtime lane (E1 to E17) |

### Roles

| Layer | Role in a training run |
| --- | --- |
| SushiRuntime | Holds the graphs and the memory, decides where each node runs by the rules it was given, moves the bytes between machines |
| SushiBLAS | Computes every operation of the step, packs tensors into flat buffers and adds the replicas' gradients in a fixed order |
| SushiAI | Owns the compiled step, splits the batch into replicas, records the exchange around the optimizer, and turns its training spec into placement rules |
| The person running it | Writes the training spec: which machine runs which replica, and any rule of their own |

### What the distributed layer is today

Every fact below was read from SushiRuntime's source on 2026-10-19.

- **A star around one master.** A master is a runtime with a `Cluster` attached; a worker is a
  process that has registered kernels by name
  (`sushiruntime/modules/extension/distributed/README.md`, "cluster.hpp"). There is no
  connection between two workers.
- **Task offload, not collectives.** A node of the master's graph leaves the machine when it
  carries a nonzero `OperationID`, an alive worker advertises that operation, every buffer it
  touches is shared and it writes no buffer twice
  (`offload_coordination/source/offload_dispatcher.cpp`, `place_fixed()`). No all-reduce,
  broadcast, gather or barrier across machines exists in the tree.
- **The master owns every shared byte.** A worker pulls each input from the master at the
  handle's version and streams each output back into the master's buffer.
- **A worker keeps what it pulled.** `WorkerBufferCache` holds buffers under a byte budget and
  asks only for a version it does not hold.
- **Host-addressable memory only.** `Cluster::share()` stops the process for a device-resident
  handle. SushiAI's `reserve_storage()` always allocates `Residency::Shared`
  (`modules/core/allocation/source/allocation.cpp`), so every tensor of a step can be shared as
  it is.
- **A kernel is a function on a queue.** A registered kernel is
  `sycl::event(sycl::queue&, span<const BufferView> inputs, span<const BufferView> outputs,
  const NodeOptions&)`, and `NodeOptions` carries up to 8 `u64` parameters.
- **Every task has a local fallback.** A task no worker takes, or one taken back after a lost
  worker or the 30 s `task_deadline`, runs on the master.
- **Nothing is measured** (sushiruntime RW-21, RW-22).

What the runtime adds for this run (placement rules that pin a class and key to a named worker,
worker labels, a per-rule deadline, the two probes) is SushiRuntime's `LAYERED_EXECUTION.md`.

### One step on five machines

```
                         master (replica 0)
            ┌───────────────────────────────────────────┐
            │ parameter pack ──► 5 replica tasks         │
            │                     │  ▲                   │
            │      ordered_sum ◄──┘  │ gradient packs    │
            │          │             │                   │
            │      optimizer ──► parameter pack (v+1)    │
            └──────┬──────▲──────────────────────────────┘
       parameters  │      │  gradients        (1 Gbps, one link)
          ┌────────┼──────┼────────┬─────────────────┐
          ▼        ▼      │        ▼                 ▼
     replica 1  replica 2 │   replica 3          replica 4
     (worker)   (worker)  │   (worker)           (worker)
     own dataset copy, own device, forward + backward on its rows
```

The master is one of the five machines and trains a part of the batch itself.

1. **Publish.** The parameters sit in one flat shared buffer on the master, the parameter pack.
   The optimizer's last task writes it, which raises its version.
2. **Dispatch.** The master's step records one replica task per replica inside the placement
   scope `sushiai.replica` with the replica's index as key. Its `NodeOptions` parameters are the
   replica index, the replica count, the run index and the seed, and its second input is the
   position pack that holds the epoch and the batch index. The
   rules the training spec produced send replica `k` to the worker named for it; one whose worker
   is lost runs on the master through the same kernel, or waits, as the spec says.
3. **Replica work.** A replica pulls the parameter pack once per step (the cache holds it until
   the version moves), unpacks it into its own model state, reads its rows from its own copy of
   the dataset, runs its micro-batches through the compiled forward and backward step,
   accumulating their gradients, and packs its gradients, its loss and its batch statistics into
   its output.
4. **Sum.** The master adds the gradient packs in replica order, each scaled by its share of the
   batch's rows. The order is fixed by the graph's edges, never by arrival.
5. **Update.** The loss scaler's screen, the gradient clip and the optimizer run on the master on
   the summed gradients, through the `Optimizer` they run through today. Batch statistics are
   summed with the same weights and blended by `BufferUpdateRecorder`.

The batch never crosses the network. The epoch permutation and the augmentation are already
functions of the seed, the epoch and the index
(`modules/core/data/include/SushiAI/data/epoch_permutation.hpp`, `docs/design/DATASETS/README.md`), so
every machine finds its own rows.

#### Devices of different sizes

The five devices hold 6 to 12 GB and run at different speeds. The cluster section of the training
spec gives each replica two numbers: its rows per step `r_k` and its micro-batch `m_k`, with `m_k`
dividing `r_k`. A 6 GB device takes a smaller micro-batch and more of them; a faster device takes
more rows. The global batch is `B = Σ r_k`, and the sum weights replica `k` by `r_k / B`, so the
update is the one a single machine would make over all `B` rows, whatever the split. A7 measures
each machine's rows per second and sets `r_k` from it.

#### What the link carries

With 4 workers and `P` bytes of gradients, the master sends `4P` and receives `4P` per step over
its one 1 Gbps link. The table is arithmetic at line rate (117 MB/s of payload), not a
measurement; the runtime's R13 and this document's A7 replace it with measured numbers.

| Model | Parameters | `P` in float32 | Each direction |
| --- | --- | --- | --- |
| YOLOv3-tiny | 8.9 M | 35 MB | 1.2 s |
| Language model, reference scale | 10.7 M | 43 MB | 1.5 s |
| YOLOv4 | 64 M | 256 MB | 8.8 s |
| GPT-2 small | 124 M | 496 MB | 17 s |

At 1 Gbps the exchange is the cost to beat. Three things lower it:

1. **More rows per exchange.** The micro-batches of step 3 let a replica take many rows for one
   exchange.
2. **Gradients in their own type.** A model whose gradients are `Half` sends half the bytes.
3. **Worker-to-worker links.** A ring all-reduce would cut each machine's traffic to `1.6P` per
   direction and spread it over five links. It needs links the layer does not have; the
   runtime's R14 holds it, and it is designed only if A7 measures the master's link as the bound.

Whether five machines beat the fastest one alone is not known until A7. A YOLOv3-tiny step on one
device has not been timed on the GPU; if it takes well under the 1.2 s above, the exchange
dominates and only the three items above can change the answer.

#### Data types

A pack holds tensors of any of SushiBLAS's eight types (`Half`, `Float32`, `Float64`,
`Complex32`, `Complex64`, `Int32`, `Int64`, `UInt32`); the exchange sums the seven that
`ordered_sum()` takes and refuses a `UInt32` result, a count type nothing computes on. It groups them into one contiguous segment per
type, and a gradient crosses the network in its tensor's own type. `ordered_sum()` accumulates
`Half` in `Float32` and rounds once after the last input; the other floating and complex types
accumulate in their own type; integers add exactly, wrap, and refuse a weight other than 1.

#### Faults

A worker lost mid-step has its replica task returned to the master, which runs it, unless the
spec's rule says `wait` or `fail`. A worker whose model differs from the master's cannot take a
replica task: the replica kernel's operation name carries the fingerprint of the model and its
pack layouts (`sushiai.replica:<fingerprint>`), so its `OperationID` differs and the master never
sees it advertised.

#### Repeatability

A run repeats bit for bit on the same hardware when replica `k` runs on the same machine every
run and the sum's order is fixed:

- the spec names a worker per replica, and the rules it produces pin `sushiai.replica` key `k`
  to that worker;
- every replica's seeded draws take the seed, the replica index and the run index;
- `ordered_sum()` adds in replica order and its kernel writes the order down.

A rule with `if_absent = local` lets a lost worker's replica run on the master, which is other
hardware, so from that step the run is not repeatable and the master logs one WARN naming the
replica and the step. A spec that must repeat writes `wait` or `fail`. Against a single-machine
run over the whole batch the result is equal within a tolerance, never bitwise.

#### Machines on two operating systems

The frames of the layer are little-endian with fixed-width fields
(`sushiruntime/modules/extension/distributed/README.md`, "Protocol versions"), and every device
of the five is little-endian. A layout's fingerprint includes the size of each type, so two
builds that disagreed on one would refuse each other instead of exchanging wrong bytes.
`sushiai_worker` builds on both systems.

### SushiBLAS

SushiBLAS's share is its own `LAYERED_EXECUTION.md`. What this run takes from it: `TensorPack`,
`ClusterBinding`, `record_remote()` and `bind_views()` in `modules/engine/cluster/`, and
`ElementwiseOps::ordered_sum()` (its B11 and B12); and an `Engine` without a graph (its B5), which
this document's A0 prepares.

### SushiAI

```
sushiai/
    modules/core/distribution/              built only when SA_ENABLE_DISTRIBUTED is on
        CMakeLists.txt
        README.md
        include/SushiAI/distribution/
            replica_plan.hpp                ReplicaPlan
            replica_draws.hpp               ReplicaDraws: the seed of a replica's draws
            replica_task.hpp                ReplicaTask: the replica node's four parameters
            replica_position.hpp            ReplicaPosition: the epoch and batch a step serves
            gradient_exchange.hpp           GradientExchange
            parameter_publish.hpp           ParameterPublish
            replica_model.hpp               ReplicaModel: the model a step runs, behind an interface
            replica_step.hpp                ReplicaStep
            replica_kernel.hpp              ReplicaKernel
            gradient_exchange.hpp           GradientExchange
            parameter_publish.hpp           ParameterPublish
            cluster_section.hpp             ClusterSection
        source/
            one .cpp per header
    applications/programs/source/
        sushiai_worker.cpp                  the worker program
    tests/core/distribution/                two replicas on InProcessFabric
```

| Brick | Does one thing | Runs on |
| --- | --- | --- |
| `ReplicaPlan` | From the spec and the batch index, says which rows replica `k` takes, in which micro-batches, with which weight | every machine |
| `ReplicaStep` | Owns a compiled forward and backward step with no optimizer, the replica's model state and its accumulated gradients | every machine |
| `ReplicaKernel` | The registered kernel: unpacks parameters, asks `ReplicaPlan` for its rows, runs `ReplicaStep` once per micro-batch, packs the result. It composes the three and decides nothing itself | every machine |
| `GradientExchange` | A `Graph::StepRecorder` that records the replica tasks inside the `sushiai.replica` scope, records the ordered sum and unpacks it into the gradient tensors the optimizer reads | master |
| `ParameterPublish` | A `Graph::StepRecorder` that packs the parameters after the optimizer | master |
| `ClusterSection` | Reads and checks the spec's cluster section (listen address, secret file, and per replica a worker name, rows and micro-batch, and what to do when its worker is absent) and turns it into the runtime's `PlacementRules` | master and worker |

`TrainingStep` gains no knowledge of machines: it takes `GradientExchange` before the optimizer
and `ParameterPublish` after it, through the recorder list it already has. `Trainer` is
unchanged. The `train` program builds the bricks when the spec has a cluster section, and
`sushiai_worker` loads the same model file and spec, builds `ReplicaStep` and `ReplicaKernel`,
and serves through `WorkerAgent`. `sa build --distributed` sets the three options.

#### What A2 settled

`ReplicaStep` runs its compiled step on a `Runtime` of its own (one scheduler thread, floating
workers), because SushiRuntime refuses a graph run of the runtime a registered kernel runs in
(its R11). It reads the model only through `ReplicaModel`, since `TrainingGraph` and `ModelState`
live in the layers tier above `core`; the worker program implements it (A4). The replica node's
parameters are, in order, the replica index, the replica count, the run index and the seed; the
epoch and the batch travel in a position pack, the node's second input, because a compiled step
replays its nodes' parameters as they were recorded and the position changes every step. The result pack holds the gradients in parameter order, the loss, then
the batch statistics. The kernel's fingerprint folds the model's own fingerprint with the
parameter and result layouts. Gradients over several micro-batches are averaged on the host, in
micro-batch order, since the pack and the gradients live in two runtimes' contexts; one
micro-batch is copied byte for byte.

#### What A3 settled

`GradientExchange` and `ParameterPublish` are `StepRecorder`s that `TrainingStep` takes as
`before_update` and `after_update`, around the optimizer. A step recorder records through one
`Graph::StepRecording`, which hands out both the SushiBLAS `TaskRecorder` and the SushiRuntime
`Recorder` of the graph, so the exchange opens placement scopes and records remote tasks without
an interface any recorder refuses. Integer gradients and statistics sum with weight 1, the only
weight `ordered_sum()` takes for integers, so their totals are exact; floating ones take
`r_k / B`. The epoch and batch come from a replay counter that resume sets (A4).

#### SOLID

| Principle | How the design holds it | Where it is weakest |
| --- | --- | --- |
| One responsibility | Each brick does one thing; `ReplicaKernel` composes three bricks and owns no rule | `ReplicaStep` owns both the compiled step and the accumulated gradients; they share one lifetime, so they stay one class |
| Open to extension | Placement is data, so a new arrangement of machines is a new spec, not code; a ring all-reduce replaces `GradientExchange` behind the same `StepRecorder` interface | |
| Substitution | `GradientExchange` and `ParameterPublish` are `StepRecorder`s like the optimizer | |
| Narrow interfaces | SushiAI sees `TensorPack`, `record_remote()` and `bind_views()`, never a raw `BufferView` address or a frame; `ClusterBinding` exposes only what the bricks use | |
| Depend on abstractions | SushiAI depends on SushiBLAS's module and the runtime's public rule model; no layer reaches `Internal::` | |
| Dependencies | The three options keep the distributed layer out of every build that does not ask for it | |

### Out of this document

- **Per-operation offload.** Ruled out by the owner on 2026-10-19.
- **Expert parallelism.** Mixture of experts in `LANGUAGE.md` phase 7 runs every expert on every
  token on one machine; under this design each replica holds all of it.
- **Model and pipeline parallelism.** A model whose replica does not fit the smallest device,
  6 GB, at a micro-batch of one row is not covered.
- **A dataset that lives on one machine.** Every machine holds a copy for now.

## Phases

Hierarchy first. Each task has one acceptance criterion and one build; the manual pages a task
makes false change in its own commit.

| Id | Phase | State |
| --- | --- | --- |
| A0 | SushiAI records only on graphs it owns: `run_recorded()` and `discard_recorded()` of `modules/core/execution` replaced, the checkpoint savers and the tests that used the engine's graph moved. Files: `modules/core/execution`, `modules/layers/io`, `tests/`. Waits on: nothing. Done when: No SushiAI file calls `Engine::execute()`, `discard_recorded()` or `Engine::recorder()`, and the suites pass | building |
| A1 | `ReplicaPlan` and `ClusterSection`, the spec's cluster section turned into placement rules. Files: `modules/core/distribution/`. Waits on: `sushiruntime` R4. Done when: A test turns a five-replica section into rows, micro-batches, weights and rules, and refuses a micro-batch that does not divide its rows | building |
| A2 | `ReplicaStep` and `ReplicaKernel`. Files: `modules/core/distribution/`. Waits on: A1, `sushiblas` B12, `sushiruntime` R11. Done when: One replica on `InProcessFabric` returns the gradients of a single-machine step over the same rows, bit for bit | building |
| A3 | `GradientExchange` and `ParameterPublish` in `TrainingStep`'s recorder list. Files: `modules/core/distribution/`, `modules/layers/train/`. Waits on: A2, `sushiblas` B11, `sushiruntime` R10, R12. Done when: Two replicas on `InProcessFabric` give summed gradients equal to one step over the whole batch within the test's tolerance, two pinned runs equal each other bit for bit, and a pack of every type crosses | building |
| A4 | `sushiai_worker` and the `train` program's cluster section; `sa build --distributed`. Files: `applications/programs/`, `cli/`. Waits on: A3. Done when: Both programs build and a two-process run on one machine trains one step from a spec | planned |
| A5 | Two machines, one Linux and one Windows. Files: `docs/guides/`. Waits on: A4. Done when: A YOLOv3-tiny step runs across the two and the guide page shows how | planned |
| A6 | Five machines, YOLOv3-tiny. Files: `docs/guides/`, this document. Waits on: A5, `sushiruntime` R13. Done when: The step time on five machines and on the fastest one alone are measured, with the link's time separated out, and `r_k` set from each machine's rows per second | planned |
| A7 | Decide on worker-to-worker links. Files: this document. Waits on: A6. Done when: The measurements of A6 are written here, and the runtime's R14 is either opened or withdrawn | planned |

A0 has no dependency and can land first; SushiBLAS's B5 waits on it. The owner chooses inline or
waves.

### Work

<!-- work:begin -->
- [2026_10_24_LAYERED_EXECUTION](../../agent/2026_10_24_LAYERED_EXECUTION/PLAN.md): A0, A1, A2, A3, A4, A5, A6, A7
<!-- work:end -->

## Open questions

None.
