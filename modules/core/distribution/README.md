# distribution

## Owns

The cluster section of a training file and the split of a batch over the replicas it names.
`ClusterSection` reads and checks the section and turns it into SushiRuntime `PlacementRules`.
`ReplicaPlan` says which rows of a batch each replica takes, in which micro-batches and with
which weight. `ReplicaStep` runs one replica's compiled forward and backward step on a runtime of
its own, and `ReplicaKernel` is the registered kernel that serves a replica task with it. The
module is built only when `SA_ENABLE_DISTRIBUTED` is on. The design is
[`docs/design/LAYERED_EXECUTION/README.md`](../../../docs/design/LAYERED_EXECUTION/README.md).

## Depends on

The module declares `config` and `failure` as public dependencies and `seam` and `hash` as private ones.
It takes `SushiRuntime::placement` as an external public dependency, since `ClusterSection`
returns `SushiRuntime::PlacementRules`. `ReplicaPlan` includes nothing of SushiRuntime and no
SYCL header, so the machines that only need the split do not compile against the runtime.

`ReplicaStep` and `ReplicaKernel` add `graph` as a public dependency, since their headers name
`Graph::Graph`, `TensorPool` and `CompiledStep`, and `execution` as a private one, for
`create_runtime()`. `GradientExchange` adds `allocation` as a private dependency, for the packs it
allocates. They take SushiBLAS's `SushiBLAS::cluster` and SushiRuntime's
`SushiRuntime::distributed` as external public dependencies, since their headers name
`PackLayout`, `BoundPack`, `KernelRegistry` and `BufferView`. The module stays in the core tier: it reads a model only through `ReplicaModel`, so
it includes nothing of `layers/train`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/distribution/replica_plan.hpp` | `K_REPLICA_PLACEMENT_CLASS`, `ReplicaShare`, `RowPositions`, `ReplicaPlan` |
| `SushiAI/distribution/cluster_section.hpp` | `K_CLUSTER_KEY`, `ClusterSection` |
| `SushiAI/distribution/replica_draws.hpp` | `ReplicaDraws`, `replica_seed()` |
| `SushiAI/distribution/replica_position.hpp` | `K_REPLICA_POSITION_BYTES`, `ReplicaPosition` |
| `SushiAI/distribution/exact_sum.hpp` | `sums_exactly()`: Int32, Int64 and UInt32 results add as counts with weight 1 |
| `SushiAI/distribution/host_access.hpp` | `is_host_visible()`, `require_host_visible()`: the check before a host copy of a replica's tensors |
| `SushiAI/distribution/gradient_exchange.hpp` | `ExchangeRun`, `GradientExchange` |
| `SushiAI/distribution/parameter_publish.hpp` | `ParameterPublish` |
| `SushiAI/distribution/replica_task.hpp` | `K_REPLICA_TASK_WORDS`, `ReplicaTask` |
| `SushiAI/distribution/replica_model.hpp` | `ReplicaRows`, `ReplicaModel`, `ReplicaModelFactory` |
| `SushiAI/distribution/replica_step.hpp` | `replica_runtime_config()`, `ReplicaStep` |
| `SushiAI/distribution/replica_kernel.hpp` | `ReplicaKernel` |

## The cluster section

The section is the object under the `cluster` key of a training file. A file without the key
trains on one machine. A file with the key states every required key of the first table, and
every entry of `replicas` states the required keys of the second. An unknown key is refused, as everywhere
in a training file.

| Key | Required | Holds |
| --- | --- | --- |
| `listen` | yes | The address the master listens on, as text. The module checks that it is not empty; the runtime's cluster reads its grammar |
| `secret_file` | yes | The file that holds the cluster's secret. A relative path resolves against the training file's directory, as `model` does. The module does not open it |
| `attach_timeout_ms` | no | How long the master waits for every worker to attach, in milliseconds, above 0; 60000 (`K_DEFAULT_ATTACH_TIMEOUT`) when absent. A larger `wait_ms` of a replica raises it |
| `replicas` | yes | A list with at least one entry. Entry `k` is replica `k` |

| Key of a replica | Required | Holds |
| --- | --- | --- |
| `worker` | yes | The name of the machine the replica runs on, the name that machine reports for itself. Two replicas cannot name the same machine |
| `rows` | yes | `r_k`, the rows the replica takes of each batch. Zero lets the replica sit out |
| `micro_batch` | yes | `m_k`, the rows of one micro-batch. At least 1, and it divides `rows` |
| `if_absent` | yes | `local`, `wait` or `fail`: what happens to the replica's task when its worker is absent |
| `wait_ms` | with `wait` | The longest wait in milliseconds, above 0. Stated beside `local` or `fail` it is refused |
| `priority` | no | The priority of the replica's rule, a whole number that fits an `int`; 0 when absent |

`if_absent` has no default. `local` runs the replica on the master, which is other hardware, so a
run that loses a worker under `local` stops being repeatable; the choice is written down in the
file rather than made by omission.

The training file states version 2 to carry this section (`K_TRAIN_CLUSTER_VERSION` in
`train_config.hpp`); a version 1 file with a `cluster` key is refused. `attach_timeout_ms` joined
the section before any release carried it, so the file stays at version 2.

```json
"cluster": {
    "listen": "0.0.0.0:7700",
    "secret_file": "cluster.secret",
    "replicas": [
        {"worker": "atlas", "rows": 12, "micro_batch": 4, "if_absent": "local"},
        {"worker": "bravo", "rows": 8, "micro_batch": 4, "if_absent": "wait", "wait_ms": 30000},
        {"worker": "cobalt", "rows": 8, "micro_batch": 2, "if_absent": "fail", "priority": 5}
    ]
}
```
Source: not compiled

## The master is replica 0

Replica 0 is the master. Its `worker` names the master's own machine, and its rule is built
like every other. The master's machine reports its own name, and the file and that name must agree. The module has no key for "the master".

## The rules

`ClusterSection::placement_rules()` adds one rule per replica, in replica order, to a fresh
`PlacementRules`. Rule `k` has these parts.

| Part | Value |
| --- | --- |
| `name` | `replica k` |
| `match.placement_class` | `sushiai.replica`, an exact class |
| `match.key` | `k` |
| `match.operation` | empty |
| `target.machine_name` | the replica's `worker`; no label and no device part |
| `priority` | the replica's `priority` |
| `if_absent` | `local` gives `AbsentAction::Local`; `wait` gives `Wait` with the bound of `wait_ms`; `fail` gives `Fail` |
| `deadline` | none, so the cluster's task deadline holds |

A refusal by `PlacementRules::add()` comes back as `InvalidDescription` that names the replica.

## The rows

With `B` the sum of every `r_k`, the batch has `B` rows, and `offset(k)` is the sum of the rows of the
replicas before replica `k`. In batch `b` of an epoch, counted from 0:

- replica `k` takes the `r_k` consecutive positions of the epoch's sample order that start at
  `b * B + offset(k)`;
- its micro-batch `i`, for `i` below `r_k / m_k`, is the `m_k` positions that start at
  `b * B + offset(k) + i * m_k`;
- its weight in the gradient sum is `r_k / B`.

The ranges of one batch follow each other in replica order and tile positions `b * B` to
`(b + 1) * B`. A position maps to a sample through the epoch permutation, so a machine that holds
the seed, the epoch and the batch index finds its own rows without being sent any.

## What `ReplicaPlan::make` refuses

Every refusal is `InvalidDescription`. `ClusterSection` prefixes it with `cluster 'replicas'`.

| Case | Message names |
| --- | --- |
| No replica | at least one replica |
| A row count below 0 | `replica k 'rows'` |
| A micro-batch below 1 | `replica k 'micro_batch'` |
| A micro-batch that does not divide its rows | `replica k 'micro_batch'` and `'rows'` |
| Rows that sum above the largest `int64` | the replica where the sum overflows |
| Rows that sum to 0 | `'rows'`, since `r_k / B` has no `B` |

`ReplicaPlan`'s accessors and `rows()` stop the process on a replica or a micro-batch outside the
plan, as a precondition broken by the caller, and on a batch whose end passes 64 bits.

## The replica node

A replica task is a registered node with two inputs, the parameter pack and the position pack,
and one output, the result pack. The kernel refuses any other count as `InvalidDescription`. Its
`NodeOptions::parameters` hold four words, which `ReplicaTask` encodes and decodes:

| Word | Holds |
| --- | --- |
| 0 | the replica index `k` |
| 1 | the replica count; it must equal the kernel's plan |
| 2 | the run index |
| 3 | the seed |

`decode()` refuses another word count, a replica count of 0 and a replica index at or above the
count, each as `InvalidDescription`.

The runtime copies a node's words when the node is recorded, and a compiled step replays them
unchanged, so the epoch and the batch travel in the position pack: 16 bytes, the epoch and then
the batch index as two little-endian 64-bit words, which `ReplicaPosition::read()` reads. The
master rewrites the pack before each replay.

## The packs

The parameter pack is `PackLayout::of(ReplicaModel::parameters())`. The result pack is
`PackLayout::of(ReplicaModel::results())`: the gradients in the order of the parameters, then the
loss, then the batch statistics. A pack groups its tensors by type, so the order holds inside each
type's segment.

The kernel registers under `sushiai.replica:<fingerprint>`, through SushiBLAS's
`remote_operation_name()`. The fingerprint is the FNV-1a hash of three 64-bit words, each
little-endian: the model's own `ReplicaModel::fingerprint()`, which the layer above defines, the
parameter layout's fingerprint and the result layout's. A worker whose model or packs differ
advertises another operation, and the master never sends it the task.

## GradientExchange

`GradientExchange` is a `Graph::StepRecorder` that `TrainingStep` runs before the optimizer. It
records through one `Graph::StepRecording`, which gives it the task recorder and the runtime
recorder of the graph. `make()` allocates and shares the position pack and one result pack per
replica, and takes an `ExchangeRun` that names the operation, the seed, the run index and the
batches of an epoch. One recording holds, in this order:

1. the gather of the position tensor into the position pack;
2. one replica task per replica, recorded in the `sushiai.replica` scope with key `k`, over the
   parameter pack and the position pack;
3. one `ordered_sum()` per type segment of the result layout that holds an element, over the
   result packs' segment views in replica order;
4. the scatter of the sum pack into the target tensors.

The weight of replica `k` in a sum is `r_k / B` for `Float16`, `Float32`, `Float64` and complex
types, and 1 for `Int32`, `Int64` and `UInt32`, which `sums_exactly()` names and the replica's
own micro-batch sum agrees with, so integer totals are exact. `make()` refuses a target of type `UInt32` as `Unsupported`, since `ordered_sum()` does not
sum it, and a target unlike its layout entry, an unshared parameter pack, a plan of no batch or
too many replicas for `ordered_sum()` as `InvalidDescription`.

`before_replay()` writes the position of the coming replay and counts it; `next_step()` and
`set_next_step()` read and set the counter. `record_into_step()` returns `2 + replica_count()`
plus the number of filled segments.

## ParameterPublish

`ParameterPublish` is a `Graph::StepRecorder` that `TrainingStep` runs after the optimizer. It
owns the parameter pack the replica tasks read. `make()` allocates the pack of a layout, shares
it through the binding and copies the parameters into it on the host. `record_into_step()`
records one task that gathers every parameter into the pack. The gather runs on the master, so
it raises the pack's version and the replica tasks of the next replay pull the new parameters.
`pack()` returns the pack.

## ReplicaPosition

`ReplicaPosition` holds the epoch and the batch index of a replay. `at_step()` maps a step count
and the batches per epoch to a position, and `read()` reads one from the first
`K_REPLICA_POSITION_BYTES` of a `BufferView`, refusing a view with no memory or a smaller one
as `InvalidDescription`.

## ReplicaStep

`ReplicaStep::make()` creates a `SushiRuntime::Runtime` with `replica_runtime_config()`: one
pool thread, not pinned, the default device filter. It owns the runtime through a
`unique_ptr`, so the address the `SushiBLAS::Engine` over it holds does not move, and destroys,
in order, the compiled step, the accumulator, the model, the engine and the runtime.

A registered kernel runs inside a graph run of the runtime that serves it, and SushiRuntime
refuses a nested run of that runtime (`docs/design/LAYERED_EXECUTION/README.md` in SushiRuntime,
"What a training run over several machines needs"). The second runtime costs a second device context, a second scheduler
pool of one thread, and one pool thread of the serving runtime held for the length of each task.

`begin()` copies the parameter pack into the model's parameters on the host. `run_micro_batch()`
feeds the rows, replays the step and adds its results. `finish()` writes the mean over the
micro-batches into the result pack. Every copy goes through the host: the pack lives in memory of
the serving runtime's context and the model in memory of the replica's, and both are
host-addressable.

### The mean over micro-batches

| Micro-batches | What `finish()` writes |
| --- | --- |
| 0 | zeros; a replica of 0 rows weighs 0 in the sum |
| 1 | the step's results, byte for byte |
| more | the sum in micro-batch order, divided by the count, in each tensor's own type; `Half` sums in `float` and rounds once; a result of `Int32`, `Int64` or `UInt32` is refused as `Unsupported` |

The objective of a micro-batch is a mean over its rows, so the mean of `r_k / m_k` micro-batches
is the mean over the replica's `r_k` rows, which the gradient sum weights by `r_k / B`.

## ReplicaKernel

`ReplicaKernel::make()` builds the step of the replica its `ReplicaDraws` names, which fixes the
fingerprint and the operation. A task runs on the calling thread: it decodes the words, checks
the replica count against the plan, reads the position pack, binds the parameter and result
views through `bind_views()`, runs the replica's micro-batches of that batch in order through
`ReplicaPlan::micro_batch()`, and writes the mean. A failure is
thrown as `std::runtime_error`, which the kernel registry's contract reports as a failed task.

The kernel keeps one step per replica index, built the first time a task names that replica or
a new seed or run index, and refuses one whose fingerprint differs as `Incompatible`. A worker
pinned to one replica builds one step. A master that runs a lost worker's replica builds one more
step, with its own runtime. Tasks are served one at a time.

## Seeded draws

Every draw of a replica takes the seed, the replica index and the run index: the factory is
called with them and seeds the model's random sites with `replica_seed()`, and each
`ReplicaRows` the feed receives carries them for the augmentation.

## Bit for bit against one machine

A replica of one micro-batch returns the bytes the same graph returns on one machine over the
same rows when three things agree: the device, the reduction order inside each kernel, and the
order of the micro-batch sum. `tests/core/distribution/test_replica_kernel.cpp` runs both on a
runtime made by `replica_runtime_config()`, compiles the same graph with the same selector, and
compares one micro-batch byte for byte. Two micro-batches of two rows against one step of four
split the sum over rows inside the matrix product, so that case compares within an absolute
1e-6 plus a relative 1e-5.

## Host copies

`ReplicaStep::begin`, the replica's micro-batch sum and `ParameterPublish::make` copy between the
packs and the model's tensors with host `memcpy`. Every side is host memory: SushiAI's pool
allocates `Residency::Shared`, a `TensorPack` allocates `Residency::Shared` on the device its
tensors live on (the publish's parameters, the exchange's parameter pack, the position tensor),
and SushiRuntime
borrows every view a worker's dispatch hands a kernel as `Residency::Shared`.
`require_host_visible()` stops the process at the copy, naming the tensor, if a model tensor
ever lives in device memory.
