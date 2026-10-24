# distribution

## Owns

The cluster section of a training file and the split of a batch over the replicas it names.
`ClusterSection` reads and checks the section and turns it into SushiRuntime `PlacementRules`.
`ReplicaPlan` says which rows of a batch each replica takes, in which micro-batches and with
which weight. The module is built only when `SA_ENABLE_DISTRIBUTED` is on. The design is
[`docs/design/LAYERED_EXECUTION/README.md`](../../../docs/design/LAYERED_EXECUTION/README.md).

## Depends on

The module declares `config` and `failure` as public dependencies and `seam` as a private one.
It takes `SushiRuntime::placement` as an external public dependency, since `ClusterSection`
returns `SushiRuntime::PlacementRules`. `ReplicaPlan` includes nothing of SushiRuntime and no
SYCL header, so the machines that only need the split do not compile against the runtime.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/distribution/replica_plan.hpp` | `K_REPLICA_PLACEMENT_CLASS`, `ReplicaShare`, `RowRange`, `ReplicaPlan` |
| `SushiAI/distribution/cluster_section.hpp` | `K_CLUSTER_KEY`, `ClusterSection` |

## The cluster section

The section is the object under the `cluster` key of a training file. A file without the key
trains on one machine. A file with the key states every key of the first table, and every entry
of `replicas` states the required keys of the second. An unknown key is refused, as everywhere
in a training file.

| Key | Required | Holds |
| --- | --- | --- |
| `listen` | yes | The address the master listens on, as text. The module checks that it is not empty; the runtime's cluster reads its grammar |
| `secret_file` | yes | The file that holds the cluster's secret. A relative path resolves against the training file's directory, as `model` does. The module does not open it |
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

The training file's own `format_version` is not changed by this module. The key joins
`train_config.hpp` in the stage that wires the `train` program (phase A4).

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
