# data

## Owns

The training data as a model reads it. `Dataset` is the interface of a dataset: it reports its
batch size, the shape and the element type of one input sample, its batch count and the
roles of its targets, and `read` fills a `Batch` the caller holds with the batch at an index, or returns why it could not. A `Batch` holds the inputs and one `BatchTarget` per role, in the order of
`Dataset::targets()`. An `InputDescription` states the per-sample shape and the element type
of the input, and a `TargetDescription` states the same two for a role and, for a role that
holds indices, how many classes they span. The module names no task: a
role is a string and a tensor, and `roles.hpp` holds the role names more than one module reads.
`SyntheticClassification` and `Rings` are the two synthetic datasets, built by
`SyntheticClassification::make` and `Rings::make`; both publish `labels`, one int32 class index
per sample. `MaterialisedDataset` holds batches that are already resident on the device.
`DatasetRegistry` maps a configured type name to a factory. `EpochAware` is the interface of a
dataset whose batches depend on the epoch, `EpochPermutation` is the order of one epoch, and
`ThreadedLoader` reads the batches of another dataset ahead on its own threads.

## Depends on

The module declares `failure`, `dtype`, `shape`, `seam`, `execution`, `allocation` and `config` as
public dependencies. It declares no private
dependency.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/data/dataset.hpp` | `Batch`, `Dataset`, `describe_input`, `SyntheticOptions`, `SyntheticClassification`, `RingsOptions`, `Rings`, `make_rings` |
| `SushiAI/data/dataset_registry.hpp` | `DatasetFactory`, `DatasetEntry`, `DatasetRegistry`, `register_datasets`, `make_dataset` |
| `SushiAI/data/epoch.hpp` | `EpochAware`, `begin_epoch` |
| `SushiAI/data/epoch_permutation.hpp` | `EpochPermutation` |
| `SushiAI/data/input.hpp` | `InputDescription` |
| `SushiAI/data/materialised_dataset.hpp` | `MaterialisedDataset`, `allocate_batch`, `share_batch`, `prepare_batch` |
| `SushiAI/data/roles.hpp` | `K_ROLE_LABELS` |
| `SushiAI/data/target.hpp` | `BatchTarget`, `TargetDescription`, `find_target` |
| `SushiAI/data/threaded_loader.hpp` | `LoaderOptions`, `ThreadedLoader` |

## The input a dataset states

A dataset describes its input with two queries. `sample_shape()` returns the shape of one
sample and `sample_dtype()` returns its element type. `sample_dtype()` returns `Float32`
unless a dataset overrides it: `MaterialisedDataset` returns the type it was built with and
`Language::Datasets::TokenDataset` returns `Int32`. `describe_input(dataset)` returns both as
an `InputDescription`, which has the fields `sample_shape` and `dtype` of a
`TargetDescription`.

`allocate_batch` and `prepare_batch` take an `InputDescription` and allocate the input tensor
in the type it names. `MaterialisedDataset` takes one in its constructor. No overload of the
three takes a bare `Shape`.

## Reading a batch

`Status Dataset::read(std::size_t index, Batch& batch) const` is the one way to a batch. The
caller owns `batch` and passes the same one on every call of a loop.

| Dataset | What `read` does | Cost per call |
| --- | --- | --- |
| Holds its batches: `SyntheticClassification`, `Rings`, `MaterialisedDataset` | `share_batch` copies the tensor handles of the held batch into `batch` | One reference count per tensor and one role name per target; no element is copied, and nothing is allocated once `batch` has held a batch of the dataset |
| Builds its batches: `Vision::Datasets::ReaderDataset`, `Language::Datasets::TokenDataset` | `prepare_batch` keeps or allocates the tensors, then the dataset writes them | The tensors are allocated on the first call and written in place after it |
| Reads another dataset ahead: `ThreadedLoader` | Exchanges `batch` with a batch one of its threads filled | No element is copied; the wait for a batch that is not ready yet |

A `SushiBLAS::Tensor` is a shape and a counted pointer to its `Storage`, so copying one shares
the elements. After a read from a dataset that holds its batches, `batch` and the dataset
reference the same storage: the caller reads it and does not write it.

`batch` may hold anything on entry. `prepare_batch` writes in place only tensors that have the
dataset's layout, which includes the element type of the input, start at offset 0 of their storage and are the storage's single reference.
Any other batch is replaced by a new allocation. A batch that shares storage with a dataset is
therefore never overwritten, and a caller that copied a tensor of the previous batch keeps its
elements. After a failed read the contents of `batch` are unspecified; it can be passed to the
next read.

An index at or past `batch_count()` is a broken precondition and stops the process through
`SA_FATAL_IF` in every implementer. The other failures are statuses.

`read` is `const` and no implementer in this repository changes state in it, so several threads
may call `read` on one dataset at once, each with its own `Batch`. Two threads must not read
into the same `Batch`. `ThreadedLoader` is the exception: one thread reads it. A dataset that
builds its batches allocates through the engine on the first read into a `Batch`, which any
thread may do; the section "The loader on threads" has the evidence.

## The epoch of a read

`read` takes no epoch. A dataset whose batches depend on the epoch implements `EpochAware`:
`begin_epoch(epoch)`, with epochs counted from 0, makes every read that follows return a batch
of that epoch. The method is `const` because a training loop holds a `const Dataset&`; an
implementer keeps the epoch in a member it may write, and the caller makes sure no read is
running on another thread during the call. `Data::begin_epoch(dataset, epoch)` calls it on a
dataset that implements the interface and does nothing for one that does not, so a loop calls
it once at the top of every epoch without knowing the dataset.

A negative epoch is a broken precondition and stops the process.

## The order of an epoch

`EpochPermutation(seed, epoch, count)` is a permutation of `[0, count)`: `at(position)` returns
the sample read at that position, and every sample comes up once. It is a function of the three
arguments alone. It stores no table, so building one costs four hashes, `at` allocates nothing,
and threads may call `at` at once.

`at` runs a balanced Feistel network of four rounds over the smallest `2^(2h)` that is at least
`count`, then repeats it while the result is `count` or more. The network is a bijection on
`[0, 2^(2h))`, so walking its cycle from a position below `count` returns to a value below
`count`, and the map is a bijection on `[0, count)`. `2^(2h)` is below `4 * count`, so the walk
takes fewer than four steps on average. The round function is the SplitMix64 finaliser of the
right half plus the round key. The round keys are that finaliser of
`base + (round + 1) * 0x9E3779B97F4A7C15`, with `base` the finaliser of the seed xored with the
finaliser of `epoch + 0x9E3779B97F4A7C15`. Every step is 64-bit integer arithmetic, so a seed
gives the same order on every platform.

The orders are not drawn uniformly from all `count!` permutations; no generator with a 64-bit
seed can do that for a `count` above 20.

`Vision::Imaging::Random` holds the same finaliser. `failure`, `dtype`, `shape`, `seam`,
`allocation` cannot include a domain module, so
the constants are written in both.

## The loader on threads

`ThreadedLoader::make(inner, options)` takes a dataset and returns a `Data::Dataset` that
reads the batches of `inner` ahead, on its own threads, and hands them over in index order.

| `LoaderOptions` field | Default | Meaning |
| --- | --- | --- |
| `workers` | `2` | Threads that call `inner.read`; at least 1 |
| `prefetch` | `2` | Batches read ahead of the caller and held; at least 1 |

The loader owns `prefetch` slots, each a `Batch` with a state: idle, queued, running or ready.
A worker sleeps on a condition variable until a slot is queued, takes the queued slot of the
lowest index, reads that index into the slot's batch with the mutex released, marks the slot
ready and wakes the caller. No thread polls.

`read(index, batch)` waits for the slot of `index`, exchanges the slot's batch with `batch` and
queues the next index that no slot holds into that slot. The exchange copies no element. The
batch the caller hands in becomes the batch a worker fills next, so after the first `prefetch`
reads the loader recycles the caller's own batches, and `Data::prepare_batch` in the inner
dataset finds their tensors and allocates nothing. The loader holds `prefetch` batches and the
caller one.

| Behaviour | Rule |
| --- | --- |
| Back-pressure | A worker starts index `j` only when a slot holds it, and the slots hold at most the `prefetch` indices after the last one the caller asked for |
| Start | Nothing is read until the first `read` or `begin_epoch` |
| End of an epoch | No slot is queued past `batch_count() - 1`; the loader idles until `begin_epoch` or a read of index 0 |
| An index out of order | The loader drops what was read ahead, waits for the reads that are running and queues from that index. The dropped work is lost |
| A failed read | Its status is returned by the `read` of that index and by no other. The workers go on; every other index stays readable |
| `begin_epoch` | Drops what was read ahead, waits for the running reads, calls `Data::begin_epoch` on the inner dataset with no read running, then queues from index 0 |
| Shutdown | The destructor sets a flag, wakes every worker and joins. A read that is running finishes first; it cannot be interrupted |
| Exceptions | What `inner.read` throws becomes the status of that index through `Detail::guard`; a `SushiAI::Error` stops the process, as it does everywhere under that guard |

One thread at a time calls `read` and `begin_epoch`. `make` returns `InvalidDescription` for a
null dataset or an option below 1, and the status of a thread that could not be started; the
threads already started are joined.

**What a worker may allocate and write.** A worker calls `inner.read`, which for a dataset that
builds its batches calls `prepare_batch` and so `allocate_tensor`, then writes the elements
through the tensor's host pointer. Both are safe from any thread:

- `allocate_tensor` (`modules/core/allocation/source/allocation.cpp`) builds a `SushiBLAS::Storage`
  with `Residency::Shared` and reads `Engine::layout()`. It touches no recorder and no
  graph of the engine.
- `Storage` calls `Runtime::buffer` (sushiblas:
  `modules/data/tensor/include/SushiBLAS/tensor/storage.hpp`).
- `Memory::USMAllocator` states "Every method may be called from any thread at any time"
  (sushiruntime: `modules/platform/memory/include/SushiRuntime/memory/usm_allocator.hpp`). Its
  `allocate` calls `sycl::aligned_alloc_shared` and records the address in a `USMRegistry`
  under a `std::mutex` (`usm_registry.hpp`).
- The reference count of a `Storage` is a `std::atomic` (sushiruntime:
  `modules/foundation/core/include/SushiRuntime/core/ref_counted.hpp`).
- Shared USM is host memory until a device kernel reads it, so a worker writes it as it writes
  any buffer.

So the loader keeps no host staging buffer and adds no copy. This was read in the sources of
the two sibling checkouts, not run: no test here calls `allocate_tensor` from two threads.

`Engine::execute` and `Engine::discard_recorded` carry the precondition "No thread is recording
on this engine". A dataset whose `read` records operations on the engine is therefore not safe
behind a loader; no dataset of this repository records in `read`.

A batch of 16 float32 images of 640 by 640 by 3 is 79 MB of shared USM, so `prefetch` 4 holds
about 315 MB beside the caller's batch.
