# modules/vision/datasets/include/SushiAI/vision/datasets/reader_dataset.hpp


| `ReaderDatasetOptions` field | Default | Meaning |
| --- | --- | --- |
| `fit` | `Resize`, no size | The input size, the fit and the letterbox settings; the caller sets the size |
| `channels` | `3` | Channels of the input. `1` decodes an image file as gray; any other count decodes it as RGB |
| `batch_size` | `16` | Samples per batch |
| `name` | `"reader"` | The name a run report prints |
| `shuffle` | `false` | `true` gives every epoch its own order of the samples |
| `seed` | `0` | The seed of those orders; read when `shuffle` is set |

`ReaderDataset::make` takes the engine, the list, the collator, a pipeline and the options. The
form without a pipeline builds a `FitPipeline` of `fit` and `channels`. `ReaderDataset` derives
from `Data::Dataset` and from `Data::EpochAware`.

Batch `i` holds positions `i * batch_size` to `(i + 1) * batch_size - 1` of the epoch. Without
`shuffle` a position is the sample of that index in the list. With it, position `p` is sample
`Data::EpochPermutation(seed, epoch, sample count).at(p)`: every sample once per epoch, in an
order that depends on the seed and the epoch alone. The positions past the last whole batch are
never read, so under `shuffle` the samples left out differ from epoch to epoch.

`begin_epoch(epoch)` stores the epoch in an atomic; it is 0 until the first call. A loop that
never calls it reads the order of epoch 0, and the draws of epoch 0, in every epoch.

`read(i, batch)` takes the epoch once, asks the pipeline for the fitted samples of the batch,
readies `batch` with `Data::prepare_batch` and calls the collator. `prepare_batch` allocates
the tensors on the first read into a `Batch` and keeps them on the next, unless something else
references them; `modules/core/data/README.md`, section "Reading a batch", has the rule. The
fitted samples, and the scratch images of the pipeline, are allocated on every call. `read`
changes no member, so threads may read at once, each into its own `Batch`: the engine's
allocator takes calls from any thread (`modules/core/data/README.md`, section "The loader on
threads").

| Code | From | When |
| --- | --- | --- |
| `InvalidDescription` | `make` | A null collator or pipeline; an input size or a channel count outside its bounds; a batch size below 1; fewer samples than one batch |
| `Incompatible` | `read` | A sample the pipeline made has another channel count than `channels`; the message opens with the file |
| Any other | `read` | What the pipeline, `Data::prepare_batch` or the collator returns |

A batch index at or past `batch_count()` is a broken precondition of `Data::Dataset::read` and
stops the process.
