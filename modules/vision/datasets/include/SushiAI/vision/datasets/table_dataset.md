# modules/vision/datasets/include/SushiAI/vision/datasets/table_dataset.hpp


`TableDataset` is the `Data::Dataset` of a list whose samples are rows of `SampleList::table`.
`make` takes the engine, the list, the collator and a `TableDatasetOptions`: `batch_size`, 16 by
default, and `name`, `"table"` by default. Its sample shape is `[feature_count]`. `read(i, batch)`
gives each sample of the batch a view of its row, readies `batch` with `Data::prepare_batch` and
calls the collator. It opens no file. Batches, the samples past the last whole batch and a batch
index out of range are as in `ReaderDataset`.

| Code | From | When |
| --- | --- | --- |
| `InvalidDescription` | `make` | A null collator; a table with no feature; a table whose number count is not the sample count times the row width; a batch size below 1; fewer samples than one batch |
| Any other | `read` | What `Data::prepare_batch` or the collator returns |

`ReaderDataset` loads and fits an image per sample and states an input of `[H, W, C]`; a row of
numbers has neither step and an input of `[F]`. The tabular case is a class of its own so that
`ReaderDataset` has no branch on the kind of its samples.
