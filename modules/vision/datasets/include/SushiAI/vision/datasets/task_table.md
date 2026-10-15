# modules/vision/datasets/include/SushiAI/vision/datasets/task_table.hpp


A row is a name, the fit of the task, the keys the task reads beside the shared ones, the
function that builds its collator and the function that builds its dataset. A collator factory
takes the section and a `CollatorRequest`: the list the reader returned, the class count and the
input size. A `DatasetBuilder` reads the keys of its dataset class, calls the reader, then the
collator factory, in that order, so a section with a bad key fails before a file is read.

| Row | Fit | Own keys | Collator | Dataset |
| --- | --- | --- | --- | --- |
| `detection` | `SampleFit::Letterbox` | `max_objects`, required, at least 1 | `DetectionCollator` | `ReaderDataset` |
| `classification` | `SampleFit::Resize` | none | `ClassificationCollator` | `ReaderDataset` |
| `binary` | `SampleFit::Resize` | `positive`: a class name of the list or a class index; the second class by default | `BinaryCollator` | `ReaderDataset` |
| `multi_label` | `SampleFit::Resize` | none | `MultiLabelCollator` | `ReaderDataset` |
| `regression` | not read | none | `RegressionCollator`, over the columns of the list's table | `TableDataset` |
| `segmentation` | `SampleFit::Letterbox` | `ignore_index`, a whole number, 255 by default. `background`, a class index, for a list of polygons | `SegmentationCollator`, over the input size | `ReaderDataset` |

`segmentation` takes its masks from files when a sample of the list names one, as the samples
of `mask_png` do, and from polygons when none does, as with `yolo_txt` and with `coco_json`
under `"polygons": true`.

The two builders are `Detail::build_image_dataset` and `Detail::build_table_dataset` of
`source/task_dataset.cpp`. The first reads `input`, `channels`, `batch_size`, `shuffle`, `seed`,
`pipeline`, `cache` and `cache_side`, refuses a reader that lists rows of numbers, and writes
and uses the cache before it builds the collator. The second reads `batch_size`, refuses those
five image keys and refuses a reader that lists no row. `binary` refuses a class count other
than 2.
