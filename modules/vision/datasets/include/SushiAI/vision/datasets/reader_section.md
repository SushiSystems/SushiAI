# modules/vision/datasets/include/SushiAI/vision/datasets/reader_section.hpp


`make_reader_dataset(engine, section, context)` has the signature of a `Data::DatasetFactory`.
`vision/registration` adds it to a `Data::DatasetRegistry` as the row `reader`, and
`Config::parse_train` gives that type to a data block that spells `reader` and no `type`.

| Key | Kind | Default | Meaning |
| --- | --- | --- | --- |
| `reader` | string | required | A row of the reader table |
| `task` | string | required | A row of the task table |
| `root` | string | required | The directory of the dataset; `Config::load_train` and `Config::load_data` resolve a relative one against the file's directory |
| `split` | string | `"train"` | The split, as the reader takes it |
| `batch_size` | whole number | `16` | Samples per batch |
| `input` | `[height, width]` | required for a task of images | The input size every image is fitted to |
| `channels` | whole number | `3` | Channels of the input, 1 to 4 |
| `classes` | whole number | the count of the list | Classes of the collator; at least the count the samples name |
| `cache` | string | none | The directory of the raw cache; the images are decoded into it when the dataset is built, and read from it |
| `cache_side` | whole number | `0` | The longest side an image is cached at; 0 keeps the stored size. Needs `cache` |
| `shuffle` | boolean | `false` | A new order of the samples in every epoch |
| `seed` | whole number | `0` | The seed of the order and of the pipeline's draws; any 64-bit pattern |
| `pipeline` | string | `"none"` | A row of the pipeline table |
| `workers` | whole number | `0` | Threads that read batches ahead; 0 reads on the caller's thread |
| `prefetch` | whole number | the value of `workers` | Batches read ahead and held; at least 1. Needs `workers` above 0 |

These fifteen are the shared keys. A section may also hold the keys of its reader's row and of
its task's row, which the two sections above list. `regression` reads neither `input` nor
`channels`, and a section that states them for it is not refused; it refuses `cache`,
`cache_side`, `shuffle`, `seed` and `pipeline`.

With `workers` above 0 the dataset is wrapped in a `Data::ThreadedLoader`, whatever the task.

A detection block with all four:

```json
{ "reader": "yolo_txt", "task": "detection", "root": "data/coco", "split": "train2017",
  "batch_size": 16, "input": [640, 640], "classes": 80, "max_objects": 256,
  "cache": "data/coco/cache_640", "cache_side": 640,
  "shuffle": true, "seed": 20261011, "pipeline": "default",
  "workers": 4, "prefetch": 4 }
```

The dataset reports the reader's name. Its sample shape is `[height, width, channels]` for a
task of images and `[features]` for `regression`.

One block per task:

```json
{ "reader": "image_folder", "task": "classification", "root": "data/flowers", "split": "train",
  "batch_size": 32, "input": [224, 224] }

{ "reader": "image_folder", "task": "binary", "root": "data/cats_dogs", "split": "train",
  "batch_size": 32, "input": [224, 224], "positive": "dog" }

{ "reader": "coco_json", "task": "multi_label", "root": "data/coco", "split": "train2017",
  "batch_size": 32, "input": [448, 448] }

{ "reader": "csv", "task": "regression", "root": "data/housing", "split": "train",
  "batch_size": 64, "features": ["rooms", "area"], "targets": ["price"] }

{ "reader": "voc_xml", "task": "detection", "root": "data/VOC2012", "split": "trainval",
  "batch_size": 16, "input": [416, 416], "max_objects": 64 }

{ "reader": "mask_png", "task": "segmentation", "root": "data/ade", "split": "train",
  "batch_size": 8, "input": [512, 512], "classes": 150, "ignore_index": 255 }
```

| Outcome | When |
| --- | --- |
| `InvalidDescription` | A reader, a task or a pipeline no table holds, with every name the table has; an `input` that is not two entries; a count below its lowest value or above the largest int32; `classes` below the count of the list; a task of images over a reader of rows, or the reverse; `prefetch` without `workers`; `cache_side` without `cache`; an image key on a section of rows; a key that neither the shared list, nor the reader, nor the task holds; a required key left out; a value of another kind |
| Any other status | What the reader, the collator's `make` or the dataset's `make` returns |
