# Datasets

**Status:** Open — all seven phases written, not yet built
**About:** How SushiAI reads training data for more than one kind of task, from the layouts data is commonly kept in, without a class per named dataset.
**Modules:** none

## Problem

How SushiAI reads training data for more than one kind of task, from the layouts data is
commonly kept in, without a class per named dataset.

## Decision

Made by the owner on 2026-10-11.

| Subject | Ruling |
| --- | --- |
| The cut | A dataset is a task and a format reader. A named dataset (MNIST, COCO) is a description, not a class |
| Breaking change | `Data::Batch` and `Data::Dataset` change shape; the single one-hot target goes |
| Tasks | Classification, binary (yes or no), multi-label, detection, semantic segmentation, regression |
| Segmentation | Inside the feature freeze of 0.1.0 |
| Image decoding | libjpeg-turbo and libpng are added as dependencies |

Decided by the engineer, open to the owner's objection:

| Subject | Decision |
| --- | --- |
| Downloading | The library reads what is on disk. It holds no network code and ships no dataset |
| Instance segmentation | Outside 0.1.0. The polygon labels are already read and kept, so it adds a task and a head later, not a reader |
| Where decoding runs | Once, into the raw cache of `vision/imaging`, as ruled for YOLO26. A training epoch reads raw pixels, never a JPEG |

## Design

### A batch carries roles

Today a batch is an input tensor, a one-hot target matrix and a class index per sample, which
fits classification alone. A batch becomes an input and a list of named targets:

```cpp
struct BatchTarget { std::string role; SushiBLAS::Tensor values; };   // resident where the step reads it
struct Batch { SushiBLAS::Tensor inputs; std::vector<BatchTarget> targets; };
```

| Task | Roles | Shape |
| --- | --- | --- |
| Classification | `labels` | `[N]` int32 |
| Binary | `labels` | `[N]` float32, 0 or 1 |
| Multi-label | `labels` | `[N, C]` float32 |
| Regression | `values` | `[N, K]` float32 |
| Detection | `boxes`, `classes`, `valid` | `[N, M, 4]` float32 corners in pixels of the input, `[N, M]` int32, `[N, M]` float32 |
| Semantic segmentation | `mask` | `[N, H, W]` int32, with an ignore index |

An objective already declares the leaves it reads by role (`Train::Objective::declare`). The
step matches the roles a dataset publishes against the roles its objective declares, once, when
it is built; a role that is missing or of another shape is a status that names it. The `data`
module still names no task: a role is a string and a tensor. The one-hot matrix classification
needs is built by its objective from `labels`, not stored by every dataset.

`Dataset` loses `num_classes` and `input_dim`, which only classification could answer, and
gains `targets()`: the roles and per-sample shapes it publishes.

A batch is read with `Status read(index, Batch&) const`. A dataset that holds its batches
shares their tensors into the caller's `Batch`; one that builds them from files writes the
caller's tensors in place, so a loop that keeps one `Batch` allocates on the first read only.
Several threads may read at once, each into its own `Batch`. An index outside the epoch is a
broken precondition and stops the process.

### The parts of a dataset

| Part | Tier | Does |
| --- | --- | --- |
| Format reader | `vision/datasets` | Lists the samples of one on-disk layout and parses each sample's annotation once, rejecting a bad one with its file and line |
| Sample | `vision/imaging` | An image reference and a `LabelSet` (class, box, polygon) or a class or a mask reference |
| Pipeline | `vision/imaging` | The transforms of one sample: letterbox, flips, colour, mosaic. Exists |
| Collator | `vision/datasets` | Turns the samples of a batch into the roles of one task |
| Loader | `core/data` | Runs readers and pipelines on threads and hands batches over in order. Names no task |

A reader knows a layout and no task; a collator knows a task and no layout. The same YOLO-txt
reader feeds detection and, through its polygons, segmentation.

| Reader | Layout | Named datasets it covers |
| --- | --- | --- |
| `image_folder` | One folder per class | ImageNet-style sets, any collected set; a two-folder set is the binary task |
| `idx` | IDX files | MNIST, Fashion-MNIST, EMNIST, KMNIST |
| `cifar` | CIFAR binary batches | CIFAR-10, CIFAR-100 |
| `yolo_txt` | An image list and one `.txt` per image | Ultralytics and Darknet sets |
| `coco_json` | One annotation JSON | COCO, LVIS, Objects365 |
| `voc_xml` | One XML per image | Pascal VOC |
| `mask_png` | An image folder and a mask folder | Cityscapes, ADE20K, VOC segmentation |
| `csv` | A header row and numeric columns | Tabular sets, regression, binary |

The training file names them:

```json
"data": { "reader": "yolo_txt", "task": "detection", "root": "../../../datasets/coco128",
          "split": "train", "batch_size": 16, "input": [640, 640], "pipeline": "default" }
```

`"type": "mnist"` stays valid as a preset that expands to `reader: idx` with its file names.
Readers, collators and presets are rows of `Data::DatasetRegistry`, added by the functions each
module registers, as nodes are.

### What the readers and collators decided

- **COCO**: crowd annotations and boxes without area are dropped; a multi-part outline keeps
  its largest part, since a label holds one polygon; images are not checked to exist.
- **Pascal VOC**: boxes are read as counted from one with both ends included; difficult
  objects are dropped unless `keep_difficult` is set; class names collected from a split are
  sorted, so `class_names` is stated for a stable index.
- **`voc_xml` reads through a parser of this repository**: elements, text, five entities,
  attributes read and dropped. Anything else is refused with its line.
- **`mask_png` pairs an image and a mask by the same relative path**, which Cityscapes' file
  names do not satisfy.
- **In a mask the byte 255 means no class**, and becomes the ignore index; a mask holds at
  most 255 classes. The mask is fitted by nearest neighbour with its image's letterbox.
- **A CSV sample's numbers live in one table of the sample list**, and a tabular task reads
  them through `TableDataset`; an image sample carries nothing for it.
- **The COCO reader streams its file** through `Config::read_json_events` and holds no
  document: about 230 MB for train2017 with boxes alone and under 1 GB with polygons, by
  estimate, where the whole-document parser needed about 8 GB.

### Image decoding

A module `vision/codecs` holds `decode_image(bytes) -> Result<Image>`, choosing the decoder from
the leading bytes: JPEG through libjpeg-turbo, PNG through libpng, and the raw cache's own
format. The two libraries are private dependencies of that module, through vcpkg, and no other
module includes their headers. The first pass over a dataset writes the raw cache; a build
without the module still trains from a cache written elsewhere.

### Segmentation

A sibling of `detect`, `vision/segment`, with namespace `SushiAI::Vision::Segment`:

- Mask geometry: a polygon rasterised to a mask, a mask resized by nearest neighbour, the
  transforms of `vision/imaging` applied to a mask as they are to an image.
- Metrics: the confusion matrix, intersection over union per class and its mean, pixel accuracy.
- Objectives, in `vision/training`: cross entropy per pixel with an ignore index, and Dice.
- A head block in `vision/blocks`: convolutions and an upsample to the input size, over any
  body the model file states.

Whether the upsample and the per-pixel cross entropy need a new operation in the core tier is
not checked yet; if they do, it is written there, in its operation's domain file.

### The binary task

A model with one output and `{ "type": "binary_cross_entropy" }` as its objective, computed from
logits. The metric is accuracy at a threshold the training file states, with precision and
recall beside it.

### What phase 5 decided

- **An objective owns its feed and its metrics** (`Objective::feed`, `Objective::measure`).
  `Trainer` names no role and no task; an objective whose targets a host pass computes between
  two runs, as the anchor objective's are, keeps the default, which reports that its own step
  feeds it.
- **The binary and multi-label losses are computed from logits in the stable form**, with
  operations the graph already had. The threshold of their metrics is a key of the objective
  section, 0.5 unless stated.

### What phase 7 decided and left open

- **The per-pixel loss reads integer labels through the row gather**, as the next-token loss
  does. One weight leaf carries both the mask of ignored pixels and the divisor: `1 / kept` on
  a kept pixel, 0 on an ignored one.
- **`N * H * W * C` must fit in int32**: 54 images of 512 by 512 over 150 classes in a batch.
- **Dice is an objective of its own**, and a training file trains on cross entropy with Dice
  through the `sum` objective: `{ "type": "sum", "terms": [ ... ] }`, each term with a
  `weight`. Dice reads two int32 index leaves; its intersection is a scatter-add of gathered
  probabilities, so it has the same int32 bound as the cross entropy.
- **The head's upsample is nearest**, the only mode the graph has, so a class boundary falls
  on a multiple of the head's scale.
- **The mask fit stays in `vision/datasets`**: its signature names that module's types, and
  moving it to `segment` would make a cycle.
- **No collator paints polygons into a mask yet**; the rasteriser exists and nothing calls it.
- **The overfit test's numbers are unverified**: 80 steps, a last loss under half the first,
  a mean IoU above 0.8.

### What the loader, the cache and the pipelines decided

- **A worker thread may allocate and fill a tensor.** Tensor storage is shared memory from an
  allocator SushiRuntime states as callable from any thread; only recording on the engine is
  bound to one thread. The loader therefore holds no host buffer and makes no extra copy:
  workers read into tensors and the consumer swaps batch handles.
- **An epoch is told to a dataset, not passed to `read`.** `Data::begin_epoch` reaches a
  dataset that implements `EpochAware`; the three training loops call it. The order of an
  epoch is a permutation computed from the seed, the epoch and a position, with no table.
- **A sample's augmentation is drawn from the seed, the epoch and its index**, so it is the
  same alone, in any batch and on any thread. A mask follows its image's draws with nearest
  sampling.
- **A cache file is named by a hash of its source's path, size and time.** A changed source
  has no cache file, so it is written again; old files are not pruned.
- **A failed read is the status of its index alone**, and later indices stay valid.
- **A mask painted from polygons**: the later object wins an overlap, and an uncovered pixel
  takes the `background` class or the ignore index.
- **COCO train2017 in the raw cache is about 99 GB**, by estimate.

## Phases

| Id | Phase | State |
| --- | --- | --- |
| 1 | `Batch` with roles, `Dataset::targets()`, the role match in the steps; MNIST and the synthetic sets carried over; classification's one-hot built by its objective | building |
| 2 | `vision/codecs` with the two libraries, and the raw cache written from a folder of images | building |
| 3 | The loader on threads, and the readers `yolo_txt`, `image_folder`, `idx`, `cifar`, `csv` | building |
| 4 | The detection collator, and `train` driving `AnchorTrainingStep` from a training file. A YOLOv7-tiny trains from `model.json`, a `.sushi` file and a folder here | building |
| 5 | The binary and multi-label objectives and their metrics | building |
| 6 | `coco_json`, `voc_xml`, `mask_png` | building |
| 7 | `vision/segment`, its objectives, its head, and a model that overfits one batch | building |

Each phase's code can be written before SushiBLAS follows SushiRuntime v0.4.0; none can be
verified before it.

### Work

<!-- work:begin -->
No open work.
<!-- work:end -->

## Open questions

- Which segmentation model 0.1.0 states as its example: a head over the YOLOv7-tiny body, or a
  small encoder and decoder written for it.
- Whether a reference implementation is run for the segmentation objectives, as Ultralytics and
  Darknet were for detection.
