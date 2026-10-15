# datasets

## Owns

The format readers of `docs/design/DATASETS.md`, the collators of its six tasks with the sample
loaders they stand on, and the MNIST dataset loader.

A format reader knows one on-disk layout and no task. It lists the samples of a root and a
split, parses each sample's annotation once, and returns a `SampleList` or a failure that names
the file and the line or record at fault. It decodes no pixel, builds no batch and runs no
transform. The eight readers are `read_image_folder`, `read_yolo_txt`, `read_idx`, `read_cifar`,
`read_coco_json`, `read_voc_xml`, `read_mask_png` and `read_csv`; each takes its options struct,
whose first two fields are `root` and `split`, and returns `Result<SampleList>`. `parse_xml` is
the XML parser `read_voc_xml` stands on. Everything is in namespace `SushiAI::Vision::Datasets`.

The MNIST loader reads the four IDX files of one split into a `Data::Dataset`, with
the tensors allocated on the `SushiBLAS::Engine` passed in. The dataset publishes one role,
`labels`: one int32 class index per sample, over 10 classes. `MNISTOptions` selects the
directory, the split, the flat or image layout, the batch size and the sample cap; `make_mnist`
builds the same options from a configuration section.

A collator knows one task and no layout. It takes the samples of one batch, already fitted to
the input size, and writes `Batch::inputs` and the roles of its task. A pipeline makes those
fitted samples: `FitPipeline` loads and fits, and `AugmentPipeline` runs the training transforms
of `vision/imaging`. `ReaderDataset` joins a `SampleList`, a pipeline and a collator: it is a
`Data::Dataset` that makes and collates one batch each time it is read, in the list's order or
in a new order per epoch. It runs no thread; `Data::ThreadedLoader` of `core/data` reads it
ahead. `TableDataset` collates a list of rows of numbers, which has no image to load or fit.

`write_sample_cache` decodes the images of a list once into raw image files, and
`use_sample_cache` points the list at them.

`make_reader_dataset` builds one of the two datasets from the data section of a training file.
The section names a reader and a task; the reader table gives the function that lists the
samples, the task table gives the collator, the fit and the dataset class, and the pipeline
table gives the pipeline.

## Depends on

- `core`, `config` and `data` (`PUBLIC_DEPENDS`).
- `vision/imaging` (`PUBLIC_DEPENDS`), for the readers: `sample.hpp` names `Imaging::FileLabels`,
  `stored_pixels.hpp` names `Imaging::Image`, and `read_yolo_txt` calls `Imaging::parse_label_text`.

- `vision/codecs`, for `load_sample_image`, `load_sample_mask` and `ReaderDataset`:
  `load_sample_image.hpp` names `Codecs::DecodeOptions` and the two sources call
  `Codecs::load_image`.
- `config`, also for `read_coco_json`, which streams its file through `Config::read_json_events`.
- `vision/segment`, for `paint_polygon_mask`, which calls `Segment::rasterise_outlines`. No
  header of this module names a type of it, so the edge is private (`PRIVATE_DEPENDS`).
- `data`, also for `Data::EpochAware`, `Data::EpochPermutation` and `Data::ThreadedLoader`.

The readers do not depend on `vision/codecs`. `has_image_extension` holds the extensions of the
formats that module decodes, and no reader opens an image file. No reader depends on an XML or
a CSV library: `source/xml_element.cpp` and `source/csv.cpp` parse their formats.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/datasets/sample.hpp>` | `Sample`, `SampleList` |
| `<SushiAI/vision/datasets/value_table.hpp>` | `ValueTable` |
| `<SushiAI/vision/datasets/xml_element.hpp>` | `K_XML_MAX_DEPTH`, `XMLElement`, `parse_xml` |
| `<SushiAI/vision/datasets/coco_json.hpp>` | `COCOSegmentation`, `COCOJSONOptions`, `read_coco_json` |
| `<SushiAI/vision/datasets/voc_xml.hpp>` | `VOCDifficult`, `VOCXMLOptions`, `read_voc_xml` |
| `<SushiAI/vision/datasets/mask_png.hpp>` | `MaskPNGOptions`, `read_mask_png` |
| `<SushiAI/vision/datasets/sample_cache.hpp>` | `SampleCacheOptions`, `SampleCacheReport`, `sample_cache_path`, `write_sample_cache`, `use_sample_cache` |
| `<SushiAI/vision/datasets/sample_pipeline.hpp>` | `SamplePipeline` |
| `<SushiAI/vision/datasets/fit_pipeline.hpp>` | `K_PIPELINE_NONE`, `FitPipeline` |
| `<SushiAI/vision/datasets/augment_pipeline.hpp>` | `K_PIPELINE_DEFAULT`, `AugmentOptions`, `AugmentPipeline` |
| `<SushiAI/vision/datasets/pipeline_table.hpp>` | `PipelineRequest`, `PipelineFactory`, `PipelineEntry`, `pipeline_table`, `find_pipeline` |
| `<SushiAI/vision/datasets/polygon_mask.hpp>` | `paint_polygon_mask` |
| `<SushiAI/vision/datasets/csv.hpp>` | `CSVOptions`, `read_csv` |
| `<SushiAI/vision/datasets/load_sample_mask.hpp>` | `load_sample_mask` |
| `<SushiAI/vision/datasets/fit_mask.hpp>` | `K_MASK_VOID`, `fit_mask` |
| `<SushiAI/vision/datasets/binary_collator.hpp>` | `K_TASK_BINARY`, `K_BINARY_CLASSES`, `BinaryCollator` |
| `<SushiAI/vision/datasets/multi_label_collator.hpp>` | `K_TASK_MULTI_LABEL`, `MultiLabelCollator` |
| `<SushiAI/vision/datasets/regression_collator.hpp>` | `K_TASK_REGRESSION`, `RegressionCollator` |
| `<SushiAI/vision/datasets/segmentation_collator.hpp>` | `K_TASK_SEGMENTATION`, `K_DEFAULT_IGNORE_INDEX`, `MaskOrigin`, `SegmentationOptions`, `SegmentationCollator` |
| `<SushiAI/vision/datasets/table_dataset.hpp>` | `TableDatasetOptions`, `TableDataset` |
| `<SushiAI/vision/datasets/stored_pixels.hpp>` | `PlaneLayout`, `StoredPixels`, `read_stored_pixels` |
| `<SushiAI/vision/datasets/image_extensions.hpp>` | `has_image_extension` |
| `<SushiAI/vision/datasets/image_folder.hpp>` | `ImageFolderOptions`, `read_image_folder` |
| `<SushiAI/vision/datasets/yolo_txt.hpp>` | `YOLOTxtOptions`, `yolo_label_path`, `read_yolo_txt` |
| `<SushiAI/vision/datasets/idx_file.hpp>` | `IDXElement`, `idx_element_bytes`, `IDXHeader`, `read_idx_header`, `read_idx_payload` |
| `<SushiAI/vision/datasets/idx.hpp>` | `IDXOptions`, `read_idx` |
| `<SushiAI/vision/datasets/cifar.hpp>` | `K_CIFAR_SIDE`, `CIFARVariant`, `CIFAROptions`, `read_cifar` |
| `<SushiAI/vision/datasets/load_sample_image.hpp>` | `load_sample_image` |
| `<SushiAI/vision/datasets/fitted_sample.hpp>` | `FittedSample` |
| `<SushiAI/vision/datasets/fit_sample.hpp>` | `SampleFit`, `SampleFitOptions`, `fit_sample` |
| `<SushiAI/vision/datasets/batch_inputs.hpp>` | `K_BYTE_SCALE`, `write_batch_inputs` |
| `<SushiAI/vision/datasets/collator.hpp>` | `Collator` |
| `<SushiAI/vision/datasets/classification_collator.hpp>` | `K_TASK_CLASSIFICATION`, `ClassificationCollator` |
| `<SushiAI/vision/datasets/detection_collator.hpp>` | `K_TASK_DETECTION`, `K_BOX_COORDINATES`, `DetectionCollator` |
| `<SushiAI/vision/datasets/reader_dataset.hpp>` | `ReaderDatasetOptions`, `ReaderDataset` |
| `<SushiAI/vision/datasets/reader_table.hpp>` | `ReaderRequest`, `SampleReader`, `ReaderEntry`, `reader_table`, `find_reader` |
| `<SushiAI/vision/datasets/task_table.hpp>` | `CollatorRequest`, `CollatorFactory`, `DatasetBuilder`, `TaskEntry`, `task_table`, `find_task` |
| `<SushiAI/vision/datasets/reader_section.hpp>` | `make_reader_dataset` |
| `<SushiAI/vision/datasets/mnist.hpp>` | `MNISTLayout`, `MNISTOptions`, `mnist_available`, `mnist_expected_paths`, `try_load_mnist`, `make_mnist` |

`try_load_mnist` returns the dataset, or a failure: `FileNotFound` when a file of the split is
absent, `InvalidFile` when one is unreadable or malformed, and `InvalidDescription` when the batch
size is not positive. `mnist_available` and `mnist_expected_paths` let a caller check the files
before it loads them.

### MNISTOptions

| Field | Default | Meaning |
| --- | --- | --- |
| `directory` | `"data/mnist"` | Holds the four standard IDX files |
| `batch_size` | `64` | Samples per batch |
| `max_samples` | `4096` | Cap on the samples loaded; `0` loads all of them |
| `test_split` | `false` | Loads the test split rather than the training split |
| `layout` | `MNISTLayout::Flat` | `Flat` gives `[784]` per sample; `Image` gives `[28, 28, 1]` |

### Configuration keys

`make_mnist` reads `directory` (required), `split` (`"train"` or `"test"`), `layout` (`"flat"` or
`"image"`), `batch_size` and `max_samples`. A relative `directory` resolves against the working
directory. A `split` or `layout` that names neither of its values fails with `InvalidDescription`.
`try_load_mnist` takes the engine and the options; `make_mnist` takes the engine, the attributes
and a context string.

### sample.hpp

[`include/SushiAI/vision/datasets/sample.md`](include/SushiAI/vision/datasets/sample.md) describes it.

### value_table.hpp

[`include/SushiAI/vision/datasets/value_table.md`](include/SushiAI/vision/datasets/value_table.md) describes it.

### stored_pixels.hpp

[`include/SushiAI/vision/datasets/stored_pixels.md`](include/SushiAI/vision/datasets/stored_pixels.md) describes it.

### image_extensions.hpp

[`include/SushiAI/vision/datasets/image_extensions.md`](include/SushiAI/vision/datasets/image_extensions.md) describes it.

### image_folder.hpp

[`include/SushiAI/vision/datasets/image_folder.md`](include/SushiAI/vision/datasets/image_folder.md) describes it.

### yolo_txt.hpp

[`include/SushiAI/vision/datasets/yolo_txt.md`](include/SushiAI/vision/datasets/yolo_txt.md) describes it.

### idx_file.hpp

[`include/SushiAI/vision/datasets/idx_file.md`](include/SushiAI/vision/datasets/idx_file.md) describes it.

### idx.hpp

[`include/SushiAI/vision/datasets/idx.md`](include/SushiAI/vision/datasets/idx.md) describes it.

### cifar.hpp

[`include/SushiAI/vision/datasets/cifar.md`](include/SushiAI/vision/datasets/cifar.md) describes it.

### coco_json.hpp

[`include/SushiAI/vision/datasets/coco_json.md`](include/SushiAI/vision/datasets/coco_json.md) describes it.

### xml_element.hpp

[`include/SushiAI/vision/datasets/xml_element.md`](include/SushiAI/vision/datasets/xml_element.md) describes it.

### voc_xml.hpp

[`include/SushiAI/vision/datasets/voc_xml.md`](include/SushiAI/vision/datasets/voc_xml.md) describes it.

### mask_png.hpp

[`include/SushiAI/vision/datasets/mask_png.md`](include/SushiAI/vision/datasets/mask_png.md) describes it.

### csv.hpp

[`include/SushiAI/vision/datasets/csv.md`](include/SushiAI/vision/datasets/csv.md) describes it.

### load_sample_image.hpp

[`include/SushiAI/vision/datasets/load_sample_image.md`](include/SushiAI/vision/datasets/load_sample_image.md) describes it.

### fit_sample.hpp

[`include/SushiAI/vision/datasets/fit_sample.md`](include/SushiAI/vision/datasets/fit_sample.md) describes it.

### load_sample_mask.hpp

[`include/SushiAI/vision/datasets/load_sample_mask.md`](include/SushiAI/vision/datasets/load_sample_mask.md) describes it.

### fit_mask.hpp

[`include/SushiAI/vision/datasets/fit_mask.md`](include/SushiAI/vision/datasets/fit_mask.md) describes it.

### batch_inputs.hpp

[`include/SushiAI/vision/datasets/batch_inputs.md`](include/SushiAI/vision/datasets/batch_inputs.md) describes it.

### collator.hpp

[`include/SushiAI/vision/datasets/collator.md`](include/SushiAI/vision/datasets/collator.md) describes it.

### polygon_mask.hpp

[`include/SushiAI/vision/datasets/polygon_mask.md`](include/SushiAI/vision/datasets/polygon_mask.md) describes it.

### sample_cache.hpp

[`include/SushiAI/vision/datasets/sample_cache.md`](include/SushiAI/vision/datasets/sample_cache.md) describes it.

### Pipelines

A `SamplePipeline` makes the fitted samples of one batch: `produce(list, samples, epoch,
fitted)` takes the indices of the samples, one per row, and fills one `FittedSample` per
index. An implementer keeps no state between calls, so threads may call it at once.

`pipeline_table()` lists the rows a data section names with `"pipeline"`.

| Row | Class | Makes |
| --- | --- | --- |
| `none` | `FitPipeline` | Each sample loaded with `load_sample_image`, fitted with `fit_sample`, its mask loaded with `load_sample_mask` and fitted with `fit_mask`. The epoch is not read. The default |
| `default` | `AugmentPipeline` | Each sample through `Imaging::TrainingPipeline`: mosaic with three partners, random affine, colour jitter and horizontal flip, at their default settings |

A new pipeline is a class that implements `SamplePipeline` and one row of `K_PIPELINES`.

`FitPipeline` decodes a mask whatever the task, so a list of `read_mask_png` under another task
than segmentation decodes masks it does not use.

`AugmentPipeline` builds one `Imaging::TrainingPipeline` and one `Imaging::TrainingMaskPipeline`
per call of `produce`, over the list, with `Imaging::DatasetPartners`. A sample's draws come
from `Imaging::make_sample_random(seed, epoch, sample index)` and its partners from the same
generator, so a sample is the same whatever batch it is in, whatever thread makes it and
whatever was made before it. The boxes and the polygons come out as the pipeline moves them. A
mask is loaded for the sample and for each partner and takes the mosaic, the warp and the flip
of the trace with nearest sampling; where no mask lands it holds `K_MASK_VOID`.

| `AugmentOptions` field | Default | Meaning |
| --- | --- | --- |
| `image_size` | `640` | The side of the square sample |
| `seed` | `0` | The seed of every draw |
| `mosaic_epochs` | the largest int32 | The first epochs that compose mosaics; from that epoch on a sample is its own image |

The row `default` takes `image_size` from `input`, which must be square, and `seed` from the
section. It makes three channels, so `channels` must be 3. No key sets `mosaic_epochs`: the data
section does not know how many epochs the run has.

| Code | From | When |
| --- | --- | --- |
| `InvalidDescription` | `FitPipeline::make` | An input size outside the bounds of an image |
| `InvalidDescription` | `AugmentPipeline::make`, the row `default` | A training size outside [1, 8192]; a negative `mosaic_epochs`; an input that is not square or not of three channels |
| `InvalidDescription` | `produce` | Another count of samples than of fitted samples; for `AugmentPipeline`, an epoch outside [0, the largest int32) and a list of no sample |
| `Incompatible` | `produce` | A mask of several channels or of another size than its image; the message opens with the mask file |
| Any other | `produce` | What `load_sample_image` or `load_sample_mask` returns; `InvalidFile` for what a transform of `vision/imaging` throws |

`AugmentPipeline` reads a failed image load from the adapter it gives the imaging pipeline: the
adapter keeps the status and hands over an image of no pixels, which the pipeline refuses.

### reader_dataset.hpp

[`include/SushiAI/vision/datasets/reader_dataset.md`](include/SushiAI/vision/datasets/reader_dataset.md) describes it.

### table_dataset.hpp

[`include/SushiAI/vision/datasets/table_dataset.md`](include/SushiAI/vision/datasets/table_dataset.md) describes it.

### reader_table.hpp

[`include/SushiAI/vision/datasets/reader_table.md`](include/SushiAI/vision/datasets/reader_table.md) describes it.

### task_table.hpp

[`include/SushiAI/vision/datasets/task_table.md`](include/SushiAI/vision/datasets/task_table.md) describes it.

### reader_section.hpp

[`include/SushiAI/vision/datasets/reader_section.md`](include/SushiAI/vision/datasets/reader_section.md) describes it.

### Failure messages

A message opens with the path in quotes, written with forward slashes, then the name of the
function that failed. A failure that has no path opens with the function. `source/disk.hpp` holds
the file system calls the readers share and `at_boundary`, which turns what a reader throws into
a status: `OutOfMemory` for `std::bad_alloc`, `InvalidFile` with the exception's text for anything
else.

## Known limits

- Nothing in the readers has run. The sources and the tests pass `clang -fsyntax-only`; no test
  has been executed.
- `make_reader_dataset` has not run. The `idx` row cannot name its two files, and no row sets
  the letterbox settings; both take their defaults.
- A `root` is not resolved against the training file, as `directory` of `mnist` is not.
- Nothing in the collators, `fit_sample` or `ReaderDataset` has run either.
- No trainer has read a `ReaderDataset`. `Train::match_roles` and `Trainer::fit` take a
  `const Data::Dataset&`, which it now is.
- No unit turns the roles of a detection batch into the `Detect::GroundTruthBatch` that
  `AnchorTrainingStep::run` takes. That type holds int64 classes and uint8 validity in vectors;
  the roles hold int32 and float32 in tensors.
- `ReaderDataset` converts no channel count. A one-channel stored image read with `channels` at
  3 is `Incompatible`; `Imaging::expand_gray` is not called.
- `Imaging::resize_bilinear` does not widen its kernel on a reduction, so a large image fitted to
  a small input aliases.
- `fit_sample` reports whatever a transform of `vision/imaging` throws as `InvalidFile`, because
  those functions throw `Error` and return no status.
- `read_yolo_txt` does not read the dataset file that names the classes, so `class_names` is
  empty and `class_count` comes from the caller.
- `read_cifar` and `read_idx` read no class name file.
- Nothing of `read_coco_json`, `read_voc_xml`, `parse_xml`, `read_mask_png`, `read_csv`,
  `fit_mask`, `load_sample_mask`, `TableDataset` or the four newer collators has run either.
- `read_coco_json` holds the text, the JSON library's document and the `Config::Value` tree at
  once. For the 450 MB `instances_train2017.json` of COCO that is several gigabytes; the figure
  has not been measured. `modules/core/config` has no streaming parser.
- `read_coco_json` keeps one outline per object, the part of the largest area, because a
  `LabelSet` holds one polygon per object. It decodes no run-length mask, keeps no crowd
  annotation and does not check that an image file exists.
- `read_voc_xml` takes every box as counted from 1. An annotation written by a tool that counts
  from 0 gives boxes one pixel narrower on the left and the top. `truncated`, `occluded`, `pose`
  and `part` are not read.
- `read_mask_png` reads masks of one byte per pixel whose bytes are class indices; a mask
  painted in colours, and a set whose ids need a mapping to train ids, are not converted.
- Nothing of the cache, the pipelines, the order of an epoch, `paint_polygon_mask` or the two
  suffixes has run. Their sources and tests pass `clang -fsyntax-only`.
- The cache prunes nothing: the file of a source that changed stays until someone deletes it.
- The cache is written on one thread when the dataset is built.
- A list with masks cannot be cached at a reduced size; the masks would need the same reduction.
- `AugmentPipeline` allocates its scratch images on every batch, and both pipelines allocate
  the fitted samples on every batch.
- `AugmentPipeline` runs the transforms at the defaults of `Imaging::TrainingSettings`; no key
  of the data section sets a probability, a gain or `mosaic_epochs`.
- A mask of a mosaic partner is taken to be the size of its image; only the sample's own mask
  is checked against its image. Each image is the sample itself once per epoch.
- `Imaging::clip_labels` clamps the vertices of a polygon to the image; it does not clip the
  polygon. An outline the image edge cuts is painted with its outside vertices moved onto the
  edge, which changes its shape near the edge.
- `TableDataset` reads in list order; `shuffle` is refused for it.
- `read_csv` keeps the whole file and the whole table in memory, fills in no missing cell and
  scales no column. No row makes a class index of a column, so the `binary` and `classification`
  tasks do not read a CSV file.
- A mask painted from polygons has one part per object: `read_coco_json` keeps the largest part
  of an outline, and a label holds one polygon.
- `SegmentationCollator` takes at most 255 classes.
- Paths are sorted with the order of `std::filesystem::path`, which compares native characters. A
  folder name outside ASCII can sort differently on Windows and on Linux.
- A path is written into a message, and a class folder into `class_names`, with
  `generic_string()`. On Windows a name the active code page cannot hold makes the reader fail
  with `InvalidFile`.

## Tests

`modules/vision/datasets/tests/`. No test needs a device and none reads a file from the
repository. Each writes files of a few bytes, or CIFAR records of 3073 or 3074 bytes, under the
system's temporary directory and removes them. `datasets_test_support.hpp` holds the scratch
directory and the IDX file builder the tests share. The MNIST loader's tests are in
`tests/vision/registration/test_registration.cpp`. `test_datasets_collators.cpp`,
`test_datasets_task_collators.cpp`, `test_datasets_reader_dataset.cpp`,
`test_datasets_table_dataset.cpp` and `test_datasets_reader_section.cpp` take an engine from
`SushiAI::Test::EngineTest`, because `Data::allocate_batch` allocates through one. A test that
needs a decodable image or mask writes a raw image file with `Support::write_raw_image`;
`vision/codecs` picks its decoder from the leading bytes, so such a file may be named `.png`.

| File | Suite | Covers |
| --- | --- | --- |
| `test_datasets_stored_pixels.cpp` | `Small_DatasetsStoredPixels` | Interleaved and planar copies, a plane longer than one read, a short file, an absent file, extents no image has |
| `test_datasets_image_folder.cpp` | `Small_DatasetsImageFolder` | The extension table, sorted classes and files, nested files, a split folder, an empty class, the refusals |
| `test_datasets_yolo_txt.cpp` | `Small_DatasetsYoloTxt` | The label path rule, the folder and the list form, box and polygon rows, an image without labels, the file and line of a bad label and of an absent listed image |
| `test_datasets_idx_file.cpp` | `Small_DatasetsIdxFile` | The header and the payload of a well-formed file, each malformed header, a size that disagrees with the header, extents that overflow |
| `test_datasets_idx.cpp` | `Small_DatasetsIdx` | The samples, offsets and classes of a split, the pixels read back, named files, four dimensions, the refusals |
| `test_datasets_cifar.cpp` | `Small_DatasetsCifar` | The five training files in order, the planes read back, the fine and the coarse label, the record of a bad label, a cut file, an absent file, another split |
| `test_datasets_load_sample_image.cpp` | `Small_DatasetsLoadSampleImage` | Stored pixels read from their offset, a raw image file decoded, a sample and a file index outside the list, an absent file |
| `test_datasets_fit_sample.cpp` | `Small_DatasetsFitSample` | A resize and its boxes, a letterbox with its padding and its boxes, an object without area removed, an input size out of bounds, a source of no pixels |
| `test_datasets_collators.cpp` | `Unit_DatasetsCollators` | The scaled inputs and their refusals, the descriptions and the values of both collators, a sample without a class, a batch without the role, an image with more than `M` objects, a class outside the set |
| `test_datasets_reader_section.cpp` | `Small_DatasetsReaderTable`, `Small_DatasetsTaskTable`, `Medium_DatasetsReaderSection` | The rows of both tables and the keys of each, the cifar row's class counts, the keys the csv, mask_png, voc_xml and coco_json rows read, the `positive` and `ignore_index` keys, a detection section over a YOLO split read into a letterboxed batch, a classification section over class folders, a regression section over a CSV file, a segmentation section over an image and a mask folder with its padding, a binary section over two folders, a multi-label section over a YOLO split, a mask of another size than its image, a task over a reader of the other sample kind, a class count above and below the list's, an unknown reader and task with the known ones, a key of another task and of another reader, a missing `max_objects`, values out of bounds, the reader's failure, a cache written and then read with the sources removed, a shuffled section under the `default` pipeline, a pipeline the table lacks, the batches of a section with `workers` equal to those without, `prefetch` without `workers`, `cache_side` without `cache`, an image key on a section of rows, a segmentation section that paints the polygons of a label file |
| `test_datasets_xml_element.cpp` | `Small_DatasetsXmlElement` | Elements, text and lines, the five entities, the declaration, comments and attributes skipped, and the line of each refusal: a closing tag of another name, an element left open, an unknown entity, markup outside the subset, a malformed attribute, no root or two, nesting past the depth |
| `test_datasets_voc_xml.cpp` | `Small_DatasetsVocXml` | A listed split with sorted classes and the box conversion, an empty split, a difficult object dropped and kept, a stated class list, and the file and line of a malformed annotation, a missing element, a number that is none, a reversed box, an unknown class; an absent list and an absent annotation with the line of the list |
| `test_datasets_coco_json.cpp` | `Small_DatasetsCocoJson` | Boxes as fractions, sparse ids as dense classes, crowd and empty boxes left out, a named file and folder, a polygon kept on request and the largest part taken, and the index of each refusal: a run-length mask, a malformed polygon, an unknown image and category, a malformed box, a shared id, an image of no size; an absent file and folder, text that is no JSON, a missing list |
| `test_datasets_mask_png.cpp` | `Small_DatasetsMaskPng` | Images paired with the masks of their stems in sorted order, named folders with no split, an image without a mask, a mask without an image, two images of one stem, a class count below 1, an absent and an empty folder, the two suffixes over names as Cityscapes spells them, an image without the image suffix, a mask of the suffix without an image |
| `test_datasets_sample_cache.cpp` | `Small_DatasetsSampleCache` | A first pass that writes and a second that reuses, the pixels read back through the cache, a longest side that reduces only what exceeds it, a source that changed found stale and cached again, stored pixels and masks left alone, the name of a cache file, the refusals |
| `test_datasets_pipelines.cpp` | `Small_DatasetsPipelineTable`, `Small_DatasetsFitPipeline`, `Small_DatasetsAugmentPipeline` | The two rows and what each builds, the inputs `default` refuses; a fit that is the same in every epoch; training samples of the training size, a sample that is the same alone, in a batch and in a reversed batch and differs in the next epoch, the objects past the mosaic epochs, a mask at the training size, an unreadable image as the failure, the refusals |
| `test_datasets_polygon_mask.cpp` | `Small_DatasetsPolygonMask` | The pixels a polygon paints, the later object winning an overlap, a set of no object, an object without a polygon, a class no byte names |
| `test_datasets_csv.cpp` | `Small_DatasetsCsv` | The default and the named columns, quoted fields, a doubled quote, a line break inside quotes, carriage returns and a byte order mark, and the line and column of a cell that is no number; a record of another field count, the three malformed quotes, a column the header lacks, a header name used twice, a file with no record |
| `test_datasets_fit_mask.cpp` | `Small_DatasetsFitMask`, `Small_DatasetsLoadSampleMask` | An enlarged and a reduced mask by nearest neighbour, a letterbox with its void padding with and without a reduction, a mask of several channels or of another size; a mask file decoded, a sample without a mask, indices outside the list, an absent file |
| `test_datasets_task_collators.cpp` | `Unit_DatasetsTaskCollators` | The description and the values of the binary, multi-label, regression and segmentation collators, either positive class, a stale row overwritten with zeros, the ignore index at its default and at -100, and the refusals: a sample without a class, a third class, a class outside the set, a sample with no row, inputs of another width, a byte that is no class with its pixel, a sample without a mask, polygons painted with and without a background, an object without a polygon, a background that is no class or stated over mask files |
| `test_datasets_table_dataset.cpp` | `Medium_DatasetsTableDataset` | The queries, a batch in list order read through `Data::Dataset`, a second read reusing the tensors, the refusals of `make`, a batch index out of range stopping the process, the collator's failure |
| `test_datasets_reader_dataset.cpp` | `Medium_DatasetsReaderDataset` | The queries, a batch in list order, a letterboxed detection batch from raw image files, the refusals of `make`, a batch index out of range stopping the process, a second read reusing the tensors, a read leaving a shared tensor alone, a read through `Data::Dataset`, a channel count that differs, a file absent at read time, a shuffled epoch in the order of its seed with every sample once, the order `begin_epoch` selects, the list order kept without shuffle, a null pipeline |

The coordinate tolerance of `test_datasets_yolo_txt.cpp` is 1e-6, absolute: the values are
decimal fractions parsed to `float`.
A byte divided by 255 is compared with an absolute tolerance of 1e-6, and a box coordinate after
a fit with 1e-4 pixels, absolute. The values a collator copies are compared exactly. A box
fraction of `read_voc_xml` and `read_coco_json` is a quotient computed in double and compared
with 1e-6, absolute. The cells of the CSV tests are numbers a `float` holds exactly and are
compared exactly.
