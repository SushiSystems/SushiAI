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
  header of this module names a type of it, so the edge can be private. `modules/vision/datasets/CMakeLists.txt`
  does not declare it yet.
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

| `SampleList` field | Holds |
| --- | --- |
| `files` | Every file a sample's pixels or its mask are in, once each |
| `samples` | One `Sample` per sample, in the order the reader lists them |
| `class_names` | One name per class; empty when the layout names no class |
| `class_count` | The bound of a sample's class index and of an object's class |
| `table` | A `ValueTable`: row `i` for sample `i` of a tabular list; no row for a list of images |

| `Sample` field | Holds |
| --- | --- |
| `file` | The index of the sample's file in `SampleList::files` |
| `stored` | The place of raw pixels inside that file; absent when the file is one encoded image |
| `class_index` | The class of the whole image; absent when the layout labels objects |
| `objects` | An `Imaging::FileLabels`: a class, a normalised box and a polygon per object |
| `mask` | The index, in `SampleList::files`, of the file with one class per pixel; absent when the layout has none |

| Reader | `files` | `stored` | `class_index` | `objects` | `mask` | `class_names` | `table` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `read_image_folder` | One per sample | Absent | Set | Empty | Absent | The folder names | Empty |
| `read_yolo_txt` | One per sample | Absent | Absent | Parsed | Absent | Empty | Empty |
| `read_idx` | The image file | Set, interleaved | Set | Empty | Absent | Empty | Empty |
| `read_cifar` | One per binary file | Set, planar | Set | Empty | Absent | Empty | Empty |
| `read_coco_json` | One per image | Absent | Absent | Boxes, and polygons on request | Absent | The category names | Empty |
| `read_voc_xml` | One per image | Absent | Absent | Boxes | Absent | Stated or collected | Empty |
| `read_mask_png` | The images, then the masks | Absent | Absent | Empty | Set | Empty | Empty |
| `read_csv` | The CSV file | Absent | Absent | Empty | Absent | Empty | One row per record |

A sample holds an index and not a path, so the 60 000 samples of an IDX file share one path.
`Sample` is 128 bytes in the release build, 8 of them `mask`. A sample of a tabular list
carries nothing but its file index; its numbers are in the table.

### value_table.hpp

A `ValueTable` holds `feature_count + target_count` floats per row, the features first, one row
after another in `values`, and the column names in the same order in `columns`. `features(row)`
and `targets(row)` return views of one row; they check no index. A list of images has a table of
no column and no row.

### stored_pixels.hpp

IDX and CIFAR keep the pixels inside the annotation file. Their samples carry a `StoredPixels`:
the byte offset, the size, the channel count and the `PlaneLayout`. `read_stored_pixels` seeks to
the offset and copies the bytes into an `Imaging::Image`. `Interleaved` bytes are copied as they
are. `Planar` bytes are read plane by plane, 4096 bytes at a time into a buffer on the stack, and
plane `c` becomes channel `c` of every pixel. One form takes a stream the caller keeps open and
the other takes a path; the second opens the file on every call.

| Code | When |
| --- | --- |
| `InvalidDescription` | A side outside [1, `Imaging::K_MAX_IMAGE_SIDE`], a channel count outside [1, 4], or an offset no stream seeks to |
| `InvalidFile` | The file cannot be opened, the seek fails or the bytes end early; the message gives the bytes read and wanted |
| `OutOfMemory` | The image cannot be allocated |

### image_extensions.hpp

`has_image_extension` is true for `.jpg`, `.jpeg`, `.png` and `.saim`, whatever the case of the
ASCII letters. The four are one table in `source/image_extensions.cpp`. `vision/codecs` picks a
decoder from a file's leading bytes and keeps no list of extensions, so the list lives here.

### image_folder.hpp

`read_image_folder` reads `root/<split>`, or `root` when `split` is empty. Each folder there is
a class; the classes are sorted by path and class `i` is folder `i`. A sample is a file with an
image extension at any depth below its class folder. A class lists its samples sorted by path. A
file that sits beside the class folders is ignored. A folder with no image is still a class, so
adding images to it later moves no class index.

| Code | When |
| --- | --- |
| `FileNotFound` | The split is not a directory |
| `InvalidFile` | The split holds no folder, the folders hold no image file, or a directory cannot be listed |
| `OutOfMemory` | An allocation fails |

### yolo_txt.hpp

A split has one of two forms, tried in this order:

1. The image list `root/<split>.txt`: one image path per line. Spaces, tabs and a carriage
   return around a path are dropped and a line with nothing else is skipped. A relative path is
   taken from `root`. The samples keep the order of the list.
2. The image folder `root/images/<split>`: every file with an image extension at any depth,
   sorted by path.

`yolo_label_path` gives the label file of an image: the last folder named `images` on the path
becomes `labels` and the extension becomes `.txt`. A path with no `images` folder keeps its
folder, so the label file sits beside the image. An image whose label file is absent has no
object. A label file that exists is read whole and parsed by `Imaging::parse_label_text`, which
takes box rows (`class cx cy w h`) and polygon rows (`class x1 y1 x2 y2 ...`), all normalised.
`class_count` comes from the caller, because the layout stores it in a file this reader does not
read.

| Code | When |
| --- | --- |
| `InvalidDescription` | `class_count` is below 1 |
| `FileNotFound` | The split has neither form; a line of the list names no file, and the message gives the list and the line |
| `InvalidFile` | The parser rejects a label file, and the message gives the label file and the parser's text with its line; the split holds no image; a file cannot be read |
| `OutOfMemory` | An allocation fails |

### idx_file.hpp

An IDX file opens with two zero bytes, an element code and a dimension count, then one 32-bit
extent per dimension with the most significant byte first, then the elements. `read_idx_header`
returns the element type, the extents and the byte count of the elements. It refuses a file
whose size is not the header plus those bytes, and extents whose product passes 64 bits.
`read_idx_payload` returns the element bytes as the file stores them; it converts no byte order.

| Element | Code | Bytes |
| --- | --- | --- |
| `UInt8` | `0x08` | 1 |
| `Int8` | `0x09` | 1 |
| `Int16` | `0x0B` | 2 |
| `Int32` | `0x0C` | 4 |
| `Float32` | `0x0D` | 4 |
| `Float64` | `0x0E` | 8 |

`source/mnist.cpp` still parses its own IDX headers. Its failure messages differ from those of
`read_idx_header`, and it reads both files inside one guarded allocation, so moving it would
change what a caller sees.

### idx.hpp

`read_idx` reads an image file and a label file of `root`. With `image_file` and `label_file`
empty it takes the names of the MNIST distribution, which Fashion-MNIST and KMNIST share:

| `split` | Image file | Label file |
| --- | --- | --- |
| `train` | `train-images-idx3-ubyte` | `train-labels-idx1-ubyte` |
| `test` | `t10k-images-idx3-ubyte` | `t10k-labels-idx1-ubyte` |

A set with other names, such as EMNIST, sets both fields, and `split` is then not read. The image
file holds unsigned bytes in three dimensions (count, height, width) or four (a channel count
last). The label file holds one unsigned byte per image. `class_count` is one more than the
largest label.

| Code | When |
| --- | --- |
| `InvalidDescription` | A file name is empty and `split` is neither `train` nor `test` |
| `FileNotFound` | A file is absent |
| `InvalidFile` | `read_idx_header` rejects a file; the image file has another element type or rank; the label file is not one dimension of unsigned bytes; the two counts differ; there is no image |
| `Unsupported` | A side outside [1, `Imaging::K_MAX_IMAGE_SIDE`] or more than four channels |
| `OutOfMemory` | An allocation fails |

### cifar.hpp

A record is its label bytes and then 3072 pixel bytes: a 32 by 32 red plane, a green one and a
blue one, each row-major.

| `CIFARVariant` | Label bytes | Label taken | Classes | `train` | `test` |
| --- | --- | --- | --- | --- | --- |
| `CIFAR10` | 1 | Byte 0 | 10 | `data_batch_1.bin` to `data_batch_5.bin` | `test_batch.bin` |
| `CIFAR100Fine` | 2 | Byte 1 | 100 | `train.bin` | `test.bin` |
| `CIFAR100Coarse` | 2 | Byte 0 | 20 | `train.bin` | `test.bin` |

The rows are `K_FORMATS` in `source/cifar.cpp`. The reader seeks to the label byte of each record
and reads it; it reads no pixel. The label that is not taken is not checked.

| Code | When |
| --- | --- |
| `InvalidDescription` | `split` is neither `train` nor `test` |
| `FileNotFound` | A file of the split is absent |
| `InvalidFile` | A file is empty or not a whole number of records; a label is at or above the class count, and the message gives the file and the one-based record |
| `OutOfMemory` | An allocation fails |

### coco_json.hpp

`read_coco_json` reads one file with the lists `images`, `categories` and `annotations`, and an
image folder. With `annotation_file` empty the file is `root/annotations/instances_<split>.json`;
with `image_folder` empty the folder is `root/<split>`. A relative path of either is taken from
`root`. A sample is one entry of `images`, in the order of the list, and its file is the folder
joined with `file_name`. No image is checked to exist.

| Member of the file | Becomes |
| --- | --- |
| `categories[].id` | A class index: the rank of the id among all ids, in rising order. `class_names` holds `name` in that order |
| `annotations[].bbox` | `[x, y, width, height]` in pixels becomes a centre and an extent as fractions of `width` and `height` of its image, computed in double |
| `annotations[].iscrowd` | A whole number other than 0 leaves the annotation out |
| `annotations[].segmentation` | Not read with `COCOSegmentation::Ignored`. With `Polygons`, a list of flat `[x1, y1, x2, y2, ...]` parts gives the part of the largest area as the object's polygon, in fractions of the image |

An annotation is left out when its box has a width or a height that is not positive. A crowd
annotation is checked for its image and its category before it is left out, and its
`segmentation` is never read.

| Code | When |
| --- | --- |
| `FileNotFound` | The image folder is not a directory; the file is absent |
| `InvalidFile` | The text is not JSON; one of the three lists is absent; `images` or `categories` is empty; two entries of a list share an id, with both indices; an image has no `file_name` or no positive size, with its index; an annotation names an image or a category the file lacks, with its index; a `bbox` is not four numbers, with its index; with `Polygons`, a `segmentation` that is a run-length mask or a part that is not at least three pairs of numbers, with the index of the annotation |
| `OutOfMemory` | An allocation fails |

An index counts from 0 in the order of the list, as in "annotation 12".

`Config::read_json_events` streams the file to the reader's handler through a 64 KiB buffer,
so no document and no tree of `Config::Value` is held. The reader keeps the three lists it
returns and nothing else of the file. A file that states `images`, `annotations` or
`categories` twice is refused.

### xml_element.hpp

`parse_xml` reads a document into its root `XMLElement`: a name, the text, the child elements
and the line of the opening tag.

| Read | How |
| --- | --- |
| A UTF-8 byte order mark and one `<?xml ...?>` declaration at the start | Skipped |
| Comments, before, inside and after the root | Skipped |
| `<name>...</name>` and `<name/>` | An element; a name opens with an ASCII letter, `_` or `:` and goes on with those, digits, `-` and `.` |
| Attributes, `name="value"` or `name='value'` | Read and dropped |
| Character data | Appended to the text of the element it is in; the text loses the white space around it |
| `&amp;` `&lt;` `&gt;` `&quot;` `&apos;` | Replaced by their characters |

Every other thing is `InvalidFile`, and the message opens with `line <n>`: another entity,
a numeric character reference, a CDATA section, a `<!...>` declaration, a `<?...?>` instruction
other than the leading declaration, a closing tag of another name, an element or a comment left
open, an attribute without `=` or without quotes, text outside the root, a second root, and more than
`K_XML_MAX_DEPTH` (64) elements open at once. Bytes above ASCII pass through text unchanged and
are not allowed in a name. `find_child` returns the first child of a name.

### voc_xml.hpp

`read_voc_xml` reads the layout of the Pascal VOC development kit below `root`:

| Path | Holds |
| --- | --- |
| `ImageSets/Main/<split>.txt` | One image stem per line; what follows the stem on its line is not read. With `split` empty the reader takes every `.xml` file of `Annotations`, sorted |
| `Annotations/<stem>.xml` | One `annotation` element: `size` with `width` and `height`, and one `object` per object with `name`, `bndbox` (`xmin`, `ymin`, `xmax`, `ymax`) and `difficult` |
| `JPEGImages/` | The images; a sample's file is this folder joined with the text of `filename`, or with `<stem>.jpg` when the annotation has none |

The layout counts pixels from 1 and a box includes both of its ends, so a box spans
`[xmin - 1, xmax]` by `[ymin - 1, ymax]` in the continuous coordinates of `vision/imaging`; a
start below 1 is taken as 1. The box is stored as fractions of `width` and `height`. An object
whose `difficult` is `1` is left out with `VOCDifficult::Dropped`, the default.

`class_names` gives class `i` at entry `i`. When it is empty the classes are the names the
objects of the split use, sorted by byte, so a split that lacks a class has fewer classes.

| Code | When |
| --- | --- |
| `FileNotFound` | The split has no list; a line of the list names a stem with no annotation, with the list and the line |
| `InvalidFile` | `parse_xml` refuses an annotation, with the file and the line; the root is not `annotation`; `size`, `width`, `height`, `name`, `bndbox` or an edge is absent, or a number is not one, with the file and the line; a box ends before it starts; an object names a class the class list lacks, with the file and the line of the object; the split holds no annotation |
| `InvalidDescription` | The class list names a class twice; the split holds no object and the class list is empty |
| `OutOfMemory` | An allocation fails |

### mask_png.hpp

`read_mask_png` lists the files with an image extension at any depth below
`root/<image_folder>/<split>`, sorted by path, and pairs each with the file of the same relative
path and the extension `.png` below `root/<mask_folder>/<split>`. The folders default to `images`
and `masks`; an empty `split` reads the two folders themselves. `files` holds the images and then
the masks, and a sample's `mask` is the index of its mask. No file is opened, so `class_count`
comes from the caller and no mask is checked against it.

`image_suffix` and `mask_suffix` pair names that differ. The mask of
`<image_folder>/<split>/a/b<image_suffix>.jpg` is `<mask_folder>/<split>/a/b<mask_suffix>.png`,
and both are empty by default. With a `mask_suffix`, only the `.png` files whose stem ends in it
are masks; the others in the mask folder are not listed. Cityscapes pairs with
`"images": "leftImg8bit"`, `"masks": "gtFine"`, `"image_suffix": "_leftImg8bit"` and
`"mask_suffix": "_gtFine_labelIds"`.

| Code | When |
| --- | --- |
| `InvalidDescription` | `class_count` is below 1 |
| `FileNotFound` | One of the two folders is not a directory; an image has no mask, with both paths |
| `InvalidFile` | The image folder holds no image; an image whose stem does not end in `image_suffix`, with its path; a mask has no image, with its path; two images share a stem and so a mask; a directory cannot be listed |
| `OutOfMemory` | An allocation fails |

A mask extension is compared as `.png` in lower case, and a suffix byte for byte in UTF-8.

### csv.hpp

`read_csv` reads `root/<split>.csv`, or `file` when it is set. The first record that is no blank
line is the header. `targets` names the target columns and defaults to the last column;
`features` names the feature columns and defaults to every column that is no target, in file
order. A column that is neither is not read, so it may hold text. A sample is one record below
the header, and its row of `SampleList::table` holds its features and then its targets.

| Syntax | Rule |
| --- | --- |
| Records | End at a line feed, a carriage return and a line feed, or the end of the file. A line with nothing on it is skipped |
| Fields | Separated by commas. A field in double quotes may hold commas and line breaks, and `""` is one quote |
| Header names | Lose the spaces and tabs around them and must differ from each other |
| Cells | Lose the spaces and tabs around them and must spell a finite number a `float` holds, as `std::from_chars` reads a double, with one leading `+` allowed |
| Start of the file | A UTF-8 byte order mark is skipped |

| Code | When |
| --- | --- |
| `FileNotFound` | The file is absent |
| `InvalidDescription` | `features` or `targets` names a column the header lacks, with the names it has; a column is named as both; no column is left as a feature |
| `InvalidFile` | A cell of a column that is read is not a finite number, with the line, the column and its name; a record has another field count than the header, with the line; a quoted field is not closed, text follows a closing quote, or a quote stands inside a field that does not open with one, with the line and the column; a header name occurs twice; the file has no header or no record |
| `OutOfMemory` | An allocation fails |

A column counts from 1. For a cell and for a field count the line is the line the record starts
on, also when a quoted field of the record holds a line break. A failure of a quote names the
line the quote is on.

### load_sample_image.hpp

`load_sample_image` takes a list and the index of a sample. A sample with `stored` set goes
through `read_stored_pixels` on its file; any other goes through `Codecs::load_image` with the
`DecodeOptions` passed in. It converts no channel and resizes nothing.

| Code | When |
| --- | --- |
| `InvalidDescription` | The sample index, or the sample's file index, is outside the list |
| Any other | What `read_stored_pixels` or `Codecs::load_image` returns |

### fit_sample.hpp

`fit_sample` turns a loaded image and its `Sample` into a `FittedSample`: the pixels at the
input size, the objects as pixel boxes of that image, and the class index and file index copied.

| `SampleFit` | Image | Objects |
| --- | --- | --- |
| `Resize` | `Imaging::resize_bilinear` to the input size; each axis has its own scale | `Imaging::to_pixel_labels` on the input size |
| `Letterbox` | `Imaging::plan_letterbox` and `Imaging::apply_letterbox` with `SampleFitOptions::letterbox` | `Imaging::to_pixel_labels` on the source size, then `Imaging::apply_letterbox` with the same plan |

After either fit the objects go through `Imaging::clip_labels` and `Imaging::remove_empty_boxes`,
so a box lies inside the input and has an area above zero. The two fits are the table `K_FITS`
in `source/fit_sample.cpp`.

| Code | When |
| --- | --- |
| `InvalidDescription` | The input size is outside [1, `Imaging::K_MAX_IMAGE_SIDE`], or the fit names no row of the table |
| `OutOfMemory` | An allocation fails |
| `InvalidFile` | A function of `vision/imaging` throws, such as for a source of no pixels; the message holds its text |

### load_sample_mask.hpp

`load_sample_mask` decodes the file a sample's `mask` names through `Codecs::load_image` with
`ChannelLayout::Stored` and `PaletteRead::Indices`, so a palette PNG gives its indices and a
gray PNG its bytes. It is `InvalidDescription` for a sample index outside the list, a sample
with no mask and a mask index outside `files`; any other failure is the decoder's.

### fit_mask.hpp

`fit_mask` fits a one-channel mask with the geometry `fit_sample` gives the image it belongs
to, by nearest neighbour: fitted column `x` of `W` takes source column
`min(floor((2x + 1) * w / (2W)), w - 1)` of `w`, the column the centre of the fitted pixel falls
in, and rows likewise. No two classes are blended.

| `SampleFit` | Mask |
| --- | --- |
| `Resize` | Each axis is scaled to the input size |
| `Letterbox` | `Imaging::plan_letterbox` over the mask's size with `SampleFitOptions::letterbox`; the mask is scaled to `plan.resized` and written from row `plan.top` and column `plan.left`; every other byte is `K_MASK_VOID`, 255 |

| Code | When |
| --- | --- |
| `InvalidDescription` | The input size is outside its bounds, or the fit names no row of the table |
| `Incompatible` | The mask has more than one channel, or another size than its image |
| `InvalidFile` | `Imaging::plan_letterbox` throws; the message holds its text |
| `OutOfMemory` | An allocation fails |

`vision/imaging` has no nearest-neighbour resize and no letterbox for a mask.
`docs/design/DATASETS.md` gives mask geometry to `vision/segment`, which does not exist yet, so
the fit is a unit of this module until that one is written.

### batch_inputs.hpp

`write_batch_inputs` writes image `n` of a batch into row `n` of a float32 `[N, H, W, C]` tensor,
each byte divided by 255, the scaling of `source/mnist.cpp`. It is `Incompatible` when the tensor
is not float32 of rank 4, when `N` is not the sample count, or when an image is not `H` by `W` of
`C` channels; the message gives the image's position in the batch.

### collator.hpp

| Member | Does |
| --- | --- |
| `task()` | Returns the task's name as a training file spells it |
| `targets()` | Returns one `Data::TargetDescription` per role |
| `collate(list, samples, batch)` | Writes `batch.inputs` and every role; `list` is read for file names |

`batch` comes from `Data::allocate_batch` over `targets()`. A batch that lacks a role, or holds
it with another element type or count, is `Incompatible`. A sample the task cannot use is
`InvalidDescription`, and the message opens with the sample's file in quotes. Every collator of
images writes the inputs through `write_batch_inputs`; `RegressionCollator` copies
`FittedSample::features` into inputs of `[N, F]` float32.

| Collator | `make` | Role | Shape | Type | `classes` |
| --- | --- | --- | --- | --- | --- |
| `ClassificationCollator` | `class_count` | `labels` | `[N]` | int32 | `class_count` |
| `DetectionCollator` | `class_count`, `max_objects` | `boxes` | `[N, M, 4]` | float32 | 0 |
| | | `classes` | `[N, M]` | int32 | `class_count` |
| | | `valid` | `[N, M]` | float32 | 0 |
| `BinaryCollator` | `positive_class` | `labels` | `[N]` | float32 | 0 |
| `MultiLabelCollator` | `class_count` | `labels` | `[N, C]` | float32 | 0 |
| `RegressionCollator` | `feature_count`, `target_count` | `values` | `[N, K]` | float32 | 0 |
| `SegmentationCollator` | `SegmentationOptions` | `mask` | `[N, H, W]` | int32 | `class_count` |

`BinaryCollator` reads a set of two classes and writes 1 for a sample whose `class_index` is
`positive_class` and 0 for the other. A sample without a class index, or with one outside
[0, 2), fails.

`MultiLabelCollator` writes row `n` as `C` flags: 1 for every class one of the sample's objects
has, and for its `class_index` when it has one; 0 elsewhere. A sample with neither has a row of
zeros. The objects are the ones `fit_sample` kept, so a class whose only box has no area after
the fit is not flagged. A class outside [0, `C`) fails.

`RegressionCollator` copies the `F` features of sample `n` into row `n` of the inputs and its
`K` targets into row `n` of `values`. A sample that does not hold `F` and `K` numbers fails, as
a sample of an image does.

`SegmentationCollator` writes pixel `(y, x)` of `FittedSample::mask` into `mask[n, y, x]`. A
byte below `class_count` is a class. The byte `K_MASK_VOID`, 255, which is also the padding of
`fit_mask`, becomes `ignore_index`, 255 by default. Any other byte fails with the file, the row
and the column. `class_count` is in [1, 255], because a mask holds bytes and 255 names no class;
`ignore_index` is any int32 outside [0, `class_count`). A sample with no mask of the input size
fails.

`SegmentationOptions::origin` selects where that mask comes from. `MaskOrigin::File`, the
default, reads `FittedSample::mask`. `MaskOrigin::Polygons` paints the polygons of
`FittedSample::objects` with `paint_polygon_mask`, after the fit, so the vertices are already in
pixels of the input and nothing is resampled. `background` is the class of a pixel no polygon
holds; left out, such a pixel is `K_MASK_VOID` and becomes `ignore_index`. The padding of a
letterbox and the border of a warp hold no polygon, so they take the background too. A
`background` that is no class, or one stated with `MaskOrigin::File`, is `InvalidDescription`.

`ClassificationCollator` copies `class_index`. A sample without one, or with one outside
[0, `class_count`), fails.

`DetectionCollator` writes object `m` of image `n` into slot `m`. A box is `x1, y1, x2, y2` in
pixels of the input, the units of `Detect::GroundTruthBatch`, which `AnchorTrainingStep::run`
takes. `valid` is 1 in a slot that holds an object and 0 in a padding slot, whose box and class
are 0. `M` is `max_objects`. An image with more than `M` objects fails and names its file; no
object is dropped. An object whose class is outside [0, `class_count`) fails the same way.

The role names are `K_ROLE_LABELS`, `K_ROLE_BOXES`, `K_ROLE_CLASSES`, `K_ROLE_VALID`,
`K_ROLE_VALUES` and `K_ROLE_MASK` of `<SushiAI/data/roles.hpp>`.

### polygon_mask.hpp

`paint_polygon_mask(objects, size, background, mask)` resets `mask` to `size` with one channel,
fills it with `background` and paints the polygon of each object with the object's class, in
the order of the label set. Where two objects overlap, the later one wins. A polygon holds the
pixels whose centres it holds by the even-odd rule of `Segment::fill_outline`.

| Code | When |
| --- | --- |
| `InvalidDescription` | An object has a box and no polygon; a class is outside [0, 254]; a size outside the bounds of an image; a vertex that is not finite |
| `OutOfMemory` | An allocation fails |

An object holds one polygon, so its outline has one part and no hole. `Segment::ClassOutline`
already takes several parts per object. A multi-part outline needs `Imaging::LabelSet` to hold
a list of polygons per object, `read_coco_json` to keep every part and the transforms of
`vision/imaging` to move each part; nothing in this unit would change but the span it passes.

### sample_cache.hpp

`write_sample_cache(list, options)` decodes each image file of a list once with
`Codecs::load_image` and writes it as a raw image file of `vision/imaging` under
`options.directory`. `use_sample_cache(list, options)` returns the list with each of those
files replaced by its cache file, so `load_sample_image` reads raw pixels. A file a sample
reads stored pixels from, and a mask file, are neither cached nor replaced.

| `SampleCacheOptions` field | Default | Meaning |
| --- | --- | --- |
| `directory` | none | Where the cache files are; created when absent |
| `longest_side` | `0` | An image whose longer side exceeds it is reduced to it with `Imaging::resize_bilinear` before it is written; 0 keeps every stored size, and a smaller image is never enlarged |
| `layout` | `Color` | The channels an image is decoded to: `Gray` gives one, `Color` three |

The cache file of a source is `<directory>/<stem>-<hash>.saim`. `<stem>` is the source's file
name without its extension. `<hash>` is sixteen hexadecimal digits of the 64-bit FNV-1a hash of
the text `<path>|<bytes>|<modified>|<longest_side>|<layout>`: the source path as the list holds
it, with forward slashes, in UTF-8; its byte count; the tick count of its modification time;
and the two options. `sample_cache_path` returns it.

A source that changed in size or in modification time has another name, so its old cache file
is not found: `use_sample_cache` returns `FileNotFound` with the source, and the next
`write_sample_cache` writes the new file. That is the whole staleness check, and it opens no
cache file. The old file stays in the directory; nothing prunes it. A path spelled differently,
for instance a root given once relative and once absolute, is another source.

A cache file is written under `<name>.partial` and renamed, so a file with its final name is
whole. The tick count is the one of `std::filesystem::file_time_type` of the platform, so a
cache copied to another operating system is written again.

A list with masks is cached at its stored size only: a `longest_side` would leave each mask at
another size than its image, which `fit_mask` refuses.

| Code | From | When |
| --- | --- | --- |
| `InvalidDescription` | `write_sample_cache` | No directory; a `longest_side` outside [0, 16384]; a `longest_side` above 0 over a list with masks; a sample that names no file of the list |
| `FileNotFound` | `use_sample_cache`, `sample_cache_path` | A source that is absent; a source whose cache file is absent |
| `Unsupported` | `write_sample_cache` | A decoded image of two or four channels |
| Any other | all | What `Codecs::load_image` returns; `InvalidFile` for what the file system or the raw image writer throws |

Disk size, by estimate and not measured: COCO train2017 has 118 287 images, most of them 640
pixels on the longer side already, about 280 000 pixels each. At three bytes a pixel that is
about 0.84 MB an image and about 99 GB for the split, at `longest_side` 640 or at the stored
size. The upper bound, every image 640 by 640, is 145 GB. The JPEG files are about 18 GB.

The pass runs on one thread and decodes one image at a time.

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

### table_dataset.hpp

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

### reader_table.hpp

A row is a name, a function that takes a `ReaderRequest`, and the keys of the section the row
reads. A `ReaderRequest` holds the root, the split, the class count the section states, 0 when
it states none, the section itself and what a diagnostic calls it. A request built without a
section holds a null one, and a row then takes the default of each of its keys.

| Row | Calls | Reads of the request | Own keys |
| --- | --- | --- | --- |
| `yolo_txt` | `read_yolo_txt` | Root, split and class count; a count of 0 is refused by the reader | none |
| `image_folder` | `read_image_folder` | Root and split | none |
| `idx` | `read_idx` | Root and split; the files take the MNIST names of the split | none |
| `cifar` | `read_cifar` | Root and split; a class count of 0 or 10 reads CIFAR-10, 100 the fine labels of CIFAR-100, 20 the coarse ones, and any other is `InvalidDescription` | none |
| `coco_json` | `read_coco_json` | Root and split | `annotations`, a string: the JSON file. `images`, a string: the image folder. `polygons`, a boolean, `false` by default: `true` keeps polygons |
| `voc_xml` | `read_voc_xml` | Root and split | `class_names`, an array of strings. `keep_difficult`, a boolean, `false` by default |
| `mask_png` | `read_mask_png` | Root, split and class count; a count of 0 is refused by the reader | `images` and `masks`, strings: the two folders, `images` and `masks` by default. `image_suffix` and `mask_suffix`, strings, empty by default |
| `csv` | `read_csv` | Root and split | `file`, a string. `features` and `targets`, arrays of strings |

A key left out, or an empty string, gives the default of the reader's options. A value of
another kind is a thrown `Error` that names the key, as it is for a shared key.

A new reader is a function of that signature and one row of `K_READERS`.

### task_table.hpp

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

### reader_section.hpp

`make_reader_dataset(engine, section, context)` has the signature of a `Data::DatasetFactory`.
`vision/registration` adds it to a `Data::DatasetRegistry` as the row `reader`, and
`Config::parse_train` gives that type to a data block that spells `reader` and no `type`.

| Key | Kind | Default | Meaning |
| --- | --- | --- | --- |
| `reader` | string | required | A row of the reader table |
| `task` | string | required | A row of the task table |
| `root` | string | required | The directory of the dataset; a relative one resolves against the working directory |
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
| `InvalidDescription` | A reader, a task or a pipeline no table holds, with every name the table has; an `input` that is not two entries; a count below its lowest value or above the largest int32; `classes` below the count of the list; a task of images over a reader of rows, or the reverse; `prefetch` without `workers`; `cache_side` without `cache`; an image key on a section of rows |
| A thrown `Error` | A key that neither the shared list, nor the reader, nor the task holds; a required key left out; a value of another kind |
| Any other status | What the reader, the collator's `make` or the dataset's `make` returns |

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
`modules/vision/registration/tests/test_datasets_mnist.cpp`. `test_datasets_collators.cpp`,
`test_datasets_task_collators.cpp`, `test_datasets_reader_dataset.cpp`,
`test_datasets_table_dataset.cpp` and `test_datasets_reader_section.cpp` take an engine from
`SushiAI::Test::EngineTest`, because `Data::allocate_batch` allocates through one. A test that
needs a decodable image or mask writes a raw image file with `Support::write_raw_image`;
`vision/codecs` picks its decoder from the leading bytes, so such a file may be named `.png`.

| File | Suite | Covers |
| --- | --- | --- |
| `test_datasets_stored_pixels.cpp` | `Unit_DatasetsStoredPixels` | Interleaved and planar copies, a plane longer than one read, a short file, an absent file, extents no image has |
| `test_datasets_image_folder.cpp` | `Unit_DatasetsImageFolder` | The extension table, sorted classes and files, nested files, a split folder, an empty class, the refusals |
| `test_datasets_yolo_txt.cpp` | `Unit_DatasetsYoloTxt` | The label path rule, the folder and the list form, box and polygon rows, an image without labels, the file and line of a bad label and of an absent listed image |
| `test_datasets_idx_file.cpp` | `Unit_DatasetsIdxFile` | The header and the payload of a well-formed file, each malformed header, a size that disagrees with the header, extents that overflow |
| `test_datasets_idx.cpp` | `Unit_DatasetsIdx` | The samples, offsets and classes of a split, the pixels read back, named files, four dimensions, the refusals |
| `test_datasets_cifar.cpp` | `Unit_DatasetsCifar` | The five training files in order, the planes read back, the fine and the coarse label, the record of a bad label, a cut file, an absent file, another split |
| `test_datasets_load_sample_image.cpp` | `Unit_DatasetsLoadSampleImage` | Stored pixels read from their offset, a raw image file decoded, a sample and a file index outside the list, an absent file |
| `test_datasets_fit_sample.cpp` | `Unit_DatasetsFitSample` | A resize and its boxes, a letterbox with its padding and its boxes, an object without area removed, an input size out of bounds, a source of no pixels |
| `test_datasets_collators.cpp` | `Unit_DatasetsCollators` | The scaled inputs and their refusals, the descriptions and the values of both collators, a sample without a class, a batch without the role, an image with more than `M` objects, a class outside the set |
| `test_datasets_reader_section.cpp` | `Unit_DatasetsReaderTable`, `Unit_DatasetsTaskTable`, `Unit_DatasetsReaderSection` | The rows of both tables and the keys of each, the cifar row's class counts, the keys the csv, mask_png, voc_xml and coco_json rows read, the `positive` and `ignore_index` keys, a detection section over a YOLO split read into a letterboxed batch, a classification section over class folders, a regression section over a CSV file, a segmentation section over an image and a mask folder with its padding, a binary section over two folders, a multi-label section over a YOLO split, a mask of another size than its image, a task over a reader of the other sample kind, a class count above and below the list's, an unknown reader and task with the known ones, a key of another task and of another reader, a missing `max_objects`, values out of bounds, the reader's failure, a cache written and then read with the sources removed, a shuffled section under the `default` pipeline, a pipeline the table lacks, the batches of a section with `workers` equal to those without, `prefetch` without `workers`, `cache_side` without `cache`, an image key on a section of rows, a segmentation section that paints the polygons of a label file |
| `test_datasets_xml_element.cpp` | `Unit_DatasetsXmlElement` | Elements, text and lines, the five entities, the declaration, comments and attributes skipped, and the line of each refusal: a closing tag of another name, an element left open, an unknown entity, markup outside the subset, a malformed attribute, no root or two, nesting past the depth |
| `test_datasets_voc_xml.cpp` | `Unit_DatasetsVocXml` | A listed split with sorted classes and the box conversion, an empty split, a difficult object dropped and kept, a stated class list, and the file and line of a malformed annotation, a missing element, a number that is none, a reversed box, an unknown class; an absent list and an absent annotation with the line of the list |
| `test_datasets_coco_json.cpp` | `Unit_DatasetsCocoJson` | Boxes as fractions, sparse ids as dense classes, crowd and empty boxes left out, a named file and folder, a polygon kept on request and the largest part taken, and the index of each refusal: a run-length mask, a malformed polygon, an unknown image and category, a malformed box, a shared id, an image of no size; an absent file and folder, text that is no JSON, a missing list |
| `test_datasets_mask_png.cpp` | `Unit_DatasetsMaskPng` | Images paired with the masks of their stems in sorted order, named folders with no split, an image without a mask, a mask without an image, two images of one stem, a class count below 1, an absent and an empty folder, the two suffixes over names as Cityscapes spells them, an image without the image suffix, a mask of the suffix without an image |
| `test_datasets_sample_cache.cpp` | `Unit_DatasetsSampleCache` | A first pass that writes and a second that reuses, the pixels read back through the cache, a longest side that reduces only what exceeds it, a source that changed found stale and cached again, stored pixels and masks left alone, the name of a cache file, the refusals |
| `test_datasets_pipelines.cpp` | `Unit_DatasetsPipelineTable`, `Unit_DatasetsFitPipeline`, `Unit_DatasetsAugmentPipeline` | The two rows and what each builds, the inputs `default` refuses; a fit that is the same in every epoch; training samples of the training size, a sample that is the same alone, in a batch and in a reversed batch and differs in the next epoch, the objects past the mosaic epochs, a mask at the training size, an unreadable image as the failure, the refusals |
| `test_datasets_polygon_mask.cpp` | `Unit_DatasetsPolygonMask` | The pixels a polygon paints, the later object winning an overlap, a set of no object, an object without a polygon, a class no byte names |
| `test_datasets_csv.cpp` | `Unit_DatasetsCsv` | The default and the named columns, quoted fields, a doubled quote, a line break inside quotes, carriage returns and a byte order mark, and the line and column of a cell that is no number; a record of another field count, the three malformed quotes, a column the header lacks, a header name used twice, a file with no record |
| `test_datasets_fit_mask.cpp` | `Unit_DatasetsFitMask`, `Unit_DatasetsLoadSampleMask` | An enlarged and a reduced mask by nearest neighbour, a letterbox with its void padding with and without a reduction, a mask of several channels or of another size; a mask file decoded, a sample without a mask, indices outside the list, an absent file |
| `test_datasets_task_collators.cpp` | `Unit_DatasetsTaskCollators` | The description and the values of the binary, multi-label, regression and segmentation collators, either positive class, a stale row overwritten with zeros, the ignore index at its default and at -100, and the refusals: a sample without a class, a third class, a class outside the set, a sample with no row, inputs of another width, a byte that is no class with its pixel, a sample without a mask, polygons painted with and without a background, an object without a polygon, a background that is no class or stated over mask files |
| `test_datasets_table_dataset.cpp` | `Unit_DatasetsTableDataset` | The queries, a batch in list order read through `Data::Dataset`, a second read reusing the tensors, the refusals of `make`, a batch index out of range stopping the process, the collator's failure |
| `test_datasets_reader_dataset.cpp` | `Unit_DatasetsReaderDataset` | The queries, a batch in list order, a letterboxed detection batch from raw image files, the refusals of `make`, a batch index out of range stopping the process, a second read reusing the tensors, a read leaving a shared tensor alone, a read through `Data::Dataset`, a channel count that differs, a file absent at read time, a shuffled epoch in the order of its seed with every sample once, the order `begin_epoch` selects, the list order kept without shuffle, a null pipeline |

The coordinate tolerance of `test_datasets_yolo_txt.cpp` is 1e-6, absolute: the values are
decimal fractions parsed to `float`.
A byte divided by 255 is compared with an absolute tolerance of 1e-6, and a box coordinate after
a fit with 1e-4 pixels, absolute. The values a collator copies are compared exactly. A box
fraction of `read_voc_xml` and `read_coco_json` is a quotient computed in double and compared
with 1e-6, absolute. The cells of the CSV tests are numbers a `float` holds exactly and are
compared exactly.
