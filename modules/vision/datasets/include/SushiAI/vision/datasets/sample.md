# modules/vision/datasets/include/SushiAI/vision/datasets/sample.hpp


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
