# modules/vision/datasets/include/SushiAI/vision/datasets/reader_table.hpp


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
