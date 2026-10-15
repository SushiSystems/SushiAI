# modules/vision/datasets/include/SushiAI/vision/datasets/coco_json.hpp


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
