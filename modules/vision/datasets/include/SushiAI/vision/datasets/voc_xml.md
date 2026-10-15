# modules/vision/datasets/include/SushiAI/vision/datasets/voc_xml.hpp


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
