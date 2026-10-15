# modules/vision/datasets/include/SushiAI/vision/datasets/yolo_txt.hpp


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
