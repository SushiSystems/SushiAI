# modules/vision/datasets/include/SushiAI/vision/datasets/idx.hpp


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
