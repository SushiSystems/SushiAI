# modules/vision/datasets/include/SushiAI/vision/datasets/image_folder.hpp


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
