# modules/vision/datasets/include/SushiAI/vision/datasets/stored_pixels.hpp


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
