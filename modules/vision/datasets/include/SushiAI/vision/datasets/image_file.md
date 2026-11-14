# modules/vision/datasets/include/SushiAI/vision/datasets/image_file.hpp


`read_image_file` lists the one image at `root/file` as a sample with no object and no class. The
image is decoded when the sample is read. It serves `sushiai inference` on one picture, where
there is no truth to compare with.

| Code | When |
| --- | --- |
| `InvalidDescription` | The class count is below 1, or the file has no image extension |
| `FileNotFound` | The file does not exist |
| `InvalidFile` | The path is no regular file |
| `OutOfMemory` | An allocation fails |

## Reading from memory

`read_image_file(options, source)` does what `read_image_file(options)` does, with the file
looked up through a `ByteSource` (`byte_source.hpp`). The one-argument form passes the file system.
