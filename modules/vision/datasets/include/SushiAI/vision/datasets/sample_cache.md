# modules/vision/datasets/include/SushiAI/vision/datasets/sample_cache.hpp


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
