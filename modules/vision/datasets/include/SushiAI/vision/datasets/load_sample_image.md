# modules/vision/datasets/include/SushiAI/vision/datasets/load_sample_image.hpp


`load_sample_image` takes a list and the index of a sample. A sample with `stored` set goes
through `read_stored_pixels` on its file; any other goes through `Codecs::load_image` with the
`DecodeOptions` passed in. It converts no channel and resizes nothing. The form that takes a
`ByteSource` reads the file through it, with `read_stored_pixels` of the source or
`Codecs::decode_image` of the bytes; the form that takes none passes the file system.

| Code | When |
| --- | --- |
| `InvalidDescription` | The sample index, or the sample's file index, is outside the list |
| Any other | What `read_stored_pixels` or `Codecs::load_image` returns |
