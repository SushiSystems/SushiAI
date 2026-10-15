# modules/vision/datasets/include/SushiAI/vision/datasets/image_extensions.hpp


`has_image_extension` is true for `.jpg`, `.jpeg`, `.png` and `.saim`, whatever the case of the
ASCII letters. The four are one table in `source/image_extensions.cpp`. `vision/codecs` picks a
decoder from a file's leading bytes and keeps no list of extensions, so the list lives here.
