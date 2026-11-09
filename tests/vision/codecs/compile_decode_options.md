# tests/vision/codecs/compile_decode_options.cpp


A compile-only unit: it holds no test and the runner never lists it. It checks at build time that `SushiAI/vision/codecs/decode_options.hpp` compiles on its own.

The file needs no device and runs nothing. It keeps `decode_options.hpp` under a file of its own in
this folder, which holds one test file per header. What each option does to a decoded image is
tested in `test_decode_image.cpp`, `test_jpeg_decoder.cpp` and `test_png_decoder.cpp`.
