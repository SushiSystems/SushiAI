# tests/vision/codecs/test_decode_options.cpp


Suites: `Small_CodecsDecodeOptions`.

Tests `DecodeOptions` in `modules/vision/codecs/include/SushiAI/vision/codecs/decode_options.hpp`.
The header declares two enumerations and a struct and has no function, so the cases pin the
defaults and the storage the header states. What each choice does to a decoded image is tested
in `test_decode_image.cpp`, `test_jpeg_decoder.cpp` and `test_png_decoder.cpp`.

## TheDefaultsKeepWhatTheFileStores

A default `DecodeOptions` holds `ChannelLayout::Stored` and `PaletteRead::Colors`, so a caller
that sets nothing gets the channels the file stores and the colours of a palette image.

## BothEnumerationsFitInOneByte

The underlying type of `ChannelLayout` and of `PaletteRead` is `std::uint8_t`.
