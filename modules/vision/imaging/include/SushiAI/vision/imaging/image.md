# modules/vision/imaging/include/SushiAI/vision/imaging/image.hpp


## ImageSize, is_valid_side, is_valid_size, is_valid_channel_count

Height first, then width, the order of the reference's `shape[:2]`. `K_MAX_IMAGE_SIDE` is 16384
and `K_MAX_CHANNELS` is 4. Both are this module's bounds, not the reference's. They let the raw
image reader refuse a header before it allocates. `K_RGB_CHANNELS` is 3.

## ImageView, Image

`ImageView` is a pointer and three extents with no ownership and no right to write. `Image` owns
a `std::vector<std::uint8_t>`. `Image::reset` changes the extents and keeps the allocation when
it is large enough; the pixel values afterwards are unspecified and every earlier view is
invalid, so a function that resets a destination writes every byte of it.
`Image::allocated_bytes` is the length of the allocation, which never shrinks. `data()` of an
image that never allocated is unspecified; test `allocated_bytes() == 0` instead.

## overlaps, check_source

`overlaps(view, image)` is true when the view's bytes and the image's allocation share a byte.
`check_source(source, destination, operation)` stops the process, in this order, when the source
has no pixels, a size `is_valid_size` rejects, a channel count outside [1, 4], or an overlap with
the destination. A view is a handle: the code that makes one from a file tests it first.
