# modules/vision/datasets/include/SushiAI/vision/datasets/polygon_mask.hpp


`paint_polygon_mask(objects, size, background, mask)` resets `mask` to `size` with one channel,
fills it with `background` and paints every polygon of each object with the object's class, in
the order of the label set. Where two objects overlap, the later one wins. The parts of one
outline hold the pixels whose centres they hold by the even-odd rule of `Segment::fill_outline`,
taken over all parts together.

| Code | When |
| --- | --- |
| `InvalidDescription` | An object has a box and no polygon; a class is outside [0, 254]; a size outside the bounds of an image; a vertex that is not finite |
| `OutOfMemory` | An allocation fails |

An object holds an `Imaging::Outline`, one polygon per part, and passes all of them to
`Segment::ClassOutline` as one span. Two parts that do not touch are both filled, and a part
inside another cuts a hole in it.
