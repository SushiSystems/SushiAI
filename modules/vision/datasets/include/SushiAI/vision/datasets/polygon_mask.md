# modules/vision/datasets/include/SushiAI/vision/datasets/polygon_mask.hpp


`paint_polygon_mask(objects, size, background, mask)` resets `mask` to `size` with one channel,
fills it with `background` and paints the polygon of each object with the object's class, in
the order of the label set. Where two objects overlap, the later one wins. A polygon holds the
pixels whose centres it holds by the even-odd rule of `Segment::fill_outline`.

| Code | When |
| --- | --- |
| `InvalidDescription` | An object has a box and no polygon; a class is outside [0, 254]; a size outside the bounds of an image; a vertex that is not finite |
| `OutOfMemory` | An allocation fails |

An object holds one polygon, so its outline has one part and no hole. `Segment::ClassOutline`
already takes several parts per object. A multi-part outline needs `Imaging::LabelSet` to hold
a list of polygons per object, `read_coco_json` to keep every part and the transforms of
`vision/imaging` to move each part; nothing in this unit would change but the span it passes.
