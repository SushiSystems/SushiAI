# tests/vision/imaging/test_fit_transform.cpp

Tests `resize_transform`, `letterbox_transform` and `box_in_source`.

| Suite | Cases |
| --- | --- |
| `Small_ImagingFitTransform` | A stretched box returns by each axis scale; a letterboxed box returns past the padding; the box of `apply_letterbox` returns exactly; a box in the padding is clipped to the source |

Every value is dyadic, so the comparisons are exact and no tolerance is stated.
