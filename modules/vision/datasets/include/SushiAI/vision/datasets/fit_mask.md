# modules/vision/datasets/include/SushiAI/vision/datasets/fit_mask.hpp


`fit_mask` fits a one-channel mask with the geometry `fit_sample` gives the image it belongs
to, by nearest neighbour: fitted column `x` of `W` takes source column
`min(floor((2x + 1) * w / (2W)), w - 1)` of `w`, the column the centre of the fitted pixel falls
in, and rows likewise. No two classes are blended.

| `SampleFit` | Mask |
| --- | --- |
| `Resize` | Each axis is scaled to the input size |
| `Letterbox` | `Imaging::plan_letterbox` over the mask's size with `SampleFitOptions::letterbox`; the mask is scaled to `plan.resized` and written from row `plan.top` and column `plan.left`; every other byte is `K_MASK_VOID`, 255 |

| Code | When |
| --- | --- |
| `InvalidDescription` | The input size is outside its bounds, or the fit names no row of the table |
| `Incompatible` | The mask has more than one channel, or another size than its image |
| `InvalidDescription` | `Imaging::plan_letterbox` refuses the sizes; the status is its own |
| `OutOfMemory` | An allocation fails |

`vision/imaging` has no nearest-neighbour resize and no letterbox for a mask.
`docs/design/DATASETS/README.md` gives mask geometry to `vision/segment`, which does not exist yet, so
the fit is a unit of this module until that one is written.
