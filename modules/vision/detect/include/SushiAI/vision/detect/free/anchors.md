# modules/vision/detect/include/SushiAI/vision/detect/free/anchors.hpp


## count_anchors

The anchor count A of a list of levels is the sum of `height * width`: one anchor per cell
(`U/utils/tal.py:453`, `h * w` rows per level). It throws on a level without cells or with a
stride that is not positive. Every other unit sizes its `[A, ...]` arrays by this count.

`K_POINT_VALUES` is 2, the width of an anchor point, x then y.

## make_anchors

One anchor per cell, at the cell centre in grid units, levels in order and row-major inside a
level. Anchor `h * W + w` of a level is element `[b, h, w, :]` of that level's NHWC head output.

```
U/utils/tal.py:449  sx = torch.arange(w).type_as(feats[0]) + grid_cell_offset  # shift x
U/utils/tal.py:450  sy = torch.arange(h).type_as(feats[0]) + grid_cell_offset  # shift y
U/utils/tal.py:451  sy, sx = torch.meshgrid(sy, sx, indexing="ij") ...
U/utils/tal.py:452  anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
U/utils/tal.py:453  stride_tensor.append(feats[0].new_full((h * w, 1), stride, dtype=dtype))
```

The offset is 0.5 at both call sites (`U/utils/loss.py:424`, `U/nn/modules/head.py:223`), so it is
the constant `K_GRID_CELL_OFFSET` and not a parameter. Strides are written as `[A]`, not `[A, 1]`.

## scale_points

The assigner takes anchor points in pixels: `anchor_points * stride_tensor`
(`U/utils/loss.py:442`).
