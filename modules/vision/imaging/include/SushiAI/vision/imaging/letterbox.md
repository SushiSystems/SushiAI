# modules/vision/imaging/include/SushiAI/vision/imaging/letterbox.hpp


## LetterboxSettings

The reference's validation transform is `LetterBox(new_shape=(imgsz, imgsz), scaleup=False)`
(`data/dataset.py:339`) with the constructor's other defaults (`data/augment.py:1704-1710`):
centred, padded with 114, no stride rounding, no stretch. `scale_up` and `padding_value` are the
two that can be set here. `auto`, `scale_fill` and an off-centre letterbox are off by default
and are not written. The training pipeline uses no letterbox, in the reference or here.

## plan_letterbox

`data/augment.py:1791-1811`:

```
ratio   = min(target.height / source.height, target.width / source.width)     (double)
ratio   = min(ratio, 1.0) unless scale_up
resized = (round_half_even(source.height * ratio), round_half_even(source.width * ratio))
dw      = target.width - resized.width          dh = target.height - resized.height
left    = floor(dw / 2)     right  = dw - left
top     = floor(dh / 2)     bottom = dh - top
```

The reference writes the split as `round(dw / 2 - 0.1)` and `round(dw / 2 + 0.1)` with Python's
`round`. For an integer `dw` those are `floor(dw / 2)` and `ceil(dw / 2)`, which is what the
lines above compute. `round_half_even` is Python's `round`.

## apply_letterbox

Image: the target is filled with `padding_value` outside the content rectangle, and the
content is the source copied, or resized by the kernel of `resize_bilinear` straight into rows
`top` onwards at column `left` when `plan.resized` differs from the source size
(`data/augment.py:1838-1849`). No scratch image is allocated.

Labels: `x * ratio + left`, `y * ratio + top` for boxes and vertices (`data/augment.py:1931-1932`,
called with `padw = left`, `padh = top` at `1893`). No clip and no filter, as in the reference.
