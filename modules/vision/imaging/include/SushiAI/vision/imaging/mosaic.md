# modules/vision/imaging/include/SushiAI/vision/imaging/mosaic.hpp


## MosaicSettings, validate_mosaic, validate_mosaic_size, draw_mosaic

`mosaic` is 1.0 (`cfg/default.yaml:126`); the canvas fill is 114 (`data/augment.py:611`).

`validate_mosaic_size` owns the rule that the 2s canvas is a valid image: `image_size` is in
[1, `K_MAX_MOSAIC_IMAGE_SIZE`], which is `K_MAX_IMAGE_SIDE / 2`. `draw_mosaic`, `plan_mosaic`, both `apply_mosaic` overloads and `validate_training` call it, so
its message opens with `validate_mosaic_size:` whoever the caller is.

Draws, in order: the three partners through the `PartnerSource`, the gate, `center_y`,
`center_x`. The reference draws the gate first (`data/augment.py:340`) and the partners after
it; the order differs here so that the source is asked for every sample. The reference skips
the mosaic when `random.uniform(0, 1) > p`; `draw_mosaic` makes it when
`bernoulli(probability)`, which differs only on a draw equal to `p`.

Centre (`data/augment.py:480`, `520`): `center_y` first, then `center_x`, each
`int(uniform(b, 2s - b))` with `b = (s + 1) / 2`. For s = 640 that is a value in 320 to 959.

## plan_mosaic

`data/augment.py:525-538`. With centre `(cx, cy)`, patch size `(h, w)` and canvas side `c = 2s`:

| Patch | Canvas `x1, y1, x2, y2` | Patch `x1, y1, x2, y2` |
| --- | --- | --- |
| 0, top left | `max(cx - w, 0), max(cy - h, 0), cx, cy` | `w - (x2a - x1a), h - (y2a - y1a), w, h` |
| 1, top right | `cx, max(cy - h, 0), min(cx + w, c), cy` | `0, h - (y2a - y1a), min(w, x2a - x1a), h` |
| 2, bottom left | `max(cx - w, 0), cy, cx, min(c, cy + h)` | `w - (x2a - x1a), 0, w, min(y2a - y1a, h)` |
| 3, bottom right | `cx, cy, min(cx + w, c), min(c, cy + h)` | `0, 0, min(w, x2a - x1a), min(y2a - y1a, h)` |

`x2a - x1a` is the width of the canvas rectangle in the same row. `offset_x = canvas_x1 -
patch_x1` and `offset_y = canvas_y1 - patch_y1`. Patch 0 is the sample; patches 1 to 3 are the
partners in draw order. Each patch is an image as `resize_long_side` returned it; the mosaic
resamples nothing.

## apply_mosaic

Image: a canvas of 2s by 2s filled with `fill_value`, then the four copies
(`data/augment.py:611-617`). The canvas is not cropped; `apply_affine` reduces it to s by s.

Labels: each patch's boxes and vertices plus its offsets (`data/augment.py:724`,
`utils/instance.py:315-317`), joined in patch order (`773-774`), clipped to [0, 2s] with
`clip_labels` (`776`), then `remove_empty_boxes` (`777-778`). The clip clamps each vertex on its
own, which is why the polygons were resampled to 1000 points at load.
