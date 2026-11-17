# modules/vision/imaging/include/SushiAI/vision/imaging/darknet_augment.hpp


## DarknetAugmentSettings, validate_darknet_augment

The four keys of a Darknet `.cfg` that augment a sample: `jitter` of the `[yolo]` section and
`hue`, `saturation` and `exposure` of `[net]`. The identity is 0, 0, 1, 1. Darknet draws a crop
edge up to `jitter` of a side beyond the image on each side, so a jitter of 0.5 or more can
leave no crop; `validate_darknet_augment` refuses it. A saturation or exposure below 1 is
refused as well: Darknet's `rand_scale` then draws between `s` and 1, which no `.cfg` of the
YOLO family states.

## plan_darknet_augment

Darknet's `load_data_detection` (`data.c` of the Darknet repository):

```
dw = ow * jitter                 dh = oh * jitter
pleft = rand_uniform(-dw, dw)    pright = rand_uniform(-dw, dw)
ptop  = rand_uniform(-dh, dh)    pbot   = rand_uniform(-dh, dh)
swidth = ow - pleft - pright     sheight = oh - ptop - pbot          (truncated to int)
crop_image(orig, pleft, ptop, swidth, sheight)                       (offsets truncated to int)
random_distort_image: dhue = rand_uniform(-hue, hue), dsat = rand_scale(s), dexp = rand_scale(e)
```

`rand_scale(s)` is `scale = rand_uniform(1, s)`, returned as it is on a coin flip and as `1 / scale`
on the other side. It is not log-uniform: the factor is uniform in [1, s] with probability 1/2
and the reciprocal of such a value otherwise, so it lies in [1 / s, s] and is dense near 1.
The plan draws the same nine numbers in the same order from `Random`; Darknet's flip draw, which
sits between the crop and the colour, is not made because the `.cfg` has no flip key here.

`swidth` and `sheight` are clamped to 1 so that a small image does not fail on an unlucky draw.

## apply_darknet_augment

Image: the crop reads the source at the truncated offsets and repeats the edge pixel for a
position outside it (`crop_image` clamps its row and column), the crop is resized to the input
with `resize_darknet`, and the colour goes through `apply_hsv`: the hue shifts by `dhue` of a
turn, saturation and value scale by `dsat` and `dexp`, with the clamps and the 8-bit hexcone of
`hsv.hpp`. Darknet distorts in float HSV; the two differ by the rounding of the 8-bit hexcone.
The Ultralytics draw of `plan_hsv` is not used; only the plan it fills and the tables and
hexcone that read it.

Labels: Darknet's `correct_boxes` moves a box by the unrounded shift, so a box coordinate is
`(x - pleft) * input / swidth`. `darknet_augment_transform` states that map for the fit record
of a sample.
