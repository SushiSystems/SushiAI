# modules/vision/detect/include/SushiAI/vision/detect/anchor/head_options.hpp


## HeadCoding, HeadOptions

`HeadCoding` names how the published values of a head encode a box. Under `Logit` the offsets,
the objectness and the classes are logits and the sizes are exponents. Under `Activated` the
layer before the head already applied a logistic, every entry arrives in (0, 1), and the decode
applies none. `HeadOptions` adds the class count and `scale_xy`, the stretch of an offset about
the cell centre.

## HeadShape, NetworkSize

`HeadShape` is the batch, the grid rows and the grid columns of a head's
`[B, H, W, A * (5 + C)]` output. `NetworkSize` is the input in pixels; the decode does not read
it, and a caller that needs pixel boxes multiplies by it.

## The entries of a slot

A head has `A` anchor slots per cell and each slot has `5 + C` entries. `K_X_ENTRY`,
`K_Y_ENTRY`, `K_WIDTH_ENTRY`, `K_HEIGHT_ENTRY` and `K_OBJECTNESS_ENTRY` are 0 to 4;
`K_FIRST_CLASS_ENTRY` is 5, and class `c` is entry `5 + c`. Entry `e` of slot `a` is channel
`a * (5 + C) + e`.

## validate_head, count_head_channels, count_head_rows

`validate_head` returns the failure of `validate_anchor_layout`, or `InvalidDescription` for a
head index outside the layout, a class count below 1, or a `scale_xy` that is not finite and
above zero. `count_head_channels` is `A * (5 + C)`, the channel count the convolution before the
head must produce. `count_head_rows` is `H * W * A`, the rows one image decodes to; it does not
multiply by the batch. Both counts are `noexcept` and stop the process on a head index outside
the layout.
