# modules/vision/detect/include/SushiAI/vision/detect/anchor/head_decode.hpp


## decode_head

Reads a head's `[B, H, W, A * (5 + C)]` output and writes `[B, H * W * A, 4]` boxes and
`[B, H * W * A, C]` scores. With `t` the entries of one slot in cell `(row, col)`, `(pw, ph)` the
slot's anchor, `s = scale_xy` and `act` the logistic under `Logit` and the identity under
`Activated`:

```
X  = act(t[x]) * s - (s - 1) / 2        Y  = act(t[y]) * s - (s - 1) / 2
bx = (col + X) / W                      by = (row + Y) / H
Logit:      bw = exp(t[w]) * pw         bh = exp(t[h]) * ph
Activated:  bw = t[w]^2 * 4 * pw        bh = t[h]^2 * 4 * ph
score_c = act(t[objectness]) * act(t[class_c])
```

A box is centre x, centre y, width and height as fractions of the network input. Row
`(row * W + col) * A + a` of an image holds slot `a` of cell `(row, col)`. The decode applies no
threshold and clamps nothing: with `scale_xy` above 1 a centre can leave its cell by
`(s - 1) / 2` of a cell, and a row whose objectness is near 0 is still written.

The arithmetic is float32 in the order written above. The two codings are two rule types with
the same pair of static functions, and the cell loop is a template over the rule, so the coding
is chosen once per call and not per entry. The call allocates nothing. The rules and the decode
of one slot's box are defined once, in the private `source/anchor/slot_decode.hpp`, and the
training delta reads a slot through the same definitions.

`decode_head` stops the process when `validate_head` fails on its arguments, when an extent of
the shape is below 1, and when a span does not hold exactly the values its shape gives.

`test_detect_anchor_decode.cpp` pins each formula with inputs at zero or at saturation, to 1e-6
absolute. A logit of 100 is the saturated input: its float32 logistic is exactly 1, and that of
-100 exactly 0. No test compares with a reference tensor.
