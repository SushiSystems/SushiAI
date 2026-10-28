# modules/vision/detect/include/SushiAI/vision/detect/free/distribution_expectation.hpp


## expect_distances

The host decode of a box output with `reg_max` bins per side, for inference. Ultralytics' DFL
layer reshapes the `4 * reg_max` channels as `x.view(b, 4, reg_max, a)`, so channel
`side * reg_max + bin` holds bin `bin` of side `side`, sides in the order left, top, right,
bottom (`U/nn/modules/block.py`, class `DFL`). It takes the softmax over each side's bins and
applies a 1x1 convolution whose weights are `0, 1, ..., reg_max - 1`, the expected bin.

The softmax runs in double, shifted by the side's largest logit, and the result is narrowed once.
With `reg_max = 1` the layer is the identity and the logits are copied. A logit that is NaN or
infinite returns `InvalidDescription`, naming its index, and writes no distance.

## fill_distribution_projection

Writes the `[reg_max, 1]` leaf the graph form multiplies the probabilities by.
