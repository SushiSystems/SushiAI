# Convolutional networks

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-27 | P2 | open | Pad convolution and average pooling asymmetrically | `Conv2d` and `AvgPool2d` take trailing pads through the builder's `_with_extents` emitters, and the model file accepts them as `trailing_padding`, as… |  |  | 2026-10-25 |
| RW-29 | P2 | open | Give the layer-name prefix of a refused window one home | `Conv2d`, both pools and `Upsample2d` stop catching and re-throwing the builder's message, and the four copies of `fit_window` become one |  |  | 2026-10-19 |
| RW-30 | P2 | open | Check settings and ranks in one order across the layers | the geometry check of `window_extents` is its own function and every layer checks in one order |  |  | 2026-10-19 |
| RW-31 | P2 | open | Check the leaky slope in the element type | a slope that narrows to 0 or 1 in float32 is refused |  |  | 2026-10-19 |
| RW-33 | P2 | open | Pass the ResNet-18 on CIFAR-10 gate | a ResNet-18 model file and a CIFAR-10 training file exist, and the gate of the August design passes on them |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
