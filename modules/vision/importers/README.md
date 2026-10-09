# importers

## Owns

The reading of a Darknet model: the `.cfg` text, the `.weights` binary, the fold of batch
normalisation into a convolution, the copy of the weights into a `ModelState`, the translation of
a parsed `.cfg` into a `Config::ModelSpec`, and the solver settings and per-head training options
a `.cfg` holds. Everything is in namespace
`SushiAI::Vision::Importers`. The formats and the behaviour are in sections 4.1 to 4.5 of
`docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md`.

Nothing here is taken from Darknet's source. A content failure is a `Status` or a `Result`;
nothing throws across the module's surface.

## Depends on

- `core`, `config`, `nn` and `vision/detect` (`PUBLIC_DEPENDS`).

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/importers/darknet_config.hpp>` | `DarknetOption`, `DarknetSection`, `DarknetConfig`, `parse_darknet_config`, `load_darknet_config` |
| `<SushiAI/vision/importers/darknet_weights.hpp>` | `DarknetWeightsHeader`, `DarknetConvolutionShape`, `DarknetConvolution`, `DarknetWeights`, `read_darknet_header`, `read_darknet_weights` |
| `<SushiAI/vision/importers/darknet_fold.hpp>` | `fold_darknet_normalisation` |
| `<SushiAI/vision/importers/darknet_store.hpp>` | `DarknetWeightTarget`, `store_darknet_weights` |
| `<SushiAI/vision/importers/darknet_model.hpp>` | `DarknetNormalisation`, `DarknetTranslation`, `DarknetHead`, `DarknetModel`, `translate_darknet_model` |
| `<SushiAI/vision/importers/darknet_training.hpp>` | `DarknetSolver`, `DarknetTraining`, `read_darknet_training` |

## The translation

`translate_darknet_model` walks the layers of a `DarknetConfig` in file order. Each section type
has one translator in its own source file, `source/darknet_layer_<type>.cpp`, and one row in the
table of `source/darknet_layer_table.cpp`. A new section type is a new file and a new row.

| Section | Rows it becomes |
| --- | --- |
| `convolutional` | `conv2d`; `batchnorm2d` with `bn_epsilon` 1e-5 when normalised and the form is `Separate`; then the activation row |
| `maxpool` | `maxpool2d` with `padding` equal to half the section's total and the `trailing_padding` its extent needs |
| `upsample` | `upsample` with `scale` equal to the section's `stride` |
| `route` | `concat` over the listed layers; no row when it lists one layer, whose row the later layers read. With `groups` above 1 each listed layer first gets a `slice_channels` row holding its group `group_id` |
| `shortcut` | `add` over the layer before and the layer `from` names |
| `yolo` | No row. It publishes the row of the layer before it as `head.<k>` and adds a head to the layout |

Activations: `leaky` becomes `leaky_relu` with slope 0.1, `mish` becomes `mish`, `swish` becomes
`silu`, `logistic` becomes `sigmoid`, and `linear` adds no row.

`DarknetNormalisation::Separate` keeps the batch-norm row and is the form training uses. `Folded`
emits a convolution with a bias and no batch-norm row; the caller runs
`fold_darknet_normalisation` on each convolution before `store_darknet_weights`. In both forms
`DarknetModel::convolutions` describes the file, so `normalised` stays true for a folded layer.

A state entry is named `<row>.<field>`, the name `NN::build_model` gives it: `weight` and `bias`
for a `conv2d` row; `weight`, `bias`, `running_mean` and `running_var` for a `batchnorm2d` row.
`DarknetModel::targets` holds these names per convolution and `row_of_layer` the row that carries
each layer's output.

The `yolo` translator divides the anchors by the network's width and height; no pixel anchor
leaves it. Every `[yolo]` section of a file must list the same anchors. The suppression measure
is the one of the last `[yolo]` section: `Iou` without `nms_kind`, `DistanceIou` with exponent 0.6
for `greedynms`, and `DistanceIou` with `beta_nms` for `diounms`.

## Refusals

A section type or a key outside SPEC 4.1 is `Unsupported`, with its line. So are `letter_box=1`,
an `[upsample]` `scale` other than 1, an activation outside the list above, an `nms_kind` other
than `greedynms` and `diounms`, and a `[shortcut]` with `weights_type` or an activation other
than `linear`. A value that is malformed,
out of range or does not fit the layer it reads is `InvalidDescription`, with its line.

The keys `jitter`, `random` and `resize` of a `[yolo]` section are accepted and not read, with the
training keys.

## Training settings

`read_darknet_training` reads `[net]` into a `DarknetSolver` and every `[yolo]` section into a
`Detect::Anchor::HeadTrainingOptions`, in file order; an absent key keeps the default of its
field. It refuses a `policy` other than `steps`, a `truth_thresh` other than 1, a non-zero
`label_smooth_eps` and an `iou_loss` other than `mse` and `ciou` as `Unsupported`. A malformed
value, `steps` and `scales` of different lengths, and options `validate_head_training` refuses are
`InvalidDescription`.
