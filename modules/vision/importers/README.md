# importers

## Owns

The reading of a Darknet model: the `.cfg` text, the `.weights` binary, the fold of batch
normalisation into a convolution, the copy of the weights into a `ModelState`, the translation of
a parsed `.cfg` into a `Config::ModelSpec`, the solver settings and per-head training options
a `.cfg` holds, and the Darknet importer, which returns all of it as an `IO::ImportedModel`.
The reading of an Ultralytics checkpoint: the walk of its module tree, the translation of its
`yaml` description into a model file, the layout of its tensors, and the Ultralytics importer.
The translation of an ONNX graph, read by `layers/interchange`, into rows and weights, and the
ONNX importer.
The translation of a Caffe net, read by `layers/interchange`, into a model file and its weights, and
the Caffe importer. Everything is in namespace
`SushiAI::Vision::Importers`. The formats and the behaviour are in sections 4.1 to 4.5 of
`docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md`.

Nothing here is taken from Darknet's or Ultralytics' source. A content failure is a `Status` or a `Result`;
nothing throws across the module's surface.

## Depends on

- `failure`, `shape`, `config`, `nn` and `vision/detect` (`PUBLIC_DEPENDS`); `shape` because
  `UltralyticsTensor` holds a `Shape`. `dtype` and `seam` are private (`PRIVATE_DEPENDS`).
- `io` (`PUBLIC_DEPENDS`), for `IO::ImportedModel` and `IO::ImportedTensor`, and privately for
  `IO::TorchArchive`, the pickle tree and `IO::copy_kernel_last`.
- `optim` (`PRIVATE_DEPENDS`), for `write_step_schedule_section`.
- `interchange` (`PUBLIC_DEPENDS`), for `Interchange::OnnxGraph`, `read_onnx_graph`, `fold_constants`
  and the axis and filter rewrites; `onnx_import.hpp` takes a graph, so the dependency is public.
- `interchange` (`PUBLIC_DEPENDS`), for `Interchange::CaffeNet`, which `caffe_import.hpp` takes, and
  for `Interchange::filter_to_record_layout` and `Interchange::channels_last_axis`.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/importers/darknet_config.hpp>` | `DarknetOption`, `DarknetSection`, `DarknetConfig`, `parse_darknet_config`, `load_darknet_config` |
| `<SushiAI/vision/importers/darknet_weights.hpp>` | `DarknetWeightsHeader`, `DarknetConvolutionShape`, `DarknetConvolution`, `DarknetWeights`, `read_darknet_header`, `read_darknet_weights` (a path, or bytes in memory) |
| `<SushiAI/vision/importers/darknet_fold.hpp>` | `fold_darknet_normalisation` |
| `<SushiAI/vision/importers/darknet_store.hpp>` | `DarknetWeightTarget`, `store_darknet_weights` |
| `<SushiAI/vision/importers/darknet_model.hpp>` | `DarknetNormalisation`, `DarknetTranslation`, `DarknetHead`, `DarknetModel`, `translate_darknet_model` |
| `<SushiAI/vision/importers/darknet_training.hpp>` | `DarknetSolver`, `DarknetTraining`, `read_darknet_training` |
| `<SushiAI/vision/importers/darknet_tensors.hpp>` | `list_darknet_tensors` |
| `<SushiAI/vision/importers/darknet_recipe.hpp>` | `DarknetRecipe`, `write_darknet_recipe` |
| `<SushiAI/vision/importers/darknet_import.hpp>` | `K_DARKNET_FORMAT`, `import_darknet_model`, `import_darknet_files` |
| `<SushiAI/vision/importers/caffe_import.hpp>` | `K_CAFFE_FORMAT`, `import_caffe_net`, `import_caffe_files` |
| `<SushiAI/vision/importers/ultralytics_checkpoint.hpp>` | `UltralyticsTensor`, `UltralyticsCheckpoint`, `read_ultralytics_checkpoint`, `load_ultralytics_checkpoint` |
| `<SushiAI/vision/importers/ultralytics_model.hpp>` | `K_ULTRALYTICS_IMAGE_SIDE`, `UltralyticsInputSize`, `describe_ultralytics_model`, `translate_ultralytics_model` |
| `<SushiAI/vision/importers/ultralytics_tensors.hpp>` | `list_ultralytics_tensors` |
| `<SushiAI/vision/importers/ultralytics_import.hpp>` | `K_ULTRALYTICS_FORMAT`, `import_ultralytics_model`, `import_ultralytics_files` |
| `<SushiAI/vision/importers/onnx_import.hpp>` | `K_ONNX_FORMAT`, `import_onnx_model`, `import_onnx_files` |

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
is the one of the last `[yolo]` section: `IoU` without `nms_kind`, `DistanceIoU` with exponent 0.6
for `greedynms`, and `DistanceIoU` with `beta_nms` for `diounms`.

## Refusals

A section type or a key outside SPEC 4.1 is `Unsupported`, with its line. So are an
`[upsample]` `scale` other than 1, an activation outside the list above, an `nms_kind` other
than `greedynms` and `diounms`, and a `[shortcut]` with `weights_type` or an activation other
than `linear`. A value that is malformed, out of range or does not fit the layer it reads is
`InvalidDescription`, with its line.

The `letter_box` of `[net]` becomes the `fit` of the model's input: 0, the default, is `resize`
and 1 is `darknet_letterbox`, the fit of Darknet's `letterbox_image`; another value is `InvalidDescription`.

The keys `random` and `resize` of a `[yolo]` section are accepted and not read, with the
training keys; `jitter` is read by `read_darknet_training`.

## Training settings

`read_darknet_training` reads `[net]` into a `DarknetSolver` and every `[yolo]` section into a
`Detect::Anchor::HeadTrainingOptions`, in file order; an absent key keeps the default of its
field. Every head takes `Detect::BoxGradient::DarknetCompatible`, the box gradient of REPORT
section 11, where the library's default is `Analytic`. `cls_normalizer` is checked to be a number
and reaches no option, because without per-class counters it scales no delta. It refuses a `policy` other than `steps`, a `truth_thresh` other than 1, a non-zero
`label_smooth_eps` and an `iou_loss` other than `mse` and `ciou` as `Unsupported`. A malformed
value, `steps` and `scales` of different lengths, and options `validate_head_training` refuses are
`InvalidDescription`.

## The importer

`import_darknet_files` is the row `Vision::register_importers` adds under the format name
`darknet`. It takes a `.cfg` and optionally a `.weights`, in that order, and refuses any other
count. It translates with `DarknetNormalisation::Separate`, the form training uses, so a
normalised convolution keeps its batch-norm row and the weight file holds the four batch-norm
vectors. `import_darknet_model` does the same from a parsed `.cfg`, its translation and
weights already in memory, and reads no file.

| Part of `IO::ImportedModel` | Comes from |
| --- | --- |
| `model` | `DarknetModel::spec`, with `task` set to what `write_anchor_task` returns for the model's classes, layout, heads and suppression |
| `weights` | `list_darknet_tensors`; empty without a `.weights` |
| `training`, `data`, `notes` | `write_darknet_recipe` over `read_darknet_training` |

`list_darknet_tensors` is `store_darknet_weights` without a state. It returns one float32
tensor per target name: the weights moved from `[out, in, kh, kw]` to `[out, kh, kw, in]`,
every other vector as `[out]`. Both functions move the weights through `copy_kernel_last` of
`source/darknet_weight_layout.hpp`, which hands them to `IO::copy_kernel_last`, and both refuse a vector of another length and a name
stated twice.

`write_darknet_recipe` returns the `training` object, with three keys, and a `data` object.

| Key | Holds |
| --- | --- |
| `objective` | `write_head_training_section` of the options every head shares; its type is `anchor` |
| `optimizer` | `sgd` with `momentum` and `weight_decay` set to the solver's `momentum` and `decay` |
| `schedule` | `write_step_schedule_section` of `learning_rate` as `base_rate`, `burn_in` as `warmup_iterations`, and `steps` and `scales` |

The `data` object is the part of the `data` block of the training file that a `.cfg` states: `pipeline`
"darknet" and the four keys of that row, `jitter`, `hue`, `saturation` and `exposure`, each written
as the shortest decimal of its float32. `read_darknet_training` reads `hue`, `saturation` and `exposure`
from `[net]`, with the defaults 0, 1 and 1, and `jitter` from the last `[yolo]` section, 0.2 when
absent, since Darknet's detector trains with the last head's jitter. An earlier head whose jitter
differs gets a note. `compose_training_file` writes the object as `data`; a person adds the reader,
the task and the root.

The objective section holds one set of options, so heads whose options differ are refused.
The comparison is made on the written sections, key by key, and the status names head 0, the
other head, the line of each `[yolo]` section, the key and both values.

A key of the `.cfg` that reaches none of the three files becomes a note with its line and
value: `batch`, `subdivisions`, `max_batches`, `angle` and `mosaic` of the first section, and
`random`, `resize` and `cls_normalizer` of a `[yolo]` section. The two lists are tables in `source/darknet_recipe.cpp`.

No run has trained a model with the written `optimizer`. The parity suites of
`modules/vision/darknet_training_cases/source/darknet_training.cpp` keep momentum and decay at 0, so
whether `weight_decay` under `sgd` equals Darknet's `decay` at a given batch size is not measured.

## The Ultralytics importer

`import_ultralytics_files` is the row `Vision::register_importers` adds under the format name
`ultralytics`. It takes one `.pt` and refuses any other count. The design is
`docs/design/MODEL_IMPORT/ULTRALYTICS.md`.

`read_ultralytics_checkpoint` opens the file with `IO::TorchArchive`, admitting the classes of the
modules `ultralytics.`, `models.common.`, `models.yolo.` (the YOLOv5 repository) and
`torch.nn.modules.` as plain objects, and takes the `ema` entry when it
is not None, else `model`. From that object it keeps the `yaml` description and walks the
`_modules` of its layer list `model`: each parameter and persistent buffer becomes an
`UltralyticsTensor` named by its path below the list, such as `0.conv.weight`, with the `groups`
of the convolution that owns it, widened to float32. Those names are the record names of a
SushiAI model of the same rows, since the vision blocks name their children as Ultralytics does.
The step counters `num_batches_tracked` are left out and counted.

`describe_ultralytics_model` follows Ultralytics' own model builder over the backbone and the
head, with the scale from its `scales` table or, in older files, from `depth_multiple` and
`width_multiple`: a width is capped at the scale's channel cap, multiplied by its width and rounded up to a
multiple of eight; a repeat count above 1 is multiplied by the depth and rounded half to even; a
C3k2 takes the inner block C3k at the scales m, l and x. Each module is a row of the table in
`source/ultralytics_module_table.cpp`, one unit each: Conv, C3, C2f, C3k2, SPPF, C2PSA, nn.Upsample,
Concat and Detect. A Conv that states its padding keeps it. A Detect without anchors becomes a
`detect` row; one with anchors, as YOLOv5 writes it, becomes a 1x1 `conv2d` per level published
as `head_<level>` and an `anchor_detection` task with each anchor's width over the input width and its height over the input height. A module without a row is refused, and the message lists every such module of
the description with its entry. The written file states the normalisation of the Ultralytics
blocks, epsilon 0.001 and momentum 0.03, as the defaults of each row type that holds one. The anchor-free
task carries Ultralytics' prediction thresholds, 0.25 for the score and 0.7 for the suppression.
The translation also returns `TensorRules`: the tensors of an anchored level are renamed to the
row of its convolution, and the fixed `dfl` ramp and the `anchors` buffer give no record. The input is a square of `imgsz` of `train_args`, 640 when absent. A list of one or two sides imports at a square of its largest, the size Ultralytics' trainer takes from it; an `imgsz` of another kind is refused. `describe_ultralytics_model` takes an `UltralyticsInputSize` of any height and width.

`list_ultralytics_tensors` moves a convolution weight `[out, in, kh, kw]` to `[out, kh, kw, in]`
and a depthwise one `[C, 1, kh, kw]` to `[kh, kw, C]`, names each record by the rules through `IO::copy_kernel_last` and
`IO::copy_depthwise_kernel_last`, keeps every other tensor as it is, and refuses a grouped
convolution that is not depthwise.

`train_args` is not translated, so `train.json` states no recipe; the import notes it.

## The Caffe importer

`import_caffe_files` is the row `Vision::register_importers` adds under the format name `caffe`.
It takes a `.prototxt` and optionally a `.caffemodel`, in that order, and refuses any other count.
It reads the two with `Interchange::read_caffe_prototxt` and `read_caffe_model`, joins them by layer
name with `attach_caffe_weights`, and calls `import_caffe_net`, which translates a `CaffeNet`
that is already in memory and reads no file. The design is phase 8 of
`docs/design/MODEL_IMPORT/README.md`.

A layer type has one translator in `source/caffe_op_<name>.cpp` and one row of the table in
`source/caffe_operator_table.cpp`. A new type is a new file and a new row.

| Layer type | Rows it becomes |
| --- | --- |
| `Input`, or `input` and `input_shape` of the net | The model input: `[N, C, H, W]` becomes `[H, W, C]`, `[N, K]` becomes `[K]`. A second input is `Unsupported` |
| `Convolution` | `conv2d`. Its weight moves from `[out, in / group, kh, kw]` to `[out, kh, kw, in / group]`, or from `[C, 1, kh, kw]` to `[kh, kw, C]` when `group` equals both channel counts, which is a depthwise row. `group` above 1 is the key `groups`, and one that divides neither channel count is `InvalidDescription`. An `axis` other than 1 is `Unsupported` |
| `Pooling` | `maxpool2d`, or `avgpool2d`, or `global_avgpool` for `global_pooling` with `AVE`. Caffe rounds the window count up, so a pool that needs it states `trailing_padding`; an average pool that would need it is `Unsupported`, because Caffe divides its last window by a smaller area |
| `ReLU` | `relu`, or `leaky_relu` when `negative_slope` is not 0 |
| `InnerProduct` | `flatten` when the input is an image, then `linear`. The inputs of a weight move from the order `[C, H, W]` to the order `[H, W, C]` a flatten row yields, through `Interchange::flatten_weights_to_channels_last`, which the ONNX `Gemm` uses too. `transpose` and an `axis` other than 1 are `Unsupported` |
| `Concat` | `concat` along the channels; any other axis is `Unsupported`. One bottom writes no row |
| `Eltwise` | `SUM`, `PROD` and `MAX` become a chain of `add`, `mul` or `maximum` rows, one per bottom after the first. A `coeff` other than 1 on a `SUM` is `Unsupported`, and any `coeff` on `PROD` or `MAX` is `InvalidDescription`, as Caffe refuses it |
| `Softmax` | One `softmax` row, which a net that ends in it publishes as the blob it writes. The `axis` moves to the channel-last order and becomes the row's per-sample `axis`, stated only when it is not the last; the batch axis is `Unsupported` |
| `Dropout` | `dropout` with `rate` equal to `dropout_ratio`, which is the identity at inference |
| `BatchNorm`, with the `Scale` that reads its top | One `batchnorm2d` row with `bn_epsilon` equal to `eps`. A `BatchNorm` alone stores a scale of 1 and a shift of 0 |

A layer type outside the table is refused before any layer is translated, and the one `Unsupported`
message lists every such layer with its name and type. A layer that `include`s only the TRAIN phase
is left out. A `Scale` that no `BatchNorm` precedes is refused by its own row.

The weights come with a `.caffemodel`. Every layer then needs its blobs, and a count that does not
fill the layer's extents is `Incompatible`. A state entry is named `<row>.<field>` as the
`conv2d`, `linear` and `batchnorm2d` rows name it. The folded row of a `BatchNorm` and a `Scale` stores
`running_mean` and `running_var` as the first two blobs times the inverse of the third, a factor
of 0 giving 0, and `weight` and `bias` as the blobs of the `Scale`.

`import_caffe_net` publishes every blob that no layer reads after the layer that wrote it, under the
blob's name. An in-place layer rewrites its bottom, so the row published is the last one that wrote
it. The importer states no task, no recipe and no `input_fit`; a Caffe net holds none of them.

## The ONNX importer

`import_onnx_files` is the row `Vision::register_importers` adds under the format name `onnx`.
It takes one `.onnx` file and refuses any other count. `import_onnx_model` does the same from an
`Interchange::OnnxGraph` already read. The design is `docs/design/MODEL_IMPORT/README.md`, phase 7.

The graph must be of opset 13 or newer, with one float32 input `[N, C, H, W]` whose C, H and W
are stated; the model's input is `[H, W, C]`. `fold_constants` runs first, so a `Constant` or a
`Shape` chain that only feeds a shape operand leaves no node. Each remaining node is looked up by
its operator in the table of `source/onnx_operator_table.cpp`. Each operator is its own file,
`source/onnx_op_<name>.cpp`, and a new operator is a new file and a new row. Every output of the
graph is published under its own name.

| Operator | Becomes |
| --- | --- |
| `Conv` | `conv2d`; the weight moves from `[out, in / group, kh, kw]` to `[out, kh, kw, in / group]`, or from `[C, 1, kh, kw]` to `[kh, kw, C]` when `group` equals both channel counts (a depthwise row, which takes no trailing pad); `group` above 1 is the key `groups`; pads below or right beyond the pads above or left become `trailing_padding` |
| `BatchNormalization` | `batchnorm2d` with `bn_epsilon`; its four vectors are the row's `weight`, `bias`, `running_mean` and `running_var`; inference only |
| `Relu`, `Sigmoid` | `relu`, `sigmoid` |
| `Sigmoid` read only by a `Mul` of itself and its own input | One `silu` row; `onnx_fusion.cpp` finds the pair before translation, so no `sigmoid` row is written |
| `Add`, `Sub`, `Div`, `Mul` | `add`, `sub`, `div`, `mul` of two computed values of equal extents; `Sub` and `Div` read them in operand order, and a `Mul` that is the pair of a `Sigmoid` and its own input is the `silu` row above instead |
| `Softmax` | `softmax`; the `axis` moves to the channel-last order and is stated only when it is not the last axis of a sample; the batch axis is `Unsupported` |
| `Transpose` | `transpose` of an image that keeps the batch axis first; `perm` moves to the channel-last order, and a `perm` that moves no axis writes no row |
| `Concat` | `concat`, along the channel axis only |
| `MaxPool` | `maxpool2d` with no dilation and `ceil_mode` 0; equal pads are `padding`, and unequal pads are `leading_padding` and `end_padding` |
| `Resize` | `upsample` for mode `nearest` and one whole factor on both axes, from `scales` or `sizes` |
| `Reshape`, `Flatten` | `flatten` of an image to one vector; a `Reshape` to the extents the value has writes no row |
| `Slice`, `Split` | `slice_channels`, one row per output, along the channel axis only |
| `Gemm`, `MatMul` | `linear` over a vector, with a constant weight |
| `Identity`, `Constant` | No row; the output is a second name for the operand, or a constant a later node reads |

A weight is named `<row>.<field>` by the row it belongs to, as `NN::build_model` names a state
entry. A vector flattened from an image lists its features as `(H, W, C)`, so `linear` moves the
columns of its weight from ONNX's `(C, H, W)` order.

A node whose operator has no row, or whose form a row cannot say, is refused with `Unsupported`.
The importer does not stop at the first: it collects every such node, skips a node that reads an
output of a refused node, and names each refused node in one message. The task section and the training recipe stay null; `sushiai import --task
<task.json>` attaches a task section to the written model file.

The fixture `tests/vision/importers/fixtures/onnx/silu_resize.onnx` is written by
`tools/reference/make_onnx_fixture.py`, with the output `onnx.reference.ReferenceEvaluator` gives
for one input beside it.
