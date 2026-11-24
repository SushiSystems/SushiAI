# exporters

## Owns

The writing of a SushiAI model as the files of another framework: the walk over the rows, the
row table of each format, the refusal of every row the format's importer cannot read back, and
the notes for what the files leave out. Each exporter is an `IO::ExporterFunction`: it takes an
`IO::ImportedModel` and a stem and returns the files as bytes in an `IO::ExportedModel`, so it
writes no file. Everything is in namespace `SushiAI::Vision::Exporters`. The design is
`docs/design/MODEL_IMPORT/EXPORT.md`; the specification is
`docs/agent/2026_11_24_MODEL_EXPORT/SPEC.md`, section 3.

The module does not depend on `vision/importers`. An exporter writes only what the importer of
the same format reads back, and its tests prove it by a round trip through that importer.

## Depends on

- `failure`, `io` and `interchange` (`PUBLIC_DEPENDS`): `onnx_export.hpp` takes an
  `IO::ImportedModel`, returns an `IO::ExportedModel` and an `Interchange::OnnxGraph`.
  `interchange` also holds `write_onnx_graph` and the channel-first rewrites the rows use.
  `caffe_export.hpp` returns an `Interchange::CaffeNet`, written by `write_caffe_prototxt` and
  `write_caffe_model`; its layer parameters are `IO::TextField` messages.
- `config` and `dtype` (`PRIVATE_DEPENDS`), for `Config::ModelSpec` and the element type of a
  weight.
- `optim` and `vision/detect` (`PRIVATE_DEPENDS`), for the Darknet exporter: the optimizer and
  step schedule the `[net]` section states, and the anchor task its `[yolo]` sections state.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/vision/exporters/onnx_export.hpp>` | `K_ONNX_EXPORT_FORMAT`, `export_onnx_graph`, `describe_onnx_losses`, `export_onnx_files` |
| `<SushiAI/vision/exporters/caffe_export.hpp>` | `K_CAFFE_EXPORT_FORMAT`, `export_caffe_net`, `describe_caffe_losses`, `export_caffe_files` |
| `<SushiAI/vision/exporters/darknet_export.hpp>` | `K_DARKNET_EXPORT_FORMAT`, `export_darknet_files` |

## Refusals

An exporter walks the rows in order and looks each `type` up in its row table. A row with no
entry, or whose keys or operands the format's importer cannot read back, is refused with
`Unsupported`. The exporter does not stop at the first: it names every refused row as
`row <i> '<type>': <reason>` in one message. A row that reads the value of a refused row is
skipped without a reason of its own. A row that reads a port other than `output`, or a named
input, is refused. A model with named inputs, an input that is not float32 or an input of a rank
the format cannot state is refused before any row.

What the files cannot hold and what does not change the computation is a note, and the export
goes on. For ONNX and Caffe these are the task section, the image fit, the input normalisation,
the training recipe and the data block. The Darknet exporter states the task as `[yolo]` sections
and the fit as `letter_box`, and refuses a model with no task (section "The Darknet exporter").

The walk is shared: `source/export_walk.cpp` gathers a row's operands, collects the refusals and
words the final message, `source/row_keys.cpp` reads a row's keys and `source/row_weights.cpp`
its weights. Each exporter passes its own wording, so a refusal names its format.

## The ONNX exporter

`export_onnx_files` is the row `Vision::register_exporters` adds under the format name `onnx`. It
writes one file, `<stem>.onnx`, through `Interchange::write_onnx_graph`. `export_onnx_graph`
returns the graph before it is written. The graph imports opset 13 (IR version 7). Its input is
`input`, float32 `[batch, C, H, W]` with `batch` symbolic, from the model's `[H, W, C]`; an input of
another rank is refused, and so is a model with no weights. Each weight is an initializer named
`<row>.<field>`. A row's value is named after the first output that publishes it, else
`<row>.output`; a second name for one row is an `Identity` node.

The table is `source/onnx_row_table.cpp`, and each row type is its own file,
`source/onnx_row_<type>.cpp`; a new row type is a new file and a new row.

| Row | Becomes | Refused when |
| --- | --- | --- |
| `conv2d` | `Conv`; the weight moves from `[out, kh, kw, in / groups]` to `[out, in / groups, kh, kw]`, or from `[kh, kw, C]` to `[C, 1, kh, kw]`; `pads` are `[top, left, top + trailing_h, left + trailing_w]`; `groups` above 1 is `group`; the bias is an initializer when the weights hold one | A dilation other than 1, a negative trailing padding, or a trailing padding on a depthwise kernel |
| `batchnorm2d` | `BatchNormalization` with `epsilon` from `bn_epsilon` and the four vectors | The row states `bn_momentum` |
| `relu`, `sigmoid` | `Relu`, `Sigmoid` | Never |
| `silu` | `Sigmoid` and the `Mul` of it with its input, which the importer fuses back into one row | Never |
| `add`, `sub`, `mul`, `div` | `Add`, `Sub`, `Mul`, `Div` in operand order | The operands differ in shape; a `mul` of a `sigmoid` row read by it alone and that row's input, which the importer reads back as `silu` |
| `softmax` | `Softmax` with `axis` moved to the channel-first order | Never |
| `transpose` | `Transpose` with `perm` moved to the channel-first order, the batch axis first | The value is not an image, or the permutation moves no axis |
| `concat` | `Concat` on axis 1 | Never |
| `maxpool2d` | `MaxPool`, `ceil_mode` 0; `padding` and `trailing_padding`, or `leading_padding` and `end_padding`, become `pads` | Never |
| `upsample` | `Resize`, mode `nearest`, `coordinate_transformation_mode` `asymmetric`, `nearest_mode` `floor`, `scales` `[1, 1, s, s]` | `mode` `bilinear` |
| `flatten` | `Flatten` on axis 1; the `linear` that reads it moves its weight columns from `(H, W, C)` to `(C, H, W)` | The model publishes it, or a row other than `linear` reads it |
| `slice_channels` | `Slice` on axis 1, with `starts`, `ends` and `axes` as initializers | Never |
| `linear` | `Gemm` with `transB` 1, the weight and the bias as initializers | The input is not a vector |

Every other row type is refused: among them `leaky_relu`, `mish`, `gelu`, `tanh`, `avgpool2d`,
`global_avgpool`, `maximum`, `dropout`, `unflatten`, `embedding`, `layernorm`, the vision blocks
and every language row.

Round-trip text equality holds only for rows in the importer's canonical form: a stated default
key, such as `"groups": 1` or a softmax `"axis": -1`, comes back without the key. A `maxpool2d`
with a trailing padding also comes back as `leading_padding` and `end_padding`, and `bn_epsilon`
as a float32 value.

## The Caffe exporter

`export_caffe_files` is the row `Vision::register_exporters` adds under the format name `caffe`.
It writes `<stem>.prototxt` through `Interchange::write_caffe_prototxt` and, when the model has
weights, `<stem>.caffemodel` through `Interchange::write_caffe_model`; a model without weights
gives the prototxt alone and a note. `export_caffe_net` returns the net before it is written.
The net opens with an `Input` layer `input` of shape `[1, C, H, W]` from `[H, W, C]`, or `[1, K]`
from `[K]`. Each row is the layer `row_<i>` (a `batchnorm2d` adds `row_<i>_scale`), and writes
the blob named after the output that publishes it, else `row_<i>`.

The Caffe importer publishes every blob no layer reads, in row order. The model must publish
exactly those rows, each once and in row order, or each output that differs is refused, and so is
a row that no row reads and no output publishes.

The table is `source/caffe_row_table.cpp`, and each row type is its own file,
`source/caffe_row_<type>.cpp`.

| Row | Becomes | Refused when |
| --- | --- | --- |
| `conv2d` | `Convolution` with `num_output`, `bias_term`, `pad_h`, `pad_w`, `kernel_h`, `kernel_w`, `stride_h`, `stride_w`, `group` above 1 and a dilation other than 1; the weight goes back to `[out, in / groups, kh, kw]` | A trailing padding other than 0; a kernel whose rank disagrees with the depthwise layout the importer chooses |
| `maxpool2d` | `Pooling` `MAX` with the kernel, stride and pad of each axis | A pad not below the kernel; an end padding other than the one Caffe's rounding up adds |
| `avgpool2d` | `Pooling` `AVE` | A trailing padding, or a window count that needs rounding up |
| `global_avgpool` | `Pooling` `AVE` with `global_pooling` true | Never |
| `relu`, `leaky_relu` | `ReLU`, with `negative_slope` for `leaky_relu` | A `leaky_relu` of slope 0, which comes back as `relu` |
| `flatten` | No layer; the `InnerProduct` of the `linear` right after it reads its input and moves its weight columns from `(H, W, C)` to `(C, H, W)` | The model publishes it, or a row other than that `linear` reads it |
| `linear` | `InnerProduct` with `num_output` and `bias_term` | The input is an image that no `flatten` reads first |
| `concat` | `Concat` on axis 1 | Fewer than two operands |
| `add`, `mul`, `maximum` | `Eltwise` `SUM`, `PROD`, `MAX` over two bottoms | An operand is not an image, or the two differ in shape |
| `softmax` | `Softmax` with `axis` moved to the channel-first order | Never |
| `dropout` | `Dropout` with `dropout_ratio` from `rate` | Never |
| `batchnorm2d` | `BatchNorm` with `eps` from `bn_epsilon` and the blobs mean, variance and a factor of 1, then a `Scale` in place with `bias_term` true and the blobs weight and bias | The row states `bn_momentum` |

Every other row type is refused, because the Caffe importer has no row for it: among them
`sigmoid`, `silu`, `mish`, `gelu`, `tanh`, `upsample`, `slice_channels`, `sub`, `div`,
`transpose`, `unflatten`, `embedding`, `layernorm`, the vision blocks and every language row.

Round-trip text equality holds only for rows in the importer's canonical form: a stated default
key, such as `"groups": 1` or a softmax `"axis": -1`, comes back without the key.

## The Darknet exporter

`export_darknet_files` is the row `Vision::register_exporters` adds under the format name
`darknet`. It writes `<stem>.cfg` and, when the model has weights, `<stem>.weights`, whose header
is version 0.2.0 with a 64-bit `seen` of 0, then each `[convolutional]` section's biases, its
scales, means and variances when it is normalised, and its kernel as `[out, in, kh, kw]`.

The `[net]` section states `width`, `height` and `channels` from the float32 `[H, W, C]` input,
and `letter_box=1` for the fit `darknet_letterbox`; the fit `letterbox` is refused. The training
recipe writes `momentum` and `decay` from an `sgd` optimizer, the learning rate, `burn_in` and the
step schedule, and the data keys `jitter`, `hue`, `saturation` and `exposure`; what a `.cfg`
cannot state is a note.

A model with no task is refused with `Unsupported` ("the model states no task"), because the
Darknet importer refuses a `.cfg` without a `[yolo]` section; so is a task of a type other than
`anchor_detection`. Each output must be `head.<k>` on port `output`, and each becomes a `[yolo]`
section after the section of its row, with `mask`, `anchors` in pixels of the input, `classes`,
`num`, `scale_x_y`, `new_coords`, `nms_kind` and `beta_nms`, and the head's training keys.

The table is `source/darknet_row_table.cpp`, and each row type is its own file,
`source/darknet_row_<type>.cpp`. A section reads the section before it; a row that reads an
earlier value is preceded by a `[route]` of that one layer.

| Row | Becomes | Refused when |
| --- | --- | --- |
| `conv2d` | `[convolutional]` with `filters`, `size`, `stride` and `pad=1` for a padding of half the kernel, else `padding`; a `batchnorm2d` and then an activation that it alone reads are the same section's `batch_normalize=1` and `activation` | A dilation, `groups` above 1, a trailing padding, a bias beside a `batchnorm2d`, no bias without one, a `bn_epsilon` other than 1e-5 |
| `leaky_relu`, `mish`, `silu`, `sigmoid` | The `activation` `leaky`, `mish`, `swish`, `logistic` of the `conv2d` they follow | They follow anything else, or a `leaky_relu` has a slope other than 0.1 |
| `batchnorm2d` | `batch_normalize=1` of the `conv2d` it follows | It follows anything else |
| `maxpool2d` | `[maxpool]` with `size`, `stride` and the padding total | The trailing padding is not the one the importer computes |
| `upsample` | `[upsample]` with `stride` from `scale` | `mode` `bilinear` |
| `concat` | `[route]` over the layers it joins | Fewer than two operands, or an operand is the input |
| `slice_channels` | `[route]` of one layer with `groups` and `group_id` | The slice is not one of equal groups of the channels |
| `add` | `[shortcut]` with `from` and `activation=linear` | An operand is the input |

Every other row type is refused, because no Darknet section the importer reads gives it back:
among them `relu`, `gelu`, `tanh`, `avgpool2d`, `global_avgpool`, `maximum`, `dropout`, `sub`,
`mul`, `div`, `softmax`, `transpose`, `flatten`, `linear`, the vision blocks and every language
row.
