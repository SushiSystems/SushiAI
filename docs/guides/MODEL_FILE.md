# The model file

A model file is a JSON document that describes an architecture: the shape of one input sample,
the layers, which values each layer reads and which values the model publishes. The `Config`
module owns the format. `modules/core/config/include/SushiAI/config/model_config.hpp` declares
the reader and the `ModelSpec` it produces, and `modules/core/config/source/model_config.cpp`
holds the schema. `write_model` in `model_writer.hpp` writes a `ModelSpec` back as the text of a
model file, for the import step of `docs/design/MODEL_IMPORT/README.md`; a person writes the others.

The current version is 2.

## A version 2 file

```json
{
    "format_version": 2,
    "input": { "shape": [28, 28, 1] },
    "defaults": {
        "conv2d": { "stride": 1, "padding": 1, "use_bias": true }
    },
    "layers": [
        { "type": "conv2d", "filters": 8, "kernel": 3 },
        { "type": "relu" },
        { "type": "conv2d", "filters": 8, "kernel": 3 },
        { "type": "concat", "from": [-1, 0] },
        { "type": "flatten" },
        { "type": "linear", "units": 10 }
    ],
    "outputs": [ { "node": 5 } ]
}
```

The reader accepts this file, and `NN::build_model` builds it. "Building a model" below says
which builder takes which file.

Keys are `snake_case`. A key the reader does not know is an error; a layer's own keys are checked
by that layer's factory.

## Top-level keys

| Key | Required | Holds |
| --- | --- | --- |
| `format_version` | Yes | The integer `2` |
| `input` | Yes | An object with one key, `shape` |
| `layers` | Yes | A list of at least one layer object, called rows below |
| `defaults` | No | A map from a layer type to keys its rows receive |
| `outputs` | No | A list of at least one output entry |
| `task` | No | An object with a string `type`; the rest belongs to that task type |

### `input.shape`

The shape of one sample, without the batch axis: a list of one or more positive whole numbers.
`[784]` is a flat vector of 784 features and `[28, 28, 1]` is a 28 by 28 image with one channel.
`ModelSpec::input_shape` holds it. The list holds at most 6 extents, and their product must fit
a signed 64-bit integer.

### `layers`

Each row is an object. `type` names the layer and is required. `from` is described below. Every
other key belongs to the layer type, and the factory for that type checks it: the reader passes
those keys on as `LayerSpec::attributes` without reading them.

Rows are numbered from 0 in file order.

### `from`

`from` names the values a row reads. It holds one entry or a list of entries, and a row without
`from` reads the row before it. An entry takes one of four forms.

| Entry | Reads |
| --- | --- |
| `-1` | The row before; at row 0, the model input |
| A whole number from 0 up | That row, which must be above the row that names it |
| `"input"` | The model input |
| `{ "node": 23, "port": "boxes" }` | The named port of a row; `node` follows the two number rules above, and `port` is optional |

A negative number other than `-1`, a row's own index, a later row's index and an empty list are
errors, and so is a `port` that is an empty string. The reader resolves every entry to an
absolute row, so `LayerSpec::from` holds a row index or `K_MODEL_INPUT`, never `-1` in the
relative sense.

### `defaults`

`defaults` maps a layer type to an object of keys. A row of that type receives each of those keys
it does not state; a key the row states keeps the row's value. The reader merges the keys before
any factory sees the row, after the row's own keys and in the order `defaults` lists them. An
entry may not carry `type` or `from`. A type that no row states is an error, so a misspelt type
cannot leave its rows without their defaults.

### `outputs`

Each entry names a row with `node`, an absolute row index.

| Entry | Publishes |
| --- | --- |
| `{ "node": 5 }` | Every port of row 5, each under the port's own name |
| `{ "node": 5, "port": "boxes" }` | That port, under its own name |
| `{ "node": 5, "port": "boxes", "name": "predictions" }` | That port, as `predictions` |

`name` without `port` is an error, and so is a `port` or a `name` that is an empty string. A file
without `outputs` publishes every port of its last row.

### `task`

`input` may state `"dtype"`: `"float32"`, which is the default, or `"int32"` for a model whose
first row is an `embedding`. A `language_head` row may state `"tie"` with a row reference in
the forms above; it then shares the table of that `embedding` row and needs no `vocabulary`.

`input` may state `"fit"`: `"resize"`, which scales each axis of an image to the input on its
own, or `"letterbox"`, which scales both by one ratio and pads the rest. `sushiai train` and
`sushiai inference` hand it to a data block that states no `fit`, and refuse one that states
another. `sushiai import darknet` writes it from the `.cfg`'s `letter_box`.

`task` says what the published outputs mean, for a model whose rows alone do not: a detector's
anchors, the coding of its heads and its suppression. It is an object, and `type` names the
task as a string. A `task` that is not an object, has no `type`, or has a `type` of another
kind is an error. A `type` no domain knows is refused by the domain that reads the section.

```json
"task": {
    "type": "anchor_detection",
    "classes": 80,
    "anchors": [[0.024, 0.031], [0.055, 0.072], [0.089, 0.061]],
    "heads": [
        { "output": "head_0", "anchors": [0, 1, 2], "coding": "logit", "scale_xy": 1.05 }
    ],
    "suppression": { "criterion": "iou" }
}
```

The reader checks `type` and nothing else. It stores the object in `ModelSpec::task`, keys in
file order, and the domain that owns the type reads the other keys. `ModelSpec::task` of a file
without the key is null. `anchor_detection` belongs to `modules/vision/detect`, whose
`README.md` lists its keys under `anchor/anchor_task.hpp`.

A version 1 file may not carry `task`.

## Building a model

Two functions build a parsed file. Both take the `ModelSpec` the reader returns.

| Function | Declared in | Builds | Returns |
| --- | --- | --- | --- |
| `NN::build_model` | `modules/layers/nn/include/SushiAI/nn/node_registry.hpp` | Any version 2 file | `Result<ModelGraph>` |
| `NN::build_layer_stack` | `modules/layers/nn/include/SushiAI/nn/layer_registry.hpp` | A chain | `LayerStack`, or it throws `Error` |

`build_model` makes one node per row and names it by the row's index, so the state of row 3 is
named `3.<leaf>`. It resolves each `from` entry to a port of an earlier node, hands the factory
the shapes of those ports, and publishes what `outputs` lists. A failure is an
`InvalidDescription` status that names the row as `layer 3 'concat'` or the output as
`output 0`.

`build_layer_stack` builds a chain and nothing else. It refuses a row that reads anything but
the row before it, and an `outputs` that publishes anything but the whole of the last row; both
messages name `build_model`. Every version 1 file is a chain.

A `type` is looked up in an `NN::NodeRegistry`, which holds node rows and layer rows; no type
has two rows.

| Row | Types | A row reads |
| --- | --- | --- |
| Node | `batchnorm2d`, `concat`, `add`, `embedding`, `layernorm`, `dropout`; the vision rows `conv`, `bottleneck`, `c3k`, `c3k2`, `c2f`, `c3`, `sppf`, `c2psa`, `detect`, `segment_head`; the language rows | One value; `concat` reads one or more |
| Layer | `linear`, `gelu`, `relu`, `tanh`, `sigmoid`, `leaky_relu`, `unflatten`, `conv2d`, `maxpool2d`, `avgpool2d`, `global_avgpool`, `flatten`, `upsample`, `silu`, `mish`, `slice_channels` | One value |

`build_model` reads both kinds of row. `build_layer_stack` reads the layer rows only, so a chain
of `conv` rows builds with `build_model`. A type the registry does not hold fails with a message
that lists every row.

A `ModelSpec` filled in code is checked as a parsed one is. Both builders treat an empty
`outputs` list as a file without `outputs`: the last row is published whole.

## Versions

| Version | Version key | Release | What changed |
| --- | --- | --- | --- |
| 1 | `sushiai_model` | Before 0.1.0, untagged | The first format: `input.features` and a chain of `layers` |
| 2 | `format_version` | Unreleased | `input.shape` replaces `input.features`; adds `from`, `defaults` and `outputs`; the version key becomes `format_version` |

`task` joined version 2 after its other keys, and the version number did not change, because
version 2 is in no release. A file without `task` reads as it did. A build from before the key
refuses a file that carries it: the reader refuses every top-level key it does not know, so
that build reports `task` as an unknown key.

## Reading an older or a newer file

The reader accepts version 1 and upgrades it in memory with `upgrade_model_v1_to_v2`. No file on
disk is rewritten. The step makes three changes:

- `"sushiai_model": 1` becomes `"format_version": 2`.
- `"input": { "features": F }` becomes `"input": { "shape": [F] }`.
- Every row gets `"from": -1`.

A version 1 file therefore describes the same model it did before: a chain over a flat input,
publishing its last row. A version 1 file that uses `from`, `defaults` or `outputs` is refused,
because those keys belong to version 2.

The reader refuses these, and each message names the file:

| File states | Message names |
| --- | --- |
| `format_version` above 2 | The file's version and version 2 |
| `format_version` 1 or lower | `format_version` and `sushiai_model`, the key a version 1 file uses |
| `sushiai_model` other than 1 | The stated number and version 1 |
| Neither key | Both keys |

## What the reader does not check

The reader knows no layer type. Whether a `type` exists, which keys it accepts, how many values
it reads and which ports it has are checked where the model is built, by the two tables above
and the factory for that type.
