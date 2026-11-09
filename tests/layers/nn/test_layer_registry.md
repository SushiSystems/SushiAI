# tests/layers/nn/test_layer_registry.cpp


Suites: `Small_LayerRegistry`, `Small_LayerRegistryMinimal`, `Small_LayerRegistryOwnType`,
`Small_LayerRegistryKernelSpelling`, `Small_LayerRegistryRefusal`, `Small_LayerRegistryGradient`,
`Small_LayerRegistryChainRefusal`.

Tests the layer registry: each row's factory, the keys it reads and the model it builds.

This file needs no device. `NN::register_layers` in `modules/layers/nn/include/SushiAI/nn/layer_registry.hpp`
adds sixteen layer rows to an `NN::NodeRegistry`, and a factory turns a per-sample input shape and a parsed
JSON object into an `NN::Layer`. A case therefore needs a `Shape`, a string of JSON and, for
the model cases, a host-side graph.

The cases fall into three groups, in file order:

- every row of the table: that it builds from a minimal object, builds its own layer type and
  refuses an unknown key
- the keys of each factory, their defaults and the values they refuse
- the convolutional classifier of `K_CONV_MODEL`, built, traced, named and differentiated

## K_CONV_MODEL

`K_CONV_MODEL` is a model file as a string: 784 input features, then `unflatten` to
28 x 28 x 1, `conv2d` with 8 filters, `relu`, `maxpool2d`, `conv2d` with 16 filters, `relu`,
`maxpool2d`, `flatten` and `linear` with 10 units. Both convolutions have kernel 3 and
padding 1, and both pools have kernel 2. The layers of
`applications/examples/mnist_cnn/model.json` are the same nine, with the optional keys spelled out.

## K_CONV_MODEL_PARAMETERS

`K_CONV_MODEL_PARAMETERS` lists the six parameters that model declares, as "name:extents":
a weight and a bias under the indices 1, 4 and 8. The test that compares names reads the
expected list from here.

## MinimalCase

`MinimalCase` holds one registry key, a layer object without its "type", the per-sample shape
that enters the layer and the per-sample shape the layer must report.

## K_MINIMAL_CASES

`K_MINIMAL_CASES` holds one `MinimalCase` per registry row, in registry order. Each object
carries the required keys of its row and no other. The rows of `Small_LayerRegistryMinimal` are these
entries.

## K_IMAGE

`K_IMAGE` is the [8, 8, 2] image most refusal rows enter the layer with.

## RefusedLayerRow

`RefusedLayerRow` holds a row name, a registry key, a layer object, the input shape, the subject the
message names (empty for the context) and up to two substrings after it.

## layer_context

`layer_context` returns "layer 0 '<type>'", the context `try_make_layer` passes to a factory.

## attributes_from

`attributes_from` parses a string as the attributes of one layer.

## try_make_layer and make_layer

`try_make_layer` looks a type up with `find_layer` in the registry `make_layer_registry` returns and calls the
row's factory with the context "layer 0 '<type>'". The cases that assert a message expect that
context in it. `make_layer` stops the test when the layer is refused.

## wrapped

`wrapped<L>` returns the layer of type `L` inside an `NN::LayerAdapter<L>`, or null when the
`Layer` wraps another type. The `dynamic_cast` is how a case tells which layer type a factory
built and reads the fields the factory set.

## square_window

`square_window` returns a geometry with a square kernel and one stride and one padding for
both axes.

## make_conv

`make_conv` returns a `Conv2d` with the given channel counts and geometry, its parameters
undeclared.

## make_conv_stack

`make_conv_stack` parses `K_CONV_MODEL` and returns the `LayerStack` that `build_layer_stack`
makes of it.

## traced_names

`traced_names` traces a model over a [4, 784] input and returns its parameters as
"name:extents" strings. It is a template, so it takes a `LayerStack` and a `Sequential`.

## TheTypeStringsAreTheOnesModelFilesSpell

The `type` fields of the layer rows are, in order, "linear", "gelu", "relu", "tanh",
"sigmoid", "leaky_relu", "unflatten", "conv2d", "maxpool2d", "avgpool2d", "global_avgpool",
"flatten", "upsample", "silu", "mish" and "slice_channels". A model file names a layer by one of these strings, so a
renamed row fails here.

## Small_LayerRegistryMinimal.TheFactoryBuildsFromAMinimalObject

One row per `K_MINIMAL_CASES` entry, named by its type. The factory returns a layer whose
`output_shape()` is the shape the case lists.

## Small_LayerRegistryMinimal.TheFactoryRefusesAnUnknownKey

One row per entry. An object holding only `"no_such_key"` returns `InvalidDescription` with a message
containing the layer's context and "'no_such_key'".

## Small_LayerRegistryOwnType.TheKeyBuildsItsOwnLayerType

One row per key of the fourteen that name a layer type here: `wrapped<L>` is not null for the
expected `L`. Two rows that pointed at one factory would fail for one of them. `mish` and
`slice_channels` have no row: the file includes neither of their headers.

## Conv2dTakesItsInputChannelsFromTheInputShape

A `conv2d` built over a [8, 8, 7] input has `in_channels` 7 and `out_channels` 5.

## Conv2dRefusesTheInChannelsKey

An object that sets `"in_channels"` is refused by name, so a model file cannot state a channel
count that disagrees with the layer before it.

## Conv2dRefusesAnInputThatIsNotAnImage

A `conv2d` given the per-sample shape [784] returns `InvalidDescription`. The message contains the context, "[784]",
"[H, W, C]" and "'unflatten'", the layer that gives a feature vector its image shape.

## Conv2dDefaultsItsOptionalKeys

With only "filters" and "kernel" set, the layer has stride 1, padding 0 and dilation 1 on both
axes, and `use_bias` is true.

## Conv2dReadsEveryKeyItAccepts

The object sets all six keys, four of them as [h, w] pairs with unequal entries. The layer
holds kernel 3 by 5, stride 2 by 1, padding 1 by 2, dilation 1 by 2 and `use_bias` false, and
over a [16, 16, 2] input it reports [8, 12, 5].

## Small_LayerRegistryKernelSpelling.TheSpellingBuildsTheWindow

Six rows, one per layer and spelling of a 3 by 3 kernel: `conv2d`, `maxpool2d` and `avgpool2d`, each
from `"kernel": 3` and from `"kernel": [3, 3]`. Over [8, 8, 2] each layer stores a 3 by 3 kernel,
holds the stride in the row (1 for the convolution, 3 for a pool, which follows its kernel) and
reports [6, 6, 5] or [2, 2, 2]. Each row checks the stored window and the shape against values,
so two layers that both stored a wrong window would fail.

## Small_LayerRegistryRefusal.TheLayerIsRefused

One test over `RefusedLayerRow`, instantiated six times. Every row returns `InvalidDescription` with
a message containing its subject (the context, unless the row names another) and its one or two
words.

- `AWindowSettingOfTheWrongForm`: a one-entry array, a three-entry array, a string and the
  real number 1.5 are each refused. The subject names the layer and the key, as in "layer 0
  'maxpool2d' 'kernel'", and the message contains "two-entry [h, w]". The two array messages also give
  the entry count, "1 entries" and "3 entries".
- `Conv2dNonPositiveExtent`: a filter count of 0 and of -4, a kernel of 0, of [3, 0] and of -3,
  a stride of 0 and of [1, -1], and a dilation of 0. Each message contains "Conv2d" and "positive".
- `PoolNonPositiveExtent`: for both pools, a kernel of 0, of [2, 0] and of -2 and a stride of 0 and of
  [-1, 2]; each message contains "positive".
- `NegativePadding`: for `conv2d` and both pools, a padding of -1, of [0, -1] and of [-1, 0]; each
  message contains "padding". The two pairs put the negative entry on each axis in turn.
- `AMissingRequiredKey`: a `conv2d` without "filters", a `conv2d` without "kernel", each pool without
  "kernel" and a `linear` without "units"; each message contains "has no" followed by the quoted key.
- `ALeakyReLUSlopeOutsideItsRange`: slopes of 0, 1 and -0.1 are refused when the layer is built, with a
  message containing "slope".

## APoolStrideDefaultsToItsKernel

A `maxpool2d` with `"kernel": [2, 3]` and no stride holds stride 2 by 3 and padding 0, and
maps [12, 12, 2] to [6, 4, 2].

## APoolReadsAnExplicitStrideAndPadding

An `avgpool2d` with kernel 3, stride 1 and padding 1 holds those values and keeps [12, 12, 2].

## UpsampleRejectsAScaleBelowOne

A scale of 0 is refused when the layer is built, with a message containing the context,
"Upsample2d" and "scale".

## LeakyReLUReadsItsOptionalSlope

A `leaky_relu` built from an empty object has slope 0.01, and one built with `"slope": 0.25`
has 0.25.

## UpsampleReadsItsOptionalScale

An `upsample` built from an empty object maps [4, 4, 2] to [8, 8, 2], and one built
with `"scale": 3` maps it to [12, 12, 2].

## UnflattenRefusesAnInputOfAnotherSize

A 28 x 28 x 1 `unflatten` given [783] returns `InvalidDescription` with a message containing the context, "[783]"
and "[784]".

## UnflattenRefusesAMissingChannelCount

An object without "channels" returns `InvalidDescription` with a message containing the context and
"channels".

## TheConvModelBuildsAndReportsItsOutputShape

The stack has nine layers and its output shape is [10]. The shapes after layers 1, 3, 6 and 7
are [28, 28, 8], [14, 14, 8], [7, 7, 16] and [784], so each factory received the shape its
predecessor reported.

## TheConvModelTracesToTheShapeItReports

The stack traces a [4, 784] input to a [4, 10] value.

## ASequentialOfTheSameLayersYieldsTheSameNames

An `NN::Sequential` of the same nine layer types is traced over the same input. Its names
equal `K_CONV_MODEL_PARAMETERS`: indices 1, 4 and 8, with gaps at the layers that own no
parameter. The stack's names equal the `Sequential`'s. A checkpoint written from one model
therefore names its parameters as the other reads them. This test also holds the list of names of
the model file.

## Small_LayerRegistryGradient.CrossEntropyDifferentiatesToTheParameter

One row per parameter of the model, in walk order: the first convolution's weight and bias, the
second's, and the linear weight and bias. The stack's output goes into `NN::CrossEntropyLoss`, and
the loss is differentiated against the six parameters the walk yields. The gradient of the row's
parameter is a value other than `K_NO_VALUE` with the shape of that parameter. The case checks that the
graph differentiates; it computes no number.

## ASiLURowTracesOneSiLUNode

The layer of a `silu` row over [8, 8, 2], traced over [4, 8, 8, 2], adds one node. Its
operation is `Ops::K_SILU` and the result keeps the shape [4, 8, 8, 2].

## AVersionTwoChainStartsFromTheInputShape

A version 2 document over the input shape [8, 8, 2] holds a `conv2d` of 5 filters and kernel
3, a `silu` and a `flatten`. `NN::build_layer_stack` builds three layers. Layer 0 reports
[6, 6, 5], which requires the factory to have received the rank-3 shape and not the count 128,
and the stack reports [180].

## Small_LayerRegistryChainRefusal.TheChainIsRefused

One row per model file `build_layer_stack` refuses, each with `InvalidDescription`:

| Row | The document | The message names |
| --- | --- | --- |
| ARowThatReadsTwoRows | row 2 with `"from": [-1, 0]` | "layer 2 'relu'", "the row before it", "build_model" |
| ARowThatSkipsBack | row 2 with `"from": 0` | "layer 2 'relu'", "the row before it" |
| ALaterRowThatReadsTheInput | row 1 with `"from": "input"` | "layer 1 'tanh'", "the row before it" |
| OutputsThatPublishAnEarlierRow | two rows, `outputs` is `[{"node": 0}]` | "'outputs'", "last row", "build_model" |
| ARowThatReadsANamedPort | row 1 with `"from": {"node": 0, "port": "output"}` | "layer 1 'tanh'", "the row before it" |

## AChainAcceptsOutputsThatPublishTheLastRow

The two-row document with `[{"node": 1}]` builds a stack of two layers.
