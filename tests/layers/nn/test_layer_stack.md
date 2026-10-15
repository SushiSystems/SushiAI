# tests/layers/nn/test_layer_stack.cpp


Suites: `Unit_LayerStack`.

Tests the runtime layer seam: LayerStack, LayerAdapter and the shape the factories thread.

This file needs no device. A LayerStack is host-lane, and the first property this file pins,
that it names its parameters exactly as Sequential does, is a property of the walk and not of
any memory. The second is the per-sample `Shape` that `NN::Layer::output_shape()` reports
and `NN::build_layer_stack` hands from each layer to the next factory.

The comparison is written as "trace both, walk both, compare the two lists" and not as "check
LayerStack produces these literal names", so it fails if either path changes its naming. A test
against literals would let both drift together and still pass.

## chain_row

`chain_row` returns one `Config::LayerSpec` of a given type and attributes that reads one given
source row.

## hand_built_chain

`hand_built_chain` fills a `Config::ModelSpec` field by field, with no parser involved: the
input shape [6] and the rows `linear` 8, `gelu` and `linear` 4. Row 0 reads
`Config::K_MODEL_INPUT` and each later row reads the row above it.

## NamesItsParametersExactlyAsSequentialDoes

The second `EXPECT_EQ` states the names once as literals too, so the `expected` list that both
walks are compared through cannot become trivially equal by both paths going empty.

## KeepsTheIndexGapAParameterlessActivationLeaves

The gap is the whole reason the naming matches: indices track layer positions, not parameter
positions. A stack that renumbered would produce "1.weight" for the second Linear and load
nothing.

## ReportsTheLastLayersOutputWidth

The stack of `make_stack` has a size of 3, is not empty and reports 4 output features.

## TracesTheLayersInOrder

`forward` of that stack over a [4, 6] input gives a [4, 4] value.

## RefusesToTraceWhenEmpty

`forward` of an empty stack raises an error holding "no layers" and "identity".

## RefusesAnOutputWidthWhenEmpty

`output_features` of an empty stack raises an error holding "no layers".

## RefusesANullLayer

`push(nullptr)` raises an error holding "null layer" and "index 0".

## ReportsAnOutOfRangeLayerIndex

On a stack of three layers `layer(2)` does not throw, and `layer(3)` raises an error holding
"layer 3" and "holds 3".

## TracingTwiceDoesNotDuplicateParameters

`Parameter::declare` is idempotent, and the run-time path must inherit that and not reintroduce
the duplicate-on-retrace bug at this layer. Both traces go into one graph, because a
parameter's id names a value in the graph that declared it. Retracing into a second graph would
reuse an index that means something else there.

## AnAdapterBuiltFromAShapeReportsIt

A `LayerAdapter<GELU>` constructed with the shape [4, 5, 3] returns that shape from
`output_shape()`.

## AnAdapterBuiltFromAWidthReportsARankOneShape

A `LayerAdapter<Linear>` constructed with the width 8 reports the shape [8] and
`output_features()` 8. `make_stack` builds its layers through this constructor.

## OutputFeaturesEqualsTheElementCountOfTheOutputShape

Through an `Layer` reference, `output_features()` of an adapter of shape [4, 5, 3] is 60 and
equals `output_shape().num_elements()`. `output_features()` is a non-virtual function of
`Layer` in `modules/layers/nn/include/SushiAI/nn/layer.hpp`, derived from the one virtual.

## ReportsTheLastLayersOutputShape

The stack of `make_stack` reports [4]. With a `GELU` adapter of shape [2, 2, 1] pushed after
it, the stack reports [2, 2, 1] and `output_features()` 4.

## RefusesAnOutputShapeWhenEmpty

`output_shape()` of an empty stack throws with a message containing "no layers" and "output
shape".

## BuildLayerStackThreadsEachOutputShapeIntoTheNextLayer

The spec of `hand_built_chain` builds three layers that report [8], [8] and [4]. The last
layer is cast to `LayerAdapter<Linear>` and its `in_features` is 8, the width the activation
before it reported. The spec never states that number, so it can only come from the shape
`build_layer_stack` hands on.

## BuildLayerStackSeedsTheChainFromTheInputShape

The spec of `hand_built_chain` has its `input_shape` replaced by [5]. The first layer's
`in_features` is 5, so `build_layer_stack` seeds the chain from the shape.

## BuildLayerStackRefusesAHandBuiltRowWithoutASource

With the `from` list of row 1 cleared, `build_layer_stack` throws with a message containing
"layer 1 'gelu'" and "the row before it".

## OutputShapeOfAnInputIsTheLastLayersOutputShape

`LayerStack::output_shape([6])` of the built chain is [4] and equals the argument-free
`output_shape()`.

## RefusesAnOutputShapeOfAnInputWhenEmpty

`output_shape([6])` of an empty stack throws with a message containing "no layers" and "output
shape".

## AnActivationFactoryKeepsARankThreeInputShape

`NN::make_gelu` given the per-sample shape [4, 4, 3] returns a layer that reports [4, 4, 3]
and 48 features. A factory that reduced its input to a width would report [48].

## ALinearFactoryRefusesARankThreeInputAndNamesFlatten

`NN::make_linear` given [4, 4, 3] throws with a message containing the context, "[4, 4, 3]"
and "flatten". Given [48], the same object builds.

## test_layer_stack.cpp, tests/core/config/test_value.cpp, tests/core/config/test_train_config.cpp, tests/core/config/test_model_config.cpp


How the config layer parses its JSON documents and which ones it refuses.

Suites: `Unit_Config`.

Tests how the config layer parses its JSON documents and which ones it refuses.

This file needs no device. The config layer is host-lane by construction: it turns text into a
typed tree and never touches an Engine.

Most of what the cases assert is a diagnostic, not a value. That is deliberate. A config
layer's job is to refuse a wrong file loudly, and a test that only checked "it threw" would keep
passing after a refactor collapsed five specific messages into one generic complaint.
`modules/core/failure_expectations/include/SushiAI/failure_expectations/throw_expectations.hpp` gives the argument in full.

The cases fall into two groups, in file order:

- the model document in both of its versions, and the layer registry it selects through
- the train and eval documents

## ParsesEveryScalarKind

An object with a whole number, a real, a string, a boolean, a null and an array of two parses
to `Kind::Object`, and each member reads back through `as_int`, `as_real`, `as_string`,
`as_bool`, `is_null` and `as_array`.

## KeepsTheKeyOrderTheFileSpelled

Key order is what lets a "this object carries ..." diagnostic read the way the file does, so
the test treats it as a property and not as an accident.

## AnIntegerReadsAsARealButARealIsNotAWholeNumber

JSON's grammar distinguishes 3 from 3.0 and the config layer keeps that, so `"units": 2.5` is
refused, not truncated. The widening goes one way only.

## AWrongKindNamesBothTheContextAndTheKindFound

`as_int("layer 0 'units'")` on a string raises an error holding "layer 0 'units'" and
"a string".

## AMissingKeyNamesTheKeyAndListsWhatIsPresent

`at("gamma", "model.json")` on an object of `alpha` and `beta` raises an error holding
"model.json", "'gamma'", "'alpha'" and "'beta'".

## ReadingAKeyOffANonObjectSaysWhatItActuallyIs

`at("units", "the model")` on an array raises an error holding "the model" and "an array".

## AnUnknownKeyIsAnErrorThatListsWhatIsAllowed

`reject_unknown_keys` with the allowed keys `type` and `units` over an object holding `unitss`
raises an error holding "layer 0 'linear'", "'unitss'", "'units'" and "'type'".

## AnAllowedKeySetIsAcceptedAndAnEmptyOneRefusesEverything

A layer that takes no attributes, such as gelu, passes an empty list. Any key at all is then an
error and is not quietly dropped.

## WithoutRemovesTheDiscriminatorAndLeavesTheRest

Removing a key that is absent is not an error, because a layer spelled with no attributes still
goes through the same call.

## AtOrFallsBackWithoutThrowing

A non-object falls back and does not throw. A layer spelled `{"type": "gelu"}`, whose attributes
are an empty object, therefore reads its optional keys through the same call as any other layer.

## MalformedJSONNamesTheOriginAndTheComplaint

`parse_json("{ not json", "train.json")` raises an error holding "train.json" and
"not valid JSON".

## AnObjectBuiltFromMismatchedVectorsIsRefused

`Value::object` of two keys and one value raises an error holding "2 keys" and "1 value".

## ParsesAModelAndSeparatesEachLayersTypeFromItsAttributes

"type" is gone from the attributes, so no factory has to allow the key that selected it.

## ChainsLayerWidthsFromTheInputFeatures

The activation preserves the width it was handed, which is why the file never states a width
for it.

## AConfiguredStackTracesToTheDeclaredOutputWidth

The stack `build_layer_stack` makes from `K_GOOD_MODEL` traces a [4, 6] input to a [4, 4]
value.

## AnUnknownLayerTypeListsEveryTypeThisBuildHas

The model names a layer type "hologram", which no layer row carries. The
message contains "hologram", "layer type", "'linear'" and "'gelu'". The unknown type has to
be a name the registry does not hold, and "conv2d" is a layer row.

## AnUnknownLayerKeyIsRejectedByTheLayersOwnFactory

The `stray` document writes a width onto an activation. An activation takes no attributes, so
that width is an error and not a line that does nothing.

## The version 2 model document

The cases from AVersionOneDocumentParsesToTheSameSpecAsItsVersionTwoRewrite to
AMalformedOutputIsAnError cover the two versions of the model document that
`Config::parse_model` reads. `Config::K_MODEL_VERSION` is 2. A version 1 document carries the key
`sushiai_model` and an `input` of `features`. A version 2 document carries `format_version`,
an `input` of `shape`, an optional `from` on each row, and the optional top-level keys
`defaults` and `outputs`. Both parse to one `Config::ModelSpec`, in which every row holds its
sources as absolute row indices and the model input is `Config::K_MODEL_INPUT`, which is -1.

## K_FIRST_VERSION_MODEL

`K_FIRST_VERSION_MODEL` is a version 1 document of five rows over 784 features: `unflatten` to
28 by 28 by 1, `conv2d` of 8 filters and kernel 3 with a bias, `relu`, `flatten` and `linear`
of 10 units.

## K_REWRITTEN_MODEL

`K_REWRITTEN_MODEL` is the same model written as version 2, with the input shape [784] and
`"from": -1` on every row.

## scalar_text

`scalar_text` renders one scalar attribute as "int n", "bool true", "bool false" or "string
s", and any other kind as its kind name.

## spec_text

`spec_text` renders a `ModelSpec` as text: one line for the input shape and feature count, one
line per row with its type, its sources as `node:'port'` and its attributes, and one line per
output. Two specs compare equal as text when every one of those fields agrees.

## current_model_from

`current_model_from` wraps a `layers` array, and optionally more top-level keys, in a version
2 document whose input shape is [6].

## ALayerFactoryRefusesANonPositiveWidth

A `linear` row with `units` 0 makes `build_layer_stack` raise an error holding
"layer 0 'linear'" and "'units'".

## UseBiasDefaultsToTrueAndCanBeTurnedOff

After a trace, a `linear` row has two parameters, and one with `use_bias` false has one.

## AMissingModelVersionIsAnErrorThatSaysWhy

A model document without `sushiai_model` raises an error holding "sushiai_model" and
"misread".

## AFutureModelVersionIsAnError

`sushiai_model` 2 raises an error holding "version 2" and "version 1".

## AnUnknownTopLevelModelKeyIsAnError

A version 1 document with the key `optimizer` raises an error holding "'optimizer'" and
"'layers'".

## AModelWithNoLayersIsAnError

An empty `layers` array raises an error holding "no layers".

## ANonPositiveInputWidthIsAnError

`features` 0 raises an error holding "'features'" and "positive".

## ALayerThatIsNotAnObjectIsAnError

A layer given as the string "linear" raises an error holding "layer 0" and "a string".

## AVersionOneDocumentParsesToTheSameSpecAsItsVersionTwoRewrite

`spec_text` of the two documents is equal. The expected text is asserted as a literal too: the
input is [784] with 784 features, row 0 reads -1, each later row reads the row above it, and
the one output is row 4 with no port and no name.

## AVersionOneDocumentFillsTheInputShapeAndTheFeatureCount

`K_GOOD_MODEL`, a version 1 document of 6 features, parses to `input_shape` [6],
whose `num_elements()` is 6. The case also asserts that `Config::K_MODEL_VERSION` is 2.

## AVersionTwoInputShapeKeepsItsRankAndCountsItsElements

The input shape [28, 28, 1] is kept at rank 3 and its `num_elements()` is 784.

## AnEmptyOrNonPositiveInputShapeIsAnError

An empty `shape` throws with a message containing "'input' 'shape'" and "empty". The shape
[28, 0, 1] throws with "'shape' entry 1" and "positive". A version 2 input that states
`features` throws with "'features'" and "'shape'".

## TheInputElementTypeIsFloat32UnlessTheFileStatesInt32

A version 2 document without `dtype`, and one with `dtype` "float32", both parse to an input
of `Float32`. One with "int32" parses to `Int32`.

## AnInputElementTypeOutsideTheListIsAnError

The `dtype` "int64" raises an error holding "'input' 'dtype'", "'int64'" and
"'float32', 'int32'". The `dtype` 32 raises one holding "'input' 'dtype'".

## AnInputShapeAboveTheRankOrTheElementLimitIsAnError

A shape of `K_MAX_RANK` + 1 extents throws with "'shape'" and "at most". The shape
[4294967296, 4294967296], whose product is 2^64, throws with "'shape'", "entry 1" and "more
elements".

## AModelRootThatIsNotAnObjectIsAnError

The document `[1]` throws with a message containing "model.json", "'format_version'" and "an
array".

## AVersionAboveTheNewestIsRefusedWithBothNumbers

`"format_version": 3` throws with a message containing "model.json", "version 3" and "version
2".

## FormatVersionOneIsRefusedAndNamesTheVersionOneKey

`"format_version": 1` throws with a message containing "'format_version'" and
"'sushiai_model'". Version 1 files carry the second key and never the first.

## AVersionOneDocumentRefusesTheVersionTwoKeys

A version 1 document with a `from` on row 1 throws with "layer 1", "'from'" and "version 1".
One with a top-level `defaults` throws with "'defaults'" and "'layers'".

## EachFromFormResolvesToAnAbsoluteRow

Six rows use every form of `from`. An absent `from` on row 0 and the string "input" resolve to
-1. The number -1 resolves to the row above. An array `[-1, 0]` on row 3 resolves to rows 2
and 0. An object `{"node": 3, "port": "boxes"}` resolves to row 3 with that port. An array may
mix objects, "input" and a relative object. The whole spec is compared as text. Row 0's source
is `K_MODEL_INPUT`, and "from" is not among row 3's attributes.

## ANegativeFromOtherThanMinusOneNamesTheRow

`"from": -2` on row 2 throws with a message containing "layer 2 of 'model.json'", "'from'" and
"row -2". `{"node": -3}` inside an array throws with the layer and "row -3".

## AForwardReferenceNamesTheRow

`"from": 2` on row 1 throws with "layer 1 of 'model.json'", "row 2" and "above row 1".
`"from": [0, 1]` on row 1, which names the row itself, throws with the layer and "row 1".

## AnUnknownKeyOrFormInAFromEntryNamesTheRow

Each of five `from` entries on row 1 throws with "layer 1 of 'model.json'" and one more
needle: the key `prot` with "'prot'" and "'port'"; the string "previous" with "'previous'" and
"'input'"; an empty array with "empty"; the number 0.5 with "it is a number"; and an empty
`port` string with "'port'" and "empty string".

## AnUnknownTopLevelKeyOfAVersionTwoDocumentIsAnError

A top-level key `output` throws with a message containing "'output'", "'outputs'" and
"'defaults'".

## DefaultsFillAnOmittedKeyAndNeverOverrideAStatedOne

The `defaults` block gives `conv` rows `bn_momentum` 3 and `kernel` 5. Row 0 states only
`filters` and ends with the keys `filters`, `bn_momentum`, `kernel` and the values 3 and 5.
Row 1 states `bn_momentum` 7 and keeps it. Row 2 is a `relu` and receives no key.

## AMalformedDefaultsBlockIsAnError

A `defaults` that is an array throws with "'defaults'" and "an array". A type whose default is
the number 3 throws with "'defaults' for 'conv'" and "object". A default that sets `from`
throws with that context and "'from'", and one that sets `type` with "'type'".

## ADefaultsEntryForATypeNoLayerStatesIsAnError

A default for "covn" in a model of `conv` and `relu` rows throws with a message containing
"'defaults'", "'covn'" and "'conv', 'relu'".

## OutputsDefaultToTheLastRowAndKeepWhatTheFileStates

A document with no `outputs` gets one output: the last row, with an empty port and an empty
name. A document that states three outputs keeps each as written: row 0 alone, row 1 with the
port "boxes", and row 1 with the port "boxes" and the name "predictions".

## AMalformedOutputIsAnError

Seven `outputs` values each throw. Row 1 in a one-row model: "'outputs' entry 0", "row 1" and
"0 to 0". The key `nme`: "'nme'" and "'name'". A `name` without a `port`: "'name'" and
"'port'". An empty array: "'outputs'" and "empty". The number 3 as an entry: "a whole number".
An empty `port` string: "'outputs' entry 0 'port'" and "empty string". An empty `name` string:
"'outputs' entry 0 'name'" and "empty string".

## Path resolution in the train and eval documents

Path resolution is the one thing in this file that needs real files on disk. The property is
that a relative path resolves against the config file's own directory and not against the
process working directory, and that cannot be asserted without a config file that exists
somewhere.

## AModelWithoutATaskHoldsANullTask

A version 2 document without `task`, and `K_FIRST_VERSION_MODEL`, both parse to a null task.

## ATaskIsKeptWholeWithItsKeysInFileOrder

A task stating `classes`, `type` and `heads` in that order parses to an object with those
keys in that order, the type "anchor_detection", 80 classes and one head.

## AMalformedTaskIsAnError

The parse raises an error for a task given as an array ("'model.json's 'task'", "an array"),
as null ("'model.json's 'task'", "null"), without a type ("'model.json's 'task'",
"no 'type'", "'classes'") and with the type 3 ("'task' 'type'", "a whole number").

## AVersionOneDocumentRefusesATask

A version 1 document with a `task` raises an error holding "'task'" and "'layers'".

## ParsesATrainDocumentIntoItsFourTypedBlocks

`K_GOOD_TRAIN` parses to the objective type "cross_entropy", the data type "rings", the
optimizer type "adamw" and 3 epochs. The optimizer attributes hold no `type` and a
`learning_rate` of 0.003, and the data attributes hold a `batch_size` of 8.

## ResolvesRelativePathsAgainstTheConfigsOwnDirectory

`load_train` of a `train.json` written under the system's temporary directory gives a model
path of `model.json` and a checkpoint of `out/model.sushi`, both under that directory.

## LeavesAnAbsolutePathAlone

`generic_string()` spells separators as '/', which JSON does not need to escape. The test
therefore builds the document by concatenation and needs no second, test-only escaper that
nobody would maintain.

## EpochsDefaultsToOneAndMustBePositive

A train document without `epochs` parses to 1 epoch and an empty checkpoint path. `epochs` 0
raises an error holding "'epochs'" and "positive".

## AnUnknownTopLevelTrainKeyIsAnError

The key `epochz` raises an error holding "'epochz'" and "'epochs'".

## AMissingTrainBlockNamesIt

A train document without `data` raises an error holding "'data'".

## ATypedBlockThatIsNotAnObjectIsAnError

An `objective` given as a string raises an error holding "'objective'" and "a string".

## AFutureTrainVersionIsAnError

`sushiai_train` 2 raises an error holding "version 2" and "version 1".

## ParsesAnEvalDocument

`parse_data` of a `sushiai_data` 1 document gives the data type "rings", a `batch_size` of
512 and a `seed` of 3735928575.

## AnEvalDocumentCarriesNoModelBlock

The architecture is a separate file that inference is handed separately. Naming one in an eval
document is an error; it is not silently preferred over the one on the command line.

## LoadingAnAbsentFileNamesThePath

`load_train("no/such/train.json")` raises an error holding that path.
