# tests/layers/io/test_weight_import.cpp

Suites: `Small_WeightImport`.

Tests `IO::import_state_weights`, in `modules/layers/io/include/SushiAI/io/weight_import.hpp`,
against records written by hand (`NN::StateEntry` with a name, a shape and an element type; no
engine, no tensor is allocated). The two files come from `weight_fixtures.hpp`, which
`test_weight_source.md` describes: `net.fc.weight` [2, 3] holding 1 to 6 and `net.fc.bias` [2]
holding 0.5 and -0.5, in a PyTorch zip and in a safetensors file. The map the tests share has the
state key `state_dict`, the rule `net.` to the empty text, and the drop entry `num_batches_tracked`.

## BothContainersGiveTheSameRecordsAndBytes

The model states `fc.weight` [2, 3] and `fc.bias` [2]. Both files import through the shared map, the
zip with its dropped counter and the safetensors file without one. Each returns two tensors in the
model's order, with the model's names, shapes and element type float32; the payload of each equals
the little-endian bytes of the values listed above, built apart from the code under test, and the
two payloads are equal.

## OneMessageNamesTheMissingTheExtraAndTheMisshapenTensor

The model states `fc.weight` as [3, 2] and `fc.missing` as [4], and no `fc.bias`. The safetensors file
fails once, with `Incompatible`. The one message names the origin, the record `fc.missing` the file
does not supply, the tensor `net.fc.bias` renamed to `fc.bias` that no record takes, and the tensor
`net.fc.weight`, which is [3, 2] in the model and [2, 3] in the file.

## TheMessageGroupsEachKindWithItsCount

With a second absent record, `fc.other`, the message holds the three headings with their counts:
"records the file does not supply (2): 'fc.missing', 'fc.other'", "tensors of the file that no
record takes (1)" and "tensors of another shape than their record (1)".

## TwoTensorsRenamingToOneRecordAreRefused

Two whole-name rules send `net.fc.weight` and `net.fc.bias` to `fc.weight`. The second is listed as a
tensor no record takes, with the tensor that already supplies the record, and `fc.bias` is listed as
a record the file does not supply.

## RefusesARecordOfAnotherElementType

A record `fc.bias` of element type int32 fails with `Unsupported`, naming the record and `int32`.
The match is clean, so the failure comes after it.

## AFloat64RecordIsFilledWithTheWidenedValues

A float64 record `fc.bias` gets 16 bytes that read back as the doubles 0.5 and -0.5.

## AnEmptyMapFillsAModelThatStatesTheFilesNames

With the map `{}` and records named `net.fc.weight` and `net.fc.bias`, the import succeeds and keeps
the file's names and bytes.

## AnOihwConvolutionWeightImportsInTheKernelLastLayout

A safetensors file holds `conv.weight` as a float32 [2, 3, 3, 3] tensor filled with 1 to 54, and the
model states `conv.weight` as [2, 3, 3, 3]. The shape fits the record both as stored and permuted, so
the permutation wins. The payload equals the bytes `copy_kernel_last` writes for the same values,
and equals the layout the index formula gives: element (o, h, w, i) is source element
`((o * 3 + i) * 3 + h) * 3 + w`.

## AnOihwWeightOfUnequalExtentsGivesTheRecordsShapeAndOrder

A [2, 3, 1, 2] weight (out 2, in 3, kh 1, kw 2) filled with 1 to 12 fills a record [2, 1, 2, 3]. The
record keeps its shape and the payload equals the index formula's layout.

## ARankFourTensorInTheRecordsShapeIsCopiedAsStored

A [2, 1, 1, 3] tensor fills a record [2, 1, 1, 3]. Its permutation would be [2, 1, 3, 1], which is not
the record's shape, so the payload equals the stored values.

## ARankFourTensorMatchingNeitherLayoutIsMisshapen

A [2, 3, 1, 1] tensor against a record [3, 1, 1, 3] fails with `Incompatible`, and the message lists
`conv.weight` under "tensors of another shape than their record" with the record's shape.

## ADepthwiseWeightImportsAsKernelHeightWidthChannels

A safetensors file holds `conv.weight` as a float32 [4, 1, 2, 3] tensor filled with 1 to 24, and the
model states `conv.weight` as the rank-3 [2, 3, 4]. The payload equals the bytes
`copy_depthwise_kernel_last` writes for the same values, and equals the index formula's layout:
element (h, w, c) is source element `(c * 2 + h) * 3 + w`.

## AGroupedWeightIsNotTakenForADepthwiseOne

A [4, 2, 2, 3] tensor against a record [2, 3, 4] has two input channels per filter, so it is not
depthwise. The import fails with `Incompatible` and lists `conv.weight` under "tensors of another
shape than their record".
