# tests/layers/interchange

The tests of the module `layers/interchange`, one file per header of
`modules/layers/interchange/include/`. The suite prefix of a case gives its ctest label. Each test
file has a `test_*.md` beside it that states what its cases prove.

`protobuf_bytes.hpp` is the helper the tests share: it assembles protobuf wire-format messages by
hand (varints, tags, length-delimited, fixed32, packed fields), so every fixture is a byte string
whose expected parse follows from the encoding rules and not from the reader under test.

## test_onnx_graph.cpp

Suites: `Small_OnnxGraph`. Reads a tiny ONNX model assembled in the test: nodes and attributes of
every type, initializers by `raw_data` (a view, with its offset) and by typed fields, graph inputs
and outputs with their shapes, opset imports and the refusals.

## test_axis_translation.cpp

Suites: `Small_ChannelsLastAxis`, `Small_AxisTranslation`. Checks the axis, permutation, shape and
Flatten rewrites, an OIHW and a depthwise weight permutation and the [C, H, W] to [H, W, C]
permutation of an inner-product weight, each by value.

## test_constant_folding.cpp

Suites: `Small_ConstantFolding`. Folds a Shape to Reshape chain, a Constant and Cast into Resize
scales and keeps every node whose input is not constant.

## test_caffe_net.cpp

Suites: `Small_CaffeNet`. Reads a prototxt and a caffemodel, V1 and V2 layers, and joins them by
layer name.

## test_tensor_bytes.cpp

Suites: `Small_TensorBytes`. Checks that a viewed tensor reports the file's bytes by address with its offset, and that an owned tensor holds the bytes it took and reports offset 0.

## test_caffe_writer.cpp

Checks the unit of the same name, in memory, with 2 cases:

- the readers read back lenet
- a layer with no type is refused

## test_onnx_writer.cpp

Checks the unit of the same name, in memory, with 2 cases:

- the reader reads back the graph
- a tensor that does not fill its extents is refused
