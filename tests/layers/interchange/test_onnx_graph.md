# tests/layers/interchange/test_onnx_graph.cpp

Suites: `Small_OnnxGraph`.

Tests `Interchange::read_onnx_graph`, in `modules/layers/interchange/include/SushiAI/interchange/onnx_graph.hpp`,
on a model assembled byte by byte in the test with the helpers of `protobuf_bytes.hpp`. The
expected values come from the choices made when the bytes were written: the field numbers are
those of `onnx.proto` (ModelProto 1, 2, 7, 8; GraphProto 1, 2, 5, 11, 12; NodeProto 1 to 5;
TensorProto 1, 2, 4, 5, 7, 8, 9, 13, 14; AttributeProto 1 to 11 and 20; ValueInfoProto 1, 2).
The tiny model holds a Conv with four attributes and a field the tables do not name (99), a Relu,
the initializers `w` (eight raw bytes: 1.5f and -2.0f), `b` (typed floats 0.25 and 0.5) and `ids`
(typed int64 7, -1 and 300), the input `x` with the shape [N, 1, 4, 4], the output `z` and two
opset imports.

## ReadsNodesWithTheirNamesAndWires

Both nodes come back in file order with operator, name, inputs and outputs, and the unknown field
99 does not disturb the read.

## ReadsTheAttributesOfANode

`kernel_shape` (ints, typed), `group` (int), `mode` (string) and `epsilon` (a float whose file
states no type, so the type is implied from field 2) read with their values.

## ReadsAnAttributeOfEveryType

One node holds a float, an int (-3, ten varint bytes), a string, a tensor, a graph, floats,
ints written as three unpacked varints, strings, a list of tensors and a list of two graphs. Each
reads into the member its type names.

## ViewsRawInitializerBytesWithoutCopyingThem

The offset the test expects is where the eight payload bytes first appear in the file text, found
with `std::string::find` and checked to be the only place. The tensor is a view, its `offset()`
is that index and its bytes start at that address in the caller's buffer.

## AssemblesTypedInitializersLittleEndian

The typed floats become eight bytes (0.25f, 0.5f) and the typed int64 values 7, -1 and 300 become
24 bytes: `07` then seven zeros, eight `ff`, and `2c 01` then six zeros.

## ReadsGraphInputsAndOutputsWithTheirShapes

`x` is a float input of rank 4 whose first extent is the symbol `N` (value -1) and the others 1,
4 and 4; `z` is the output [N, 2, 4, 4].

## ReadsTheOpsetImports

Two imports read: the default domain at 13, which answers to "" and "ai.onnx", and
`com.vendor` at 2; a domain the model does not import answers 0.

## RefusesExternalDataByTheNameOfTheTensor

A tensor marked `data_location` EXTERNAL and one that lists `external_data` each fail with
`Unsupported`, naming the file and the tensor.

## RefusesAModelWithNoGraph

A model of one field fails with `InvalidFile`.

## RefusesATruncatedMessage

The model with its last three bytes cut fails with `InvalidFile` naming the file.

## RefusesRawDataThatDoesNotFillTheExtents

Eight raw bytes for three floats fail with `InvalidFile` naming the tensor, 8 bytes and 12 needed.

## RefusesAFieldOfTheWrongWireType

A varint where `op_type` expects a length-delimited value fails naming `NodeProto` and field 4.

## RefusesWhatTheTablesDoNotRead

A string tensor, a sparse initializer and an attribute holding a sparse tensor fail with
`Unsupported`, naming the tensor or the attribute.
