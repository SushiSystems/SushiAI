# interchange

## Owns

The interchange module holds what the importers need to know about the file formats of other
frameworks and nothing about vision or language models. Each unit takes bytes, text or values and
returns values; none writes a file or names a SushiAI layer. The operator tables and the importers
that turn these values into model rows stay in the domain that owns the models, as ruled in
`docs/design/MODEL_IMPORT/README.md`.

`OnnxGraph` is the graph of an ONNX model, and `read_onnx_graph` builds it from the file's bytes
through `IO::ProtobufWire`. The field-number tables of `ModelProto`, `OperatorSetIdProto`,
`GraphProto`, `NodeProto`, `TensorProto`, `AttributeProto`, `ValueInfoProto`, `TypeProto`,
`TypeProto.Tensor`, `TensorShapeProto` and its `Dimension` are `enum class` lists in
`source/onnx_graph.cpp`; a field no table names is skipped by its wire type. The result holds the
nodes (operator, name, domain, inputs, outputs and attributes of every kind: float, int, string,
tensor, subgraph and the lists of each), the initializers, the graph inputs and outputs with their
element types and declared shapes (a symbolic extent keeps its symbol), and the opset imports. An
attribute whose file states no `type` has it implied from the payload field. An initializer stored
in `raw_data` is a `TensorBytes` view into the file with its offset, and no byte is copied; one
stored in `float_data`, `int32_data`, `int64_data`, `double_data` or `uint64_data` is assembled
once into the little-endian layout of its element type. The reader refuses with `Unsupported` a
tensor kept in an external file (named by tensor), a string tensor, a sparse initializer and an
attribute that holds a sparse tensor or a type; it refuses with `InvalidFile` a model with no
graph, a tensor whose bytes do not fill its extents, a field of the wrong wire type and a
message cut short. Subgraphs nest at most 16 deep.

`axis_translation.hpp` rewrites what the channel-first layout of ONNX and Caffe (`NCHW`) states into
SushiAI's channel-last one (`NHWC`), with pure functions on values. The channel axis moves to the
end and the axes after it move one place down. `channels_last_axis` and `channels_last_axes`
move axes (a negative axis counts from the end first), `channels_last_permutation` rewrites a
Transpose's `perm`, `channels_last_shape` rewrites a Reshape target and keeps its 0 and -1,
and `channels_last_flatten_axis` rewrites a Flatten axis only where it leaves the batch on one
side, refusing the rest with `Unsupported`. `axis_attribute_kind` and `translate_axis_attribute`
look an (operator, attribute) pair up in one table (Concat, Softmax, LogSoftmax, Split, Gather,
ArgMax, ArgMin, Slice, the reductions, Transpose, Reshape, Flatten) and return the integers
unchanged for a pair the table does not list. `filter_to_record_layout` permutes a convolution
weight [out, in, kh, kw] to the record layout [out, kh, kw, in], and a depthwise weight
[C, 1, kh, kw] to [kh, kw, C], through `IO::copy_kernel_last` and
`IO::copy_depthwise_kernel_last`; `filter_record_dimensions` gives the extents.
`flatten_weights_to_channels_last` moves the inputs of every output row of an inner-product weight
from the order [C, H, W] of a channel-first blob to the order [H, W, C] that a `flatten` row
yields; the Caffe `InnerProduct` and the ONNX `Gemm` translators both call it, so the permutation
lives once.

`fold_constants` evaluates the shape arithmetic of an ONNX export on the node list of an
`OnnxGraph`. It computes `Constant`, `Shape`, `Gather`, `Unsqueeze`, `Concat` and `Cast` nodes
whose inputs are constant and removes each one whose results are read only by such nodes and by
`Reshape`, `Resize` and `Expand`; the result is a `FoldedGraph` of the kept nodes in order and the
constants those nodes now read, by tensor name. A `Shape` is constant when its input is a graph
input with every extent stated or an initializer. An initializer is read as a constant when it is
int32, int64 or float32 and holds at most 1024 elements. A node with a non-constant input, a result
that is a graph output or that another operator reads stays untouched. Inputs that disagree
(an index past the extent, extents that do not concatenate) fail with `InvalidDescription` naming
the node.

`CaffeNet` is a net read from its two files. `read_caffe_prototxt` parses the prototxt through
`IO::parse_protobuf_text` and returns the layers in file order with name, type, bottoms, tops and
the layer's own message (`parameters`, such as `convolution_param`), plus the net-level inputs
with their shapes (`input_dim` and `input_shape`). The `layers` spelling of Caffe's V1 reads as a
layer too: its type enumerator, `INNER_PRODUCT`, becomes the string `InnerProduct` through a table
in `source/caffe_v1_types.cpp` (the types 1 to 39 of `V1LayerParameter.LayerType`), and the layer is
marked legacy. `read_caffe_model` reads the layers
of a caffemodel through `IO::ProtobufWire` (`NetParameter` fields 1, 2 and 100, `LayerParameter`,
`V1LayerParameter`, `BlobProto` and `BlobShape`) with their blobs: extents from `shape` or from the
four legacy extents, floats as a `TensorBytes` that views a packed field in place and owns the
bytes of an unpacked one. `attach_caffe_weights` joins the two by layer name. It fails with
`Incompatible` and changes nothing when a weights layer that carries blobs names no layer of the
net (the message lists every such name), names several, differs from the net's layer in type, or
meets a layer that already holds blobs. A weights layer with no blobs and no place in the net is
not an error.

`TensorBytes` holds the bytes of one tensor as a view into the file or as an owned buffer. Both
`OnnxTensor` and `CaffeBlob` use it, so an importer copies a tensor once, into the `.sushi` writer.

## Depends on

`CMakeLists.txt` lists `PUBLIC_DEPENDS failure io`, `EXTERNAL_PUBLIC SushiRuntime::core` and
`EXTERNAL_PRIVATE_LINKS SushiLog::text`.

- `failure`: `Status`, `Result` and the macros every function returns its errors with.
- `io`: `IO::ProtobufWire` and `IO::float_from_bits` (every reader), `IO::TextField` and
  `IO::parse_protobuf_text` (`caffe_net.hpp`, whose `CaffeLayer` holds a `TextField`), and
  `IO::copy_kernel_last` and `IO::copy_depthwise_kernel_last` (`axis_translation.cpp`).
- `SushiRuntime::core`: `SushiRuntime::span`, which the readers take and `TensorBytes` returns.

The private headers `source/wire_values.hpp` (the reader of one typed field value, shared by the
two message tables) and `source/caffe_v1_types.hpp` hold what no caller needs.

## Public entry points

The headers sit in `modules/layers/interchange/include/SushiAI/interchange/`.

| Header | Declares |
| --- | --- |
| `tensor_bytes.hpp` | `TensorBytes` |
| `onnx_graph.hpp` | `OnnxDataType`, `onnx_element_bytes`, `OnnxDimension`, `OnnxValueInfo`, `OnnxTensor`, `OnnxAttributeType`, `OnnxAttribute`, `OnnxNode`, `OnnxOpset`, `OnnxGraph`, `read_onnx_graph` |
| `axis_translation.hpp` | `channels_last_axis`, `channels_last_axes`, `channels_last_permutation`, `channels_last_shape`, `channels_last_flatten_axis`, `AxisAttributeKind`, `axis_attribute_kind`, `translate_axis_attribute`, `filter_record_dimensions`, `filter_to_record_layout`, `flatten_weights_to_channels_last`, and the channel-first counterparts `channels_first_axis`, `channels_first_permutation`, `filter_to_source_layout`, `flatten_weights_to_channels_first` |
| `onnx_writer.hpp` | `write_onnx_graph`, the inverse of `read_onnx_graph`: tensors as `raw_data`, value infos with their tensor type and shape |
| `caffe_writer.hpp` | `write_caffe_prototxt`, `write_caffe_model`, the inverses of the two Caffe readers; every layer is written as a V2 `layer` |
| `constant_folding.hpp` | `ConstantKind`, `ConstantValue`, `NamedConstant`, `FoldedGraph`, `fold_constants` |
| `caffe_net.hpp` | `CaffeBlob`, `CaffeLayer`, `CaffeNet`, `read_caffe_prototxt`, `read_caffe_model`, `attach_caffe_weights` |
