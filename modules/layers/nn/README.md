# nn

The layer library. A layer here describes a computation by tracing itself into a
`Graph::Graph` through a `Graph::GraphBuilder`; it runs nothing. The module holds the layer
types, the YOLO26 blocks built from them, the two state types `Parameter` and `Buffer`, the
`Trace` a layer is traced under, the `Node` seam, the `ModelGraph` container and the two
registries a model file selects through.

The module depends on `core` (`Shape`, `Status`, `Result`, `Error`), on `graph` (the IR, the
builder and `Graph::StateId`) and on `config` (`Config::Value`, `Config::ModelSpec`). `init.hpp`
and `model_state.hpp` also take a `SushiBLAS::Engine`; they are the only headers here that do.

Two designs are the source of most headers. `MG` below is
`docs/agent/2026_10_07_MODEL_GRAPH/SPEC.md` and `CV` is
`docs/agent/2026_10_07_CV_CLOSE/SPEC.md`. `PLAN` is
`docs/agent/2026_10_07_MODEL_GRAPH/PLAN.md`, which lists the public symbols of each unit.

Two error contracts meet in this module. `Buffer::define`, `Trace::update`, every `Node`,
`ModelOutputs`, `ModelGraph`, the node factories, `make_node` and `build_model` return `Status`
or `Result<T>` for a failure a description can cause, and stop the process for a misuse. The
layer types, `LayerStack`, the layer factories and `build_layer_stack` throw `Error`.
`ModuleNode` calls its layer through `Detail::bridge_throwing` in `throwing_bridge.hpp`, which
turns the `Error` into a `Status`. Five sources catch an `Error` from a shape or window check they
call and throw it again with the layer's name: `conv2d.cpp`, `max_pool2d.cpp`, `avg_pool2d.cpp`,
`upsample2d.cpp` and the adapter in `layer_registry.cpp`.

Every layer type and block has the same members: its fields, a static `fields()`, one of the
three trace forms, `Shape output_shape(const Shape&) const`, and one `static_assert` on
`Module`, `TraceModule` or `ModalModule`. Two types differ. `CrossEntropyLoss` has a builder
`forward` and none of the other members. `Concat` is a `Node` over several inputs and has no
`output_shape`.

| Header | Holds |
| --- | --- |
| `module.hpp` | `Parameter`, `ParameterRole`, `ParameterInit`, `field`, `buffer`, `child`, `StateVisitor`, `ParameterVisitor`, `walk_state`, `for_each_parameter`, `for_each_buffer`, `parameter_count`, `buffer_count`, `parameter_elements`, `Module`, `ModuleContainer` |
| `buffer.hpp` | `Buffer` |
| `module_list.hpp` | `ModuleList<T>` |
| `model_state.hpp` | `StateKind`, `StateEntry`, `ModelState` |
| `trace.hpp` | `TraceMode`, `BufferUpdate`, `BufferUpdateSink`, `BufferUpdateList`, `discard_updates`, `Trace`, `TraceModule`, `ModalModule`, `Traceable`, `trace_layer` |
| `layers.hpp` | `Linear`, `ReLU`, `GELU`, `Tanh`, `Sigmoid`, `Sequential` |
| `leaky_relu.hpp` | `LeakyReLU` |
| `silu.hpp` | `SiLU` |
| `identity.hpp` | `Identity` |
| `unflatten.hpp` | `Unflatten` |
| `flatten.hpp` | `Flatten` |
| `conv2d.hpp` | `Conv2d` |
| `max_pool2d.hpp` | `MaxPool2d` |
| `avg_pool2d.hpp` | `AvgPool2d` |
| `global_avg_pool.hpp` | `GlobalAvgPool` |
| `upsample2d.hpp` | `Upsample2d` |
| `batch_norm2d.hpp` | `NormOptions`, `BatchNorm2d`, `make_batch_norm2d_node`, `updated_statistic` |
| `blocks/conv_norm.hpp` | `ConvNorm<Convolution, Activation>`, `Conv`, `ConvLinear`, `make_conv_node` |
| `blocks/bottleneck.hpp` | `Bottleneck`, `make_bottleneck_node` |
| `blocks/c3k.hpp` | `C3k`, `make_c3k_node` |
| `blocks/c3k2.hpp` | `C3k2<Inner>`, `C3k2InnerEntry`, `K_C3K2_INNER`, `make_c3k2_node`, `make_c3k2_of_bottleneck`, `make_c3k2_of_c3k` |
| `blocks/sppf.hpp` | `SPPF`, `make_sppf_node` |
| `loss.hpp` | `CrossEntropyLoss` |
| `init.hpp` | `Nonlinearity`, `calculate_gain`, `InitFamily`, `InitPolicy`, `constant`, `kaiming_normal`, `kaiming_uniform`, `xavier_normal`, `xavier_uniform`, `zeros`, `initialize` |
| `layer.hpp` | `Layer`, `LayerAdapter<L>` |
| `layer_stack.hpp` | `LayerStack` |
| `layer_registry.hpp` | `LayerFactory`, `LayerEntry`, the fourteen `make_*` factories, `K_LAYER_REGISTRY`, `build_layer_stack` |
| `node.hpp` | `Node`, `Port`, `ValueSpan`, `ShapeSpan`, `K_SINGLE_PORT_NAME`, `NodeFactory`, `NodeEntry` |
| `module_node.hpp` | `ModuleNode<L>` |
| `layer_node.hpp` | `LayerNode` |
| `concat.hpp` | `Concat`, `make_concat_node` |
| `throwing_bridge.hpp` | `Detail::bridge_throwing` |
| `model_outputs.hpp` | `ModelOutputs` |
| `model_graph.hpp` | `NodeRef`, `ModelGraph` |
| `node_registry.hpp` | `K_NODE_REGISTRY`, `make_node`, `build_model` |

## State

### module.hpp

A `Parameter` holds a learnable tensor's description and a `Graph::StateId`, and no graph.
`declare()` adds a leaf to the graph of the builder it is given unless that graph already
holds this parameter, so one parameter has one leaf in every graph it is traced into. A copy
is new state and draws its own identity. A move carries the identity away. `ParameterRole`
selects the fill: fan-scaled for `Weight`, zero for `Bias`, one for `Scale`. A second
`declare()` form takes a `ParameterInit` as well. `state()` returns the identity, or
`K_NO_STATE` when none was drawn. `id_in(graph)` returns the parameter's leaf in `graph`, or
`Graph::K_NO_VALUE` when that graph holds none.

A layer lists its state in a static `fields()`: `field` for a `Parameter`, `buffer` for a
`Buffer`, `child` for a member that is walked in turn. `walk_state` visits every declared
parameter and defined buffer in one order. It tries, in this order, the type's own
`walk_state(prefix, visitor)`, then `walk_parameters(prefix, visitor)`, then `visit_children`
named by index, then `fields()` entry by entry. `for_each_parameter` and `for_each_buffer` each
yield one kind. `parameter_count`, `buffer_count` and `parameter_elements` count what those
walks yield; the last sums the element count of every parameter's shape. `ParameterVisitor`
refers to its callable and owns none; `walk_parameters` takes one.

Source: MG 2.2, with the identity from MG 2.1. PLAN unit U2.

### buffer.hpp

A `Buffer` is a tensor of model state that no optimizer moves. `define()` fixes the shape, the
element type and the starting value, draws the identity and returns a `Status`. `read()`
declares the buffer's leaf in the builder's graph once per graph. A buffer that is defined and
never read adds no leaf. A copy is new state, and a move leaves the source undefined.

Source: MG 2.2. PLAN unit U2.

### module_list.hpp

`ModuleList<T>` holds a run-time count of children of one type, and the walk names each by its
index. `T` must be nothrow move constructible, so that growing the list keeps each child's
state identity.

Source: MG 2.2, and its use in MG 2.10. PLAN unit U2.

### model_state.hpp

`ModelState` owns one tensor per entry of a model, allocated once on a `SushiBLAS::Engine`.
An entry is each declared parameter and each defined buffer. `ModelState::create` walks a traced
model and describes every entry as a `StateEntry`: its dotted name, its `StateKind`, its
identity, shape, dtype, `ParameterInit` and fans. The walk skips what no trace declared, so a
model no trace has reached gives an empty state. `from_entries` allocates the tensors of a list
of entries, in order, and refuses a repeated name or identity with `InvalidDescription`.
`tensor()` returns an entry's tensor by name or by identity. `attach` binds the storage of every
state leaf of a graph to a `TensorPool`, so the pool and the state read the same numbers; a leaf
with no entry, or one whose shape or dtype differs, gives `Incompatible`. The state is move-only.

Source: PLAN unit U14, task T14.1.

## Tracing

### trace.hpp

A `Trace` carries what a layer needs to trace itself: the builder, the mode and the sink of
buffer updates. It refers to its builder and its sink and owns neither. It comes from one of
two factories, `Trace::training(builder, updates)` and `Trace::evaluation(builder)`, and cannot
be copied or moved. A layer that refuses its input calls `fail()` and returns its input; the
caller reads `status()` afterwards, and `fail()` marks the graph refused.

A `BufferUpdate` describes
`target = (1 - momentum) * target + momentum * source_scale * source`. A training trace hands
each update to its `BufferUpdateSink`. `BufferUpdateList` collects them in order, and
`discard_updates()` returns the sink that drops them. `update()` under an evaluation trace
returns a failure.

A layer has one of three forms. A `Module` has `forward(const GraphBuilder&, ValueId)`. A
`TraceModule` has `forward(const Trace&, ValueId)`. A `ModalModule` has `trace_training` and
`trace_evaluation` and no `forward`. `trace_layer` calls whichever form a layer has, and a type
that is both a `TraceModule` and a `Module` is traced as a `TraceModule`.

Source: MG 2.3. PLAN unit U6.

## Layers

### layers.hpp

`Linear` is the affine map Y = X * W^T + b. `ReLU`, `GELU`, `Tanh` and `Sigmoid` are
elementwise activations with no parameter. `Sequential` applies its layers in order and names
each child's state by its index. It has a builder `forward` only when every layer is a
`Module`, and a trace `forward` that traces each child through `trace_layer`.

Source: the per-sample `output_shape(const Shape&)` of each type is CV section 2, decision 6.
`Sequential`'s trace `forward` and its `output_shape` are MG 2.3, PLAN unit U6. The types
themselves predate both specs.

### leaky_relu.hpp

`LeakyReLU` applies x where x is positive and `slope * x` elsewhere. The slope must lie in
(0, 1).

Source: CV section 3, with the name `leaky_relu` from CV section 2, decision 9.

### silu.hpp

`SiLU` applies x * sigmoid(x), elementwise, over an input of any rank. It has no parameter.

Source: MG 2.10. PLAN unit U8.

### identity.hpp

`Identity` passes its input through and emits no node. It stands where a block takes no
activation: `ConvLinear` is `ConvNorm<Conv2d, Identity>`.

Source: MG 2.10. PLAN unit U8.

### unflatten.hpp

`Unflatten` views an [N, height * width * channels] value as [N, height, width, channels]
through one K_RESHAPE.

Source: CV section 2, decision 1. MG 2.9 keeps it as a layer and a registry row.

### flatten.hpp

`Flatten` folds [N, d1, ..., dk] into [N, d1 * ... * dk] without moving data. It traces one
K_RESHAPE, or nothing for a rank-2 input.

Source: CV section 3.

### conv2d.hpp

`Conv2d` convolves an NHWC activation with a bank of filters. The weight is rank 4,
[out_channels, kernel_h, kernel_w, in_channels], and `GraphBuilder::conv2d` owns the node
sequence. `Conv2d::same_padded(in, out, kernel, stride)` builds a bias-free layer padded by
kernel / 2 on every side. `bias_init` is the value every bias element starts from.

Source: CV section 2, decisions 3 and 4. `same_padded` is MG 2.10 and `bias_init` is MG 2.8;
both are PLAN unit U9.

### max_pool2d.hpp

`MaxPool2d` takes the maximum of every window of an NHWC activation. The window is a
`Graph::ConvGeometry` whose top and left pads apply to both sides of an axis.

Source: CV section 3.

### avg_pool2d.hpp

`AvgPool2d` averages every window of an NHWC image and divides by the full window area. Its
window is a `Graph::ConvGeometry`, as `MaxPool2d`'s is.

Source: CV section 3.

### global_avg_pool.hpp

`GlobalAvgPool` averages an [N, H, W, C] activation over H and W and gives [N, C].

Source: CV section 3.

### upsample2d.hpp

`Upsample2d` repeats every pixel of an NHWC image over a scale-by-scale block. The scale
applies to both axes and must be at least 1.

Source: CV section 3.

### batch_norm2d.hpp

`BatchNorm2d` normalises an NHWC activation per channel and is a `ModalModule`.
`trace_training` traces K_BATCHNORM2D over batch statistics and requests the two running
updates. `trace_evaluation` traces K_BATCHNORM_FOLD over the running statistics and then
K_CHANNEL_AFFINE. The layer holds gamma as the `SCALE` parameter `weight`, beta as `bias`, and
the buffers `running_mean` and `running_var`, which start at zero and one. `NormOptions` holds
`epsilon`, 1e-5 by default, and `momentum`, 0.1 by default.

`updated_statistic` returns `(1 - momentum) * running + momentum * source_scale * batch` on
the host. `make_batch_norm2d_node` builds the node of a `batchnorm2d` row with the optional
keys `bn_epsilon` and `bn_momentum`.

Source: MG 2.7. PLAN unit U7.

## Blocks

The five headers under `blocks/` are the composite blocks of YOLO26. Each block is a
`TraceModule`, and each has a factory with the `NodeFactory` signature. Source for all five:
MG 2.10, with the state names of MG 2.11. PLAN unit U11.

### blocks/conv_norm.hpp

`ConvNorm<Convolution, Activation>` applies a bias-free convolution, a batch normalisation and
an activation, and lists its state under "conv" and "bn". `Conv` is `ConvNorm<Conv2d, SiLU>`
and `ConvLinear` is `ConvNorm<Conv2d, Identity>`. `make_conv_node` reads `filters`, the
optional `kernel` and `stride`, which are 1 when absent, and the two norm keys.

### blocks/bottleneck.hpp

`Bottleneck` traces `cv1`, then `cv2`, then an add of its input. The add is made only when a
shortcut is asked for and the input width equals the output width; `adds_input` tells which.
Both windows are 3 wide unless the caller states them. `make_bottleneck_node` reads `filters`
and `hidden`, and the optional `shortcut`, which is true when absent.

### blocks/c3k.hpp

`C3k` traces `cv1` and a chain of bottlenecks, then `cv2` on a second branch, then `cv3` over
the two results joined. The hidden width is half the output width, rounded down. The chain is
the `ModuleList<Bottleneck>` named `m`. `make_c3k_node` reads `filters` and the optional
`repeats`, 1 when absent, and `shortcut`, true when absent.

### blocks/c3k2.hpp

`C3k2<Inner>` traces `cv1`, cuts its result into two halves of `hidden` channels, chains the
inner blocks of `m` off the second half, and traces `cv2` over both halves and every inner
result. `K_C3K2_INNER` lists the inner blocks a `c3k2` row may name: "bottleneck" and "c3k".
`make_c3k2_node` reads `filters` and `hidden`, and the optional `inner`, "bottleneck" when
absent, `repeats`, 1 when absent, and `shortcut`, true when absent.

### blocks/sppf.hpp

`SPPF` traces `cv1`, a `ConvLinear` to half the input width, then a stride-1 max pool applied
`pools` times in a chain, then `cv2` over `cv1`'s result and every pooled value joined, then
an add of its input when `adds_input` holds. `make_sppf_node` reads `filters` and the optional
`kernel`, 5 when absent, `pools`, 3 when absent, and `shortcut`, false when absent. The kernel
must be odd.

## Loss and initialisation

### loss.hpp

`CrossEntropyLoss` emits the mean cross-entropy of a batch from an [N, C] value of raw logits
and an [N, C] one-hot target.

Source: neither spec. The header predates both.

### init.hpp

`initialize` records the initial fill of every entry of a `ModelState` on a `SushiBLAS::Engine`,
buffers included, and then runs the recording. An entry whose `ParameterInit` is a constant takes
`constant`. The last fan-scaled entry is zeroed with `zeros` when `zero_final_layer` holds. Every
other fan-scaled entry takes `kaiming_normal`, `kaiming_uniform`, `xavier_normal` or
`xavier_uniform`, as `family` and `uniform` select, with the gain `calculate_gain` gives for
`nonlinearity`. The random stream is seeded with `seed`, so one seed gives one set of values.

Source: PLAN unit U14, task T14.2.

## The run-time layer seam

### layer.hpp

`Layer` is one layer of a model whose architecture came from a file. `output_shape()` is the
virtual that reports the per-sample shape, and `output_features()` is derived from it.
`LayerAdapter<L>` adapts a `Module` to `Layer`. It asserts at compile time that `L` lists no
buffer anywhere in its `fields()` tree, because the adapter walks parameters only.

Source: CV section 2, decision 2. The buffer assertion is MG 2.2, PLAN unit U8.

### layer_stack.hpp

`LayerStack` holds a run-time list of `Layer`s, traces them in order and names each layer's
parameters by its index, as `Sequential` does. It reports the last layer's per-sample output
shape.

Source: CV section 3, the shape-typed layer seam.

### layer_registry.hpp

`K_LAYER_REGISTRY` is the constant table of the fourteen layer types a model file may name:
`linear`, `gelu`, `relu`, `tanh`, `sigmoid`, `leaky_relu`, `unflatten`, `conv2d`, `maxpool2d`,
`avgpool2d`, `global_avgpool`, `flatten`, `upsample` and `silu`. A `LayerFactory` takes the
per-sample input shape, the row's attributes and a context string, and refuses any key it
does not know. `build_layer_stack` builds a chain: every row must read the row before it, row
0 the model input, and `outputs` must be empty or publish the last row whole.

Source: CV section 2, decisions 2 and 9. The chain rule of `build_layer_stack` is MG 3.5, and
the `silu` row is PLAN unit U12.

## The node seam

### node.hpp

`Node` is one node of a model graph: several values in, several values out. `forward` traces
the node and appends one id per entry of `ports()`. `input_count()` is how many values it
reads, and `walk_state` hands its state to a visitor. A `Port` is a name and a per-sample
shape, and a node with one result names its port `K_SINGLE_PORT_NAME`, "output". A
`NodeFactory` builds one node from the per-sample shapes of a row's sources, the row's
attributes and a context string.

Source: MG 2.4. PLAN unit U8.

### module_node.hpp

`ModuleNode<L>` presents a `Traceable` layer that has `output_shape(const Shape&)` as a
one-in one-out node. `create` reads the port's shape from the layer, and `forward` traces the
layer in the form `trace_layer` picks.

Source: MG 2.4. PLAN unit U8.

### layer_node.hpp

`LayerNode` presents a run-time `Layer` as a one-in one-out node. It owns the layer, traces
it through the trace's builder and walks its parameters; it has no buffer.

Source: MG 2.4. PLAN unit U8.

### concat.hpp

`Concat` joins its [H, W, C_i] inputs into one [H, W, sum(C_i)] result through one
K_CONCAT_CHANNELS and holds no state. `Concat::create` refuses an empty list, an input that is
not rank 3, and two inputs that disagree on H or W. `make_concat_node` takes no key.

Source: MG 2.5 and MG 2.10. PLAN unit U8.

### throwing_bridge.hpp

`Detail::bridge_throwing` calls a callable that reports a failure by throwing and returns a
`Result`. An `Error` becomes `InvalidDescription`, `std::bad_alloc` becomes `OutOfMemory`, and
any other exception is fatal.

Source: PLAN unit U8 and the paragraph of PLAN that lists its callers.

## The model graph

### model_outputs.hpp

`ModelOutputs` holds the values one trace of a model produced, by name, in the order they were
published. A name is unique within one set. The set stores graph ids and owns no graph.
`at(name)` returns one value and `only()` returns the value of a set that holds exactly one.

Source: MG 2.5. PLAN unit U5.

### model_graph.hpp

`ModelGraph` holds named nodes and the outputs the model publishes. A node reads only what was
added before it, so the nodes are in topological order and no cycle can be written. The graph
has one input and owns its nodes. A `NodeRef` names one port of one node, or the input; it is
an index pair, so it stays valid when its graph moves.

`add` appends a node, or a `Traceable` layer wrapped in a `ModuleNode`, under a name. `port`
selects a named port, `output` publishes one value and `output_ports` publishes every port of
a node. `trace` traces every node in the order added and returns a `ModelOutputs`, and
`forward` returns the single published output. `walk_state` names each node's state under the
node's name. `ModelGraph` is itself a `TraceModule`.

Source: MG 2.5. PLAN unit U10.

### node_registry.hpp

`K_NODE_REGISTRY` lists the seven node types `K_LAYER_REGISTRY` cannot hold, which its comment
gives as several ports or a mode: `batchnorm2d`, `concat`, `conv`, `bottleneck`, `c3k`, `c3k2`
and `sppf`. `make_node` builds a node of a type from that table, or else from a `K_LAYER_REGISTRY`
row wrapped in a `LayerNode`; a type in neither table fails with a message that lists both.
`build_model` builds the `ModelGraph` a `Config::ModelSpec` describes and names each row by
its index. An empty outputs list publishes every port of the last row.

Source: MG 2.4 for the table and `make_node`, MG 2.5 and MG 3.5 for `build_model`, and MG 2.6
for the file it reads. PLAN unit U12.
