# nn

The layer library. A layer here describes a computation by tracing itself into a
`Graph::Graph` through a `Graph::GraphBuilder`; it runs nothing. The module holds the layer
types, the two state types `Parameter` and `Buffer`, the
`Trace` a layer is traced under, the `Node` seam, the `ModelGraph` container and the two
registries a model file selects through.

The module depends on `core` (`Shape`, `Status`, `Result`, `Error`), on `graph` (the IR, the
builder and `Graph::StateID`) and on `config` (`Config::Value`, `Config::ModelSpec`). `init.hpp`
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
call and throw it again with the layer's name: `source/conv2d.cpp`, `source/max_pool2d.cpp`,
`source/avg_pool2d.cpp`, `source/upsample2d.cpp` and the adapter in `source/layer_registry.cpp`.

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
| `host_table.hpp` | `HostTable`, `HostTableSink`, `HostTableList` |
| `layers.hpp` | `Linear`, `ReLU`, `GELU`, `Tanh`, `Sigmoid`, `Sequential` |
| `leaky_relu.hpp` | `LeakyReLU` |
| `silu.hpp` | `SiLU` |
| `mish.hpp` | `Mish` |
| `identity.hpp` | `Identity` |
| `unflatten.hpp` | `Unflatten` |
| `flatten.hpp` | `Flatten` |
| `conv2d.hpp` | `Conv2d` |
| `depthwise_conv2d.hpp` | `DepthwiseConv2d` |
| `max_pool2d.hpp` | `MaxPool2d` |
| `avg_pool2d.hpp` | `AvgPool2d` |
| `global_avg_pool.hpp` | `GlobalAvgPool` |
| `upsample2d.hpp` | `Upsample2d` |
| `channel_slice.hpp` | `ChannelSlice` |
| `batch_norm2d.hpp` | `NormOptions`, `BatchNorm2d`, `make_batch_norm2d_node`, `updated_statistic` |
| `embedding.hpp` | `Embedding`, `make_embedding_node`, `find_embedding` |
| `layer_norm.hpp` | `K_LAYER_NORM_EPSILON`, `LayerNorm`, `make_layer_norm_node` |
| `dropout.hpp` | `Dropout`, `make_dropout_node` |
| `loss.hpp` | `CrossEntropyLoss` |
| `init.hpp` | `Nonlinearity`, `calculate_gain`, `InitFamily`, `InitPolicy`, `constant`, `kaiming_normal`, `kaiming_uniform`, `xavier_normal`, `xavier_uniform`, `zeros`, `initialize` |
| `layer.hpp` | `Layer`, `LayerAdapter<L>` |
| `layer_stack.hpp` | `LayerStack` |
| `layer_registry.hpp` | `LayerFactory`, `LayerEntry`, the sixteen `make_*` factories, `register_layers`, `build_layer_stack` |
| `node.hpp` | `Node`, `Port`, `ValueSpan`, `ShapeSpan`, `K_SINGLE_PORT_NAME`, `NodeFactory`, `NodeEntry`, `LinkedNodeFactory`, `LinkedNodeEntry` |
| `built_rows.hpp` | `BuiltRows` |
| `module_node.hpp` | `ModuleNode<L>` |
| `layer_node.hpp` | `LayerNode` |
| `concat.hpp` | `Concat`, `make_concat_node` |
| `add.hpp` | `Add`, `make_add_node` |
| `throwing_bridge.hpp` | `Detail::bridge_throwing` |
| `model_outputs.hpp` | `ModelOutputs` |
| `model_graph.hpp` | `NodeRef`, `ModelGraph` |
| `node_registry.hpp` | `NodeRegistry`, `make_node`, `build_model` |

## State

### module.hpp

A `Parameter` holds a learnable tensor's description and a `Graph::StateID`, and no graph.
`declare()` adds a leaf to the graph of the builder it is given unless that graph already
holds this parameter, so one parameter has one leaf in every graph it is traced into. A copy
is new state and draws its own identity. A move carries the identity away. `ParameterRole`
selects the fill: fan-scaled for `Weight` and for `Embedding`, zero for `Bias`, one for
`Scale`. A second `declare()` form takes a `ParameterInit` as well: `fan_scaled()`,
`constant_value(value)` or `normal(deviation)`, a zero-mean normal of one stated deviation
that the fans do not scale. The role is also what `Train::takes_weight_decay` reads, so
`Embedding` marks a table that weight decay leaves alone. `state()` returns the identity, or
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

A `Trace` carries what a layer needs to trace itself: the builder, the mode and its sinks. It
refers to its builder and its sinks and owns none of them. It comes from one of three
factories, `Trace::training(builder, updates)`, `Trace::evaluation(builder)` and
`Trace::carrying(builder, fed, carried)`, and cannot be copied or moved. A layer that refuses its input calls `fail()` and returns its input; the
caller reads `status()` afterwards, and `fail()` marks the graph refused.

A `BufferUpdate` describes
`target = (1 - momentum) * target + momentum * source_scale * source`. A training trace hands
each update to its `BufferUpdateSink`. `BufferUpdateList` collects them in order, and
`discard_updates()` returns the sink that drops them. `update()` under an evaluation trace
returns a failure.

`Trace::training` takes a seed as an optional third argument; `seed()` returns it, and 0 for
every other trace. `claim_random_site()` returns 0, 1, 2 and so on in call order, so a layer
that draws at each run of the step can tell its site from the others of the trace.

A layer has one of three forms. A `Module` has `forward(const GraphBuilder&, ValueID)`. A
`TraceModule` has `forward(const Trace&, ValueID)`. A `ModalModule` has `trace_training` and
`trace_evaluation` and no `forward`. `trace_layer` calls whichever form a layer has, and a type
that is both a `TraceModule` and a `Module` is traced as a `TraceModule`.

A carrying trace is an evaluation trace for a step that keeps values between its runs. Its
mode is `Evaluation`, so a `ModalModule` traces its evaluation form and `update()` fails as it
does under `Trace::evaluation`. It adds two requests, and both fail with `InvalidDescription`
in the other two forms:

| Call | Declares | Reaches |
| --- | --- | --- |
| `feed(name, shape, dtype)` | An input leaf the caller of the step fills on every run. A second call with the same name returns the same leaf, and one with another shape or type is refused | `FedLeafSink`, as a `FedLeaf` of the name and the leaf |
| `carried_leaf(shape, debug_name, dtype)` | An input leaf the step itself rewrites | Nothing yet |
| `carry(leaf, next, initial_value)` | That `leaf` holds `next` after each run, and `initial_value` before the first one | `CarriedValueSink`, as a `CarriedValue` |

`FedLeafList` and `CarriedValueList` collect them in order. A carried value is a leaf of the
graph and no `Buffer`, so it is no entry of a `ModelState` and no checkpoint holds it. The
name given to `feed` and to `carried_leaf` must outlive the graph, as a debug name does.

A layer may have a fourth entry beside its form:
`forward_incremental(const Trace&, ValueID x, const Shape& sample)`. It traces the newest
slice of what `forward` reads whole, and `sample` is the per-sample shape of the whole value.
`IncrementalModule<T>` holds for such a layer and `trace_layer_incremental` calls it.
`Embedding` and `LayerNorm` have the entry and forward it to `forward`, since a row reads no
other row.

`Trace::host_table(shape, debug_name, dtype, values)` declares a leaf whose numbers the host
computed during the trace, float32 or float64, one value per element. It works in every form
of trace once `collect_tables(sink)` has named a `HostTableSink`, and fails with
`InvalidDescription` before that. The leaf is a `Lifetime::Buffer` value bound to no state:
it is no `Parameter` and no `Buffer`, so no `ModelState`, optimizer or checkpoint holds it.
The step built from the trace writes the values once. Two declarations are two leaves, even
under one debug name. `RotaryEmbedding` of `language/blocks` declares its cosines and sines
this way. `tests/test_host_table.cpp` covers the declaration and its refusals.

Source: MG 2.3. PLAN unit U6. The carrying form: `docs/design/LANGUAGE.md`, phase 5.

## Layers

### layers.hpp

`Linear` is the affine map Y = X * W^T + b. Its `weight_init` is the fill of the weight,
fan-scaled unless a model states another. `ReLU`, `GELU`, `Tanh` and `Sigmoid` are
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

### mish.hpp

`Mish` applies x * tanh(softplus(x)), elementwise, over an input of any rank. It has no
parameter. `forward` emits SOFTPLUS, TANH and MUL, and the result has the shape and the layout
of the input.

Source: `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md`, section 2.1.

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

### depthwise_conv2d.hpp

`DepthwiseConv2d` filters each channel of an NHWC activation with its own window of taps. The
weight is rank 3, [kernel_h, kernel_w, channels], and `GraphBuilder::depthwise_conv2d` owns the
node. `DepthwiseConv2d::same_padded(in, out, kernel, stride)` builds a layer padded by
kernel / 2 on every side; it throws when `in` and `out` differ. The layer has no bias. Both fans
of the weight are kernel_h * kernel_w, since each channel reads and writes only its own taps.

Source: YOLO26 plan, `docs/agent/2026_10_07_YOLO26/PLAN.md`, unit U5.

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

### channel_slice.hpp

`ChannelSlice` keeps the channels [`offset`, `offset + count`) of an NHWC image through one
K_SLICE_CHANNELS and has no parameter. `offset` must be at least 0 and below the input's channel
count, and `count` must be at least 1 and must not run past it. `output_shape` maps [H, W, C] to
[H, W, `count`].

Source: `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md`, section 2.1.

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

### embedding.hpp

`Embedding` maps each INT32 index of its input to a row of a learned `[num_embeddings,
embedding_dim]` table and is a `TraceModule`. `forward` traces one K_GATHER_ROWS over the table
and the input, so an input of `[N, T]` gives `[N, T, embedding_dim]`. The table is the
parameter `weight`, of role `Embedding`, drawn from a normal of deviation
`K_EMBEDDING_DEVIATION`, 0.02, unless `weight_init` states another fill; `dtype` names its
element type, FLOAT32 by default, since the input cannot. The layer refuses an input that is
not INT32 or whose rank leaves no axis for the row. It does not read the indices: one outside
`[0, num_embeddings)` is not caught.

`make_embedding_node` builds the node of an `embedding` row. The row reads one per-sample shape
of rank 1 to 4 and takes two required keys, both whole numbers above zero:

| Key | Holds |
| --- | --- |
| `num_embeddings` | The row count of the table |
| `embedding_dim` | The length of one row, which becomes the last extent of the result |

Source: `docs/design/LANGUAGE.md`, phase 1.

### layer_norm.hpp

`LayerNorm` normalises each `[dim]` row of a `[..., dim]` value by the row's own mean and
variance and is a `TraceModule`. `forward` traces one K_LAYER_NORM, which is the same node under
a training and an evaluation trace, since the layer has no running statistic. The gain is the
`Scale` parameter `weight`, which starts at one, and `bias` starts at zero; both take the
input's element type. `epsilon` is `K_LAYER_NORM_EPSILON`, 1e-5, unless stated. The layer
refuses an input whose last extent is not `dim` or that is not a real float.

`make_layer_norm_node` builds the node of a `layernorm` row. The row reads one per-sample shape
of rank 1 to 5, takes `dim` from its last extent, and has one optional key:

| Key | Holds |
| --- | --- |
| `ln_epsilon` | The constant added to a row's variance, a positive number; 1e-5 when absent |

Source: `docs/design/LANGUAGE.md`, phase 2.

### dropout.hpp

`Dropout` holds a drop `rate` in `[0, 1)` and is a `ModalModule` with no state. Its training
form emits one `K_DROPOUT` node over its input: each element is zeroed with probability `rate`
and scaled by `1 / (1 - rate)` otherwise, with a new mask at each run of the compiled step. Its
evaluation form returns its input and emits no node, and so does the training form at a rate
of 0. A rate outside the interval fails the trace.

Each emitted node is one random site. The layer asks the trace for the site's order with
`Trace::claim_random_site()` and gives the node the seed
`Graph::dropout_site_seed(trace.seed(), order)`, so two sites of one model draw different masks
and a model traced twice with one seed draws the same ones. `Trace::training` takes the seed as
its third argument, 0 when absent; `Train::GraphOptions::seed` carries it there. The mask has
the element type of the operand: the host draws FLOAT32 ones and zeros and a converting copy
carries them into a HALF or FLOAT64 graph. A step that is compiled again starts at run 0. A run
resumed from a checkpoint compiles its step again and then restores the run count with
`Graph::CompiledStep::restore_run_count`, so under the seed of the saved run it draws the masks
that run would have drawn next.

`make_dropout_node` builds the node of a `dropout` row. The row reads one per-sample shape of
any rank and has one required key:

| Key | Holds |
| --- | --- |
| `rate` | The probability of dropping an element, a number in `[0, 1)` |

Source: `docs/design/LANGUAGE.md`, the layers that go to `nn`.

## Loss and initialisation

### loss.hpp

`CrossEntropyLoss` emits the mean cross-entropy of a batch from an [N, C] value of raw logits
and an [N, C] one-hot target.

Source: neither spec. The header predates both.

### init.hpp

`initialize` records the initial fill of every entry of a `ModelState` on a `SushiBLAS::Engine`,
buffers included, and then runs the recording. An entry whose `ParameterInit` is a constant takes
`constant`, and one whose kind is `Normal` takes `normal`, which refuses a deviation that is
not positive and finite. Neither is the final layer, whatever its position. The last fan-scaled entry is zeroed with `zeros` when `zero_final_layer` holds. Every
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

`register_layers` adds to a `NodeRegistry` the sixteen layer types a model file may name,
`linear`, `gelu`, `relu`, `tanh`, `sigmoid`, `leaky_relu`, `unflatten`, `conv2d`, `maxpool2d`,
`avgpool2d`, `global_avgpool`, `flatten`, `upsample`, `silu`, `mish` and `slice_channels`, then the node types `batchnorm2d`, `concat`, `add`, `embedding`,
`layernorm` and `dropout`. A `LayerFactory` takes the
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

`forward_incremental` is the second entry of a node: its incremental form, traced in a
carrying trace over the newest slice of each input. The base returns `Unsupported`, "the node
has no incremental form", and `ModelGraph::trace_incremental` puts the node's name in front.
A node overrides it when it has such a form.

A `LinkedNodeFactory` takes a fourth argument, the `BuiltRows` above the row, for a row that
names another row: a head that scores with the table of an embedding row is one. It is a
second factory type beside `NodeFactory`, held in a `LinkedNodeEntry`, and not a new argument
of `NodeFactory`, because the vision blocks call a `NodeEntry::make` with three arguments.

### built_rows.hpp

`BuiltRows` views the nodes of the rows above the one being built. `build_model` names node
`i` of its graph after row `i`, so the view is the graph: `size()` is the index of the row
being built and `node(i)` is the node of row `i`. A default view holds no row, which is what
`make_node` passes when it builds a node outside a model. A node it returns is owned by the
`ModelGraph` through a `std::unique_ptr`, so its address holds when the graph moves.

`Detail::read_row_above` in `node_attributes.hpp` reads a reference to such a row from a
row's attributes: a row number, `-1` for the row before, or an object with `node`. It refuses
a row that is not above, the string `input` and an object with `port`, since a row reference
names a node and not a value.

`find_embedding` in `embedding.hpp` returns the `Embedding` a node built by
`make_embedding_node` holds, or null for any other node.

Source: MG 2.4. PLAN unit U8.

### module_node.hpp

`ModuleNode<L>` presents a `Traceable` layer that has `output_shape(const Shape&)` as a
one-in one-out node. `create` reads the port's shape from the layer, and `forward` traces the
layer in the form `trace_layer` picks. `forward_incremental` calls the layer's own
`forward_incremental` with the per-sample shape `create` was given, and returns the base's
`Unsupported` for a layer that has none.

Source: MG 2.4. PLAN unit U8.

### layer_node.hpp

`LayerNode` presents a run-time `Layer` as a one-in one-out node. It owns the layer, traces
it through the trace's builder and walks its parameters; it has no buffer. It keeps the
base's `forward_incremental`, so a layer row has no incremental form.

Source: MG 2.4. PLAN unit U8.

### concat.hpp

`Concat` joins its [H, W, C_i] inputs into one [H, W, sum(C_i)] result through one
K_CONCAT_CHANNELS and holds no state. `Concat::create` refuses an empty list, an input that is
not rank 3, and two inputs that disagree on H or W. `make_concat_node` takes no key.

Source: MG 2.5 and MG 2.10. PLAN unit U8.

### add.hpp

`Add` sums its two [H, W, C] inputs elementwise into one [H, W, C] result through one K_ADD and
holds no state. `Add::create` refuses an input count other than two, an input that is not
rank 3, and two inputs of different shapes. `make_add_node` takes no key.

Source: `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md`, section 2.1.

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

`trace_incremental` walks the nodes in the same order through `Node::forward_incremental`
and needs a carrying trace. It compares no port with its declared per-sample shape, because a
slice has the shape its node gives it; `trace` checks every port as before. The first node
without an incremental form fails the walk as `Unsupported` under "node '<name>'".
`forward_incremental` returns the single published output of that walk, so a nested graph is
a node that has the form.

Source: MG 2.5. PLAN unit U10.

### node_registry.hpp

`NodeRegistry` holds the rows a model file may name, by type. A node row is a type a layer
cannot hold: several ports or a mode. The vision and language registration add their own node
rows, such as `conv`, `c3k2` and `sppf`. `make_node` builds a node of a type from a node row,
or else from a layer row wrapped in a `LayerNode`; a type the registry does not hold fails with a
message that lists every row.
`build_model` builds the `ModelGraph` a `Config::ModelSpec` describes and names each row by
its index. An empty outputs list publishes every port of the last row.

A `NodeRegistry` holds a third kind of row, the linked node rows, added with
`add(const LinkedNodeEntry&)` and read with `find_linked` and `linked()`. A type has one row
among the three kinds. `make_node` looks a type up among the node rows, then the linked node
rows, then the layer rows, and hands a linked factory the `BuiltRows` it was given;
`build_model` passes the rows above each row. The message for an unknown type lists the
linked types after the node types.

Source: MG 2.4 for the table and `make_node`, MG 2.5 and MG 3.5 for `build_model`, and MG 2.6
for the file it reads. PLAN unit U12.
