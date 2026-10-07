# Model graph: plan and state of the work

**Status:** In progress. Waves 0 to 6 written and syntax-checked only.

The design is `SPEC.md` in this folder. No file of waves 0 to 6 has been built or run: the owner
builds once the programme's code is written (`docs/design/PROGRAMME.md:30`). Every changed C++
file was parsed alone with `clang -fsyntax-only`, with the flags `build/compile_commands.json`
gives `src/graph/fusion.cpp` and `tests/functional/unit/test_ir.cpp`.

## 1. What each unit put on disk

| Unit | Files | Public declarations |
| --- | --- | --- |
| U1 | `graph/state_id.hpp`, `graph/ir.hpp`, `graph/builder.hpp`, `core/ops.hpp` | `Graph::StateId`, `K_NO_STATE`, `StateIdentity` (`get`, `peek`; a copy holds none, a move carries it). `Graph::bind_state_leaf`, `state_leaf`, `state_leaves`, `without_nodes`. `Lifetime::BUFFER`, `GraphBuilder::buffer`. `BatchNormResult::batch_variance`, the fourth `BATCHNORM2D` output. `BATCHNORM_FOLD`, `GraphBuilder::batchnorm_fold`, `BatchNormFold` |
| U2 | `nn/buffer.hpp`, `nn/module_list.hpp`, `nn/module.hpp` | `Buffer` (`define` returning `Status`, `read`, `id_in`, `state`, `declared`, `shape`, `dtype`, `initial_value`). `ModuleList<T>`. `ParameterInit`, `ParameterRole::SCALE`, the two `Parameter::declare` forms, `id_in`, `state`, `init`. `BufferField`, `ChildField`, `buffer()`, `child()`, `StateVisitor`, `walk_state`, `for_each_buffer`, `buffer_count` |
| U3 | `config/model_config.hpp`, `src/config/model_config.cpp` | `MODEL_VERSION` 2, `K_MODEL_INPUT`, `SourceSpec`, `OutputSpec`, `LayerSpec::from`, `ModelSpec::input_shape` and `outputs`, `parse_model`, `load_model`; the version 1 upgrade runs in memory |
| U4 | `graph/step_recorder.hpp` | `StepRecorderList` (`add`, the forwarding `record_into_step`) |
| U5 | `nn/model_outputs.hpp`, `src/nn/model_outputs.cpp` | `ModelOutputs` (`add`, `at`, `only`, `all`) |
| U6 | `nn/trace.hpp`, `src/nn/trace.cpp`, `nn/layers.hpp` | `TraceMode`, `BufferUpdate`, `BufferUpdateSink`, `BufferUpdateList`, `discard_updates`, `Trace` (`training`, `evaluation`, `builder`, `mode`, `update`, `fail`, `status`, `value_or_fail`), `TraceModule`, `ModalModule`, `Traceable`, `trace_layer`, `Sequential::forward(const Trace&, ValueId)` and `Sequential::output_shape(const Shape&)` |
| U7 | `nn/batch_norm2d.hpp`, `src/nn/batch_norm2d.cpp` | `NormOptions`, `BatchNorm2d` (`trace_training`, `trace_evaluation`, `output_shape`), `make_batch_norm2d_node`, `updated_statistic` |
| U8 | `nn/node.hpp`, `nn/module_node.hpp`, `nn/layer_node.hpp`, `nn/concat.hpp`, `nn/silu.hpp`, `nn/identity.hpp`, `nn/throwing_bridge.hpp` and sources | `Node` (`forward`, `input_count`, `walk_state`, `ports`), `Port`, `ValueSpan`, `ShapeSpan`, `NodeFactory`, `NodeEntry`, `ModuleNode<L>::create`, `LayerNode`, `Concat::create`, `make_concat_node`, `SiLU`, `Identity`, `Detail::bridge_throwing` |
| U9 | `nn/conv2d.hpp`, `src/nn/conv2d.cpp` | `Conv2d::same_padded`, `Conv2d::bias_init` |
| U10 | `nn/model_graph.hpp`, `src/nn/model_graph.cpp` | `NodeRef`, `ModelGraph` (`input`, both `add` forms, `port`, `output`, `output_ports`, `trace`, `forward`, `output_shape`, `walk_state`, `size`, `shape_of`, `node`) |
| U11 | `nn/blocks/*.hpp`, `src/nn/blocks/*.cpp` | `ConvNorm<Convolution, Activation>`, `Conv`, `ConvLinear`, `Bottleneck`, `C3k`, `C3k2<Inner>`, `SPPF`, `K_C3K2_INNER`, and `make_conv_node`, `make_bottleneck_node`, `make_c3k_node`, `make_c3k2_node`, `make_c3k2_of_bottleneck`, `make_c3k2_of_c3k`, `make_sppf_node` |
| U12 | `nn/node_registry.hpp`, `src/nn/node_registry.cpp`, `nn/layer_registry.hpp`, `examples/yolo26/model.json` | `K_NODE_REGISTRY` (7 rows), `make_node`, `build_model`, the `silu` row of `LAYER_REGISTRY`, the chain guard of `build_layer_stack`, the 24-row YOLO26n file |

Every layer type and block has the same members: its fields, `static fields()`, `forward` or
the pair `trace_training` and `trace_evaluation`, `Shape output_shape(const Shape&) const`, and
one `static_assert` on `Module`, `TraceModule` or `ModalModule`. Every node factory has the
`NodeFactory` signature, and a new type is one row of `K_NODE_REGISTRY` or `K_C3K2_INNER`; no
function switches over node kinds.

## 2. The decision the exporter run settled

The running variance moves toward the unbiased batch variance: the biased variance of the
graph times n/(n-1), where n is the number of values one channel's statistics are taken over.
`tests/fixtures/yolo26/MANIFEST.json` records it under `batchnorm.sacp`, from the seed 0 run of
Ultralytics commit `ce13994` on torch 2.14.1.

| Case | n | n/(n-1) | Largest difference, biased | Largest difference, unbiased | Tolerance |
| --- | --- | --- | --- | --- | --- |
| `unit`, `torch.nn.BatchNorm2d` over `[2, 3, 2, 2]` | 8 | 1.1428571428571428 | 0.0082741379737854 | 0.0 | 5.28635837326874e-07 |
| `model.9.cv1.bn` over `[1, 128, 20, 20]` | 400 | 1.0025062656641603 | 1.3887882232666016e-05 | 1.1920928955078125e-07 | 6.951018463041692e-07 |

Momentum is 0.03 and epsilon 0.001 in both cases. The manifest names `unbiased` as the matching
candidate of both. `BatchNorm2d::trace_training` therefore requests the variance update with
`source_scale` n/(n-1) and the mean update with 1.

## 3. Departures from the SPEC that stand

| SPEC | On disk | Reason |
| --- | --- | --- |
| 2.1 `Status bind_state_leaf(StateId, ValueId)` (SPEC line 110) | `void`, with `SA_THROW_IF` at four misuse sites | `graph/ir.hpp` predates this work and keeps its contract; the status conversion wave changes it |
| 2.2 `void Buffer::define(...)` | Returns `Status` | A second description that differs comes from the input a user traces the model over, so it is content. `Parameter::declare_described` throws for the same cause; it is in a file that predates this work and keeps that file's contract until its conversion |
| 2.2 a buffer draws its identity on first read | `define()` draws it | `Trace::update` never meets a buffer without one. A copy of a defined buffer draws its own; a moved-from buffer is undefined |
| 2.3 a layer form returns a value id and throws | `Trace::fail`, `Trace::status`, `Trace::value_or_fail` | A file created by this work does not throw. A refusing layer records the failure, returns its input, and the node that called it returns the status |
| 2.3 does not state it | `Trace` has no copy and no move; a `forward` that takes a `Trace` by value is not a `TraceModule` | A copy holds its own failure, which the caller of the original never reads |
| 2.4 `Node` has `forward`, `walk_state`, `ports` | Also `input_count()` | `ModelGraph::add` refuses a wrong source count, as 2.5 says |
| 2.4 nine rows in `K_NODE_REGISTRY` | Seven | `c2psa` and `detect` wait for wave 8 |
| 2.4 key spelling of a bare `batchnorm2d` row is not given | `bn_epsilon`, `bn_momentum` | The same two keys every block row reads |
| 2.5 `ModuleNode(L, Shape)`, `explicit Concat(ShapeSpan)` | `ModuleNode<L>::create`, `Concat::create`, both returning `Result` | Construction can fail on a shape |
| 2.5 `add` returns a `NodeRef` | `Result<NodeRef>` | The example needs `.value()` |
| 2.5 `add` fails when a layer does not fit its source | Not met for two layer types. `BatchNorm2d::output_shape` returns its input unchecked, so `add` accepts a 3-channel layer over a 4-channel source. `ModelGraph::output_shape` ignores its argument and returns an empty shape for a graph without exactly one output. Both refuse at trace, under the node's name. Every other layer type refuses at `add` | `output_shape` returns `Shape` in every layer type and a file created by this work does not throw, so these two have no way to refuse. Section 7 holds the two ways out; the owner has not ruled |
| 2.5 does not state it | `add` refuses a name with `.`; null node, no port and a repeated port name are fatal | A dotted name could collide with a child's state name |
| 2.6 does not state it | A `defaults` entry no row uses, an empty `port` or `name`, more than 6 extents and an overflowing element count are errors | Each would otherwise read as a default |
| 2.6 `SourceSpec::node` -1 is the previous row | -1 is `K_MODEL_INPUT`; the parser resolves the relative form | One meaning after the parse |
| 2.6 `build_model` and an empty `outputs` | Both builders publish the last row whole | The rule of a file without `outputs` |
| 2.7 `BatchNorm2d::declare_state` is a private member | A file-local function | It needs no access to private state |
| 2.7 body | Also refuses channels, epsilon, momentum and a non-float input | `GraphBuilder::batchnorm2d` has no float check |
| 2.10 `K_C3K2_INNER` has three rows (SPEC line 1314) | Two | `bottleneck_psa` waits for wave 8 (row 22) |
| 2.10 `Parameter::declare(init)` reports BIAS for a constant | `declare(role, init)`; `ParameterRole::SCALE` fills with one | See section 4, U14 |
| `Parameter::id()` marked `[[deprecated]]` | Not marked | Three frozen callers use it under `-Werror` |
| `SA_REFUSE_IF` in `graph/traced.hpp` | In `graph/builder.hpp` | U1's move; no other user |

`Detail::bridge_throwing` in `nn/throwing_bridge.hpp` holds the only `try` of the new files. It
is called where a function that returns `Status` or `Result` calls one that throws:
`ModuleNode::create` and `forward`, `LayerNode::forward`, `Concat::forward`, `make_node` for a
`LAYER_REGISTRY` factory, and six calls inside `batch_norm2d.cpp` (the two `declare` calls, the
two `Buffer::read` calls, `batchnorm2d`, and `batchnorm_fold` with `channel_affine`). When those
callees return a status, the bridge and its eleven calls are deleted.

`ModuleNode::forward` reads `trace.status()` before the bridge's result, so a failure a layer
recorded is returned with its own code when the layer, or a `Sequential` around it, throws
afterwards.

Two calls into throwing code sit outside the bridge, because the function around them returns
a `ValueId`: `Buffer::read` (`nn/buffer.hpp`) calls `GraphBuilder::buffer` and
`Graph::bind_state_leaf`, and `Parameter::declare_described` (`nn/module.hpp`) calls
`bind_state_leaf`. The conversion wave changes both. `checked_image` in `batch_norm2d.cpp`
reads `Graph::value(x)` only after a fatal check that the graph holds `x`.

## 4. What waits for the migration

| Unit | Change |
| --- | --- |
| U13 | `src/graph/lowering.cpp:1241` checks three `BATCHNORM2D` outputs and the builder emits four, so every batch-norm case of `tests/functional/integration/test_forward.cpp` and `test_autograd_numeric.cpp` fails at run time until the lowering writes the biased variance to output 3. `BATCHNORM_FOLD` has no lowering, so no evaluation graph of a model with a `BatchNorm2d` runs |
| U14 | `src/nn/init.cpp:100` zeroes a parameter whose role is BIAS and fan-scales every other, and `include/SushiAI/nn/init.hpp:178` walks parameters only. It must (1) fill from `parameter.init()`: a `Constant` kind with `init().constant`, a `FanScaled` kind as today, in place of the test on `role()`; (2) keep the search for the last WEIGHT as it is; (3) walk `for_each_buffer` and fill each buffer's tensor with `initial_value()`. Until then a gamma is fan-scaled where it should be one, `Conv2d::bias_init` is ignored, and no buffer is filled. `Unit_BatchNorm2d.TheScaleStartsAtOneAndTheShiftAtZero` and `Unit_Module.ARoleDeclareFillsByItsRole` pin what `init()` reports |
| U15 | `save_checkpoint` writes parameters only; it must write buffers under the names `walk_state` gives. `import_weights` reads `0.conv.weight` where the fixture has `model.0.conv.weight` |
| U16 | `Trainer::build_graph` calls `module.forward(builder, x)`, so a model that holds a `BatchNorm2d` does not instantiate. `TrainingStep` must not place one recorder in two lists of one tree, nor nest lists in a cycle: `StepRecorderList::add` checks direct members only. A compiled step must refuse a graph whose `has_refusal()` holds |
| U17 | Both applications move to `build_model`. A copied, untraced `Parameter` reports `declared()` and the original's `id()`; `Unit_Module.ACopiedLayerHasANewStateId` pins it and U17 decides whether it stays |

The status conversion of the files that predate this work is a separate wave. It removes
`bridge_throwing`, turns `output_shape` into `Result<Shape>` in every layer type, makes the layer
forms return `Graph::Traced` so that `Trace::fail`, `status` and `value_or_fail` go, and gives
`layer_registry.cpp` and `node_registry.cpp` one function for the text `layer <i> '<type>'`.

`src/nn/blocks/block_attributes.hpp` is read by `src/nn/batch_norm2d.cpp` as well as by the
blocks. It belongs one folder up, as `src/nn/row_attributes.hpp` and `.cpp`; the move needs one
line of `src/CMakeLists.txt`.

## 5. What wave 8 needs

Three rows of `examples/yolo26/model.json` fail by design, each with a status that names it:

| Row | Row text | Fails with |
| --- | --- | --- |
| 10 | `c2psa` | `layer 10 'c2psa'` has a type this build does not have |
| 22 | `c3k2` with `"inner": "bottleneck_psa"` | `'bottleneck_psa'` names no inner block |
| 23 | `detect` | `layer 23 'detect'` has a type this build does not have |

`build_model` stops at row 10, so rows 11 to 21 build only with a stand-in there;
`tests/functional/unit/test_node_registry.cpp` uses a `silu` row.

SPEC 2.10 names four emitters the blocks of U18 need. Three are missing: a depthwise
convolution (`ConvGeometry` has no group count), a rank-3 matrix product (`MATMUL` takes rank
2) and a differentiable softmax (the builder has `log_softmax`). The fourth exists: the builder
emits a stop-gradient today (`GraphBuilder::stop_gradient`, `src/graph/builder.cpp:1126`;
`Ops::STOP_GRADIENT`, `include/SushiAI/core/ops.hpp:199`). Whether the frozen
`src/graph/lowering.cpp` lowers it was not checked.

## 6. Lines of the SPEC that are wrong against the reference

The reference is Ultralytics commit `ce13994`.

`cfg/models/26/yolo26.yaml`:

```
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 2, C3k2, [512, False, 0.25]]
```

`nn/tasks.py`, `parse_model`:

```
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
```

`nn/modules/block.py`, `C3k2.__init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1,
shortcut=True)`:

```
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            ...
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
```

`Bottleneck.__init__`: `self.add = shortcut and c1 == c2`.

Rows 2 and 4 therefore call `C3k2(c1, c2, 1, False, 0.25)`: the third yaml argument is `e`, the
second is `c3k`, and `shortcut` keeps its default `True`. The inner block is
`Bottleneck(c, c, True, 1)` with `c1 == c2`, so it adds its input. With `c3k` true the inner
block is `C3k(c, c, 2, True, 1)`, whose two bottlenecks add as well.

| SPEC line | Text | Should read |
| --- | --- | --- |
| 619 | row 2, `"shortcut": false` | `"shortcut": true` |
| 621 | row 4, `"shortcut": false` | `"shortcut": true` |
| 1311 | `inner.push_back(NN::Bottleneck(16, 16, 8, false, norm));` | `NN::Bottleneck(16, 16, 8, true, norm)` |

`examples/yolo26/model.json`, the `c3k2` factory default and
`Unit_NodeRegistry.Yolo26RowsTwoAndFourAddTheirBottlenecksInput` follow the reference.

Three more passages of the SPEC are stale against the code or the sources:

| SPEC line | Text | Should read |
| --- | --- | --- |
| 1314 | `make_c3k2_node` looks `"inner"` up in "a constant table of three rows" | Two rows until wave 8 adds `bottleneck_psa` |
| 1389-1390 | "a stop-gradient (no `STOP_GRADIENT` in `ops.hpp`)"; line 1749 says "the four emitters" | `STOP_GRADIENT` is at `ops.hpp:199` and the builder emits it; three emitters are missing |
| 1922-1927 | `shortcut`, `pools`, row 9's unactivated `cv1` and its final add come from sources "which were not opened" | Read from `nn/modules/block.py` at `ce13994`, where SPPF has `Conv(c1, c_, 1, 1, act=False)`, `self.add = shortcut and c1 == c2` and `y + x`. No traced forward has been compared with `forward.sacp` |

## 7. Open outside this unit's files

| Item | File that changes | Who holds it |
| --- | --- | --- |
| `nn/throwing_bridge.hpp` and `test_throwing_bridge.cpp` are in no build list, and the configure check stops on a header the umbrella does not name | `include/SushiAI/SushiAI.h`: `#include "nn/throwing_bridge.hpp"`. `tests/CMakeLists.txt`: `functional/unit/test_throwing_bridge.cpp` | Orchestrator |
| `add` does not refuse a `BatchNorm2d` or a nested `ModelGraph` that does not fit (section 3, row 2.5). Either `ModuleNode<L>::create` calls an optional `Status fits(const Shape&) const` that these two provide, or both refuse at trace until `output_shape` returns `Result<Shape>` | `nn/module_node.hpp`, `nn/trace.hpp`, `nn/batch_norm2d.hpp` and `nn/model_graph.hpp`, or none | Owner's ruling |
| No entry for the cases this work added or renamed, among them the suites `Unit_TraceDeathTest`, `Unit_BatchNorm2dDeathTest` and `Unit_ModelOutputsDeathTest`. The heading `BatchNorm2dKeepsItsShapeAndDeclaresTwoPerChannelStatistics` at line 6770 names a case that is gone | `tests/functional/unit/README.md` | The documentation session |
| `walk`, `record`, `first_node`, `count_nodes`, `one_input`, `attributes_of` and the constant `K_YOLO_NORM` are repeated in the five `test_block_*.cpp` files | A new `tests/common/state_records.hpp` | Orchestrator; `tests/common/` is outside this unit |
| The files created by this work have no licence box | `tools/licensing/write_license_block.py` writes it | Orchestrator |
| `C3k2`'s constructor writes the literal `1, 1` for the kernel and the stride of `cv1` and `cv2` (`nn/blocks/c3k2.hpp`). `K_POINTWISE` is private to `src/nn/blocks/block_attributes.hpp`, which a public header cannot include | A public home for `K_POINTWISE`, decided with the move of `block_attributes.hpp` in section 4 | Owner; a public constant is interface |
| `tools/naming/rename_map.tsv` has rows for `ParameterRole::WEIGHT` and `BIAS` (lines 205 and 206) and none for `ParameterRole::SCALE`, so the enumerator run would leave `Weight`, `Bias`, `SCALE` | `tools/naming/build_rename_map.py` is rerun before the enumerator run | The naming wave |
