# graph

## Owns

The intermediate representation of a computation and the passes that act on it. `Graph` holds
`Value`s and `Node`s, addressed by `ValueId` and `NodeId`, and `GraphBuilder` emits
shape-checked nodes into one. The module also owns the passes over a graph (cast insertion,
static subtree fusion, the dynamic fusion selector, precision policies and im2col
rematerialisation), the memory plan with its `ArenaSet`, and the `TensorPool` that holds the
tensors behind the values. Lowering turns a graph into SushiBLAS tasks through the tables under
`lowering/`, and `CompiledStep` compiles and replays one step. `StateId` names model state
across graphs, and `Traced` carries a traced value or the failure that stopped the trace.

## Depends on

The module declares `core` as a public dependency. It declares no private dependency.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/graph/arena.hpp` | `ArenaSet` |
| `SushiAI/graph/builder.hpp` | `ConvGeometry`, `output_extent`, `SpatialExtents`, `PoolResult`, `BatchNormResult`, `BatchNormFold`, `BatchNormGrads`, `GraphBuilder` |
| `SushiAI/graph/cast_insertion.hpp` | `CastOptions`, `InsertedCast`, `CastReport`, `insert_casts`, `DTypeMismatch`, `find_dtype_mismatches`, `describe` |
| `SushiAI/graph/compiled_step.hpp` | `CompiledStep`, with its `compile` and `run` |
| `SushiAI/graph/fusion.hpp` | `FusedForm`, `fused_form_name`, `fused_form_from_name`, `fused_form_op`, `fused_form_of`, `FusionDecline`, `FusionOptions`, `FusedSubtree`, `DeclinedFusion`, `FusionReport`, `fuse` |
| `SushiAI/graph/fusion_selector.hpp` | `FusionQuery`, `FusionChoice`, `FusionSelector`, `DeviceFusionSelector`, `FusionSelection` |
| `SushiAI/graph/ir.hpp` | `ValueId`, `NodeId`, `Lifetime`, `lifetime_name`, `is_persistent`, `Attributes`, `TensorLayout`, `Value`, `Node`, `Graph` |
| `SushiAI/graph/lowering.hpp` | `Lowering`, with its `record` |
| `SushiAI/graph/memory_plan.hpp` | `ArenaId`, `K_NO_ARENA`, `Arena`, `Placement`, `PlanOptions`, `MemoryPlan` |
| `SushiAI/graph/precision.hpp` | `PrecisionQuery`, `PrecisionChoice`, `PrecisionPolicy`, `UniformPrecisionPolicy`, `MixedPrecisionPolicy`, `narrows_under_mixed_precision` |
| `SushiAI/graph/rematerialise.hpp` | `rematerialise_im2col` |
| `SushiAI/graph/state_id.hpp` | `StateId`, `K_NO_STATE`, `StateIdentity` |
| `SushiAI/graph/step_recorder.hpp` | `StepRecorder`, `StepRecorderList` |
| `SushiAI/graph/tensor_pool.hpp` | `TensorPool` |
| `SushiAI/graph/traced.hpp` | `Traced` |
| `SushiAI/graph/lowering/activation.hpp` | `K_ACTIVATION_LOWERINGS` |
| `SushiAI/graph/lowering/all.hpp` | `K_LOWERINGS` |
| `SushiAI/graph/lowering/context.hpp` | `LoweringContext` |
| `SushiAI/graph/lowering/elementwise.hpp` | `K_ELEMENTWISE_LOWERINGS` |
| `SushiAI/graph/lowering/fused.hpp` | `K_FUSED_LOWERINGS` |
| `SushiAI/graph/lowering/linalg.hpp` | `K_LINALG_LOWERINGS` |
| `SushiAI/graph/lowering/loss.hpp` | `K_LOSS_LOWERINGS` |
| `SushiAI/graph/lowering/norm.hpp` | `K_NORM_LOWERINGS` |
| `SushiAI/graph/lowering/rule.hpp` | `LoweringRule`, `LoweringEntry`, `concat_lowerings` |
| `SushiAI/graph/lowering/shape.hpp` | `K_SHAPE_LOWERINGS` |
| `SushiAI/graph/lowering/spatial.hpp` | `K_SPATIAL_LOWERINGS` |
| `SushiAI/graph/lowering/unlowered.hpp` | `K_UNLOWERED` |
