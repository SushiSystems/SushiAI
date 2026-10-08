# Model graph: plan and state of the work

**Status:** In progress. U1 to U13 are built and pass; U14 to U17 are planned, not written.

The design is `SPEC.md` in this folder. This file is the plan of the four engine units that
waited for the SushiBLAS migration. It was rewritten on 2026-10-09 against HEAD `d6bdf7b`, which
builds and passes 1803 tests against SushiBLAS `7f04834`
(`docs/agent/2026_10_08_SUSHIBLAS_MIGRATION/REPORT.md:979-990`). An implementer needs this file
and the files a task names. Nothing here has been built or run. Section 12 lists what was not
verified.

## 1. Done: what U1 to U13 put on disk

| Unit | Holds | Where |
| --- | --- | --- |
| U1 | `Graph::StateId`, `StateIdentity`, the state-leaf table, `without_nodes`, the refusal flag, the buffer lifetime, the fourth `BATCHNORM2D` output, `BATCHNORM_FOLD` | `graph/state_id.hpp`, `graph/ir.hpp:381-458`, `graph/builder.hpp`, `core/ops.hpp` |
| U2 | `Buffer`, `ModuleList<T>`, `ParameterInit`, `ParameterRole::Scale`, `walk_state`, `for_each_buffer` | `nn/buffer.hpp`, `nn/module_list.hpp`, `nn/module.hpp` |
| U3 | Model file version 2, `K_MODEL_VERSION`, `SourceSpec`, `OutputSpec`, `ModelSpec::input_shape` | `config/model_config.hpp:44-105` |
| U4 | `StepRecorderList`; `add` returns `Status` | `graph/step_recorder.hpp:79-147` |
| U5 | `ModelOutputs` | `nn/model_outputs.hpp:47-80` |
| U6 | `Trace`, `BufferUpdate`, `BufferUpdateList`, `discard_updates`, the three layer forms, `trace_layer` | `nn/trace.hpp` |
| U7 | `BatchNorm2d`, `NormOptions`, `updated_statistic` | `nn/batch_norm2d.hpp` |
| U8 | `Node`, `Port`, `NodeFactory`, `ModuleNode`, `LayerNode`, `Concat`, `SiLU`, `Identity`, `Detail::bridge_throwing` | `nn/node.hpp`, `nn/throwing_bridge.hpp` and siblings |
| U9 | `Conv2d::same_padded`, `Conv2d::bias_init` | `nn/conv2d.hpp` |
| U10 | `NodeRef`, `ModelGraph` | `nn/model_graph.hpp:58-298` |
| U11 | `ConvNorm`, `Bottleneck`, `C3k`, `C3k2`, `SPPF` and their factories | `nn/blocks/` |
| U12 | `K_NODE_REGISTRY` (7 rows), `make_node`, `build_model`, the 24-row YOLO26n file | `nn/node_registry.hpp:51-81`, `examples/yolo26/model.json` |
| U13 | The lowering of four `BATCHNORM2D` outputs and of `BATCHNORM_FOLD` | `src/graph/lowering/norm.cpp` |
| U14, items 1 and 2 | `apply_initializer` fills from `parameter.init()` before it tests the final layer | `src/nn/init.cpp:131-157` |

The running variance moves toward the unbiased batch variance, n/(n-1) times the graph's biased
one; `tests/fixtures/yolo26/MANIFEST.json` records the exporter run that settled it.

## 2. Done: departures from the SPEC that stand

| SPEC | On disk |
| --- | --- |
| 2.2 `void Buffer::define` | Returns `Status` and draws the identity (`nn/buffer.hpp:118-139`) |
| 2.3 a layer form throws | `Trace::fail`, `status`, `value_or_fail`; a `Trace` has no copy and no move (`nn/trace.hpp:141-172`) |
| 2.4 nine registry rows, three `K_C3K2_INNER` rows | Seven and two; `c2psa`, `detect` and `bottleneck_psa` are U18 |
| 2.5 constructors, `add` returning `NodeRef` | `ModuleNode<L>::create`, `Concat::create`, `Result<NodeRef>` (`nn/model_graph.hpp:157-167`) |
| 2.5 `add` refuses a layer that does not fit | `BatchNorm2d` and a nested `ModelGraph` refuse at trace, not at `add` |
| 2.6 `SourceSpec::node` -1 is the previous row | -1 is `K_MODEL_INPUT`; the parser resolves the relative form |
| Rows 2 and 4 of the model file, `"shortcut": false` | `true`, as Ultralytics `ce13994` has it |

## 3. Done: what the migration changed under this plan

- `TensorPool::bind`, `alias`, `request_constant` and `StepRecorderList::add` return `Status`
  (`graph/tensor_pool.hpp:102`, `:147`, `:175`; `graph/step_recorder.hpp:89`).
- A step recorder records through `Result<std::size_t> record_into_step(SushiBLAS::TaskRecorder&)`
  (`graph/step_recorder.hpp:60-61`). `CompiledStep::compile` refuses a graph whose `has_refusal()`
  holds (`src/graph/compiled_step.cpp:51`), which closes that item of the old U16 row.
- `Trainer::make`, `run_step`, `fit`, `bind_parameters`, `initialize`, `save_checkpoint` and
  `load_checkpoint` return `Status` or `Result` (`train/trainer.hpp:225`, `:337-362`;
  `nn/init.hpp:179`; `io/checkpoint.hpp:115`, `:154`).
- Still throwing, and left so by this plan: `GraphBuilder`, the layer forms, `Parameter::declare`
  (`nn/module.hpp:217`), `Graph::bind_state_leaf` (`graph/ir.hpp:397-421`), `Objective` and
  `ObjectiveLeaves` (`src/train/objective.cpp:40`, `:66`, `:86`), `Config::load_model`,
  `make_objective`, `build_layer_stack`, `Shape`'s constructors (`core/shape.hpp:56`, `:73`).
  `SushiAI::Detail::guard` stops the process on a `SushiAI::Error` (`src/core/seam.cpp:105`), so
  a status-returning function reaches throwing code through `NN::Detail::bridge_throwing`
  (`nn/throwing_bridge.hpp:51-78`) and through nothing else.

## 4. The contract every task below follows

1. A content failure is a `Status` or a `Result<T>`; a misuse is `SA_FATAL_IF`. A new file holds
   no `try`, `catch` or `throw`. Propagate with `SA_TRY`, `SA_TRY_ASSIGN`, `SA_FAIL_IF` and
   `SA_ANNOTATE` (`core/status.hpp:249-277`, `core/result.hpp:232`).
2. A host allocation that can fail runs inside
   `SushiAI::Detail::guard(SeamPhase::Allocate, ...)` or `guard_value`; a recorded BLAS call
   inside `guard(SeamPhase::Record, ...)`; recorded work is run with `Detail::run_recorded`
   (`core/seam.hpp:67-79`). `nn/init.hpp:183-211` shows all three.
3. A call into code of section 3's last item runs inside `NN::Detail::bridge_throwing`, whose
   callable returns a value and not `void`.
4. Status codes: `InvalidDescription` for a wrong model or document, `Incompatible` when two
   valid things disagree, `FileNotFound`, `InvalidFile`, `OutOfMemory`
   (`core/status.hpp:41-51`).
5. Every status-returning function is `[[nodiscard]]`. A comment block is at most eight lines, a
   line at most 100 columns, a `//` comment one line that states an invariant. Every new file
   opens with the licence box and the `@file` block of `nn/model_outputs.hpp`.
6. Tests use `SA_EXPECT_OK`, `SA_ASSERT_OK_AND_ASSIGN`, `SA_EXPECT_FAILURE(expr, code, text...)`
   and `SA_EXPECT_FATAL` (`tests/common/test_status.hpp:124-159`), and the fixture
   `SaTest::EngineTest` (`tests/common/test_utils.hpp:46`) when they need a device.
7. No task edits a `CMakeLists.txt`, `include/SushiAI/SushiAI.h` or a README. Section 10 lists
   what the orchestrator adds.

The order a driver follows after this plan, which every unit below serves:

```cpp
SA_TRY_ASSIGN(NN::ModelGraph model, NN::build_model(model_spec));
SA_TRY_ASSIGN(Train::TrainingGraph built,
              Train::build_training_graph(model, *objective, batch_shape));
SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));   // after a trace
Graph::TensorPool pool(engine);
SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
              Train::bind_parameters(built, state, pool));
SA_TRY(NN::initialize(engine, state));
// make the optimizer from slots
SA_TRY_ASSIGN(std::vector<Train::BufferUpdateSlot> updates,
              Train::bind_buffer_updates(built.updates, state, pool));
SA_TRY_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
              Train::Trainer::make(engine, built.graph, pool, built.spec, *optimizer, options,
                                   std::move(updates)));
```

`ModelState::create` runs after the first trace because the walk skips a parameter that is not
declared and a buffer that is not defined (`nn/module.hpp:587-588`, `:599-600`), and a trace is
what declares them.

## 5. U14: `ModelState`, and `initialize` over it

### T14.1 `NN::ModelState`

Files: `include/SushiAI/nn/model_state.hpp`, `src/nn/model_state.cpp`,
`tests/functional/integration/test_model_state.cpp` (all new).

```cpp
namespace SushiAI::NN
{
    /** @brief Names what an entry of a ModelState is. */
    enum class StateKind : std::uint8_t { Parameter, Buffer };
    /** @brief Describes one tensor a ModelState owns. */
    struct StateEntry
    {
        std::string name;
        StateKind kind = StateKind::Parameter;
        Graph::StateId state = Graph::K_NO_STATE;
        Shape shape;
        DType dtype = DType::Float32;
        ParameterInit init;      // a buffer holds constant_value(initial_value())
        int64_t fan_in = 0;
        int64_t fan_out = 0;
    };
    /** @brief Owns one tensor per declared parameter and defined buffer of a model. */
    class ModelState final
    {
        public:
            template<typename M>
            [[nodiscard]] static Result<ModelState> create(SushiBLAS::Engine& engine, M& model,
                                                           std::string_view prefix = {});
            [[nodiscard]] static Result<ModelState> from_entries(SushiBLAS::Engine& engine,
                                                                 std::vector<StateEntry> entries);
            ModelState(ModelState&&) noexcept;
            ModelState& operator=(ModelState&&) noexcept;
            ModelState(const ModelState&) = delete;
            ModelState& operator=(const ModelState&) = delete;
            ~ModelState();
            [[nodiscard]] Status attach(const Graph::Graph& graph, Graph::TensorPool& pool) const;
            [[nodiscard]] SushiRuntime::span<const StateEntry> entries() const noexcept;
            [[nodiscard]] Result<SushiBLAS::Tensor*> tensor(std::string_view name);
            [[nodiscard]] Result<SushiBLAS::Tensor*> tensor(Graph::StateId state);
    };
}
```

Behaviour:

- `create` walks `model` once with a `StateVisitor` over two lambdas (`nn/module.hpp:500-548`),
  inside `guard(SeamPhase::Allocate, ...)`, collects one `StateEntry` per visit in walk order,
  and returns `from_entries`. A parameter entry copies `shape()`, `dtype()`, `init()`,
  `fan_in()`, `fan_out()` and `state()` (`nn/module.hpp:178-202`); a buffer entry copies
  `shape()`, `dtype()`, `state()` and `ParameterInit::constant_value(initial_value())`
  (`nn/buffer.hpp:170-197`). The template is the only part in the header.
- `from_entries` fails with `InvalidDescription` and the name when two entries share a name or
  two share a `StateId`; an entry whose state is `K_NO_STATE` is fatal. It allocates each
  tensor with `allocate_tensor(engine, shape, dtype)` (`core/allocation.hpp:60-62`) and builds
  two hashed indexes, name to position and `StateId` to position. A model with no entry gives a
  valid empty state.
- `attach` reads `graph.state_leaves()` (`graph/ir.hpp:424-428`). For each pair it finds the
  entry by `StateId`, fails with `Incompatible` when there is none or when the leaf's shape or
  dtype in `graph.value(leaf)` differs from the entry's, and calls `pool.bind(leaf, tensor)`
  with a copy of the tensor, which shares storage. Precondition, stated in the header: no leaf
  of `graph` is bound in `pool` yet; `bind` stops the process otherwise
  (`src/graph/tensor_pool.cpp:89-90`).
- Both `tensor` overloads fail with `InvalidDescription`; the name form lists nothing, it names
  the name asked for and the entry count.

Consumes: `walk_state`, `allocate_tensor`, `TensorPool::bind`. Produces: the owner of every
state tensor, read by T14.2, T15.1, T15.2, T16.3, T16.4 and T16.6.

Tests, suite `Integration_ModelState`:

| Case | Pins |
| --- | --- |
| `CreateListsParametersInWalkOrder` | A traced `Sequential<Linear, GELU, Linear>` gives `0.weight`, `0.bias`, `2.weight`, `2.bias`, all `Parameter`, with the layers' shapes |
| `CreateListsBuffersAfterATrainingTrace` | A `ModelGraph` with one `BatchNorm2d`, traced with `Trace::training`, gives `weight`, `bias`, `running_mean`, `running_var` under the row's name, the last two of kind `Buffer` with constant inits 0 and 1 |
| `AnUntracedModelGivesAnEmptyState` | `entries()` is empty and `create` succeeds |
| `APrefixIsJoinedToEveryName` | `create(engine, model, "model")` gives `model.0.weight` |
| `TwoEntriesWithOneNameAreRefused` | `from_entries` returns `InvalidDescription` naming the name |
| `TensorByNameAndByStateAreTheSameStorage` | Both lookups return the same pointer |
| `AnUnknownNameIsRefusedAndNamed` | `InvalidDescription` with the name |
| `AttachBindsEveryStateLeafToTheStatesStorage` | After `attach`, `pool.tensor(leaf).data()` equals the state tensor's `data()` for every leaf |
| `AttachRefusesAGraphOfAnotherModel` | `Incompatible` |
| `TwoPoolsAttachedToOneStateShareStorage` | A host write through one pool's tensor is read through the other's |

Acceptance: the three files pass the single-file syntax check; no `try`, `catch` or `throw`;
every case above exists under that name. Dispatch row: spec-complete one or two files.

### T14.2 `initialize` over a state

Files: `include/SushiAI/nn/init.hpp`, `src/nn/init.cpp`,
`tests/functional/integration/test_init_state.cpp` (new). After T14.1.

```cpp
namespace SushiAI::NN
{
    namespace Detail
    {
        [[nodiscard]] Status apply_initializer(SushiBLAS::Engine& engine,
                                               SushiBLAS::Tensor& tensor, ParameterInit init,
                                               int64_t fan_in, int64_t fan_out,
                                               const InitPolicy& policy, bool is_final_layer);
    }
    /** @brief Records the initial fill of every entry of state and runs the recording. */
    [[nodiscard]] Status initialize(SushiBLAS::Engine& engine, ModelState& state,
                                    const InitPolicy& policy = {});
}
```

- `apply_initializer` takes the three numbers in place of `const Parameter&`
  (`nn/init.hpp:164-168`, `src/nn/init.cpp:131-157`); its body is unchanged. The module form of
  `initialize` (`nn/init.hpp:178-212`) passes `parameters[i]->init()`, `->fan_in()` and
  `->fan_out()` and is otherwise untouched; T17.6 deletes it.
- The state form is not a template and lives in `init.cpp`. It seeds as `init.hpp:183-188`
  does, finds the last entry whose `init.kind` is `FanScaled`, calls `apply_initializer` for
  every entry in order with `is_final = policy.zero_final_layer && i == final_weight`
  (SPEC 3.6, lines 1652-1665), and ends with `Detail::run_recorded(engine)`. A buffer is a
  `Constant` entry, so it is filled by the same call; this is item 3 of the old U14 row.
- `InitPolicy::zero_final_layer`'s comment (`nn/init.hpp:149`) becomes "Zero the last
  fan-scaled entry".

Tests, suite `Integration_InitState`:

| Case | Pins |
| --- | --- |
| `AScaleIsOneAShiftIsZeroAndABufferHoldsItsInitialValue` | Over the `BatchNorm2d` model of T14.1: gamma all 1, beta all 0, `running_mean` all 0, `running_var` all 1 |
| `ZeroFinalLayerZeroesOnlyTheLastFanScaledEntry` | Convolution then `BatchNorm2d`: the convolution weight is 0, gamma is 1 |
| `WithoutZeroFinalLayerTheLastWeightIsNotZero` | At least one element differs from 0 |
| `TheSameSeedGivesTheSameBits` | Two states of two equal models, one seed, equal bit patterns |
| `TheStateFormMatchesTheModuleFormBitForBit` | `Sequential<Linear, GELU, Linear>`, `zero_final_layer` false, one seed: the pool-owned tensors of the module form equal the state's. T17.6 deletes this case with the module form |

Acceptance: syntax check of the three files; the five cases exist. Dispatch row: spec-complete
one or two files.

## 6. U15: checkpoints over `ModelState`, and `import_weights`

### T15.1 `import_weights`, the strict loader

Files: `include/SushiAI/io/import_weights.hpp`, `src/io/import_weights.cpp`,
`tests/functional/integration/test_import_weights.cpp` (all new). After T14.1.

```cpp
namespace SushiAI::IO
{
    /** @brief Rewrites a record name that starts with from_prefix. */
    struct RenameRule
    {
        std::string_view from_prefix;
        std::string_view to_prefix;
    };
    /** @brief Returns name under the first rule whose from_prefix it starts with, else name. */
    [[nodiscard]] std::string renamed(std::string_view name,
                                      SushiRuntime::span<const RenameRule> rules);
    /** @brief Loads an SACP file into state after renaming each record; strict both ways. */
    [[nodiscard]] Status import_weights(const std::filesystem::path& path, NN::ModelState& state,
                                        SushiRuntime::span<const RenameRule> rules);
}
```

`import_weights` is the body of today's `load_checkpoint` (`io/checkpoint.hpp:163-218`) with
three changes: the record's tensor is `state.tensor(renamed(record.name, rules))` (SPEC 3.7), a
hashed set of renamed names replaces `filled`, and the closing loop walks `state.entries()`.
The header includes `io/checkpoint_format.hpp` and `nn/model_state.hpp`, and not
`io/checkpoint.hpp`. Failures, all with `path.string()` first:

| Cause | Code | Text holds |
| --- | --- | --- |
| The file does not open | `FileNotFound` | `could not open '{}' for reading` |
| A record no entry matches | `Incompatible` | The record's name in the file and its renamed form |
| A renamed name seen twice | `Incompatible` | `supplies '{}' twice` |
| A dtype or a shape differs | `Incompatible` | Both dtypes, or both shapes |
| An entry no record supplied | `Incompatible` | `does not supply this model's '{}'` |

Tests: suite `Unit_RenameRule` (`RewritesAMatchingPrefix`, `LeavesANameNoRuleMatches`,
`TheFirstMatchingRuleWins`, `AnEmptyToPrefixStripsThePrefix`) and suite
`Integration_ImportWeights`:

| Case | Pins |
| --- | --- |
| `ARuleStripsAForeignPrefixAndEveryRecordLands` | A file written with `CheckpointWriter` under `model.` names loads with `{"model.", ""}`, bit for bit |
| `WithoutTheRuleTheFirstRecordIsRefused` | `Incompatible`, naming `model.0.weight` |
| `ARecordTheStateLacksIsRefusedUnderBothNames` | The text holds the file's name and the renamed one |
| `AnEntryTheFileLacksIsRefusedAndNamed` | `Incompatible` |
| `AShapeMismatchIsRefusedWithBothShapes` | `Incompatible` |
| `AMissingFileIsFileNotFound` | The code |

The fixture `tests/fixtures/yolo26/weights.sacp` is not loaded here: `build_model` stops at row
10 of the YOLO26n file until U18, and the load is strict.

Acceptance: syntax check of the three files; the ten cases exist. Dispatch row: spec-complete
one or two files.

### T15.2 `save_checkpoint` and `load_checkpoint` over a state

Files: `include/SushiAI/io/checkpoint.hpp`, `src/io/checkpoint.cpp` (new),
`tests/functional/integration/test_state_checkpoint.cpp` (new). After T14.1; written beside
T15.1 from the signature above.

```cpp
namespace SushiAI::IO
{
    /** @brief Writes every entry of state in walk order, parameters and buffers alike. */
    [[nodiscard]] Status save_checkpoint(const std::filesystem::path& path,
                                         NN::ModelState& state, SushiBLAS::Engine& engine);
    /** @brief Loads a checkpoint into state; strict both ways, and needs no graph or pool. */
    [[nodiscard]] Status load_checkpoint(const std::filesystem::path& path,
                                         NN::ModelState& state);
}
```

`save_checkpoint` follows `io/checkpoint.hpp:119-142`: `run_recorded`, open, one record per
entry through `Detail::record_for` and `CheckpointWriter::write`, `finish`, the two
`InvalidFile` checks. `load_checkpoint` returns `import_weights(path, state, {})`. The two
module templates and `Detail::bind` (`io/checkpoint.hpp:57-86`, `:114-219`) stay until T17.6.
SACP stays at version 1.

Tests, suite `Integration_StateCheckpoint`:

| Case | Pins |
| --- | --- |
| `SavesAndLoadsParametersAndBuffersBitForBit` | The `BatchNorm2d` model, statistics set to distinct values first |
| `TheFileListsEveryEntryInWalkOrder` | `CheckpointReader` yields the names of `entries()`, in order |
| `ACheckpointWithoutBuffersIsRefusedAndNamesTheRunningMean` | A file with the two parameters only: `Incompatible`, the text holds `running_mean` |
| `AModuleFormCheckpointLoadsIntoAState` | A file the module form wrote for the MLP loads into that model's state, bit for bit. T17.6 deletes this case |
| `AnUnwritablePathIsInvalidFile` | The code |

Acceptance: syntax check of the three files; the five cases exist. Dispatch row: spec-complete
one or two files.

## 7. U16: the objective, the step builders, the steps, the trainer's feed

Wave 3 writes library files only, and the tree does not parse as a whole until wave 3 is
complete: `trainer.hpp` loses what `training_graph.hpp` and `training_step.hpp` gain.

### T16.1 The objective over `ModelOutputs`

Files: `include/SushiAI/train/objective.hpp`, `src/train/objective.cpp`,
`tests/functional/unit/test_objective.cpp`.

```cpp
virtual Graph::ValueId loss(const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                            const NN::ModelOutputs& outputs,
                            const ObjectiveLeaves& leaves) const = 0;
```

Only the third parameter changes (`train/objective.hpp:138-141`, `:180-183`); the header
includes `nn/model_outputs.hpp`. `CrossEntropyObjective::loss` takes `outputs.only()` and
throws its message through `SA_THROW_IF` when it failed, then runs today's body
(`src/train/objective.cpp:84-90`). The contract stays throwing, as SPEC decision 1a rules for a
contract older than this work. The class comment `@class IObjective` (`objective.hpp:114`)
becomes `@class Objective`.

Tests in `test_objective.cpp`: every `build_graph` call becomes `build_training_graph(model,
objective, Shape{batch, features})` with `SA_ASSERT_OK_AND_ASSIGN`, and every `SA_EXPECT_ERROR`
around one becomes `SA_EXPECT_FAILURE(..., StatusCode::InvalidDescription, <same text>)`. New
cases: `Unit_Objective.CrossEntropyReadsTheOnlyOutput` (the loss node reads the id published
as `output`) and `Unit_Objective.CrossEntropyRefusesTwoOutputs` (`InvalidDescription`, the text
holds both names).

Acceptance: syntax check after wave 3. Dispatch row: spec-complete one or two files.

### T16.2 The traced graphs and their three builders

Files: `include/SushiAI/train/training_graph.hpp`, `include/SushiAI/train/steps.hpp`,
`src/train/steps.cpp`, `tests/functional/unit/test_steps.cpp` (all new).

`training_graph.hpp` takes `StepSpec`, `ParameterBinding` and `TrainingGraph` from
`train/trainer.hpp:61-126` as they are, and `TrainingGraph` gains three members:

```cpp
NN::ModelOutputs outputs;                  // every value the model published
NN::BufferUpdateList updates;              // what the training trace requested
std::vector<Graph::ValueId> pinned;        // values the planner must not pack
```

`steps.hpp`:

```cpp
namespace SushiAI::Train
{
    /** @brief States how a step's graph is built. */
    struct GraphOptions
    {
        DType dtype = DType::Float32;
        Graph::FusionOptions fusion{};
        const Graph::PrecisionPolicy* precision = nullptr;
        bool loss_scaling = false;          // read by build_training_graph only
    };
    /** @brief Holds one forward-only graph and what a step reads from it. */
    struct TracedStep
    {
        Graph::Graph graph;
        Graph::ValueId input = Graph::K_NO_VALUE;
        NN::ModelOutputs outputs;
        Graph::CastReport casts;
        Graph::FusionReport fusion;
    };
    /** @brief Returns [batch, sample...]. */
    [[nodiscard]] Result<Shape> batch_shape(int64_t batch, const Shape& sample);
    /** @brief Appends the source of every update to both lists. */
    [[nodiscard]] Status retain(const NN::BufferUpdateList& updates,
                                std::vector<Graph::ValueId>& observed,
                                std::vector<Graph::ValueId>& pinned);
    template<typename M>
    [[nodiscard]] Result<TrainingGraph> build_training_graph(M& model,
                                                             const Objective& objective,
                                                             const Shape& batch_shape,
                                                             GraphOptions options = {});
    template<typename M>
    [[nodiscard]] Result<TracedStep> build_probe_graph(M& model, const Shape& batch_shape,
                                                       GraphOptions options = {});
    template<typename M>
    [[nodiscard]] Result<TracedStep> build_evaluation_graph(M& model, const Shape& batch_shape,
                                                            GraphOptions options = {});
    template<typename M>
    [[nodiscard]] Result<TrainingGraph> build_classifier(
        M& module, int64_t batch, int64_t features, int64_t classes,
        DType dtype = DType::Float32, Graph::FusionOptions fusion = {},
        const Graph::PrecisionPolicy* precision = nullptr, bool loss_scaling = false);
}
```

Behaviour:

- One function template `Detail::trace_outputs(M& model, const NN::Trace& trace,
  Graph::ValueId input) -> Result<NN::ModelOutputs>` serves all three builders. For a type with
  `trace(const Trace&, ValueId)` returning `Result<ModelOutputs>` (`nn/model_graph.hpp:196`) it
  returns that call; for any other `Traceable` it calls `NN::trace_layer`
  (`nn/trace.hpp:330-372`) inside `bridge_throwing`, returns `trace.status()` when that failed,
  and publishes the one id under `NN::K_SINGLE_PORT_NAME` (`nn/node.hpp:60`).
- `build_training_graph` keeps the order of `build_graph` (`train/trainer.hpp:144-194`): input
  leaf over `batch_shape`, `objective.declare`, the model under `Trace::training(builder,
  built.updates)`, `objective.loss` over `built.outputs`, `spec.targets` when the leaf exists,
  `spec.logits` from `outputs.only()` when the model publishes one value and `K_NO_VALUE`
  otherwise, the parameter list from `parameter.id_in(built.graph)`, the loss-scale leaf,
  `differentiate`, the observed list, `insert_casts`, `fuse`. The observed list gains every
  published output and, through `retain`, every update source; `retain` also fills
  `built.pinned`. A declared parameter with no leaf in the graph fails with
  `InvalidDescription` naming it. Everything from the parameter walk on is one function in
  `steps.cpp`, so the template holds the trace alone; each call into throwing code is bridged.
- `build_probe_graph` traces under `Trace::training(builder, NN::discard_updates())`, and
  `build_evaluation_graph` under `Trace::evaluation(builder)`. Both observe the input and every
  output, then run `insert_casts` and `fuse`. Neither differentiates.
- `build_classifier` builds a `CrossEntropyObjective` and forwards `Shape{batch, features}` and
  the four options. `build_graph(batch, features)` is not carried over.
- `batch_shape` fails with `InvalidDescription` for a batch that is not positive or a sample of
  `K_MAX_RANK` extents (`core/shape.hpp:56`).

Tests, suite `Unit_Steps`, no device:

| Case | Pins |
| --- | --- |
| `ATrainingGraphOfAnMlpHasTheNodesBuildGraphGave` | Node count and op sequence equal a golden recorded from `build_graph` at `d6bdf7b` for `Sequential<Linear, GELU, Linear>` |
| `ATrainingGraphListsEveryParameterWithItsLeafInThatGraph` | `parameters[i].value == parameter.id_in(built.graph)` |
| `ABatchNormModelYieldsTwoUpdatesAndPinsTheirSources` | `updates.updates().size() == 2`; `pinned` equals `updates.sources()` |
| `AProbeGraphHasBatchStatisticsAndNoUpdate` | One `BATCHNORM2D` node, no buffer leaf |
| `AnEvaluationGraphFoldsTheStoredStatistics` | One `BATCHNORM_FOLD`, two buffer leaves, no `BATCHNORM2D` |
| `OneModelTracedThreeTimesKeepsOneStateIdPerParameter` | The same `StateId` has a leaf in each of the three graphs |
| `AModelGraphWithTwoOutputsLeavesLogitsUnset` | `spec.logits == K_NO_VALUE`, `outputs.all().size() == 2`, with an objective written in the test that reads one of them by name |
| `ARefusingLayerFailsTheBuilderWithItsStatus` | `InvalidDescription` with the layer's text |
| `BatchShapePrependsTheBatch` | `{16}` and `{28, 28, 1}` give `{16, 28, 28, 1}`; batch 0 fails |

Acceptance: syntax check of `steps.cpp` alone in wave 3, of the test after wave 3. Dispatch
row: integration judgment.

### T16.3 `bind_parameters` over a state, and the trainer

Files: `include/SushiAI/train/trainer.hpp`, `src/train/trainer.cpp`.

```cpp
[[nodiscard]] Result<std::vector<Optim::ParameterSlot>> bind_parameters(
    const TrainingGraph& built, const NN::ModelState& state, Graph::TensorPool& pool,
    Graph::PlanOptions options = {});
[[nodiscard]] static Result<std::unique_ptr<Trainer>> Trainer::make(
    SushiBLAS::Engine& engine, const Graph::Graph& graph, Graph::TensorPool& pool,
    StepSpec spec, Optim::Optimizer& optimizer, TrainerOptions options = {},
    std::vector<BufferUpdateSlot> updates = {});
```

- `trainer.hpp` drops `StepSpec`, `ParameterBinding`, `TrainingGraph`, `build_graph`,
  `build_classifier` and `OptimizerPlacement`, and includes `train/steps.hpp`,
  `train/training_step.hpp` and `train/feed.hpp`.
- The state form of `bind_parameters` adopts a plan whose `pinned` is `options.pinned` followed
  by `built.pinned`, calls `state.attach(built.graph, pool)`, then `materialise_leaves`, then
  the gradient loop of `src/train/trainer.cpp:55-89`. The pool form keeps its signature and
  shares that loop through one file-local function; T17.6 deletes the pool form.
- `Trainer` holds a `std::unique_ptr<TrainingStep>` in place of the `CompiledStep`. `make`
  keeps its checks (`trainer.cpp:181-195`) with one change: `spec.logits` may be `K_NO_VALUE`
  when `options.track_accuracy` is false; `rows_` is `dim(0)` of the inputs leaf, and
  `classes_` is `dim(1)` of the logits when they are set and of rank 2, else 0. `make` refuses
  accuracy tracking over logits that are not rank 2 with `InvalidDescription`.
- `run_step` checks both tensors with `check_feed` in place of the element counts
  (`trainer.cpp:290-296`), copies, and calls `step_->run()`, which owns the per-step optimizer
  call. `fit` skips the class-count check when `classes_` is 0.
- `optimizer_is_in_plan()` returns `step_->optimizer_is_in_plan()` (SPEC 3.2);
  `compile_count()` and `step()` forward to `step_->step()`.

Tests live in the files T16.7 owns. New cases there, in `test_training.cpp`, suite
`Integration_Training`: `RunningMeanMovesAcrossTwoStepsAsUpdatedStatisticSays`,
`APerStepOptimizerStillStepsWhenTheModelHoldsABatchNorm`,
`ABatchOfAnotherShapeIsRefusedWithBothShapes`.

Acceptance: syntax check after wave 3. Dispatch row: integration judgment.

### T16.4 `BufferUpdateRecorder`

Files: `include/SushiAI/train/buffer_update_recorder.hpp`,
`src/train/buffer_update_recorder.cpp`,
`tests/functional/integration/test_buffer_update_recorder.cpp` (all new). After T14.1.

```cpp
namespace SushiAI::Train
{
    /** @brief Holds one running-statistic update as the two tensors it touches. */
    struct BufferUpdateSlot
    {
        SushiBLAS::Tensor target;
        SushiBLAS::Tensor source;
        double momentum = 0.0;
        double source_scale = 1.0;
    };
    /** @brief Resolves each update's target in state and its source in pool. */
    [[nodiscard]] Result<std::vector<BufferUpdateSlot>> bind_buffer_updates(
        const NN::BufferUpdateList& updates, NN::ModelState& state,
        const Graph::TensorPool& pool);
    /** @brief Records target = (1 - momentum) * target + momentum * source_scale * source. */
    class BufferUpdateRecorder final : public Graph::StepRecorder
    {
        public:
            explicit BufferUpdateRecorder(std::vector<BufferUpdateSlot> slots) noexcept;
            [[nodiscard]] Result<std::size_t> record_into_step(
                SushiBLAS::TaskRecorder& recorder) override;
            [[nodiscard]] std::size_t slot_count() const noexcept;
    };
}
```

`bind_buffer_updates` fails with `Incompatible` when the state holds no entry for a target or
the two tensors differ in shape or dtype; a source that `pool.is_bound` denies is fatal, since
the pool must have materialised the graph first. `record_into_step` builds
`SushiBLAS::BLASOps blas(recorder)` and records `blas.scal(1 - momentum, target)` then
`blas.axpy(momentum * source_scale, source, target)` per slot, inside
`guard(SeamPhase::Record, ...)`, and returns twice the slot count
(`src/optim/sgd.cpp:256-304` is the form; migration SPEC line 1075). It takes no `Engine&`, and
`before_replay` and `after_replay` keep their empty defaults.

Tests, suite `Integration_BufferUpdateRecorder`: `RecordsTwoTasksPerSlot`,
`OneReplayBlendsTheSourceIntoTheTarget` (hand-set tensors against `NN::updated_statistic`),
`NoSlotRecordsNothing`, `BindRefusesATargetTheStateLacks`, `BindRefusesADtypeMismatch`.

Acceptance: syntax check of the three files; the five cases exist. Dispatch row: spec-complete
one or two files.

### T16.5 `TrainingStep`

Files: `include/SushiAI/train/training_step.hpp`, `src/train/training_step.cpp` (new).

```cpp
namespace SushiAI::Train
{
    enum class OptimizerPlacement : std::uint8_t { InPlan, PerStep };   // from trainer.hpp:232
    /** @brief States how a training step is compiled. */
    struct TrainingStepOptions
    {
        OptimizerPlacement optimizer_placement = OptimizerPlacement::InPlan;
        const Graph::FusionSelector* fusion_selector = nullptr;
    };
    /** @brief Owns one compiled training replay and the recorders behind the graph's work. */
    class TrainingStep final
    {
        public:
            [[nodiscard]] static Result<std::unique_ptr<TrainingStep>> make(
                SushiBLAS::Engine& engine, const Graph::Graph& graph, Graph::TensorPool& pool,
                Optim::Optimizer& optimizer, std::vector<BufferUpdateSlot> updates,
                TrainingStepOptions options = {});
            [[nodiscard]] Status run();
            [[nodiscard]] bool optimizer_is_in_plan() const noexcept;
            [[nodiscard]] Graph::CompiledStep& step() noexcept;
            [[nodiscard]] const Graph::CompiledStep& step() const noexcept;
    };
}
```

`make` moves `placement_recorder` and its message from `src/train/trainer.cpp:130-146`. It
owns one `BufferUpdateRecorder` and one `StepRecorderList`, adds the optimizer first when the
placement is `InPlan` and its own recorder second when it holds a slot, and passes the list to
`CompiledStep::compile`, or null when the list is empty, so a model without buffers under
`PerStep` compiles as it does today. The list is flat and holds at most these two members, and
an `Optimizer` is not a list (`optim/optimizer.hpp:103`, `:163`), so no recorder is held twice
and no list reaches itself; `StepRecorderList::add` leaves that to its caller
(`graph/step_recorder.hpp:84-87`). `run` replays, then calls `optimizer.step()` when the
optimizer is not in the plan. The type has no copy and no move. `optimizer_is_in_plan` is a
stored flag.

Tests are T16.3's three cases and the existing suites, which reach `TrainingStep` through
`Trainer`. Acceptance: syntax check of `training_step.cpp` alone in wave 3. Dispatch row:
spec-complete one or two files.

### T16.6 `check_feed` and `EvaluationStep`

Files: `include/SushiAI/train/feed.hpp`, `src/train/feed.cpp`,
`include/SushiAI/train/evaluation_step.hpp`, `src/train/evaluation_step.cpp`,
`tests/functional/integration/test_evaluation_step.cpp` (all new). Two dispatches of two files
each; the test goes with the second.

```cpp
namespace SushiAI::Train
{
    /** @brief Returns Incompatible unless source has the rank, extents and dtype of destination. */
    [[nodiscard]] Status check_feed(const SushiBLAS::Tensor& source,
                                    const SushiBLAS::Tensor& destination, std::string_view what);
    /** @brief Owns one compiled forward-only replay of a traced graph. */
    class EvaluationStep final
    {
        public:
            [[nodiscard]] static Result<std::unique_ptr<EvaluationStep>> make(
                SushiBLAS::Engine& engine, const TracedStep& traced,
                const NN::ModelState& state, Graph::TensorPool& pool,
                const Graph::FusionSelector* selector = nullptr);
            [[nodiscard]] Status run(const SushiBLAS::Tensor& inputs);
            [[nodiscard]] Result<const SushiBLAS::Tensor*> output(std::string_view name) const;
            [[nodiscard]] const Graph::CompiledStep& step() const noexcept;
    };
}
```

`check_feed`'s text is `The batch's {what} have shape {} but the compiled step expects {}`,
with both shapes printed through `Shape::to_string()`. `EvaluationStep::make` adopts a plan
when the pool has none, attaches the state, materialises the leaves and compiles with no
trailing recorder. `run` checks the feed, records one `blas.copy` into the input leaf, runs it
and replays. `output` resolves the name through `traced.outputs.at`. The class serves a probe
graph as well as an evaluation graph.

Tests, suite `Integration_EvaluationStep`: `TheStepCompilesOnce`,
`AnEvaluationStepSeesAWeightTheTrainingStepJustMoved` (one state, two pools, one trainer step
between two evaluation runs: the outputs differ), `AProbeRunLeavesBothRunningVectorsUnchanged`,
`AnInputOfAnotherShapeIsRefusedWithBothShapes`, `AnUnknownOutputNameListsTheNames`. The second
case is the device test SPEC section 10 asks for on two plans over shared storage.

Acceptance: syntax check after wave 3. Dispatch row: spec-complete one or two files for
`feed`, integration judgment for `EvaluationStep`.

### T16.7 The callers of `build_graph` and `build_classifier`

Files, all edited in wave 4: `tests/functional/integration/test_training.cpp`,
`test_resilience.cpp`, `test_mixed_precision.cpp`; `tests/functional/regression/`
`test_compile_once.cpp`, `test_static_memory.cpp`, `test_determinism.cpp`,
`test_fused_optimizer.cpp`, `test_mixed_precision.cpp`, `test_loss_scaling.cpp`,
`test_learning_curve.cpp`, `test_fusion_numerics.cpp`; `examples/example.cpp`,
`examples/demo_mlp.cpp`; `apps/train.cpp`. A search for `build_graph(` and `build_classifier(`
in the working tree finds 38 call sites in these 14 files.

The edit is one rule. `Train::TrainingGraph built = Train::build_classifier(args);` becomes
`SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built, Train::build_classifier(args));` in a test
body and `SA_TRY_ASSIGN(sa::Train::TrainingGraph built, sa::Train::build_classifier(args));` in
a function that returns a status (`examples/example.cpp:95`, `examples/demo_mlp.cpp:627`). A
helper that returns a value takes `std::move(result).value()` after `ASSERT_TRUE` on
`result.is_ok()` in its caller, or returns the `Result`. `apps/train.cpp:98-100` calls
`build_training_graph(model, *objective, Shape{batch, features})`. Two more edits:
`test_resilience.cpp:341` expects `Incompatible` with the shapes `[4, 6]` style text of
`check_feed` in place of `24 input elements`, and T16.3's three new cases are added to
`test_training.cpp`.

Acceptance: every file passes the single-file syntax check; no call of `build_graph` is left in
the tree. Dispatch row: spec-complete many files.

## 8. U17: the dataset's sample shape, the applications, the example, the deletions

### T17.1 `Dataset::sample_shape`

Files: `include/SushiAI/data/dataset.hpp`, `src/data/dataset.cpp`,
`tests/functional/integration/test_resilience.cpp` (one override, for `NoBatches` at line 120),
`tests/functional/integration/test_registries.cpp`.

```cpp
/** @brief Returns the shape of one sample, batch axis left out. */
virtual Shape sample_shape() const = 0;                       // Dataset, beside input_dim()
enum class MnistLayout : std::uint8_t { Flat, Image };        // [784] or [28, 28, 1]
MnistLayout MnistOptions::layout = MnistLayout::Flat;
```

`SyntheticClassification` returns `{features}`, `Rings` `{2 + distractor_features}`.
`MaterialisedDataset` (`src/data/dataset.cpp:211-239`) stores a `Shape` in place of
`features_`, and `input_dim()` returns its element count. `allocate_batch`
(`dataset.cpp:125-145`) takes the sample shape and allocates `[rows, sample...]`. `make_mnist`
reads the key `"layout"`, `"flat"` or `"image"`, default `"flat"`, and refuses any other text
with `InvalidDescription`. `Batch::inputs`' comment becomes "An [N, sample...] tensor".

Tests in `test_registries.cpp`: `Integration_Registries.RingsReportItsSampleShape`,
`AnMnistLayoutOtherThanFlatOrImageIsRefused`, and, skipped without the IDX files as the
existing MNIST cases are, `AnImageLayoutYieldsRankFourBatches`.

Acceptance: syntax check of the four files. Dispatch row: spec-complete one or two files.

### T17.2 `sushiai train` on `build_model`

Files: `apps/train.cpp`. The body follows the listing of section 4. The check at lines 85-89
compares `dataset->sample_shape()` with `model_spec.input_shape` and prints both shapes. The
model is `NN::build_model(model_spec)`; the batch shape is `batch_shape(batch_size,
sample_shape)`. `save_checkpoint(spec.checkpoint, state, engine)` writes it. The printed lines
keep their text (`model.size()` and `parameter_elements(model)` both exist on a `ModelGraph`),
and the line `compiles: 1` keeps its form, which a CTest case pins
(`tests/CMakeLists.txt:227-233`).

Acceptance: syntax check; `tests/fixtures/tiny_train.json.in` needs no edit. Dispatch row:
integration judgment.

### T17.3 `sushiai inference` on `build_model`

Files: `apps/inference.cpp`. `build_model`, `build_evaluation_graph`, `ModelState::create`,
`load_checkpoint(checkpoint_path, state)`, `EvaluationStep::make`, then per batch
`step->run(batch.inputs)` in place of the `memcpy` at line 144. The output width is `dim(1)` of
the one published output; a model that publishes several values or a value that is not rank 2
fails with `InvalidDescription` and says the report counts classes. The compile-count line
keeps its form (`tests/CMakeLists.txt:238-248`).

Acceptance: syntax check. Dispatch row: integration judgment.

### T17.4 The MNIST example in version 2

Files: `examples/mnist_cnn/model.json`, `train.json`, `eval.json`, `README.md`. The model file
takes the header of `examples/yolo26/model.json:2-3`, `"format_version": 2` and
`"input": { "shape": [28, 28, 1] }`, and loses its `unflatten` row; the eight rows after it are
unchanged. Both data blocks gain `"layout": "image"`. The README states the two changes.

Acceptance: `Config::load_model` on the file is part of the owner's run; no syntax check
applies. Dispatch row: spec-complete one or two files.

### T17.5 The callers of the pool-owned forms

Files, wave 6: the eleven test files of T16.7, `tests/functional/integration/test_checkpoint.cpp`,
`tests/functional/unit/test_module.cpp`, `test_conv2d.cpp`, `test_backward.cpp`,
`test_layer_registry.cpp`, `test_loss_roots.cpp`, `test_module_list.cpp`, `test_config.cpp`,
`test_layer_stack.cpp`, `examples/example.cpp`, `examples/demo_mlp.cpp`.

| Before | After |
| --- | --- |
| `bind_parameters(built, pool)` | `ModelState::create(engine, model)` on the line before, then `bind_parameters(built, state, pool)` |
| `NN::initialize(engine, model, pool, policy)` | `NN::initialize(engine, state, policy)` |
| `IO::save_checkpoint(path, model, pool, engine)`, `load_checkpoint(path, model, pool)` | The state forms |
| `parameter.id()` read for a leaf of a known graph | `parameter.id_in(graph)` |
| `pool.tensor(parameter.id())` in `test_checkpoint.cpp:105` | `state.tensor(name)` |
| `model.input_features` | `model.input_shape.num_elements()` |

Cases that pin `id()` itself go: `tests/functional/unit/test_module.cpp:591-617` (the leaf of
the most recent declare) and the `id()` half of line 545; the `id_in` half of each stays.
`Integration_Checkpoint`'s six cases keep their names over the state forms. Where
`zero_final_layer` is on and a bias follows the last weight, the result is unchanged: a bias is
a constant zero under both rules (SPEC lines 1667-1670).

Acceptance: every file passes the syntax check; a search for `\.id\(\)` on a `Parameter`,
`input_features`, and the four-argument `save_checkpoint` finds nothing outside the files
T17.6 owns. Dispatch row: spec-complete many files.

### T17.6 The deletions

Files, wave 7: `include/SushiAI/nn/module.hpp` (`Parameter::id()` and `id_`, lines 184, 238,
242, 255), `include/SushiAI/config/model_config.hpp:103-104` and
`src/config/model_config.cpp:479` (`input_features`), `include/SushiAI/io/checkpoint.hpp`
(`Detail::BoundParameter`, `Detail::bind`, both module templates), `include/SushiAI/nn/init.hpp`
(the module template), `include/SushiAI/train/trainer.hpp` and `src/train/trainer.cpp` (the
pool form of `bind_parameters`), and the two cases sections 5 and 6 mark.

Acceptance: each file passes the syntax check; `tools/contract/check_contract.py` is run by the
orchestrator. Dispatch row: spec-complete many files.

## 9. Waves

File sets within a wave are disjoint. A wave starts when the one before it is on disk.

| Wave | Tasks | Tree parses after it |
| --- | --- | --- |
| 1 | T14.1 | Yes |
| 2 | T14.2, T15.1, T15.2, T16.4, T17.1 | Yes |
| 3 | T16.1 (library files), T16.2, T16.3, T16.5, T16.6 | Library yes; the callers of `build_graph` no |
| 4 | T16.7, `test_objective.cpp` of T16.1, the tests of T16.2 and T16.6 | Yes |
| 5 | T17.2, T17.3, T17.4 | Yes |
| 6 | T17.5 | Yes |
| 7 | T17.6 | Yes |

`test_resilience.cpp` is edited in waves 2, 4 and 6, by one task each time. `apps/train.cpp`
is edited in wave 4 (one call) and rewritten in wave 5. The owner's build belongs after wave 4
and again after wave 7; between waves 3 and 4 the test target does not compile.

## 10. What the orchestrator adds to the build lists

`src/CMakeLists.txt`, the source list at lines 15-137: `nn/model_state.cpp`,
`io/import_weights.cpp`, `io/checkpoint.cpp`, `train/steps.cpp`,
`train/buffer_update_recorder.cpp`, `train/training_step.cpp`, `train/feed.cpp`,
`train/evaluation_step.cpp`.

`include/SushiAI/SushiAI.h`, which the configure check compares with the header glob
(`src/CMakeLists.txt:253-255`): `nn/model_state.hpp`, `io/import_weights.hpp`,
`train/training_graph.hpp`, `train/steps.hpp`, `train/buffer_update_recorder.hpp`,
`train/training_step.hpp`, `train/feed.hpp`, `train/evaluation_step.hpp`. `train/steps.hpp`
goes before `train/trainer.hpp`.

`tests/CMakeLists.txt`: `functional/unit/test_steps.cpp`; `functional/integration/`
`test_model_state.cpp`, `test_init_state.cpp`, `test_import_weights.cpp`,
`test_state_checkpoint.cpp`, `test_buffer_update_recorder.cpp`, `test_evaluation_step.cpp`.

Also the orchestrator's: the licence boxes, the rows of `tools/contract/site_classes.tsv`, the
READMEs of `nn`, `train`, `io`, `data`, `apps` and the three test folders,
`docs/guides/CLI_GUIDE.md` for the `layout` key, and the changelog lines.

## 11. Questions for the owner

Each is a module boundary, a public interface or data ownership that the SPEC leaves open. The
plan is written under the recommendation.

1. **Who owns a step's pool.** SPEC line 996 sketches `TrainingStep` with "its pool". The
   optimizer needs the parameter slots, which need the pool, before the step is compiled.
   - a. The caller owns the pool, as `Trainer::make` and `CompiledStep::compile` have it today.
   - b. The step owns the pool and is built in two phases, bind and compile.
   - Recommendation: a. No type gets a half-built state.
2. **How the trainer receives buffer updates.** a. A trailing parameter
   `std::vector<BufferUpdateSlot> updates = {}` on `Trainer::make`, so the 36 existing call
   sites compile unchanged. b. A second `make` over `TrainingGraph` and `ModelState`.
   Recommendation: a.
3. **`ModelState::from_entries` public.** It lets a test and a later importer build a state
   without a model. Recommendation: public.
4. **Where the strict loader lives.** `import_weights` is the loader and `load_checkpoint` is
   the call with no rule, so `io/checkpoint.hpp` includes `io/import_weights.hpp`. The
   alternative is a third file both include. Recommendation: as planned.
5. **`RenameRule` order.** The first rule that matches wins. Recommendation: that.
6. **`GraphOptions`.** The SPEC names the type and gives no member. Recommendation: the four
   trailing parameters of `build_graph` (`train/trainer.hpp:139-142`).
7. **`build_classifier` survives** with its parameter list and a `Result` return; the SPEC's
   deletion list names `build_graph` only. Recommendation: keep it.
8. **`StepSpec::logits` for a model with several outputs.** Recommendation: unset, and
   `Trainer::make` refuses accuracy tracking then; the objective owns the metric (SPEC lines
   1081-1083).
9. **MNIST and the shape-typed feed.** A rank-4 MNIST batch does not fit the `[N, 784]` leaf
   of `sushiai_demo_mlp` (`examples/demo_mlp.cpp:779-788`).
   - a. A `layout` key, `flat` by default, `image` in the example.
   - b. MNIST is always rank 4 and the demo's model starts with a `Flatten`.
   - c. The feed accepts equal element counts, against SPEC line 1079.
   - Recommendation: a.
10. **`train/` calls `NN::Detail::bridge_throwing`.** It is another module's `Detail`. The
    alternative is to move the bridge to `core/`. Recommendation: call it where it is; the
    status conversion of the builder and the layers deletes it.
11. **`TracedStep` has no `updates` member.** A probe trace discards them and an evaluation
    trace cannot request one, so the SPEC's member (line 981) would always be empty.
    Recommendation: leave it out.
12. **One header per step type.** `training_graph.hpp`, `training_step.hpp`,
    `evaluation_step.hpp` and `feed.hpp` beside the SPEC's `steps.hpp`, by the rule of one
    primary type per header. Recommendation: as planned.
13. **The cases that pin `Parameter::id()`** (`test_module.cpp:545`, `:591-617`) are deleted
    with the member. A copied, untraced `Parameter` still reports `declared()`; no case is
    planned to change that. Recommendation: delete the cases, keep `declared()` as it is.
14. **Effort above `low`.** T16.2 and T16.3 are the two tasks where a wrong order of passes
    compiles and trains wrong. Whether either is a `high` dispatch is the owner's call.

## 12. Not verified

- **Two compiled plans over one storage.** Not read in SushiRuntime and not run. T16.6's
  second case settles it. If it fails, the fallback of SPEC lines 1022-1025 keeps every
  signature here and adds a copy between steps.
- **That `materialise_leaves` keeps a bound leaf.** `src/graph/tensor_pool.cpp:159-168` was
  read to its fourth line; the SPEC states it of the pre-migration file.
- **That published outputs survive planning without a pin.** Inferred from
  `apps/inference.cpp:149`, which reads the output after the run, and from its passing CTest
  case; `src/graph/memory_plan.cpp` was not read.
- **`src/graph/lowering/norm.cpp`, `src/nn/batch_norm2d.cpp`, `src/nn/model_graph.cpp`,
  `src/nn/node_registry.cpp`, `src/config/model_config.cpp`** were not read. That
  `build_model` builds a version 1 chain with the names `LayerStack` gives rests on
  `nn/node_registry.hpp:62-63` and on the U10 test the SPEC lists at line 1805.
- **The call counts in T16.7 and T17.5** are text searches at `d6bdf7b`. The working tree has
  uncommitted edits in `src/train/trainer.cpp`, `test_checkpoint.cpp`,
  `test_learning_curve.cpp` and `test_step_recorder.cpp`; the line numbers cited are of the
  working tree.
- **Whether a test pins the text `needs all four`** or the `parameter` wording of the
  checkpoint failures: the search for the first found none, the second was not searched.
- **`SushiBLAS::BLASOps::scal` and `axpy` over a `TaskRecorder`**: the declarations
  (`sushiblas/include/SushiBLAS/engine/blas/level1.hpp:47`, `:53`) and one caller were read,
  not the constructor of `BLASOps`.
- **Mixed precision and the update sources.** They are observed, so the cast pass should leave
  them in the master type; `bind_buffer_updates` refuses a dtype mismatch if it does not.
- **`examples/demo_mlp.cpp`** was read at lines 596-655 and 750-794 only, and
  `examples/example.cpp` through search results.
- **No file of this plan exists yet**, so no syntax check was run for it.
