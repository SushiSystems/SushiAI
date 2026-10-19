# Model graph and batch normalisation

**Status:** Shipped — 2026-10-09. Built and run; see `REPORT.md`.

## Decided

The owner left these decisions to the engineer on 2026-10-07; each takes the recommendation of
section 9. Decision 1 follows from his ruling on the error contract: the status type lands first
(wave 1 of `docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md`) and the seams of this design
are written against it. Decision 13, the run-time arity gap of `BATCHNORM2D`, goes to the
migration work. Decision 8, the exporter rerun that settles the n/(n-1) factor of the running
variance, is done before the `BatchNorm2d` unit is accepted.

This document designs the model graph, the node seam, buffers, the two trace modes and
`BatchNorm2d` for SushiAI, and the YOLO26 blocks written over them. Section 9 lists the choices
that are the owner's; the rest of the document is written under the recommendation given there.
Line numbers are those of the working tree on 2026-10-07. Files under `src/graph/`,
`include/SushiAI/graph/`, `include/SushiAI/core/` and `src/autograd/` are being edited by other
work, so a cited line can move; none of the files read looked half-edited.

## 1. What the ruling requires

The rulings are in `docs/design/PROGRAMME.md:22-38`. Five of them bind this design.

| Ruling | Where | What it fixes here |
| --- | --- | --- |
| Batch normalisation | `PROGRAMME.md:38` | The batch variance is an output of `BATCHNORM2D`. The running statistics are updated beside the graph, by a recorder, not by a node |
| Error contract | `PROGRAMME.md:28` | Every operation that can fail returns a status. No status type exists under `include/` or `src/` today (a search for `Status`, `Result`, `Expected` and `Checked` as a class or struct finds nothing) |
| Naming | `PROGRAMME.md:29` | New names take `K_` constants, PascalCase enumerators and no `I` prefix. An enumerator added to an existing enumeration keeps that enumeration's spelling until the naming wave renames the whole enumeration |
| Eager surface | `PROGRAMME.md:25` | `Session`, `Step` and `Scope` must not break after they ship. The eager spec has a caller write `encoder.forward(step.builder(), x)` and says no layer or builder method changes (`docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md:40`, `:60-61`) |
| References | `PROGRAMME.md:37` | A reference value comes from a run of the pinned Ultralytics commit. The n/(n-1) factor of the running variance has no such run yet (section 2.7) |

Two more shape the order of work: the owner builds once the programme's code is written
(`PROGRAMME.md:30`), and the first YOLO26 run is mixed precision (`PROGRAMME.md:32`).

The code today cannot express what stage 3 and stage 5 need:

- A layer takes one value and returns one (`include/SushiAI/nn/ilayer.hpp:57-58`,
  `include/SushiAI/nn/module.hpp:218-225`), and `LayerStack` keeps only the running value
  (`src/nn/layer_stack.cpp:50-52`).
- A parameter holds one graph id and returns it on every later declare
  (`module.hpp:84`), so a module traced into a second graph gets the first graph's ids.
- There is no buffer type, no trace mode and no named child. The walk has three arms
  (`module.hpp:306-325`), and the field form lists `Parameter` members only (`module.hpp:131-138`).
- `BATCHNORM2D` emits `{y, mean, inv_std}` (`src/graph/builder.cpp:980`). The variance before
  epsilon exists only as a scratch tensor in the lowering (`src/graph/lowering.cpp:1273`).
- The model file holds `sushiai_model`, `input.features` and `layers`
  (`src/config/model_config.cpp:40-43`), and `build_layer_stack` folds one shape through the
  rows (`src/nn/layer_registry.cpp:401-415`).

## 2. The design

Declarations below are in the status form the error contract requires. `Status` and `Result<T>`
are provisional names for the type stage 2 defines; this design needs two properties from it:
`[[nodiscard]]`, and no access to the value without the status. Examples and block bodies are in
today's throwing form (`SA_THROW_IF`), which is what a host-lane wave can write before the status
header exists. Decision 1 in section 9 settles which form the first wave writes.

### 2.1 State identity in the graph

A parameter or a buffer is one piece of model state. It is traced into several graphs: the
training graph, the forward-only graph for label assignment, the evaluation graph, and one graph
per step under the eager front end (`EAGER_FRONT_END/SPEC.md:73`). Each graph records which of
its leaves stands for which state.

```cpp
// include/SushiAI/graph/state_id.hpp   (new, host lane)
namespace SushiAI::Graph
{
    /** @brief Identifies one piece of model state across every graph it is traced into. */
    enum class StateId : std::uint64_t {};

    /** @brief Marks state that has not been assigned an identity. */
    inline constexpr StateId K_NO_STATE{0};

    /** @brief Holds a state identity that a copy does not share and a move carries away. */
    class StateIdentity
    {
        public:
            StateIdentity() noexcept = default;
            StateIdentity(const StateIdentity&) noexcept {}
            StateIdentity& operator=(const StateIdentity&) noexcept;
            StateIdentity(StateIdentity&& other) noexcept;
            StateIdentity& operator=(StateIdentity&& other) noexcept;

            /** @brief Returns the identity, drawing one from the process-wide counter if none. */
            StateId get() noexcept;

            /** @brief Returns the identity, or K_NO_STATE when none was drawn. */
            StateId peek() const noexcept;
    };
}

// include/SushiAI/graph/ir.hpp   (additions to Graph, and one Lifetime enumerator)
enum class Lifetime : std::uint8_t
{
    PARAMETER, GRADIENT, OPTIMIZER_STATE, INPUT,
    BUFFER,      /**< @brief Model state with no gradient, written beside the graph. */
    ACTIVATION, SCRATCH,
};
// lifetime_name gains "buffer"; is_persistent gains BUFFER (ir.hpp:74-100).

class Graph
{
    public:
        /** @brief Returns this graph's leaf for @p state, or NO_VALUE when it has none. */
        ValueId state_leaf(StateId state) const noexcept;

        /** @brief Records @p leaf as @p state; fails if either is already recorded. */
        Status bind_state_leaf(StateId state, ValueId leaf);

        /** @brief Lists every recorded pair, in the order they were bound. */
        const std::vector<std::pair<StateId, ValueId>>& state_leaves() const noexcept;

        /** @brief Returns a graph holding this graph's values and state leaves and no node. */
        Graph without_nodes() const;
};

// include/SushiAI/graph/builder.hpp
/** @brief Declares a buffer leaf: persistent, no gradient, not rewritten per step. */
ValueId buffer(Shape shape, std::string_view debug_name, DType dtype = DType::FLOAT32) const;
```

`StateIdentity` is a member of `Parameter` and of `Buffer`, so both keep their implicit copy and
move. A copied layer is new state, which is what a second head with its own weights needs. A
moved layer keeps its identity, so a model returned by value after tracing still matches its
graphs. The counter is one `std::atomic<std::uint64_t>` in `state_id.hpp`.

`without_nodes` exists because both graph-rebuilding passes construct a new `Graph` and move it
over the traced one (`src/graph/cast_insertion.cpp:231`, `:322`; `src/graph/fusion.cpp:585`,
`:653`). They copy each value field by field and already have to remember `requires_grad` and
`layout` by hand (`fusion.cpp:590-600`, `cast_insertion.cpp:236-246`). A table added to `Graph`
is lost there unless both passes carry it. `build_graph` runs both passes after the trace
(`include/SushiAI/train/trainer.hpp:187-188`), so without this edit every state lookup after the
trace returns `NO_VALUE`. Value ids survive both passes (`fusion.cpp:593-595`,
`cast_insertion.cpp:239-241`), so the copied table stays correct.

A buffer cannot reuse `Lifetime::PARAMETER`: the differentiator retags the gradient of a
`PARAMETER` value (`src/autograd/differentiator.cpp:228-230`) and `GraphBuilder::parameter` sets
`requires_grad` (`src/graph/builder.cpp:144`). `Autograd::differentiate` rejects a `wrt` value
whose lifetime is `BUFFER`.

### 2.2 Parameter, Buffer and one walk

```cpp
// include/SushiAI/nn/module.hpp
namespace SushiAI::NN
{
    /** @brief Describes how one parameter is filled before training. */
    struct ParameterInit
    {
        enum class Kind : std::uint8_t { FanScaled, Constant };

        Kind kind = Kind::FanScaled;
        double constant = 0.0;

        static constexpr ParameterInit fan_scaled() noexcept;
        static constexpr ParameterInit constant_value(double value) noexcept;
    };

    class Parameter
    {
        public:
            /** @brief Declares into builder.graph() unless that graph already holds this
             *         parameter's leaf; a later declare with another shape or dtype fails. */
            Graph::ValueId declare(const Graph::GraphBuilder& builder, Shape shape,
                                   ParameterRole role, int64_t fan_in, int64_t fan_out,
                                   std::string_view debug_name, DType dtype = DType::FLOAT32);

            /** @brief Declares as above with a stated initial fill. */
            Graph::ValueId declare(const Graph::GraphBuilder& builder, Shape shape,
                                   ParameterInit init, int64_t fan_in, int64_t fan_out,
                                   std::string_view debug_name, DType dtype = DType::FLOAT32);

            Graph::ValueId id_in(const Graph::Graph& graph) const noexcept;
            Graph::StateId state() const noexcept;
            ParameterInit init() const noexcept;
            bool declared() const noexcept;          // a declare has fixed the shape
            // shape(), dtype(), role(), fan_in(), fan_out() unchanged.

            [[deprecated("use id_in(graph)")]] Graph::ValueId id() const noexcept;
    };

    /** @brief Describes a tensor of model state that no optimizer moves. */
    class Buffer
    {
        public:
            /** @brief Fixes shape, dtype and starting value; a later, different call fails. */
            void define(Shape shape, double initial_value, DType dtype = DType::FLOAT32);

            /** @brief Declares this buffer's leaf in the builder's graph, once per graph. */
            Graph::ValueId read(const Graph::GraphBuilder& builder, std::string_view debug_name);

            Graph::ValueId id_in(const Graph::Graph& graph) const noexcept;
            Graph::StateId state() const noexcept;
            bool declared() const noexcept;          // define() was called
            const Shape& shape() const noexcept;
            DType dtype() const noexcept;
            double initial_value() const noexcept;
    };

    template<typename M> struct BufferField { std::string_view name; Buffer M::* member; };
    template<typename M, typename C> struct ChildField { std::string_view name; C M::* member; };

    template<typename M>
    constexpr BufferField<M> buffer(std::string_view name, Buffer M::* member) noexcept;
    template<typename M, typename C>
    constexpr ChildField<M, C> child(std::string_view name, C M::* member) noexcept;

    /** @brief Receives every piece of state a walk reaches; non-owning. */
    class StateVisitor
    {
        public:
            template<typename OnParameter, typename OnBuffer>
            StateVisitor(OnParameter& on_parameter, OnBuffer& on_buffer) noexcept;

            void parameter(const std::string& name, Parameter& parameter) const;
            void buffer(const std::string& name, Buffer& buffer) const;
    };

    template<typename M>
    void walk_state(M& module, std::string_view prefix, const StateVisitor& visitor);

    template<typename M, typename Fn>
    void for_each_parameter(M& module, std::string_view prefix, Fn&& visitor);   // kept
    template<typename M, typename Fn>
    void for_each_buffer(M& module, std::string_view prefix, Fn&& visitor);      // new
    template<typename M> std::size_t buffer_count(M& module);
}

// include/SushiAI/nn/module_list.hpp   (new)
/** @brief Holds a run-time count of same-typed children, named by index. */
template<typename T>
class ModuleList
{
    public:
        void push_back(T child);
        std::size_t size() const noexcept;
        T& operator[](std::size_t index);
        auto begin() noexcept;
        auto end() noexcept;
        template<typename Fn> void visit_children(Fn&& visitor);   // visitor(index, T&)
};
```

`walk_state` is the only walk. `for_each_parameter` and `for_each_buffer` call it with one of
the two callbacks empty, so the two cannot differ in name or order. Its arms, tried in order:

1. The type's own `walk_state(prefix, visitor)`. `ModelGraph` and every `Node` use this arm.
2. The legacy `walk_parameters(prefix, visitor)` (`module.hpp:306-311`), which yields parameters
   only. `LayerStack` and `ILayer` use it (`src/nn/layer_stack.cpp:55-60`,
   `ilayer.hpp:110-115`).
3. `visit_children`, named by index (`module.hpp:312-318`). `Sequential` and `ModuleList`.
4. `fields()`, entry by entry in declaration order: a `ParameterField` yields a parameter, a
   `BufferField` a buffer, a `ChildField` recurses under `join(prefix, name)`.

One `fields()` tuple carries all three entry kinds, so declaration order is walk order and is
checkpoint order. `Detail::visit_field` (`module.hpp:270-277`) gains one overload per entry
kind. An undeclared parameter is skipped as today (`module.hpp:275`), and so is an undefined
buffer. Two entries of one `fields()` with the same name fail a `static_assert`. Existing types
list only `ParameterField` entries, so their walks yield what they yield today.

`define` and `read` are separate because a training graph never reads the running statistics. A
buffer that is defined and not read adds no leaf to that graph, so no pass and no memory plan
meets a leaf that no node reads.

A buffer cannot hide behind arm 2. `LayerAdapter` already refuses a type without
`forward(const GraphBuilder&, ValueId)` (`ilayer.hpp:84-88`); it gains a second assertion that
refuses a type whose `fields()` tree holds a `BufferField`.

`Parameter::id()` stays until its three callers move to another source
(`trainer.hpp:160-161`, `include/SushiAI/io/checkpoint.hpp:75`,
`include/SushiAI/nn/init.hpp:190`). All three include the engine and are frozen until the
SushiBLAS migration. Until then `id()` returns the leaf of the most recent declare, which is
what it returns today for a module traced once.

### 2.3 Trace modes

```cpp
// include/SushiAI/nn/trace.hpp   (new, host lane)
namespace SushiAI::NN
{
    enum class TraceMode : std::uint8_t
    {
        Training,     /**< Batch statistics; state updates are requested. */
        Evaluation,   /**< Stored statistics; requesting an update fails. */
    };

    /** @brief Describes target = (1 - momentum) * target + momentum * source_scale * source. */
    struct BufferUpdate
    {
        Graph::StateId target = Graph::K_NO_STATE;
        Graph::ValueId source = Graph::NO_VALUE;
        double momentum = 0.0;
        double source_scale = 1.0;
    };

    class BufferUpdateSink
    {
        public:
            virtual ~BufferUpdateSink() = default;
            virtual void add(const BufferUpdate& update) = 0;
    };

    /** @brief Collects the updates of one trace. */
    class BufferUpdateList final : public BufferUpdateSink
    {
        public:
            void add(const BufferUpdate& update) override;
            SushiRuntime::span<const BufferUpdate> updates() const noexcept;

            /** @brief Lists every source value, for the observed and the pinned lists. */
            SushiRuntime::span<const Graph::ValueId> sources() const noexcept;
    };

    /** @brief Returns the sink that drops every update, for a forward-only pass. */
    BufferUpdateSink& discard_updates() noexcept;

    class Trace
    {
        public:
            static Trace training(const Graph::GraphBuilder& builder,
                                  BufferUpdateSink& updates) noexcept;
            static Trace evaluation(const Graph::GraphBuilder& builder) noexcept;

            const Graph::GraphBuilder& builder() const noexcept;
            TraceMode mode() const noexcept;

            /** @brief Hands one update to the sink; fails in Evaluation. */
            Status update(Buffer& target, Graph::ValueId source,
                          double momentum, double source_scale) const;
    };

    /** @brief True when T has forward(const Trace&, ValueId) -> ValueId and fields(). */
    template<typename T> inline constexpr bool TraceModule = /* detection, as IsModule */;

    /** @brief True when T has trace_training and trace_evaluation, both
     *         (const Trace&, ValueId) -> ValueId, fields(), and no forward. */
    template<typename T> inline constexpr bool ModalModule = /* detection */;

    template<typename T>
    inline constexpr bool Traceable = Module<T> || TraceModule<T> || ModalModule<T>;

    /** @brief Traces one layer in whichever of the three forms it has. */
    template<typename L>
    Graph::ValueId trace_layer(L& layer, const Trace& trace, Graph::ValueId x);
}
```

A layer says what it emits in each mode by which members it has.

| Form | Members | Who |
| --- | --- | --- |
| `Module` | `forward(const GraphBuilder&, ValueId)` | Every layer in the tree today. It cannot ask the mode |
| `ModalModule` | `trace_training(const Trace&, ValueId)` and `trace_evaluation(const Trace&, ValueId)`, no `forward` | A leaf whose emission depends on the mode: `BatchNorm2d`, later dropout |
| `TraceModule` | `forward(const Trace&, ValueId)` | A composite that may hold a modal child. It passes the trace on and never reads the mode |

`trace_layer` is the one function that reads `mode()` to dispatch. A type with exactly one of
the pair fails a `static_assert`, so a layer that forgets its evaluation form does not compile.
A third mode adds one enumerator and one member to each modal leaf, and the compiler names every
leaf that lacks it.

`Trace::update` records the target by `StateId`, never by address, so a model moved after the
trace leaves no dangling target. The same call is the only way a source value enters
`BufferUpdateList::sources()`, which the step builders hand to the observed list and to the
planner (section 2.8). The forward-only pass for label assignment (`PROGRAMME.md:34`) is
`Trace::training(builder, NN::discard_updates())`: batch statistics, and no update is recorded.

`Sequential` gains a second overload, `forward(const Trace&, ValueId)`, that calls `trace_layer`
per child. Its assertion at `include/SushiAI/nn/layers.hpp:188` becomes `Traceable<Ls> && ...`,
and its existing overload (`layers.hpp:210-215`) is enabled only when every child is a `Module`.
`Module<Sequential<Linear, ReLU>>` stays true (`layers.hpp:254`). A `Sequential` holding a
`BatchNorm2d` is a `TraceModule` and not a `Module`, so a builder-only caller fails to compile
instead of tracing in an unstated mode.

**What the eager front end must supply.** The eager call form
`layer.forward(step.builder(), x)` (`EAGER_FRONT_END/SPEC.md:40`) keeps working for every
`Module`. A model that holds batch normalisation is traced as
`model.forward(step.trace(), x)`. That needs three additions to the eager spec before its
surface ships:

```cpp
// Eager::Session
Step begin_step(NN::TraceMode mode = NN::TraceMode::Training);

// Eager::Step
/** @brief Returns the trace context of this step: its builder, its mode, its update list. */
const NN::Trace& trace() const noexcept;
```

The third is one region for the buffer updates, ordered after every forward region. The eager
spec's key table (`EAGER_FRONT_END/SPEC.md:115-120`) gives the optimizer update the highest key,
`2n + 2`; the buffer updates take `2n + 3`, since they read batch statistics the forward regions
produce and write buffers no other region of the step reads. The sentence "No layer, rule or
builder method changes" (`SPEC.md:60-61`) stays true: no existing method changes, and `Step`
gains one accessor. This is decision 3.

### 2.4 The node seam

```cpp
// include/SushiAI/nn/node.hpp   (new, host lane)
namespace SushiAI::NN
{
    using ValueSpan = SushiRuntime::span<const Graph::ValueId>;
    using ShapeSpan = SushiRuntime::span<const Shape>;

    /** @brief Describes one result of a node. */
    struct Port
    {
        std::string name;   /**< @brief "output" for a single result, else the role name. */
        Shape shape;        /**< @brief Per-sample shape, batch axis excluded. */
    };

    /** @brief One node of a model graph: several values in, several values out. */
    class Node
    {
        public:
            virtual ~Node() = default;

            /** @brief Traces the node; appends one id per entry of ports(), in order. */
            virtual Status forward(const Trace& trace, ValueSpan inputs,
                                   std::vector<Graph::ValueId>& outputs) = 0;

            virtual void walk_state(std::string_view prefix, const StateVisitor& visitor) = 0;

            /** @brief Lists what forward() appends. */
            virtual SushiRuntime::span<const Port> ports() const noexcept = 0;
    };

    /** @brief Presents any Traceable type as a one-in one-out node. */
    template<typename L>
    class ModuleNode final : public Node
    {
        public:
            ModuleNode(L layer, const Shape& input);   // port "output", layer.output_shape(input)
            L& layer() noexcept;
    };

    /** @brief Presents a run-time ILayer as a one-in one-out node. */
    class LayerNode final : public Node
    {
        public:
            explicit LayerNode(std::unique_ptr<ILayer> layer);
    };

    using NodeFactory = Result<std::unique_ptr<Node>> (*)(ShapeSpan inputs,
                                                          const Config::Value& attributes,
                                                          std::string_view context);

    /** @brief One row of K_NODE_REGISTRY. */
    struct NodeEntry { std::string_view type; NodeFactory make; };
}

// include/SushiAI/nn/node_registry.hpp   (new, host lane)
/** @brief Node types that LAYER_REGISTRY cannot hold: several ports, or a trace mode. */
inline constexpr std::array<NodeEntry, 9> K_NODE_REGISTRY = {
    NodeEntry{"batchnorm2d", &make_batch_norm2d_node},
    NodeEntry{"concat",      &make_concat_node},
    NodeEntry{"conv",        &make_conv_node},
    NodeEntry{"bottleneck",  &make_bottleneck_node},
    NodeEntry{"c3k",         &make_c3k_node},
    NodeEntry{"c3k2",        &make_c3k2_node},
    NodeEntry{"sppf",        &make_sppf_node},
    NodeEntry{"c2psa",       &make_c2psa_node},
    NodeEntry{"detect",      &make_detect_node},
};

/** @brief Builds a node of @p type: a K_NODE_REGISTRY row, else a LAYER_REGISTRY row in a
 *         LayerNode; an unknown type fails and lists both tables. */
Result<std::unique_ptr<Node>> make_node(std::string_view type, ShapeSpan inputs,
                                        const Config::Value& attributes,
                                        std::string_view context);
```

The container calls `forward`, `walk_state` and `ports` and never tests a node's kind. A new
layer type is a class, a factory and one row:

- A one-in one-out layer with no trace mode stays one `LAYER_REGISTRY` row
  (`include/SushiAI/nn/layer_registry.hpp:180-194`). `make_node` wraps it in a `LayerNode`, so
  the 13 existing rows are not listed twice and the layer works in a `LayerStack` and in a model
  graph. `SiLU` is a 14th row there.
- A layer with several ports or a trace mode is one `K_NODE_REGISTRY` row.

Each factory is declared in its layer's own header. Today every `make_*` sits in
`layer_registry.hpp:75-172`; new rows do not continue that. `LayerNode` passes
`trace.builder()` to the layer, which is correct because an `ILayer` cannot hold a modal type
(`ilayer.hpp:84`). `ILayer`, `LayerAdapter`, `LayerStack` and `LAYER_REGISTRY` keep their
shape.

`K_NODE_REGISTRY` names `detect`. `include/SushiAI/detect/` includes nothing of `nn/` or
`graph/` today: its headers and sources include `core/common.hpp` and each other only. The head
layer therefore lives under `nn/blocks/`, and `detect/` stays the host arithmetic of detection
(anchors, assignment, box coding) that an objective calls. `nn/` does not include `detect/`.
This is decision 6.

### 2.5 The model-graph container

```cpp
// include/SushiAI/nn/model_outputs.hpp   (new, host lane)
/** @brief The values a model produced, by name. */
class ModelOutputs
{
    public:
        Status add(std::string name, Graph::ValueId id);             // fails on a duplicate
        Result<Graph::ValueId> at(std::string_view name) const;      // lists what exists
        Result<Graph::ValueId> only() const;                         // fails unless exactly one
        const std::vector<std::pair<std::string, Graph::ValueId>>& all() const noexcept;
};

// include/SushiAI/nn/model_graph.hpp   (new, host lane)
/** @brief Names one port of one node of a ModelGraph, or the graph's input. */
class NodeRef
{
    public:
        std::size_t node() const noexcept;
        std::size_t port() const noexcept;
        bool is_input() const noexcept;
};

class ModelGraph
{
    public:
        explicit ModelGraph(Shape input_sample);

        NodeRef input() const noexcept;

        /** @brief Appends @p node under @p name, reading @p from; returns its port 0.
         *         Fails on a duplicate name, an unknown source or a wrong input count. */
        Result<NodeRef> add(std::string name, std::unique_ptr<Node> node,
                            SushiRuntime::span<const NodeRef> from);

        /** @brief Appends a Traceable layer reading one value. */
        template<typename L>
        Result<NodeRef> add(std::string name, L layer, NodeRef from);

        /** @brief Selects a named port of a node. */
        Result<NodeRef> port(NodeRef node, std::string_view name) const;

        /** @brief Publishes one value as a model output. */
        Status output(std::string name, NodeRef from);

        /** @brief Publishes every port of a node under the port's own name. */
        Status output_ports(NodeRef node);

        Result<ModelOutputs> trace(const Trace& trace, Graph::ValueId input);

        /** @brief Traces and returns the single published output. */
        Graph::ValueId forward(const Trace& trace, Graph::ValueId input);

        void walk_state(std::string_view prefix, const StateVisitor& visitor);

        static constexpr auto fields() { return std::tuple<>{}; }
        std::size_t size() const noexcept;
        const Shape& shape_of(NodeRef ref) const;
        Node& node(std::size_t index) const;
};

/** @brief Builds the graph a model file describes. */
Result<ModelGraph> build_model(const Config::ModelSpec& model);
```

`add` accepts only sources that already exist, so the node list is in topological order by
construction and a cycle cannot be written. `trace` keeps every node's ids for the length of the
trace; the ids are integers, so nothing has to be freed early. `trace` compares each traced
value's shape, batch axis removed, with the port's declared shape and fails on a mismatch.
`LayerAdapter` states that equality and does not check it (`ilayer.hpp:92`).

A branch and a merge written by hand, in today's throwing form:

```cpp
NN::ModelGraph model(Shape{640, 640, 3});
const NN::NormOptions norm{1e-3, 0.03};

const NN::NodeRef p3 = model.add("stem", NN::Conv(3, 64, 3, 2, norm), model.input());
const NN::NodeRef p4 = model.add("down", NN::Conv(64, 128, 3, 2, norm), p3);
const NN::NodeRef up = model.add("up", NN::Upsample2d{2}, p4);

const NN::NodeRef both[] = {up, p3};
const Shape shapes[] = {model.shape_of(up), model.shape_of(p3)};
const NN::NodeRef merged = model.add("merge", std::make_unique<NN::Concat>(shapes), both);

model.output("features", model.add("mix", NN::Conv(192, 64, 1, 1, norm), merged));
```

Its walk yields `stem.conv.weight`, `stem.bn.running_mean` and so on. `build_model` runs the
same `add` for a file: for each row it resolves `from` to `NodeRef`s, collects their shapes,
calls `make_node`, and passes the row index as text for the name. A file-built graph therefore
names its rows `0`, `1`, ..., which is what `LayerStack` does (`src/nn/layer_stack.cpp:58-59`),
and a checkpoint of a chain loads into the same chain built either way.

Composition with what exists:

- `ModelGraph` has `fields()` and `forward(const Trace&, ValueId)`, so it is a `TraceModule` and
  nests in a `Sequential`. Its state is reached through arm 1 of the walk.
- A `Sequential`, a `LayerStack` or any block enters a `ModelGraph` through the templated `add`,
  which wraps it in a `ModuleNode`.
- `LayerStack` and `build_layer_stack` are not replaced. `build_layer_stack` refuses a spec in
  which any row's `from` is not the previous row (section 3.5), so a branching file is never
  built as a chain.

The container has one input. YOLO26 needs one.

### 2.6 The model file, version 2

```json
{
    "format_version": 2,
    "input": { "shape": [640, 640, 3] },
    "defaults": {
        "conv":   { "bn_epsilon": 0.001, "bn_momentum": 0.03 },
        "c3k2":   { "bn_epsilon": 0.001, "bn_momentum": 0.03 },
        "sppf":   { "bn_epsilon": 0.001, "bn_momentum": 0.03 },
        "c2psa":  { "bn_epsilon": 0.001, "bn_momentum": 0.03 },
        "detect": { "bn_epsilon": 0.001, "bn_momentum": 0.03 }
    },
    "layers": [
        { "type": "conv", "filters": 16, "kernel": 3, "stride": 2 },
        { "type": "conv", "filters": 32, "kernel": 3, "stride": 2 },
        { "type": "c3k2", "filters": 64, "hidden": 16, "repeats": 1, "inner": "bottleneck", "shortcut": false },
        { "type": "conv", "filters": 64, "kernel": 3, "stride": 2 },
        { "type": "c3k2", "filters": 128, "hidden": 32, "repeats": 1, "inner": "bottleneck", "shortcut": false },
        { "type": "conv", "filters": 128, "kernel": 3, "stride": 2 },
        { "type": "c3k2", "filters": 128, "hidden": 64, "repeats": 1, "inner": "c3k", "shortcut": true },
        { "type": "conv", "filters": 256, "kernel": 3, "stride": 2 },
        { "type": "c3k2", "filters": 256, "hidden": 128, "repeats": 1, "inner": "c3k", "shortcut": true },
        { "type": "sppf", "filters": 256, "kernel": 5, "pools": 3, "shortcut": true },
        { "type": "c2psa", "filters": 256, "hidden": 128, "repeats": 1, "heads": 2 },
        { "type": "upsample", "scale": 2 },
        { "type": "concat", "from": [-1, 6] },
        { "type": "c3k2", "filters": 128, "hidden": 64, "repeats": 1, "inner": "c3k", "shortcut": true },
        { "type": "upsample", "scale": 2 },
        { "type": "concat", "from": [-1, 4] },
        { "type": "c3k2", "filters": 64, "hidden": 32, "repeats": 1, "inner": "c3k", "shortcut": true },
        { "type": "conv", "filters": 64, "kernel": 3, "stride": 2 },
        { "type": "concat", "from": [-1, 13] },
        { "type": "c3k2", "filters": 128, "hidden": 64, "repeats": 1, "inner": "c3k", "shortcut": true },
        { "type": "conv", "filters": 128, "kernel": 3, "stride": 2 },
        { "type": "concat", "from": [-1, 10] },
        { "type": "c3k2", "filters": 256, "hidden": 128, "repeats": 1, "inner": "bottleneck_psa", "shortcut": true },
        { "type": "detect", "from": [16, 19, 22], "classes": 80 }
    ],
    "outputs": [ { "node": 23 } ]
}
```

This is the whole YOLO26n file, 24 rows. Every `filters` and `hidden` value is read from the
convolution weight shapes in `tests/fixtures/yolo26/MANIFEST.json` (for example
`model.2.cv1.conv.weight` is `[32, 1, 1, 32]` and `model.2.cv2.conv.weight` is `[64, 1, 1, 48]`,
so row 2 has hidden 16 and filters 64). Every `from` index is confirmed by channel arithmetic
against the same shapes: row 13's `cv1` reads 384 channels, which is row 10's 256 plus row 6's
128; row 16's reads 256, rows 13 and 4; row 19's reads 192, rows 17 and 13; row 22's reads 384,
rows 20 and 10; and the first head branch reads 64, row 16. The fixture cannot confirm `stride`,
`shortcut`, `pools` or `heads`, because none of them changes a weight shape; section 10 lists
them.

The rules of the format:

- **Version key.** `format_version`, which the `file-formats` rule asks for. The version 1 key
  is `sushiai_model` (`src/config/model_config.cpp:55`).
- **`input.shape`.** The per-sample shape. Version 1 holds `input.features`
  (`model_config.cpp:43`, `:64`).
- **`from`.** Optional; absent means `-1`. An entry is `-1` (the previous row, or the model
  input at row 0), an absolute row index below the current row, the string `"input"`, or
  `{ "node": 23, "port": "one2one.level0.boxes" }`. Any other negative number is an error.
- **`outputs`.** Optional; absent means the last row. `{ "node": i }` publishes every port of
  that row under the port's own name. `{ "node": i, "port": p, "name": n }` publishes one.
- **`defaults`.** Optional. A map from a type string to keys; a row of that type receives each
  key it does not state. The reader merges them before the factory sees the row, so no factory
  and no seam knows about defaults. This keeps the batch-normalisation constants of YOLO26 out
  of the blocks: a block takes them as arguments (section 2.10).
- **Widths are integers.** A block row states `hidden`, not an expansion ratio, so no width is
  computed in floating point.
- **Upgrade.** `upgrade_model_v1_to_v2` rewrites a version 1 document in memory:
  `sushiai_model: 1` becomes `format_version: 2`, `input.features: F` becomes
  `input.shape: [F]`, and each row gets `from: -1`. `examples/mnist_cnn/model.json` therefore
  loads unchanged, its `unflatten` row included. A version above 2 is refused with both numbers
  named. `require_version` accepts one exact value (`src/config/value.cpp:294-310`), so the
  model reader reads the version itself.

```cpp
// include/SushiAI/config/model_config.hpp
struct SourceSpec { std::int64_t node = -1; std::string port; };      // node -1: previous row
struct OutputSpec { std::int64_t node = 0; std::string port; std::string name; };

struct LayerSpec
{
    std::string type;
    Value attributes;                 // "type" and "from" removed, defaults merged in
    std::vector<SourceSpec> from;     // resolved to absolute rows by the parser
};

struct ModelSpec
{
    Shape input_shape;
    std::vector<LayerSpec> layers;
    std::vector<OutputSpec> outputs;
    std::int64_t input_features = 0;  // input_shape's element count; deleted in wave 7
};
```

### 2.7 Batch normalisation

**The graph surface.**

```cpp
// include/SushiAI/graph/builder.hpp
struct BatchNormResult
{
    ValueId output;           /**< @brief The normalised [N,H,W,C] value. */
    ValueId saved_mean;       /**< @brief The batch mean, [C]. */
    ValueId saved_inv_std;    /**< @brief 1/sqrt(variance + epsilon), [C]. */
    ValueId batch_variance;   /**< @brief The biased batch variance before epsilon, [C]. */
};

struct BatchNormFold { ValueId scale; ValueId shift; };   // both [C]

/** @brief Emits BATCHNORM_FOLD: scale = gamma / sqrt(variance + epsilon),
 *         shift = beta - mean * scale. */
BatchNormFold batchnorm_fold(ValueId gamma, ValueId beta, ValueId mean, ValueId variance,
                             double epsilon) const;
```

- `batchnorm2d` declares a fourth `[C]` output and emits `{y, mean, inv_std, variance}`
  (three today, `src/graph/builder.cpp:980`).
- The rule's guard covers output 3 as well (`src/autograd/rules/norm.cpp:37-38`). The backward
  call still reads outputs 1 and 2 (`norm.cpp:48-49`).
- `BATCHNORM_FOLD` is a new identity in `include/SushiAI/core/ops.hpp`, declared with `op()` as
  `BATCHNORM2D_BACKWARD` is (`ops.hpp:209`), and added to `Ops::ALL` (`ops.hpp:266-285`). It has
  no backward rule, so asking a gradient of gamma through it fails at differentiation. A
  fine-tune on frozen statistics needs that rule and is outside this design.
- `BATCHNORM_UPDATE_STATS` stays an identity with no node (`ops.hpp:215`).

**The layer.**

```cpp
// include/SushiAI/nn/batch_norm2d.hpp   (new, host lane)
namespace SushiAI::NN
{
    /** @brief Holds the two constants of a batch normalisation. */
    struct NormOptions
    {
        double epsilon = 1e-5;
        double momentum = 0.1;
    };

    /** @brief Normalises an NHWC activation per channel. */
    struct BatchNorm2d
    {
        int64_t channels = 0;
        NormOptions options{};

        Parameter weight{};        // gamma, [channels], starts at one
        Parameter bias{};          // beta, [channels], starts at zero
        Buffer running_mean{};     // [channels], starts at zero
        Buffer running_var{};      // [channels], starts at one

        BatchNorm2d() = default;
        BatchNorm2d(int64_t channels, NormOptions options);

        static constexpr auto fields()
        {
            return std::tuple{
                field("weight", &BatchNorm2d::weight),
                field("bias", &BatchNorm2d::bias),
                buffer("running_mean", &BatchNorm2d::running_mean),
                buffer("running_var", &BatchNorm2d::running_var),
            };
        }

        Graph::ValueId trace_training(const Trace& trace, Graph::ValueId x);
        Graph::ValueId trace_evaluation(const Trace& trace, Graph::ValueId x);
        Shape output_shape(const Shape& input) const;
    };

    /** @brief Returns the next running statistic; the host form of one update. */
    double updated_statistic(double running, double batch, double momentum,
                             double source_scale) noexcept;
}
```

```cpp
// src/nn/batch_norm2d.cpp
namespace
{
    /** @brief Holds the leaves both trace forms declare. */
    struct Declared { Graph::ValueId gamma; Graph::ValueId beta; Shape image; };

    /** @brief Returns the factor the batch variance carries into the running variance. */
    double running_variance_scale(int64_t count);     // section "The n/(n-1) question"
}

Declared BatchNorm2d::declare_state(const Graph::GraphBuilder& b, Graph::ValueId x)
{
    const Shape image = b.graph().value(x).shape;     // by value: declaring grows the table
    const DType dtype = b.graph().value(x).dtype;
    SA_THROW_IF(image.rank() != 4 || image[3] != channels,
                "BatchNorm2d({}) needs a rank-4 [N, H, W, {}] input; got {}",
                channels, channels, image.to_string());

    const Shape per_channel{channels};
    const Graph::ValueId gamma = weight.declare(b, per_channel, ParameterInit::constant_value(1.0),
                                                channels, channels, "weight", dtype);
    const Graph::ValueId beta = bias.declare(b, per_channel, ParameterInit::constant_value(0.0),
                                             channels, channels, "bias", dtype);
    running_mean.define(per_channel, 0.0, dtype);
    running_var.define(per_channel, 1.0, dtype);
    return Declared{gamma, beta, image};
}

Graph::ValueId BatchNorm2d::trace_training(const Trace& trace, Graph::ValueId x)
{
    const Graph::GraphBuilder& b = trace.builder();
    const Declared state = declare_state(b, x);

    const int64_t count = state.image[0] * state.image[1] * state.image[2];
    SA_THROW_IF(count < 2,
                "BatchNorm2d({}) needs more than one value per channel to train; got {}",
                channels, count);

    const Graph::BatchNormResult r = b.batchnorm2d(x, state.gamma, state.beta, options.epsilon);
    trace.update(running_mean, r.saved_mean, options.momentum, 1.0);
    trace.update(running_var, r.batch_variance, options.momentum,
                 running_variance_scale(count));
    return r.output;
}

Graph::ValueId BatchNorm2d::trace_evaluation(const Trace& trace, Graph::ValueId x)
{
    const Graph::GraphBuilder& b = trace.builder();
    const Declared state = declare_state(b, x);

    const Graph::BatchNormFold folded = b.batchnorm_fold(
        state.gamma, state.beta, running_mean.read(b, "running_mean"),
        running_var.read(b, "running_var"), options.epsilon);
    return b.channel_affine(x, folded.scale, folded.shift);
}
```

`declare_state` is a private member; it copies the shape and the type before it declares, as
`Conv2d::forward` does (`src/nn/conv2d.cpp:76-78`), because `add_value` pushes into the vector a
`const Value&` would point into (`include/SushiAI/graph/ir.hpp:222`, `:437`).

**Element types under mixed precision.** A model is traced in its master type and the cast pass
narrows afterwards (`trainer.hpp:143`, `:187`). Four facts follow from
`src/graph/cast_insertion.cpp`:

- Gamma, beta and both buffers keep the master type, because the pass pins every persistent
  value (`cast_insertion.cpp:206-208`) and `is_persistent` gains `BUFFER`.
- `GraphBuilder::batchnorm2d` requires gamma and beta to share the input's type
  (`builder.cpp:963-966`). That holds at trace time, when everything is in the master type. The
  pass re-adds nodes directly (`cast_insertion.cpp:315`) and does not go through that check.
- The two update sources are in the observed list, so the pass keeps them in the master type
  and widens into them (`cast_insertion.cpp:210-216`, `:287-318`). A momentum-0.03 update
  therefore accumulates in fp32 even when the node computes in fp16.
- `BATCHNORM_FOLD` must stay in the master type: it takes a square root of a variance plus
  1e-3. `MixedPrecisionPolicy::choose` keeps the declared type for an operation that
  `narrows_under_mixed_precision` does not name (`src/graph/precision.cpp:87-90`). That
  function's body was not read; the fold must be outside it.

**The n/(n-1) question.** The node outputs the biased variance, which is what its own
normalisation uses (`lowering.cpp:1286-1289`). Whether the running variance takes the unbiased
form is a property of the update: it is `source_scale`, one coefficient, and costs no task. The
factor is `running_variance_scale(count)`, one function in `batch_norm2d.cpp` that no header
exposes, so the answer is one function body and not a field of the layer or a key of the file.

The fixture as exported cannot give the answer. `loss.sacp` holds the running statistics of one
layer, `model.0.bn` (`MANIFEST.json:10721`, `:10736`), after one pass over a `[1, 640, 640, 3]`
image (`tools/reference/README.md:44`). That layer sees n = 320 * 320 = 102400 values per
channel, so n/(n-1) - 1 is 9.8e-6, and at momentum 0.03 (`MANIFEST.json:160`) the two candidates
differ by 2.9e-7 of the batch variance. The stored values lie between 0.59 and 1.45
(`MANIFEST.json:10746-10747`), where float32 spacing is 6e-8 to 1.2e-7. The comparison the
YOLO26 spec relies on (`docs/agent/2026_10_07_YOLO26/SPEC.md:474`) cannot separate the two.

Two additions to the exporter settle it under the references ruling (decision 8):

- A unit case: one `torch.nn.BatchNorm2d(3)` over an input written as `[2, 2, 2, 3]` NHWC
  (torch receives `[2, 3, 2, 2]`), with the input, both running vectors before and after one
  training pass, and the output. There n is 8 and the factor is 8/7.
- `model.9.cv1.bn` added to `loss.sacp`. It sits at stride 32, so n is 400 and the two
  candidates differ by 7.5e-5 of the batch variance, inside the real model.

Until that run exists the body of `running_variance_scale` is unverified. The design writes it
as `count / (count - 1)`, from memory of torch, and the test that pins it reads the exporter's
case, so the unit is not accepted before the fixture exists.

**The update recorder.**

```cpp
// include/SushiAI/graph/step_recorder.hpp   (addition, host lane)
/** @brief Runs several recorders as one, in the order they were added. */
class StepRecorderList final : public IStepRecorder
{
    public:
        void add(IStepRecorder& recorder);
        bool empty() const noexcept;
        std::size_t record_into_step() override;
        void before_replay() override;
        void after_replay() override;
};

// include/SushiAI/train/buffer_update_recorder.hpp   (new, engine lane)
/** @brief Records each BufferUpdate as device work behind the step's graph. */
class BufferUpdateRecorder final : public Graph::IStepRecorder
{
    public:
        BufferUpdateRecorder(SushiBLAS::Engine& engine,
                             SushiRuntime::span<const NN::BufferUpdate> updates,
                             const Graph::TensorPool& step_pool, NN::ModelState& state);
        std::size_t record_into_step() override;
};
```

`CompiledStep` keeps its one `IStepRecorder*` (`include/SushiAI/graph/compiled_step.hpp:60-62`)
and records it after the lowering (`src/graph/compiled_step.cpp:39-47`), so the recorder reads
what `BATCHNORM2D` wrote. It receives the list: buffer updates first, then the optimizer when it
is in the plan. Each update is `scal(1 - m, running)` then `axpy(m * source_scale, batch,
running)`; both scalars are fixed at record time, so the work is replay-invariant. The lowering
already uses both calls (`lowering.cpp:1279`, `:1292`).

Two things change in `Trainer` because a second recorder exists. Section 3.2 shows the first.

- `Trainer::run_step` steps a `PER_STEP` optimizer only when `!step_.has_trailing_work()`
  (`src/train/trainer.cpp:272`), and that is `trailing_ != nullptr` (`compiled_step.hpp:75`).
  With a statistics recorder attached the optimizer never steps and nothing reports it.
  `Trainer` holds its own flag.
- A training step fails to construct when its trace holds updates and its recorder list holds no
  `BufferUpdateRecorder`. `TrainingStep` builds that recorder itself from the trace
  (section 2.8), so the check guards the hand-assembled path.

### 2.8 Steps, storage and what a checkpoint writes

```cpp
// include/SushiAI/nn/model_state.hpp   (new, engine lane)
namespace SushiAI::NN
{
    enum class StateKind : std::uint8_t { Parameter, Buffer };

    /** @brief Describes one tensor a ModelState owns. */
    struct StateEntry
    {
        std::string name;
        StateKind kind;
        Graph::StateId state;
        Shape shape;
        DType dtype;
        ParameterInit init;        // a buffer's initial value is a Constant init
        int64_t fan_in;
        int64_t fan_out;
    };

    /** @brief Owns one tensor per declared parameter and defined buffer of a model. */
    class ModelState
    {
        public:
            /** @brief Walks @p model once; fails on two entries with one name. */
            template<typename M>
            static Result<ModelState> create(SushiBLAS::Engine& engine, M& model,
                                             std::string_view prefix = {});

            /** @brief Binds every state leaf of @p graph into @p pool, sharing storage;
             *         fails on a leaf whose state this object does not hold. */
            Status attach(const Graph::Graph& graph, Graph::TensorPool& pool) const;

            SushiRuntime::span<const StateEntry> entries() const noexcept;   // walk order
            Result<SushiBLAS::Tensor*> tensor(std::string_view name);        // hashed lookup
            Result<SushiBLAS::Tensor*> tensor(Graph::StateId state);
    };
}

// include/SushiAI/train/steps.hpp   (new, engine lane)
namespace SushiAI::Train
{
    /** @brief Holds one traced graph and what a step reads from it. */
    struct TracedStep
    {
        Graph::Graph graph;
        Graph::ValueId input = Graph::NO_VALUE;
        NN::ModelOutputs outputs;
        NN::BufferUpdateList updates;          // empty unless traced for training
        Graph::CastReport casts;
        Graph::FusionReport fusion;
    };

    template<typename M>
    Result<TrainingGraph> build_training_graph(M& model, const IObjective& objective,
                                               const Shape& batch_shape, GraphOptions options = {});
    template<typename M>
    Result<TracedStep> build_probe_graph(M& model, const Shape& batch_shape,
                                         GraphOptions options = {});
    template<typename M>
    Result<TracedStep> build_evaluation_graph(M& model, const Shape& batch_shape,
                                              GraphOptions options = {});

    class TrainingStep;      // CompiledStep, its pool, its StepRecorderList
    class EvaluationStep;    // CompiledStep and its pool, no recorder
}
```

`TrainingGraph` (`trainer.hpp:96-122`) gains `NN::ModelOutputs outputs` and
`NN::BufferUpdateList updates`, and its `ParameterBinding` list (`trainer.hpp:80-90`) is filled
from `parameter.id_in(built.graph)`. Each of the three builders adds
`updates.sources()` to the observed list, which the cast pass and the fusion pass both read
(`trainer.hpp:176-185`), and to `PlanOptions::pinned`
(`include/SushiAI/graph/memory_plan.hpp:119`, `src/train/trainer.cpp:46-51`). The sources are
`ACTIVATION` values (`builder.cpp:973-976`) that the recorder reads after the whole graph has
run; without the pin the planner may reuse their buffers, and the statistics are then wrong with
no error. One function, `Train::retain(updates, observed, pinned)`, does both.

**Training and evaluation as two compiled steps (CV-8).** One model instance is traced up to
three times: training, the forward-only probe, evaluation. Each trace has its own graph, its own
pool and its own `CompiledStep`, because a pool adopts one plan (`lowering.cpp:48-52`) and is
keyed by `ValueId` (`lowering.cpp:72-77`). `ModelState` owns the state tensors beside all three
pools. `attach` calls `TensorPool::bind` (`lowering.cpp:65-70`) for each pair in
`graph.state_leaves()` before `materialise_leaves`, which keeps a tensor that is already bound
(`lowering.cpp:98-99`). A copied `Tensor` shares storage: `storage` is a `sushi_ptr<Storage>`
(`sushiblas/include/SushiBLAS/tensor.hpp:98`), and `materialise` already aliases by copy
(`lowering.cpp:107-108`).

No pool owns state, so no pool has to outlive another, and the training graph needs no buffer
leaf. Whether two compiled plans on one engine may alias storage and replay correctly is
unverified (section 10). If they may not, the fallback is that each step's pool owns its state
leaves and `ModelState` copies in before a step of another kind runs: 594 device copies per
switch for YOLO26n, and the same interface.

**Initial values.** `materialise` returns unset storage (`include/SushiAI/graph/lowering.hpp:126`
per the inventory; `lowering.cpp:121-122`). `NN::initialize(engine, state, policy)` fills each
entry from its `ParameterInit`: a `Constant` entry through the constant fill, a `FanScaled`
entry through the family the policy names. `zero_final_layer` zeroes the last `FanScaled`
entry and nothing after it (section 3.6). The head biases of YOLO26 are `Constant` entries:
`Conv2d` gains `double bias_init = 0.0`, which its `forward` passes as
`ParameterInit::constant_value(bias_init)`.

**Checkpoints.** `save_checkpoint(path, state, engine)` writes every entry in walk order: one
record per parameter and per buffer, in one namespace. SACP stays at version 1, because a record
is a name, a type, extents and a payload and carries no kind
(`tools/reference/README.md:84-93`). The model's walk says which name is a buffer. Folded scale
and shift, batch statistics and gradients are never written. `load_checkpoint(path, state)`
stays strict both ways (`checkpoint.hpp:157-183`) and needs no graph and no pool. Two changes
come with 594 records:

- The lookup of a record's name is hashed. Today it scans every bound name per record
  (`checkpoint.hpp:153-155`).
- Two entries with one name fail when the `ModelState` is created. Today only a name that
  appears twice in the file is caught (`checkpoint.hpp:160`).

A checkpoint written before a model had buffers fails to load into it with "does not supply
this model's '...running_mean'" (`checkpoint.hpp:180-183`). That is the wanted behaviour.

**The weight importer.**

```cpp
// include/SushiAI/io/import_weights.hpp   (new, engine lane)
/** @brief Rewrites a record name that starts with from_prefix. */
struct RenameRule { std::string_view from_prefix; std::string_view to_prefix; };

/** @brief Loads a foreign SACP file into a state after renaming each record; strict. */
Status import_weights(const std::filesystem::path& path, NN::ModelState& state,
                      SushiRuntime::span<const RenameRule> rules);
```

`weights.sacp` is already SACP version 1 with Ultralytics names and SushiAI layouts
(`tools/reference/README.md:43`, `:53-55`), and `num_batches_tracked` is not written
(`README.md:62-63`). For YOLO26 the table is one rule, `{"model.", ""}`, and the load is the
strict load above. The layout transform is the exporter's three rows (`README.md:53-55`); the
importer checks its result through the per-record shape comparison. A weight file in torch
layout needs those three permutations as a second table keyed by rank, which this fixture does
not use and this design does not declare.

### 2.9 The rank-4 input, Unflatten, and several outputs

- **Input leaf.** The three builders take `batch_shape`, for example
  `Shape{16, 640, 640, 3}`, and declare the input leaf with it. `GraphBuilder::input` takes any
  shape (`builder.cpp:134-139`) and a new value is tagged NHWC (`ir.hpp:222`). `build_graph`'s
  `int64_t batch, int64_t features` overload (`trainer.hpp:133-143`) forwards
  `Shape{batch, features}`, is marked deprecated, and is deleted in wave 7.
- **Dataset and feed.** `Dataset` gains `Shape sample_shape()` beside `input_dim()`
  (`include/SushiAI/data/dataset.hpp:82`). The feed compares shapes; today it compares element
  counts (`trainer.cpp:251-256`).
- **Trainer.** `rows_` and `classes_` come from `logits.dim(0)` and `dim(1)`
  (`trainer.cpp:188-190`). A rank-4 output would put H there with no error. The objective owns
  the class check and the metric.
- **`Unflatten`** stays a layer and a registry row (`layer_registry.hpp:187`). An upgraded
  version 1 file has `input.shape: [784]` and still needs it. A version 2 file with
  `"shape": [28, 28, 1]` does not use it; `examples/mnist_cnn/model.json` moves to that form in
  wave 7, once `Dataset` yields rank 4.
- **Outputs.** `IObjective::loss` takes `const NN::ModelOutputs& outputs` where it takes one
  `ValueId` today, and keeps its `const Graph::Graph&` parameter
  (`include/SushiAI/train/objective.hpp:138-141`). `CrossEntropyObjective` calls
  `outputs.only()`. `ModelOutputs` lives under `nn/` because `objective.hpp` includes the engine
  (`objective.hpp:38`); the signature change itself is engine lane. `IObjective` keeps its name
  until the naming wave.
- **The head.** `DetectHead` is a `Node` with three inputs and twelve ports named as the fixture
  names them: `one2many.level<l>.boxes`, `one2many.level<l>.scores`,
  `one2one.level<l>.boxes` and `one2one.level<l>.scores` for l in 0 to 2
  (`MANIFEST.json:9968-10237`). The detection objective asks `outputs.at(name)` for each, and a
  wrong name fails with the list of names that exist. The probe graph publishes the same twelve
  for the host assigner, and the evaluation graph publishes them for both validation heads
  (`PROGRAMME.md:35`). One name table serves the objective and the parity tests.

### 2.10 The YOLO26 blocks

The blocks live in `include/SushiAI/nn/blocks/`, one header each, in namespace `NN`. `Conv`,
`Bottleneck`, `C3k2` and `SPPF` are shown whole; bodies are in today's throwing form.

```cpp
// nn/silu.hpp, nn/identity.hpp
struct SiLU
{
    static constexpr auto fields() { return std::tuple<>{}; }
    Graph::ValueId forward(const Graph::GraphBuilder& b, Graph::ValueId x) const { return b.silu(x); }
    Shape output_shape(const Shape& input) const { return input; }
};

struct Identity
{
    static constexpr auto fields() { return std::tuple<>{}; }
    Graph::ValueId forward(const Graph::GraphBuilder&, Graph::ValueId x) const { return x; }
    Shape output_shape(const Shape& input) const { return input; }
};

// nn/conv2d.hpp: one new static
/** @brief Builds a bias-free convolution padded by kernel / 2 on every side. */
static Conv2d same_padded(int64_t in_channels, int64_t out_channels,
                          int64_t kernel, int64_t stride);

// nn/blocks/conv_norm.hpp
/** @brief Applies a bias-free convolution, a batch normalisation and an activation. */
template<typename Convolution, typename Activation>
struct ConvNorm
{
    Convolution conv;
    BatchNorm2d bn;
    Activation act{};

    ConvNorm(int64_t in_channels, int64_t out_channels, int64_t kernel, int64_t stride,
             NormOptions norm)
        : conv(Convolution::same_padded(in_channels, out_channels, kernel, stride)),
          bn(out_channels, norm)
    {
    }

    static constexpr auto fields()
    {
        return std::tuple{child("conv", &ConvNorm::conv), child("bn", &ConvNorm::bn)};
    }

    Graph::ValueId forward(const Trace& trace, Graph::ValueId x)
    {
        x = trace_layer(conv, trace, x);
        x = trace_layer(bn, trace, x);
        return trace_layer(act, trace, x);
    }

    Shape output_shape(const Shape& input) const { return conv.output_shape(input); }
};

using Conv = ConvNorm<Conv2d, SiLU>;
using ConvLinear = ConvNorm<Conv2d, Identity>;
using DWConv = ConvNorm<DepthwiseConv2d, SiLU>;
using DWConvLinear = ConvNorm<DepthwiseConv2d, Identity>;

// nn/blocks/bottleneck.hpp
/** @brief Applies two convolutions and adds the input when the widths allow it. */
struct Bottleneck
{
    Conv cv1;
    Conv cv2;
    bool adds_input = false;

    Bottleneck(int64_t in_channels, int64_t out_channels, int64_t hidden_channels,
               bool shortcut, NormOptions norm, int64_t kernel1 = 3, int64_t kernel2 = 3)
        : cv1(in_channels, hidden_channels, kernel1, 1, norm),
          cv2(hidden_channels, out_channels, kernel2, 1, norm),
          adds_input(shortcut && in_channels == out_channels)
    {
    }

    static constexpr auto fields()
    {
        return std::tuple{child("cv1", &Bottleneck::cv1), child("cv2", &Bottleneck::cv2)};
    }

    Graph::ValueId forward(const Trace& trace, Graph::ValueId x)
    {
        const Graph::ValueId y = cv2.forward(trace, cv1.forward(trace, x));
        return adds_input ? trace.builder().add(x, y) : y;
    }

    Shape output_shape(const Shape& input) const
    {
        return cv2.output_shape(cv1.output_shape(input));
    }
};

// nn/blocks/c3k2.hpp
/** @brief Splits a stem in two, chains inner blocks off the second half, joins all parts. */
template<typename Inner>
struct C3k2
{
    int64_t hidden = 0;
    Conv cv1;
    Conv cv2;
    ModuleList<Inner> m;

    /** @param inner The inner blocks, each mapping hidden channels to hidden channels. */
    C3k2(int64_t in_channels, int64_t out_channels, int64_t hidden_channels,
         ModuleList<Inner> inner, NormOptions norm)
        : hidden(hidden_channels),
          cv1(in_channels, 2 * hidden_channels, 1, 1, norm),
          cv2((2 + static_cast<int64_t>(inner.size())) * hidden_channels, out_channels, 1, 1, norm),
          m(std::move(inner))
    {
    }

    static constexpr auto fields()
    {
        return std::tuple{child("cv1", &C3k2::cv1), child("cv2", &C3k2::cv2),
                          child("m", &C3k2::m)};
    }

    Graph::ValueId forward(const Trace& trace, Graph::ValueId x)
    {
        const Graph::GraphBuilder& b = trace.builder();
        const Graph::ValueId stem = cv1.forward(trace, x);

        std::vector<Graph::ValueId> parts{b.slice_channels(stem, 0, hidden),
                                          b.slice_channels(stem, hidden, hidden)};
        for (Inner& block : m) parts.push_back(trace_layer(block, trace, parts.back()));

        return cv2.forward(trace, b.concat_channels(
            SushiRuntime::span<const Graph::ValueId>(parts.data(), parts.size())));
    }

    Shape output_shape(const Shape& input) const
    {
        const Shape stem = cv1.output_shape(input);            // checks [H, W, in_channels]
        return cv2.output_shape(Shape{stem[0], stem[1], cv2.conv.in_channels});
    }
};

// nn/blocks/sppf.hpp
/** @brief Pools a narrowed activation repeatedly and joins every stage. */
struct SPPF
{
    ConvLinear cv1;
    Conv cv2;
    MaxPool2d pool;
    int64_t pools = 3;
    bool adds_input = false;

    SPPF(int64_t in_channels, int64_t out_channels, int64_t kernel, int64_t pool_count,
         bool shortcut, NormOptions norm)
        : cv1(in_channels, in_channels / 2, 1, 1, norm),
          cv2((in_channels / 2) * (pool_count + 1), out_channels, 1, 1, norm),
          pool{Graph::ConvGeometry{kernel, kernel, 1, 1, 1, 1, kernel / 2, kernel / 2}},
          pools(pool_count),
          adds_input(shortcut && in_channels == out_channels)
    {
    }

    static constexpr auto fields()
    {
        return std::tuple{child("cv1", &SPPF::cv1), child("cv2", &SPPF::cv2)};
    }

    Graph::ValueId forward(const Trace& trace, Graph::ValueId x)
    {
        const Graph::GraphBuilder& b = trace.builder();

        std::vector<Graph::ValueId> parts{cv1.forward(trace, x)};
        for (int64_t i = 0; i < pools; ++i) parts.push_back(pool.forward(b, parts.back()));

        const Graph::ValueId y = cv2.forward(trace, b.concat_channels(
            SushiRuntime::span<const Graph::ValueId>(parts.data(), parts.size())));
        return adds_input ? b.add(y, x) : y;
    }

    Shape output_shape(const Shape& input) const
    {
        const Shape narrowed = cv1.output_shape(input);        // checks [H, W, in_channels]
        return cv2.output_shape(Shape{narrowed[0], narrowed[1], cv2.conv.in_channels});
    }
};
```

Facts the bodies rest on:

- `Conv2d::output_shape` rejects a per-sample shape whose third extent is not `in_channels`
  (`src/nn/conv2d.cpp:51`, reached from `:109`). `C3k2::output_shape` and `SPPF::output_shape`
  therefore build the `[H, W, cv2.conv.in_channels]` shape that `cv2` expects; passing the
  block's own input shape to `cv2` throws, and `ModelGraph::add` calls `output_shape` when a row
  is built.
- The geometry's top and left pads apply to both sides of an axis
  (`include/SushiAI/nn/conv2d.hpp:50`, `include/SushiAI/nn/max_pool2d.hpp:43`), so
  `kernel / 2` is the symmetric padding. The brace list in `SPPF` follows the field order at
  `include/SushiAI/graph/builder.hpp:48-55`. A default `ConvGeometry` has kernel 1
  (`builder.hpp:48-49`), so a pool whose geometry is not set is the identity and traces without
  an error; the constructor sets it.
- `concat_channels` requires equal N, H and W (`builder.cpp:865-870`), and the pool with kernel
  5, stride 1 and pad 2 keeps H and W.
- `MaxPool2d::forward` is `const` and takes the builder (`max_pool2d.hpp:57`).
- The shapes match the fixture: row 2 is `C3k2<Bottleneck>(32, 64, 16, ...)` with `cv1`
  32 to 32, `cv2` 48 to 64 and an inner `Bottleneck(16, 16, 8, ...)`
  (`model.2.m.0.cv1.conv.weight` is `[8, 3, 3, 16]`); row 9 is `SPPF(256, 256, 5, 3, ...)` with
  `cv1` 256 to 128 and `cv2` 512 to 256.

A call site for row 2:

```cpp
NN::ModuleList<NN::Bottleneck> inner;
inner.push_back(NN::Bottleneck(16, 16, 8, false, norm));
NN::C3k2<NN::Bottleneck> row2(32, 64, 16, std::move(inner), norm);
```

`C3k2` receives its inner blocks; it does not choose them. `make_c3k2_node` reads `"inner"` and
looks it up in a constant table of three rows, each a function that builds a
`ModuleNode<C3k2<Inner>>` for one `Inner`:

```cpp
// nn/blocks/c3k2.hpp
struct C3k2InnerEntry { std::string_view name; NodeFactory make; };

inline constexpr std::array<C3k2InnerEntry, 3> K_C3K2_INNER = {
    C3k2InnerEntry{"bottleneck",     &make_c3k2_of_bottleneck},       // rows 2, 4
    C3k2InnerEntry{"c3k",            &make_c3k2_of_c3k},              // rows 6, 8, 13, 16, 19
    C3k2InnerEntry{"bottleneck_psa", &make_c3k2_of_bottleneck_psa},   // row 22
};
```

A new inner block is a row there and a class, with one `c3k2` row in `K_NODE_REGISTRY` and no
branch over inner kinds.

The rest, as signatures:

```cpp
struct C3k                  // fields: cv1, cv2, cv3, m (ModuleList<Bottleneck>)
{
    C3k(int64_t in_channels, int64_t out_channels, int64_t repeats, bool shortcut,
        NormOptions norm);                       // hidden is out_channels / 2
    Graph::ValueId forward(const Trace& trace, Graph::ValueId x);
    Shape output_shape(const Shape& input) const;
};

struct DepthwiseConv2d      // fields: weight, [kernel, kernel, channels]
{
    static DepthwiseConv2d same_padded(int64_t in_channels, int64_t out_channels,
                                       int64_t kernel, int64_t stride);   // in != out fails
    Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x);
    Shape output_shape(const Shape& input) const;
};

struct Attention            // fields: qkv (ConvLinear), proj (ConvLinear), pe (DWConvLinear)
{
    Attention(int64_t channels, int64_t heads, NormOptions norm);
    Graph::ValueId forward(const Trace& trace, Graph::ValueId x);
};

struct PSABlock             // fields: attn (Attention), ffn (Sequential<Conv, ConvLinear>)
{
    PSABlock(int64_t channels, int64_t heads, NormOptions norm);
    Graph::ValueId forward(const Trace& trace, Graph::ValueId x);
};

struct C2PSA                // fields: cv1, cv2, m (ModuleList<PSABlock>)
{
    C2PSA(int64_t in_channels, int64_t out_channels, int64_t hidden_channels,
          int64_t repeats, int64_t heads, NormOptions norm);
    Graph::ValueId forward(const Trace& trace, Graph::ValueId x);
};

class Concat final : public Node       // n inputs, port "output"
{
    public:
        explicit Concat(ShapeSpan inputs);         // unequal H or W fails
};

using BoxBranch = Sequential<Conv, Conv, Conv2d>;
using ClassBranch = Sequential<Sequential<DWConv, Conv>, Sequential<DWConv, Conv>, Conv2d>;

class DetectHead final : public Node   // 3 inputs, the 12 ports of section 2.9
{
    public:
        DetectHead(ShapeSpan levels, int64_t classes, NormOptions norm);
        // state: cv2, cv3, one2one_cv2, one2one_cv3, each a ModuleList over the levels
};
```

`Attention`, `PSABlock`, `C2PSA`, `DepthwiseConv2d` and `DetectHead` wait for builder emitters
that do not exist: a depthwise convolution (`ConvGeometry` has no group count,
`builder.hpp:46-56`), a rank-3 matrix product, a differentiable softmax, and a stop-gradient (no
`STOP_GRADIENT` in `ops.hpp`). They need the emitters, not the kernels, to be traced, and a
parameter is named only after a trace has declared it (`module.hpp:275`). Rows 10, 22 and 23 of
the model file wait with them.

### 2.11 The names YOLO26's weights get

The walk of the file-built graph under the prefix `model` gives the names below. Row indices
come from `ModelGraph`, struct members from `child`, list positions from `ModuleList` and
`Sequential`.

| Owner | Segments it adds | Example record in `MANIFEST.json` |
| --- | --- | --- |
| `ModelGraph` row | `<row>` | `model.0.conv.weight` |
| `ConvNorm` | `conv`, `bn` | `model.0.bn.running_var` |
| `Conv2d` (`conv2d.hpp:63-69`) | `weight`, `bias` | `model.23.cv2.0.2.weight`, `[4, 1, 1, 16]` |
| `BatchNorm2d` | parameters `weight`, `bias`; buffers `running_mean`, `running_var` | `model.9.cv1.bn.running_mean` (`:2731`) |
| `Bottleneck`, `SPPF` | `cv1`, `cv2` | `model.9.cv2.conv.weight`, `[256, 1, 1, 512]` |
| `C3k2<Bottleneck>` | `cv1`, `cv2`, `m.<i>` | `model.2.m.0.cv2.conv.weight`, `[16, 3, 3, 8]` |
| `C3k` inside `C3k2` | `m.<i>.cv1`, `.cv2`, `.cv3`, `.m.<j>` | `model.6.m.0.m.1.cv2.conv.weight`, `[32, 3, 3, 32]` |
| `Sequential<Bottleneck, PSABlock>` inside `C3k2` | `m.<i>.0`, `m.<i>.1` | `model.22.m.0.1.attn.qkv.conv.weight`, `[256, 1, 1, 128]` |
| `Attention` | `qkv`, `proj`, `pe` | `model.10.m.0.attn.pe.conv.weight`, `[3, 3, 128]` |
| `PSABlock` | `attn`, `ffn.<0 or 1>` | `model.10.m.0.ffn.1.conv.weight`, `[128, 1, 1, 256]` |
| `DetectHead` box branch | `cv2.<level>.<0..2>` | `model.23.cv2.0.1.conv.weight`, `[16, 3, 3, 16]` |
| `DetectHead` class branch | `cv3.<level>.<0..1>.<0..1>`, `cv3.<level>.2` | `model.23.cv3.0.0.0.conv.weight`, `[3, 3, 64]`; `model.23.cv3.0.2.weight`, `[80, 1, 1, 80]` |
| One-to-one copies | `one2one_cv2...`, `one2one_cv3...` | mirror the two rows above |

The fixture holds 594 records under `model.`, with a longest name of 42 bytes: 366 parameters
(112 rank-4 convolution weights, 14 rank-3 depthwise weights, 12 head biases, 114 gammas, 114
betas) and 228 buffers. These counts are a recount of the manifest. They fit SACP's limits of
4096 records and 256-byte names (`include/SushiAI/io/checkpoint_format.hpp:60`, `:63`). Rows 11,
12, 14, 15, 18 and 21 hold no state. A SushiAI checkpoint of the same model has the same names
without `model.`, which is why the importer's rename table has one row.

### 2.12 What is public

The ruling at `PROGRAMME.md:25` names the eager surface. From this design a caller writes inside
an eager step: `step.trace()`, `model.forward(trace, x)`, and a layer's own members. Everything
under `include/` and every file format is public from the 0.1.0 tag. The surfaces whose shape
this design fixes:

| Surface | Why it cannot move after it ships |
| --- | --- |
| `Trace`, its two factories, `TraceMode`, `discard_updates` | Every modal layer and every composite takes it |
| The three layer forms, `fields()`, `field`, `buffer`, `child`, `ModuleList` | Every layer and block lists itself with them |
| `Parameter::declare`, `id_in`; `Buffer::define`, `read` | Every layer body |
| `walk_state`, `StateVisitor`, `for_each_parameter`, `for_each_buffer` | Checkpoint, initialiser and optimizer binding read state through them |
| `Node`, `Port`, `NodeFactory`, `NodeEntry` | Every node class and factory is written against them |
| `ModelGraph`, `NodeRef`, `ModelOutputs`, `build_model` | Hand-written models and objectives |
| `BATCHNORM2D`'s four outputs, `BATCHNORM_FOLD`'s two, `Lifetime::BUFFER`, `StateId` | The rule table, the lowering and the eager region signature read the IR |
| `IObjective::loss` over `ModelOutputs`; the three `build_*_graph` functions; `ModelState` | Every objective and driver |
| Model file version 2: `format_version`, `input.shape`, `defaults`, `from`, `outputs`, the type strings, each row's keys, the port names | Files on disk |
| Checkpoint names: row index or `add` name, member name, list index, joined by `.`; `weight`, `bias`, `running_mean`, `running_var`; that a checkpoint holds parameters and buffers | Files on disk. A renamed member orphans every saved model and breaks the importer |

Not promised: `ModuleNode`, `LayerNode`, `make_node`, `StepRecorderList`, `BufferUpdateRecorder`,
`BufferUpdateList`'s storage, `StateIdentity`, `K_C3K2_INNER`'s functions, and `Detail`.

## 3. Before and after of real functions

### 3.1 `Parameter::declare`

Before (`include/SushiAI/nn/module.hpp:76-93`): the stored id is returned whatever graph the
builder wraps.

```cpp
if (id_ != Graph::NO_VALUE) return id_;

shape_ = shape;
role_ = role;
fan_in_ = fan_in;
fan_out_ = fan_out;
dtype_ = dtype;
id_ = builder.parameter(shape, debug_name, dtype);
return id_;
```

After:

```cpp
Graph::Graph& graph = builder.graph();
const Graph::ValueId held = graph.state_leaf(identity_.peek());
if (held != Graph::NO_VALUE) return held;

if (described_)
{
    SA_THROW_IF(shape != shape_ || dtype != dtype_,
                "Parameter '{}' was declared as {} {} and is now traced as {} {}",
                debug_name, shape_.to_string(), dtype_name(dtype_),
                shape.to_string(), dtype_name(dtype));
}
else
{
    shape_ = shape;
    init_ = init;
    fan_in_ = fan_in;
    fan_out_ = fan_out;
    dtype_ = dtype;
    described_ = true;
}

id_ = builder.parameter(shape, debug_name, dtype);      // kept for the deprecated id()
graph.bind_state_leaf(identity_.get(), id_);
return id_;
```

### 3.2 `Trainer::run_step` and `optimizer_is_in_plan`

Before (`src/train/trainer.cpp:272`, `include/SushiAI/train/trainer.hpp:374`):

```cpp
if (!step_.has_trailing_work()) optimizer_->step();

bool optimizer_is_in_plan() const noexcept { return step_.has_trailing_work(); }
```

After, with `optimizer_in_plan_` set in the constructor from what `placement_recorder` returns
(`trainer.cpp:126-140`):

```cpp
if (!optimizer_in_plan_) optimizer_->step();

bool optimizer_is_in_plan() const noexcept { return optimizer_in_plan_; }
```

### 3.3 `GraphBuilder::batchnorm2d`

Before (`src/graph/builder.cpp:947`, `:969-981`): `in` is a reference into the value table, and
it is read after `declare_like` has pushed into that table.

```cpp
const Value& in = graph_->value(x);
// ...
const Shape per_channel({in.shape[3]});
const ValueId y = declare_like(x, in.shape, in.dtype, Lifetime::ACTIVATION, "bn");
const ValueId mean = declare_like(x, per_channel, in.dtype, Lifetime::ACTIVATION,
                                  "bn_saved_mean");
const ValueId inv_std = declare_like(x, per_channel, in.dtype, Lifetime::ACTIVATION,
                                     "bn_saved_inv_std");

Attributes a{};
a.values[0] = epsilon;
graph_->add_node(Ops::BATCHNORM2D.id, {x, gamma, beta}, {y, mean, inv_std}, a);
return BatchNormResult{y, mean, inv_std};
```

After: the shape and the type are copied once the checks have passed, and the fourth output is
declared.

```cpp
const Shape image = graph_->value(x).shape;       // by value: declaring grows the table
const DType dtype = graph_->value(x).dtype;
const Shape per_channel({image[3]});

const ValueId y = declare_like(x, image, dtype, Lifetime::ACTIVATION, "bn");
const ValueId mean = declare_like(x, per_channel, dtype, Lifetime::ACTIVATION,
                                  "bn_saved_mean");
const ValueId inv_std = declare_like(x, per_channel, dtype, Lifetime::ACTIVATION,
                                     "bn_saved_inv_std");
const ValueId variance = declare_like(x, per_channel, dtype, Lifetime::ACTIVATION,
                                      "bn_batch_variance");

Attributes a{};
a.values[0] = epsilon;
graph_->add_node(Ops::BATCHNORM2D.id, {x, gamma, beta}, {y, mean, inv_std, variance}, a);
return BatchNormResult{y, mean, inv_std, variance};
```

### 3.4 The rebuild in `fuse` and in `insert_casts`

Before (`src/graph/fusion.cpp:585-601`; `src/graph/cast_insertion.cpp:231-247` has the same
loop with `final_dtype[id]`):

```cpp
Graph rebuilt;

for (ValueId id = 0; id < static_cast<ValueId>(graph.value_count()); ++id)
{
    const Value& value = graph.value(id);
    const ValueId fresh = rebuilt.add_value(value.shape, value.dtype,
                                            value.lifetime, value.debug_name);
    SA_THROW_IF(fresh != id, /* ... */);
    rebuilt.value(fresh).requires_grad = value.requires_grad;
    rebuilt.value(fresh).layout = value.layout;
}
```

After, in `fuse`:

```cpp
Graph rebuilt = graph.without_nodes();
```

After, in `insert_casts`:

```cpp
Graph rebuilt = graph.without_nodes();
for (ValueId id = 0; id < static_cast<ValueId>(value_count); ++id)
    rebuilt.value(id).dtype = final_dtype[id];
```

`without_nodes` copies every value with its producer reset to `NO_NODE`, and the state-leaf
table. A field added to `Value` later is carried by both passes without an edit to either.

### 3.5 `build_layer_stack`

Before (`src/nn/layer_registry.cpp:401-416`):

```cpp
Shape shape{model.input_features};
for (std::size_t i = 0; i < model.layers.size(); ++i)
{
    const Config::LayerSpec& spec = model.layers[i];
    const std::string context = "layer " + std::to_string(i) + " '" + spec.type + "'";
    const LayerEntry& entry = Config::find_entry(LAYER_REGISTRY, spec.type, "layer type");
    std::unique_ptr<ILayer> layer = entry.make(shape, spec.attributes, context);
    shape = layer->output_shape();
    stack.push(std::move(layer));
}
```

After: the seed is the input shape, and a row that does not read the previous row is refused.

```cpp
Shape shape = model.input_shape;
for (std::size_t i = 0; i < model.layers.size(); ++i)
{
    const Config::LayerSpec& spec = model.layers[i];
    const std::string context = "layer " + std::to_string(i) + " '" + spec.type + "'";

    SA_THROW_IF(!reads_previous_row_only(spec, i),
                "{} reads something other than the row before it; a LayerStack is a chain. "
                "Build this file with build_model.", context);

    const LayerEntry& entry = Config::find_entry(LAYER_REGISTRY, spec.type, "layer type");
    std::unique_ptr<ILayer> layer = entry.make(shape, spec.attributes, context);
    shape = layer->output_shape();
    stack.push(std::move(layer));
}
```

### 3.6 `initialize`

Before (`include/SushiAI/nn/init.hpp:183-192`, with `src/nn/init.cpp:100`): everything from the
last `WEIGHT` on is zeroed, whatever its role, and tensors are read through the module's id.

```cpp
std::size_t final_weight = parameters.size();
for (std::size_t i = 0; i < parameters.size(); ++i)
    if (parameters[i]->role() == ParameterRole::WEIGHT) final_weight = i;

for (std::size_t i = 0; i < parameters.size(); ++i)
{
    const bool is_final = policy.zero_final_layer && i >= final_weight;
    Detail::apply_initializer(engine, pool.tensor(parameters[i]->id()),
                              *parameters[i], policy, is_final);
}
```

After: the state object supplies the tensors, and only the last fan-scaled entry is zeroed.

```cpp
const SushiRuntime::span<const StateEntry> entries = state.entries();

std::size_t final_weight = entries.size();
for (std::size_t i = 0; i < entries.size(); ++i)
    if (entries[i].init.kind == ParameterInit::Kind::FanScaled) final_weight = i;

for (std::size_t i = 0; i < entries.size(); ++i)
{
    const bool is_final = policy.zero_final_layer && i == final_weight;
    Detail::apply_initializer(engine, *state.tensor(entries[i].state).value(),
                              entries[i], policy, is_final);
}
```

For every model in the tree today the result is the same: the entries after the last weight are
biases, which are constant zero already (`init.cpp:100`). With a batch normalisation after the
last weight, the old rule zeroes its gamma and the model outputs a constant; the new rule leaves
gamma at one.

### 3.7 The record lookup in `load_checkpoint`

Before (`include/SushiAI/io/checkpoint.hpp:153-155`):

```cpp
std::size_t index = bound.size();
for (std::size_t i = 0; i < bound.size(); ++i)
    if (bound[i].name == record.name) { index = i; break; }
```

After:

```cpp
const Result<SushiBLAS::Tensor*> found = state.tensor(renamed(record.name, rules));
```

## 4. What crosses the seam to SushiBLAS and SushiRuntime

**Calls this design makes, all of which the tree makes already.**

| Call | Used for | Used today at |
| --- | --- | --- |
| `blas().scal`, `blas().axpy` | The buffer update | `lowering.cpp:1279`, `:1292` |
| `blas().copy`, `elementwise().sqrt`, `reciprocal`, `mul` | The `BATCHNORM_FOLD` lowering | `lowering.cpp:1294-1299` |
| `random().constant` through `TensorPool::request_constant` | Buffer and constant-parameter fills | `lowering.cpp:182-198` |
| A copied `SushiBLAS::Tensor` sharing its `Storage` | `ModelState::attach` | `lowering.cpp:107-108`; `sushiblas/include/SushiBLAS/tensor.hpp:98` |
| `engine.get_graph().compile()` once per step | Each of the three steps | `compiled_step.cpp:50` |

The recorder records through the engine today. The eager spec moves recording to a
`TaskRecorder` over a caller-supplied `TaskGraph&` and takes `Engine::get_graph()` off the
public surface (`EAGER_FRONT_END/SPEC.md:19`). `BufferUpdateRecorder` and the two lowerings are
written once, after that migration, against whichever of the two forms it leaves.

**What this design asks of SushiBLAS.**

- A depthwise convolution kernel that accepts a `[kh, kw, C]` weight. The ruling places it in
  SushiBLAS (`PROGRAMME.md:32`), and the fixture fixes the shape before the kernel exists
  (`tools/reference/README.md:54`).
- Optional, later: one fused kernel for the evaluation fold and one for the running-statistic
  update. Section 8 gives the task counts without them.

**What this design asks of SushiRuntime.** Nothing new. It needs one answer: whether two
compiled plans on one engine may read and write the same storage and replay in either order.
CV-8 as designed rests on it.

**What does not cross.** The trace mode, state identity, the update list, the node seam, the
model file and the name grammar are SushiAI's own. Neither sibling learns of them.

## 5. Order of work

Host lane means no engine call and no include of `SushiBLAS/engine.hpp`. Waves 1 to 6 are host
lane and can be written before the SushiBLAS migration. Within a wave the units have disjoint
file sets and can run side by side. No build runs between waves (`PROGRAMME.md:30`); "leaves
compiling" below means each file passes a single-file syntax check against the tree as that wave
leaves it.

Every new test file needs a line in `tests/CMakeLists.txt`, which lists tests one by one
(`tests/CMakeLists.txt:48`, `:65`). That is a build configuration file; the owner adds the lines.

| Wave | Unit | Files | Leaves compiling |
| --- | --- | --- | --- |
| 0 | Owner's decisions; the exporter rerun; `core/status.hpp` if decision 1 says so | `tools/reference/`, `tests/fixtures/yolo26/`, `include/SushiAI/core/status.hpp` | Unchanged tree |
| 1 | U1 graph surface: `StateId`, the state-leaf table, `without_nodes`, `Lifetime::BUFFER`, `GraphBuilder::buffer`, the fourth `BATCHNORM2D` output, `BATCHNORM_FOLD` and its emitter, the rule guard, the `wrt` check | `graph/state_id.hpp` (new), `graph/ir.hpp`, `graph/builder.hpp`, `src/graph/builder.cpp`, `core/ops.hpp`, `src/graph/fusion.cpp`, `src/graph/cast_insertion.cpp`, `src/autograd/rules/norm.cpp`, `src/autograd/differentiator.cpp`; tests `test_ir.cpp`, `test_spatial_shapes.cpp`, `test_backward.cpp`, `test_ops.cpp`, `test_precision.cpp`, `test_fusion.cpp`, `test_rule_table.cpp` | Everything. See the note on run-time arity below |
| 2 | U2 state in `nn/`: per-graph `Parameter`, `ParameterInit`, `Buffer`, the field kinds, `ModuleList`, `walk_state` | `nn/module.hpp`, `nn/buffer.hpp` (new), `nn/module_list.hpp` (new), `test_module.cpp` | Everything; `Parameter::id()` still serves its three callers |
| 2 | U3 model file version 2 and the upgrade step | `config/model_config.hpp`, `src/config/model_config.cpp`, `test_config.cpp` | Everything; `ModelSpec::input_features` is still filled |
| 2 | U4 `StepRecorderList` | `graph/step_recorder.hpp`, `test_step_recorder.cpp` (new) | Everything |
| 2 | U5 `ModelOutputs` | `nn/model_outputs.hpp`, `src/nn/model_outputs.cpp`, test (all new) | Everything |
| 3 | U6 `Trace`, the update sinks, the layer forms, `trace_layer`, `Sequential`'s second overload | `nn/trace.hpp`, `src/nn/trace.cpp` (new), `nn/layers.hpp`, `test_trace.cpp` (new) | Everything; `Sequential<Linear, ReLU>` is still a `Module` |
| 4 | U7 `BatchNorm2d`, `NormOptions`, `updated_statistic` | `nn/batch_norm2d.hpp`, `src/nn/batch_norm2d.cpp`, test (all new) | Everything |
| 4 | U8 `Node`, `Port`, `ModuleNode`, `LayerNode`, `Concat`, `SiLU`, `Identity`, the `LayerAdapter` assertion | `nn/node.hpp`, `nn/module_node.hpp`, `nn/layer_node.hpp`, `src/nn/layer_node.cpp`, `nn/concat.hpp`, `src/nn/concat.cpp`, `nn/silu.hpp`, `nn/identity.hpp` (new), `nn/ilayer.hpp`, tests | Everything |
| 4 | U9 `Conv2d::same_padded`, `Conv2d::bias_init` | `nn/conv2d.hpp`, `src/nn/conv2d.cpp`, `test_conv2d.cpp` | Everything |
| 5 | U10 `ModelGraph`, `NodeRef` | `nn/model_graph.hpp`, `src/nn/model_graph.cpp`, test (all new) | Everything |
| 5 | U11 blocks `ConvNorm`, `Bottleneck`, `C3k`, `C3k2`, `SPPF`, and each one's factory | `nn/blocks/*.hpp`, `src/nn/blocks/*.cpp`, tests (all new) | Everything |
| 6 | U12 `K_NODE_REGISTRY`, `make_node`, `build_model`, the `silu` row, the chain guard, the YOLO26n model file | `nn/node_registry.hpp`, `src/nn/node_registry.cpp` (new), `nn/layer_registry.hpp`, `src/nn/layer_registry.cpp`, `examples/yolo26/model.json` (new), tests | Everything; rows 10, 22 and 23 of the file fail to build with a named missing type until wave 8 |
| 7 | U13 lowerings: four `BATCHNORM2D` outputs, `BATCHNORM_FOLD` | The split lowering tables the migration leaves | After the migration |
| 7 | U14 `ModelState`, `initialize` over it | `nn/model_state.hpp`, `src/nn/model_state.cpp` (new), `nn/init.hpp`, `src/nn/init.cpp` | After the migration |
| 7 | U15 checkpoint over `ModelState`, `import_weights` (after U14) | `io/checkpoint.hpp`, `io/import_weights.hpp`, `src/io/import_weights.cpp` (new) | After the migration |
| 7 | U16 `IObjective` over `ModelOutputs`, the three builders, `retain`, `TrainingStep`, `EvaluationStep`, `BufferUpdateRecorder`, the trainer's flag and shape-typed feed (after U14) | `train/objective.hpp`, `src/train/objective.cpp`, `train/trainer.hpp`, `src/train/trainer.cpp`, `train/steps.hpp`, `train/buffer_update_recorder.hpp` and sources (new) | After the migration |
| 7 | U17 `Dataset::sample_shape`, both applications on `build_model`, the MNIST example, the deletions (after U16) | `data/dataset.hpp` and sources, `apps/train.cpp`, `apps/inference.cpp`, `examples/mnist_cnn/model.json`, `nn/module.hpp`, `config/model_config.hpp` | After the migration |
| 8 | U18 `DepthwiseConv2d`, `Attention`, `PSABlock`, `C2PSA`, `DetectHead`, their rows, the whole-model name test | `nn/blocks/` | After the four emitters exist |

**The run-time arity gap.** Wave 1 makes the builder emit four outputs while the lowering still
checks three (`src/graph/lowering.cpp:1241`). That is a run-time check, so the tree compiles,
and no existing layer emits the node. Two integration tests lower it,
`tests/functional/integration/test_forward.cpp` and `test_autograd_numeric.cpp`, and would fail
if the tree were built between wave 1 and U13. The gap closes in one of two ways: U13 lands
before the owner's build, or the migration work takes the three-line change (the arity, the
variance written to output 3, epsilon added to the `inv_std` copy at `lowering.cpp:1292-1294`)
into its own batch-normalisation table. The second is the smaller risk and is asked of that
work, not done here.

**The arity sites.** `BATCHNORM2D`, `batchnorm2d`, `BatchNormResult` or `saved_inv_std` appear
in 13 files, six of them tests: `autograd/rules/norm.hpp`, `core/ops.hpp`, `graph/builder.hpp`,
`graph/lowering.hpp`, `src/autograd/rules/norm.cpp`, `src/graph/builder.cpp`,
`src/graph/lowering.cpp`, and the tests `test_autograd_numeric.cpp`, `test_forward.cpp`,
`test_backward.cpp`, `test_ops.cpp`, `test_precision.cpp`, `test_spatial_shapes.cpp`. The count
is 67 lines by that pattern, which includes the backward emitter. `test_ops.cpp`,
`test_backward.cpp` and `test_rule_table.cpp` are modified in the working tree by other work
now, as are `builder.hpp`, `builder.cpp` and `ops.hpp`; U1 runs behind that work, not beside it.

**Deletions, all in U17.** `Parameter::id()`, `ModelSpec::input_features`, the
`build_graph(batch, features)` overload, and `Detail::bind` in `checkpoint.hpp:68-82`. Each is
deleted only after its last caller has moved.

## 6. What a script does and what needs judgement

| Work | Script | Judgement |
| --- | --- | --- |
| Changing the 14 existing `BatchNormResult` and `{y, mean, inv_std}` sites to four outputs | Finds every site by the pattern of section 5 | Each test's expected arity and which output it asserts on |
| Replacing `parameter.id()` at its three callers | Finds them (`trainer.hpp:160-161`, `checkpoint.hpp:75`, `init.hpp:190`) | Which source replaces it: `id_in(graph)` for the trainer, `ModelState` for the other two |
| Converting the throwing bodies of this design to the status form in stage 2 | Wraps each fallible call in the propagation form and changes each return type | The message each failure carries, and which failures are a caller's error against a defect |
| Renaming `IObjective`, `LAYER_REGISTRY`, `MODEL_VERSION`, `Lifetime` and `ParameterRole` enumerators in the naming wave | The whole rename | None beyond the rule |
| Producing the 24-row model file | Derives `filters`, `hidden` and each `from` from the manifest's weight shapes | `stride`, `shortcut`, `pools` and `heads`, which no weight shape carries |
| The golden lists (type strings, state names, port names) | Generates the state-name list from `MANIFEST.json` at test time | Whether a changed list is a break or a fix |
| Upgrading version 1 model files | `upgrade_model_v1_to_v2` runs in the reader; no file on disk is rewritten | Whether `examples/mnist_cnn/model.json` moves to rank 4 (wave 7) |
| Designing a block's `forward` | Nothing | Every body in section 2.10, and the five blocks that are signatures |
| Choosing what the eager surface exposes | Nothing | Decision 3 |

## 7. The tests that pin it

Every test in this table runs without a device.

| Unit | Test pins |
| --- | --- |
| U1 | A buffer leaf has `requires_grad == false` and `is_persistent(BUFFER)` holds. `bind_state_leaf` twice with one state fails. `state_leaf` returns the same id before and after `insert_casts` and after `fuse`. `without_nodes` keeps `layout`, `requires_grad` and `dtype` of every value and holds no node. `batchnorm2d` emits four outputs, three of shape `[C]`. A gradient on output 3 throws. `batchnorm_fold` rejects a shape mismatch and a non-positive epsilon, and differentiating through it fails. `differentiate` with a buffer in `wrt` fails |
| U2 | One parameter traced into two graphs has a leaf in each and `id_in` returns each graph's own. A second declare in one graph returns the first id; a declare with another shape fails. A copied layer has a new `StateId`; a moved layer keeps it. `walk_state` over a struct with all three field kinds yields declaration order. `for_each_parameter` never yields a buffer. An undefined buffer is skipped. `ModuleList` names by index. Every existing `Unit_Module` name is unchanged. Two fields with one name fail a `static_assert` (compile-fail case, recorded in the header) |
| U3 | A version 1 document parses to the same `ModelSpec` as its version 2 rewrite. Version 3 is refused with both numbers. Each `from` form parses; `-2`, a forward reference and an unknown key each fail with the row named. `defaults` reach a row that omits the key and do not override a row that states it |
| U4 | `StepRecorderList` sums the counts and calls its members in the order added, with fake recorders |
| U5 | `add` rejects a duplicate. `at` on a missing name lists the names. `only` fails on zero and on two |
| U6 | `update` under `Trace::evaluation` fails. `discard_updates` records nothing. Each `update` under a list adds one update and one source. `Module<Sequential<Conv2d, BatchNorm2d>>` is false and `TraceModule` of it is true, as `static_assert`s. A type with `trace_training` and no `trace_evaluation` fails to compile (compile-fail case) |
| U7 | A training trace has one `BATCHNORM2D`, no buffer leaf, and two updates whose sources are outputs 1 and 3 and whose targets are the two buffers' `StateId`s. An evaluation trace has one `BATCHNORM_FOLD`, one `CHANNEL_AFFINE`, two buffer leaves and no `BATCHNORM2D`. A count of 1 fails. `updated_statistic` and `running_variance_scale` reproduce the exporter's `[2, 2, 2, 3]` case; this test waits for that fixture |
| U8 | `LayerNode` over `make_conv2d` reports the layer's shape. A one-input node given two inputs fails. `Concat` sums channels and rejects unequal H or W. `LayerAdapter` over a type with a `BufferField` fails to compile |
| U9 | `same_padded(c, c, 3, 1)` keeps H and W; with stride 2 it halves an even extent. A non-zero `bias_init` reaches the parameter's `init()` |
| U10 | The hand-built branch and merge of section 2.5 traces to the expected node sequence with one `CONCAT_CHANNELS` reading both branches. A duplicate name, an unknown source and a declared shape that disagrees with the traced one each fail. `forward` with two outputs fails and `trace` returns both by name. A chain built by `ModelGraph` and by `LayerStack` yields the same names |
| U11 | Per block: output shape, node count, and state names with shapes equal to the manifest's records for rows 0, 2, 6 and 9, read from `MANIFEST.json` at test time. Row 9 has no `SILU` after `cv1` |
| U12 | Every registry row builds from a minimal spec; an unknown key fails. `build_model` of a file and the hand-built graph give equal node lists and equal state names. `build_layer_stack` refuses a file with `from: [-1, 0]`. Golden lists for the type strings of both tables and for `K_C3K2_INNER`. Rows 0 to 9 of the YOLO26n file yield exactly the manifest's `model.0.` to `model.9.` records, names and shapes, both ways |
| U18 | The walk of the whole YOLO26n file under `model` equals the 594 `weights.sacp` records in `MANIFEST.json`, names and shapes, both ways. The twelve port names equal the fixture's. This is the importer's strictness tested without a device |

Tests that need a device, all in wave 7: `running_mean` changes across two training steps and
matches `updated_statistic`; an evaluation forward on the fixture's weights matches
`forward.sacp`; the training step and the evaluation step each compile once and the evaluation
step sees a weight the training step just moved; a probe pass leaves both running vectors
unchanged; a `PER_STEP` optimizer still steps when the model holds a batch normalisation.

## 8. Costs

- **Device tasks.** YOLO26n has 114 batch normalisations (the manifest lists 114 gammas). The
  buffer update is two calls per statistic: 456 tasks per training step. The evaluation fold, as
  lowered from existing calls, is 8 per layer (copy, axpy, sqrt, reciprocal, two mul, copy,
  axpy): 912 per evaluation replay. A fused kernel for each brings them to 228 and 114.
- **Three compiled plans.** Training, probe and evaluation each hold a graph, a pool and a
  plan. The probe plan's arena is about the forward part of the training plan's. Its size and
  the three compile times on the 6 GB device are not measured; `TensorPool`'s arena bytes at
  batch 16 and 640 px, read once per plan, give the number. No design here lets the probe pass
  reuse the training plan's forward values: the assigner's result is an input of the loss, so
  the forward must have run before the training plan's inputs are complete.
- **Contract width.** A layer has one of three static forms, and there is one run-time seam with
  two adapters. That is one more form than a design with a mode switch inside each modal layer,
  and it buys a compile error for a layer that lacks its evaluation form.
- **Edits to existing types.** `Parameter`, `Sequential`, `LayerAdapter` (one assertion),
  `Conv2d` (one static, one field), `BatchNormResult`, `Lifetime`, `Graph`, `ModelSpec`,
  `IObjective`, `TrainingGraph`, `Trainer`, `initialize`, the checkpoint functions, and both
  rebuild loops. `ILayer`, `LayerStack` and `LAYER_REGISTRY` keep their shape; the registry gains
  one row.
- **Breaking changes before the first tag.** The model file's version key and `input` object;
  `BATCHNORM2D`'s arity; `IObjective::loss`; the checkpoint and initialiser signatures. A
  checkpoint written by a model without buffers does not load into the same model with them.
- **A process-wide counter.** `StateIdentity` draws from one atomic. It is the only global this
  design adds.
- **`BatchNorm2d` in a model file needs `build_model`.** It is not a `Module`, so `LayerAdapter`
  and `build_layer_stack` cannot hold it. Stage 3's convolutional network with batch
  normalisation from JSON goes through `build_model`, and both applications move to it in U17.
- **One rewrite in stage 2.** Under decision 1's alternative, every signature in section 2 is
  written throwing and converted once.

## 9. Decisions for the owner

Each is a public interface, a module boundary, data ownership or a deletion. The document is
written under the recommendation.

1. **Does the status type exist before wave 1?**
   - a. Stage 2's first unit, one header with `Status` and `Result<T>`, lands in wave 0. New
     seams with no existing sibling (`Node`, `NodeFactory`, `ModelGraph`, `build_model`,
     `ModelState`, `import_weights`) are written in status form. Contracts shared with existing
     layers (`forward`, `Parameter::declare`) stay throwing until the stage 2 wave converts them
     with `GraphBuilder`.
   - b. Everything is written throwing, and stage 2 converts all of it.
   - Recommendation: a. It needs the owner to name the type and its propagation form. Until
     `GraphBuilder` is converted, an exception from a builder emitter still passes through a
     status-returning `Node::forward`; a does not hide that, it limits the later rewrite to
     bodies.
2. **The node seam (D5, open at `YOLO26/SPEC.md:457`).**
   - a. `NN::Node` as in section 2.4, with `ILayer` kept and wrapped.
   - b. Change `ILayer::forward` to take a `Trace` and several values, and edit every existing
     layer and test.
   - Recommendation: a.
3. **How a layer learns its mode, and what the eager surface gains.**
   - a. A `Trace` argument and three layer forms (section 2.3). `Eager::Step` gains `trace()`
     and `Session::begin_step` gains a mode. `forward(step.builder(), x)` keeps working for
     every mode-free layer.
   - b. Mode and update list on `GraphBuilder`. No layer signature changes and the eager call
     form is the only one. `GraphBuilder` then carries trace state as well as emitting IR, a
     default-constructed builder has a default mode, and a modal layer reached through an old
     entry point traces in that mode with no error.
   - Recommendation: a, decided before the eager surface ships.
4. **Where a parameter's graph id lives.**
   - a. In the graph, keyed by `StateId`, carried through both rebuilding passes by
     `Graph::without_nodes` (section 2.1). Costs one process-wide counter.
   - b. One id on the module, cleared by a `release_trace` call before each new trace.
   - Recommendation: a. Under b a forgotten call returns another graph's id with no error, and
     the eager front end would need the call on every step.
5. **Who owns parameter and buffer tensors.**
   - a. A `ModelState` beside every pool, bound into each (section 2.8).
   - b. The training pool, with other pools binding from it. Smaller, and it needs buffer leaves
     that no training node reads and a pool that outlives the others.
   - Recommendation: a. Both rest on two plans aliasing storage, which is unverified.
6. **Where the blocks and the head live.**
   - a. `nn/blocks/`, head included; `detect/` stays free of `nn/` and `graph/`, as it is today.
   - b. The head under `detect/`, with the node registry reaching into it or made open.
   - Recommendation: a.
7. **The evaluation form.**
   - a. A non-differentiable `BATCHNORM_FOLD` feeding `CHANNEL_AFFINE`. One new op, emitter and
     lowering; nothing derived is stored, so nothing goes stale after a checkpoint load.
   - b. Two host-computed leaves per layer, refreshed in `before_replay`. No new op; an
     evaluation step built without the refresher reads unset memory with no error.
   - c. One `BATCHNORM2D_INFERENCE` node. One op, and `CHANNEL_AFFINE` is not the applied form.
   - Recommendation: a.
8. **The exporter rerun for n/(n-1).** Add the `[2, 2, 2, 3]` unit case and `model.9.cv1.bn` to
   `loss.sacp` (section 2.7), or accept the unbiased form unverified. Recommendation: rerun
   before U7 is accepted.
9. **Model file version 2.** The key `format_version`, `input.shape`, `from`, `outputs`,
   `defaults`, integer `hidden` widths in place of expansion ratios, and `layers` kept as the
   list key (section 2.6). The alternative for the key is to keep `sushiai_model` at 2.
   Recommendation: as written.
10. **Checkpoint contents.** Parameters and buffers in one namespace, SACP version 1 unchanged,
    against a version 2 record with a kind field. Recommendation: one namespace.
11. **One `c3k2` row with an inner-kind table, or three rows** (`c3k2`, `c3k2_c3k`,
    `c3k2_attention`). Recommendation: one row and `K_C3K2_INNER`; the file then mirrors the
    Ultralytics row.
12. **Deletions in U17.** `Parameter::id()`, `ModelSpec::input_features`, the
    `build_graph(batch, features)` overload and `IO::Detail::bind`. Recommendation: delete, each
    after its last caller moves.
13. **Who closes the run-time arity gap** (section 5): the migration work takes the three-line
    lowering change, or U13 does and the owner does not build in between. Recommendation: the
    migration work.

## 10. Unverified

- **Two plans over shared storage.** Not read in SushiRuntime and not run. The whole
  training and evaluation split rests on it; section 2.8 gives the fallback. A device test in
  wave 7 settles it.
- **The n/(n-1) factor.** From memory of torch. The exporter rerun of decision 8 settles it.
- **`stride`, `shortcut`, `pools` and `heads` in the model file.** The strides agree with the
  fixture's stride list of 8, 16 and 32 (`MANIFEST.json:161-165`) and with the levels the head
  reads. `shortcut`, `pools` and `heads` come from the inventory's reading of the Ultralytics
  yaml and sources, which were not opened. That row 9's `cv1` is unactivated and that the row
  ends in an add are from the same reading. Comparing a traced forward with `forward.sacp`
  settles all of them.
- **The five blocks given as signatures.** Their compositions are the inventory's. Their state
  names are checked against the manifest by type alias only: the names in section 2.11 follow
  from `Sequential` and `ModuleList` indexing, and no walk has produced them.
- **The head's bias constants.** The mechanism is `Conv2d::bias_init`; the values are not
  stated here. `weights.sacp` holds the model before either pass
  (`tools/reference/README.md:43`), so its twelve bias records give them.
- **`narrows_under_mixed_precision`.** Its body was not read. `BATCHNORM_FOLD` must not be in
  it.
- **Fusion and the fourth output.** Value ids survive `fuse` (`fusion.cpp:593-595`), and the
  update sources are in the observed list. Whether any catalogue entry matches a chain through
  `BATCHNORM2D` was not read; an entry that did would have to keep output 3.
- **The memory planner and `Lifetime::BUFFER`.** `is_persistent` makes a buffer unpackable
  (`src/graph/memory_plan.cpp:318`). Other places that branch on a lifetime by name in the
  planner, the arenas and the lowering were not searched.
- **`blas().copy` into a rank-4 leaf.** The feed checks element counts and then copies
  (`trainer.cpp:251-260`). Whether the copy accepts a rank-4 destination was not read.
- **Whether `src/` sources are listed one by one.** `tests/CMakeLists.txt` lists tests one by
  one. `src/CMakeLists.txt:190` globs the public headers; how it collects sources was not read.
  If it lists them, every new `.cpp` here needs a line the owner adds.
- **`apps/inference.cpp`, `checkpoint_format.cpp`, `memory_plan.hpp` beyond line 119, the
  optimizer's recorder and `dataset.hpp` beyond its `input_dim` lines** were not read. Claims
  about them carry the inventory's line numbers or none.
- **The eager region key for buffer updates.** `2n + 3` follows the ordering rule at
  `EAGER_FRONT_END/SPEC.md:110-124`. `Session` and `Step` do not exist in the tree yet
  (`include/SushiAI/eager/` holds the partitioner, the scope and the trace units), so nothing
  was checked against code.
- **`Sequential`'s conditional overload.** Enabling the builder overload only when every child
  is a `Module` changes how `Module<Sequential<...>>` is detected (`module.hpp:218-225`). The
  assertions at `layers.hpp:254-266` must still hold; this is argued, not compiled.
