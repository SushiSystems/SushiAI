# Error contract and naming wave

**Status:** Open — the naming half is done; the status contract is RW-11.

## Decided

The owner ruled on three of the decisions of section 9 on 2026-10-07 and left the rest to the
engineer, who takes the recommendation written beside each. Section 9 still states the options.

| Decision | Outcome | By |
| --- | --- | --- |
| D1, a wrong model written in C++ | Content is a status, from JSON and from C++ alike; a bad handle, a wrong call order and broken internal state are fatal | Owner |
| D6, gradient-rule signature | Rules stay `void`; a refusal inside a rule is marked in the graph and `differentiate()` returns it. A signed departure from "every operation that can fail returns a status": the public boundary is `differentiate()`, not a rule | Owner |
| D11, sibling enumerators | SushiBLAS v0.1.0 is asked to rename the enumerators of `DataType`, `Layout` and `ExecutionMode` to PascalCase; the request was sent to the sushiblas session, with D17's other two requests | Owner |
| D2 to D5, D7 to D10, D12 to D16 | The recommendation of section 9 in each case | Engineer |
| D7, death tests | Taken with its condition: the start-up time of one death test is measured at the owner's first build before the fatal-class tests are written in bulk | Engineer |

Stage 2 of the programme (`docs/design/PROGRAMME.md:46`) ends when no exception crosses the public
surface and the library has one naming form. Stage 1, the SushiBLAS v0.1.0 migration, touches the
same files and is planned as the same pass (`docs/design/PROGRAMME.md:52`). This document specifies
the status and result types, where each failure in the tree goes, the conversion at the sibling
seam, the rename, and the order of work. Section 9 lists the choices that are the owner's. The
rest of the document is written under the recommendation given for each.

## 1. What the ruling requires

The two rulings, as written:

| Subject | Ruling | Where |
| --- | --- | --- |
| Error contract | Status values everywhere: every operation that can fail returns a status, and no exception crosses the public surface | `docs/design/PROGRAMME.md:28` |
| Naming | The `cpp-code-style` rule for the whole library, applied in one wave: `K_` constants, PascalCase enumerators, no `I` prefix | `docs/design/PROGRAMME.md:29` |
| Builds | The owner builds and tests once the programme's code is written. Until then work is checked by reading and by single-file syntax checks | `docs/design/PROGRAMME.md:30` |
| Eager surface | `Session`, `Step` and `Scope` must not break after they ship | `docs/design/PROGRAMME.md:25` |

`cpp-code-style` splits failures in two: a broken precondition goes to the assert family, and a
failure with an outside cause (file, parse, device, network) returns a status. The ruling says
"every operation that can fail". The two meet only once "can fail" has a definition, and section
2.1 gives one.

### 1.1 The tree as read on 2026-10-07

The tree changes while it is read. Two scans a few minutes apart returned 508 and then 510 lines
matching `SA_THROW_IF(` under `include/` and `src/`. The figures below are from the second scan.

| Fact | Value | Where |
| --- | --- | --- |
| Lines matching `SA_THROW_IF(` | 510 in 71 files, the definition and comment mentions included | definition at `include/SushiAI/core/common.hpp:128-133` |
| Per module | autograd 24, config 24, core 14, data 18, detect 30, eager 16, graph 176, imaging 85, io 30, nn 44, optim 32, train 17 | |
| Raw `throw` | 2 | `src/core/allocation.cpp:99`, `:165` |
| `try` blocks in the library | 8 | `src/config/value.cpp:317`, `src/core/allocation.cpp:90`, `:156`, `src/nn/conv2d.cpp:57`, `avg_pool2d.cpp:53`, `max_pool2d.cpp:53`, `upsample2d.cpp:54`, `layer_registry.cpp:214` |
| `try` blocks in apps and examples | 7 | `apps/train.cpp:58`, `apps/inference.cpp:69`, `examples/example.cpp:42`, `examples/demo_mlp.cpp:161`, `:184`, `:207`, `:679` |
| Test lines that expect a throw | 117 `EXPECT_THROW` or `ASSERT_THROW` in 20 files, 322 naming `SA_EXPECT_ERROR` in 38 files | helper at `tests/common/test_throws.hpp:119` |
| Test lines that expect no throw | 25 in 15 files | |
| Calls of `.at(` on a standard container | 24 calls on 22 lines | section 2.7 |
| Status type, result type, assert macro | none | `include/SushiAI/core/common.hpp` holds only `Error` (`:111`) and `SA_THROW_IF` |

The inventory this design starts from classifies 390 sites. It predates `src/imaging` (85 lines)
and most of `src/detect` (30 lines), so at least 115 sites have no class yet. The imaging module's
own README records that it is written under the old contract on purpose
(`include/SushiAI/imaging/README.md:63`).

No file looked half-edited when read. `src/autograd/differentiator.cpp`, `src/graph/builder.cpp`
and `include/SushiAI/graph/ir.hpp` are modified in the working tree, so the line numbers cited for
them are the ones found at reading time.

## 2. The design

### 2.1 The dividing rule

> A check on the **content of a description** returns a status. A check on **handle validity,
> call order or object state** is fatal.

Content means a shape, a rank, an element type, a layout, a window geometry, a hyperparameter
range, a document field or the bytes of a file. A handle is a `ValueId`, a `NodeId` or an index.
The rule is applied by reading the check alone, without tracing who called.

A wrong model description written by a user is content, so it returns a status whether the user
wrote it in JSON or in C++. Four facts in the tree decide this:

- The same check serves both paths. `require_positive_features` is called from `Linear::forward`
  and from `Linear::output_shape` (`src/nn/layers.cpp:47`, `:78`).
- Extents reach an emitter from data. A dataset whose inner extent disagrees with the model
  reaches the `ka != kb` check in `GraphBuilder::matmul` (`src/graph/builder.cpp:172`) in a
  program with no defect.
- The eager front end traces on every step (`docs/design/PROGRAMME.md:14-16`), so an abort in an
  emitter ends a training run over a batch of the wrong size.
- The alternative keeps each check twice, once fatal for C++ and once as a status for JSON. The
  tree already shows where that leads: four layers catch the emitter's exception to re-word it
  (`src/nn/conv2d.cpp:57-68`, `avg_pool2d.cpp:53`, `max_pool2d.cpp:53`, `upsample2d.cpp:54`).

A caller cannot test "these two operands contract" before calling `matmul` without repeating its
shape inference (`src/graph/builder.cpp:160-176`). A caller can test "this id came from this
graph" and "this trace is not sealed". That difference is why the first is a status and the
second is fatal.

### 2.2 The four classes

| Class | Example | Mechanism |
| --- | --- | --- |
| Outside cause | a file that does not open (`src/config/value.cpp:332`), a refused allocation (`src/core/allocation.cpp:99`) | `Status` or `Result<T>` |
| Wrong description | a JSON field (`src/config/value.cpp:195`), operand shapes in an emitter (`src/graph/builder.cpp:160`), optimizer options | `Status`, `Result<T>` or `Graph::Traced`, code `InvalidDescription` |
| Broken precondition on a handle, an order or a state | an id the graph never issued (`include/SushiAI/graph/ir.hpp:328`), a sealed trace reused (`src/eager/scope_trace.cpp:55`), a tape index (`src/autograd/tape.cpp:49`) | `SA_FATAL_IF` |
| Defect in SushiAI | a pass that changed an id (`src/graph/cast_insertion.cpp:239`), a missing lowering | `SA_FATAL_IF` or `SA_FATAL` |

No per-site table exists. The split between status and fatal is not known until each of the 510
lines is read against the rule; wave 0 produces that table and every cost in section 8 depends on
it.

### 2.3 Status

`include/SushiAI/core/status.hpp`:

```cpp
namespace SushiAI
{
    /** @brief Who has to act on a failure. */
    enum class StatusCode : std::uint8_t
    {
        Ok,
        InvalidDescription, // a document or a traced model is wrong: fix the description
        FileNotFound,       // a named file is absent
        InvalidFile,        // unreadable, truncated, wrong magic, wrong version
        Incompatible,       // two valid things disagree: checkpoint and model, dataset and step
        Unsupported,        // this build, device or execution mode lacks the capability
        OutOfMemory,        // a device or arena allocation was refused
        DeviceFailed,       // a kernel faulted, the device was lost, a task threw
    };

    /** @brief A lower-case name for @p code, e.g. "invalid_description". */
    [[nodiscard]] constexpr const char* status_code_name(StatusCode code) noexcept;

    class [[nodiscard]] Status final
    {
        public:
            /** @brief Success: one null pointer, no allocation. */
            Status() noexcept = default;

            Status(const Status& other) noexcept;
            Status(Status&& other) noexcept;
            Status& operator=(const Status& other) noexcept;
            Status& operator=(Status&& other) noexcept;
            ~Status();

            /**
             * @brief Builds a failure whose message @p compose returns.
             * @param compose Called once, inside this function; returns std::string.
             */
            template<typename Compose>
            [[nodiscard]] static Status from_message(StatusCode code, Compose&& compose) noexcept;

            /** @brief A copy whose message is "<prefix>: <message>"; @p compose returns the prefix. */
            template<typename Compose>
            [[nodiscard]] Status annotated(Compose&& compose) const noexcept;

            [[nodiscard]] bool is_ok() const noexcept;
            [[nodiscard]] StatusCode code() const noexcept;
            [[nodiscard]] std::string_view message() const noexcept;   // empty on success

        private:
            const Detail::Failure* failure_ = nullptr;
    };
} // namespace SushiAI

#define SA_FAIL_IF(condition, code, ...)     /* if (condition) return a failed Status */
#define SA_ANNOTATE(status, ...)             /* (status).annotated(deferred format)    */
#define SA_TRY(expression)                   /* return the failure of a Status or Result, if any */
```

`[[nodiscard]]` sits on the class, so every function that returns a `Status` inherits it.

`Detail::Failure` holds a `std::atomic<std::uint32_t>` reference count, the code and a
`std::string`. It is immutable after construction, so copies of one failure may live on several
threads.

**A message without an allocation on a hot path.** Success constructs nothing and allocates
nothing. `SA_FAIL_IF` formats inside the taken branch only, as `SA_THROW_IF` does today
(`include/SushiAI/core/common.hpp:128-133`). A failure allocates one block for the count, the code
and the text.

**The failure path under `noexcept`.** The macros pass the format call as a lambda:

```cpp
#define SA_FAIL_IF(condition, code, ...)                                               \
    do {                                                                               \
        if (condition) {                                                               \
            return ::SushiAI::Status::from_message(                                    \
                (code), [&]() { return ::SushiBLAS::Core::format(__VA_ARGS__); });     \
        }                                                                              \
    } while (0)
```

`from_message` calls the lambda and allocates the block inside one `try`, with `new
(std::nothrow)` for the block. On any exception it returns a statically allocated failure with
code `OutOfMemory` and a literal message. The arguments, `shape.to_string()` among them, are
evaluated inside the lambda, so their allocations are covered too. `SushiBLAS::Core::format`
itself throws only when its `std::ostringstream` cannot allocate or an argument's `operator<<`
throws: it drops a field with no argument and emits an unterminated field verbatim
(`sushiblas/include/SushiBLAS/core/format.hpp:258-304`).

**What a user can act on.** The message keeps the form the tree already uses: what was given,
what is needed, what to do (`src/graph/builder.cpp:45-48`). The code names who has to act. No
code names a layer, an op or a dataset, so a new kind adds no enumerator and edits no switch.

### 2.4 Result

`include/SushiAI/core/result.hpp`:

```cpp
template<typename T>
class [[nodiscard]] Result final
{
    public:
        Result(T value) noexcept(std::is_nothrow_move_constructible<T>::value); // implicit
        Result(Status failure) noexcept;                    // implicit; fatal when failure.is_ok()

        [[nodiscard]] bool has_value() const noexcept;
        [[nodiscard]] const Status& status() const noexcept; // Ok when a value is held
        [[nodiscard]] T& value() & noexcept;                 // fatal when no value is held
        [[nodiscard]] const T& value() const& noexcept;
        [[nodiscard]] T&& value() && noexcept;

    private:
        std::optional<T> value_;
        Status status_;
};

/** @brief The reference form: holds T* or a failure. */
template<typename T>
class [[nodiscard]] Result<T&> final;

#define SA_TRY_ASSIGN(declaration, expression)  /* bind the value, or return the failure */
```

- There is no `Result<void>`; `Status` is that type.
- `Result<T>` has no default constructor. A failed result with no status would have no message.
- The two implicit constructors break the `cpp-code-style` rule that single-argument
  constructors are `explicit`. They exist so that `return value;` and `return status;` both work.
  The header states the exception.
- `Result<T&>` serves the accessors that fail on content: `Config::Value::at`, `as_string`,
  `as_array` and `find_entry`. It removes the four dummy returns that follow a throw today
  (`include/SushiAI/config/registry.hpp:64-65`, `src/config/value.cpp:241-243`, `:324-326`,
  `src/train/objective.cpp:66-68`).
- `SA_TRY_ASSIGN` expands to two statements, so it cannot be the body of an unbraced `if`. The
  checker in section 6 flags that use.

SushiRuntime's `Result<T, E>` is not reused. It carries no `[[nodiscard]]`
(`sushiruntime/include/SushiRuntime/core/result.hpp:66-67`) and its `value()` is `std::get<0>`
(`:120`), which throws `std::bad_variant_access` on the wrong alternative.

### 2.5 The fatal family

`include/SushiAI/core/check.hpp`:

```cpp
#define SA_FATAL_IF(condition, ...)   /* always on; same polarity as SA_THROW_IF */
#define SA_FATAL(...)                 /* always on; an unreachable default        */
#define SA_ASSERT(condition)          /* debug builds only                         */

namespace SushiAI::Detail
{
    /** @brief Writes "sushiai: fatal: <message> (<file>:<line>)" to stderr, flushes, aborts. */
    [[noreturn]] void fail_fast(const char* file, int line, std::string_view message) noexcept;
}
```

- `SA_FATAL_IF` keeps the condition as `SA_THROW_IF` has it, so a fatal site changes by one
  token. An assert-polarity `SA_VERIFY` would need each condition negated with no compiler to
  check the result.
- `fail_fast` calls `std::fflush(nullptr)` before `std::abort()`, so output a program wrote to
  `stdout` before the failure is not lost.
- `SA_FATAL_IF` stays on in release builds. A compiled-out rank check in `Shape`'s constructor
  writes past `dims_` (`include/SushiAI/core/shape.hpp:62`).
- `SA_ASSERT` is used only inside a per-element or per-step loop, beside a comment naming the
  caller that already checked.

### 2.6 Constructors and RAII types

A constructor never fails. A type whose construction fails on content or on an outside cause gets
a private constructor over validated parts and a `[[nodiscard]] static Result<...> make(...)
noexcept`.

| Type | Throws today at | After |
| --- | --- | --- |
| `Shape` | `include/SushiAI/core/shape.hpp:56-78` | Both constructors use `SA_FATAL_IF`. `static Result<Shape> make(span)` serves extents read from a document or a file |
| `CheckpointReader`, `CheckpointWriter` | `src/io/checkpoint_format.cpp:124`, `:172-182` | `static Result<CheckpointReader> open(std::istream&)`, `static Result<CheckpointWriter> create(std::ostream&, count)` |
| `AdamW`, `SGD`, `Rings`, `SyntheticClassification`, `LossScaleState`, `LossScaler`, `StepScalars`, `MixedPrecisionPolicy` | their option checks | `make(options)`; the options are checked in that one place |
| `CompiledStep` | lowering, compile and flush run in the constructor (`src/graph/compiled_step.cpp:34-59`); all four copy and move members are deleted (`include/SushiAI/graph/compiled_step.hpp:64-67`) | `static Result<std::unique_ptr<CompiledStep>> compile(...) noexcept`; `[[nodiscard]] Status run() noexcept` |
| `Trainer` | `src/train/trainer.cpp:184`, `:192` | `static Result<std::unique_ptr<Trainer>> make(...) noexcept`; `Result<StepMetrics> run_step(...)`, `Result<RunReport> fit(...)` |

The existing `make_*` factories already return `std::unique_ptr`; they return
`Result<std::unique_ptr<...>>`. `try_load_mnist` returns `nullptr` for absent files and throws for
malformed ones (`src/data/dataset.cpp:287-358`, `:487`); it returns one `Result` with
`FileNotFound` or `InvalidFile`.

Accessors that return a reference and fail only on an index keep their shape and use
`SA_FATAL_IF`: `Graph::value`, `Graph::node` (`include/SushiAI/graph/ir.hpp:326-349`),
`TensorPool::tensor`, `MemoryPlan::placement`, `LayerStack::layer`, `Dataset::batch(index)`.

**Two uses of the `Shape` span constructor take extents from a file**:
`src/io/checkpoint_format.cpp:226` and `include/SushiAI/io/checkpoint.hpp:96`. The first checks
rank and every extent before it constructs (`src/io/checkpoint_format.cpp:213-225`). `Shape{`
appears on 35 lines of `include/` and `src/`; wave 0 lists each one whose extents are not
literals or values already inside a `Shape`, and each listed one moves to `Shape::make`.

**`Eager::Scope`.** Its constructor calls `ScopeTrace::open` and its destructor calls
`ScopeTrace::close` (`include/SushiAI/eager/scope.hpp:51-60`); both throw today
(`src/eager/scope_trace.cpp:68-89`), and a throw from a destructor is `std::terminate` with no
text. After the change:

- `ScopeTrace::open` and `close` stay public. Calling either on a sealed trace, or `close` with
  no scope open, is call order and is fatal.
- The name check in `open` (`src/eager/scope_trace.cpp:71-76`) is content. A guard's constructor
  has no return value, so `Scope` is the one place where a content check is fatal. `ScopeTrace`
  publishes `static bool is_legal_scope_name(std::string_view) noexcept`, as the builder already
  publishes `is_legal_leaky_relu_slope` (`src/graph/builder.cpp:102-105`), so a caller whose name
  comes from data can test it first.
- `seal()` with scopes still open is call order and is fatal; `seal()` keeps returning
  `std::vector<Segment>`.
- The destructor is `noexcept` and its failure is a message followed by an abort.

### 2.7 The emitter chain

`include/SushiAI/graph/traced.hpp`:

```cpp
namespace SushiAI::Graph
{
    /** @brief A traced value, or the failure that stopped the trace. */
    class [[nodiscard]] Traced final
    {
        public:
            Traced() noexcept = default;             // holds K_NO_VALUE: an absent operand
            Traced(ValueId id) noexcept;             // implicit by design
            Traced(Status failure) noexcept;         // implicit by design; fatal when failure.is_ok()

            [[nodiscard]] bool has_value() const noexcept;
            [[nodiscard]] const Status& status() const noexcept;
            [[nodiscard]] ValueId value() const noexcept;   // fatal when a failure is held

        private:
            ValueId id_ = K_NO_VALUE;
            Status status_;
    };
}
```

`Traced` is its own class and not an alias of `Result<ValueId>`. `ValueId` is `std::uint32_t`, so
an alias would be the same type as any other `Result<std::uint32_t>`. The default constructor
holds `K_NO_VALUE`, the representation the IR already uses for an absent operand
(`src/nn/conv2d.cpp:97`), so a rule that declares a local and assigns it in a branch
(`src/autograd/rules/linalg.cpp:49-53`) changes only the type name.

Every emitter that checks content returns `Traced` and takes `const Traced&` operands:

```cpp
[[nodiscard]] Traced matmul(const Traced& a, const Traced& b,
                            bool transpose_a = false, bool transpose_b = false) const noexcept;
[[nodiscard]] Traced bias_add(const Traced& x, const Traced& bias) const noexcept;
[[nodiscard]] Result<PoolResult> maxpool2d(const Traced& x, const ConvGeometry& g) const noexcept;
```

- An emitter handed a failed operand returns that failure and emits nothing, so nested calls and
  reassignment read as they do today.
- `input` and `parameter` hold no check (`src/graph/builder.cpp:134-147`) and keep returning
  `ValueId`. So does `NN::Parameter::declare` (`include/SushiAI/nn/module.hpp:76-93`).
- The multi-result structs `PoolResult`, `BatchNormResult` and `BatchNormGrads`
  (`include/SushiAI/graph/builder.hpp:125-139`) keep their `ValueId` fields and travel inside a
  `Result`.

**A dropped failure.** A failed `Traced` that nothing reads is caught at build time only by
`[[nodiscard]]`, and builds are off. The graph therefore carries one flag:

```cpp
// Graph
void mark_refused() noexcept;
[[nodiscard]] bool has_refusal() const noexcept;

// GraphBuilder
[[nodiscard]] Status refuse(Status failure) const noexcept;      // marks the graph, returns failure
[[nodiscard]] Status observe(Status outcome) const noexcept;     // marks when outcome is a failure
template<typename T>
[[nodiscard]] Result<T> observe(Result<T> outcome) const noexcept;

#define SA_REFUSE_IF(builder, condition, ...)   /* if (condition) return (builder).refuse(...) */
```

Every failure created on the trace surface, in an emitter or in a layer's `forward`, goes through
`refuse` or `observe`. `CompiledStep::compile` and `Autograd::differentiate` return
`InvalidDescription` for a graph whose flag is set. The message stays in the `Traced`; the graph
stores one `bool` and no text. A const builder can set the flag because `GraphBuilder::graph()`
already returns a mutable graph (`include/SushiAI/graph/builder.hpp:155`).

**Standard-container `.at(`.** `Node::inputs` and `Node::outputs` are vectors, and the rules and
the lowering index them with `.at(`: `src/autograd/rules/linalg.cpp:41-42`, `:84`,
`loss.cpp:56-57`, `:78`, `elementwise.cpp:87-223` (12 calls), `activation.cpp:51-52`,
`src/graph/cast_insertion.cpp:201`, `src/graph/lowering.cpp:817`, `:881`,
`src/graph/memory_plan.cpp:340`. Under a `noexcept` surface each `std::out_of_range` is
`std::terminate` with no message. Each becomes `[]` behind one `SA_FATAL_IF` on the node's arity
at the top of the function. The other 16 `.at(` lines are `Config::Value::at` and
`ObjectiveLeaves::at`, which are SushiAI's own functions.

### 2.8 Layers

`forward` takes and returns `Traced`:

```cpp
// nn/layer.hpp (today nn/ilayer.hpp:57)
[[nodiscard]] virtual Graph::Traced forward(const Graph::GraphBuilder& builder,
                                            const Graph::Traced& input) noexcept = 0;
```

The `Module` trait pins the exact types of `forward` (`include/SushiAI/nn/module.hpp:218-225`),
and `LayerAdapter` repeats them in a `static_assert` and in its override
(`include/SushiAI/nn/ilayer.hpp:84-88`, `:104-108`). All three change in the same commit. The
interface's `output_shape()` takes no argument and is `noexcept` already (`ilayer.hpp:69`); the
concrete `output_shape(const Shape&)` of each layer returns `Result<Shape>`. `LayerFactory`
returns `Result<std::unique_ptr<Layer>>`. Adding a layer stays one file and one registry row.

The programme adds multi-input layers in stage 3 (`docs/design/PROGRAMME.md:47`). The form of a
multi-input `forward` is settled before this wave edits `forward`, or every layer, the trait and
the adapter are edited twice.

### 2.9 Gradient rules

```cpp
// autograd/rule.hpp (today :40)
using BackwardRule = void (*)(const GradientTape&) noexcept;

// autograd/tape.hpp (today :84-93)
void contribute(Graph::ValueId primal, const Graph::Traced& gradient) const noexcept;
void contribute_input(std::size_t index, const Graph::Traced& gradient) const noexcept;
void refuse(const Status& failure) const noexcept;

// autograd/backward.hpp
[[nodiscard]] Result<std::vector<Graph::ValueId>> differentiate(...) noexcept;
```

The rule signature keeps returning `void`. That departs from the wording "every operation that
can fail returns a status", and section 9 puts it to the owner. The reasons:

- A rule's only output is a contribution. `contribute` given a failed `Traced` stores the first
  failure in the `Differentiator` and ignores later contributions. `differentiate()` is the
  `Differentiator`'s single exit and returns that failure, annotated with the op and node of the
  rule that produced it. A rule has no status to forget.
- With `Status`-returning rules every contribution needs an `SA_TRY`, and a missing one drops a
  gradient with nothing reported.

The failure keeps the code its emitter gave it. `Graph::add_node` is public and checks neither
arity nor shape (`include/SushiAI/graph/ir.hpp:235-261`), so a forward graph the emitters never
saw can reach a rule; the failure is then a wrong description. For a graph built only through the
builder the same failure is a defect in the rule, reported under `InvalidDescription` with a
message that names the rule.

Other changes in `autograd`:

- The tape's index checks (`src/autograd/tape.cpp:49`, `:57`, `:65`, `:78`) and the ordering and
  shape checks in `Differentiator::contribute` (`src/autograd/differentiator.cpp:61`, `:70`) are
  fatal.
- The non-scalar root, the root nothing produces, the root not produced by `NLL_LOSS` and the
  missing rule (`src/autograd/differentiator.cpp:138-154`, `:200`) are content and return
  statuses.
- A rule that unpacks a multi-result emitter calls `tape.refuse(result.status())` and returns.
- `src/autograd/rules` holds 56 `Graph::ValueId` local declarations. Twelve are initialised from
  an emitter call and two are assigned from one after an uninitialised declaration
  (`linalg.cpp:49`, `:59`); those fourteen change type to `Graph::Traced`.

### 2.10 The surface is `noexcept`

Every public function is `noexcept` once its module is converted. An exception that would leave a
public function, `std::bad_alloc` and `std::length_error` from SushiAI's own containers among
them, ends the process through `std::terminate`. Host heap exhaustion therefore never becomes a
status, and no exception crosses the surface. Section 9 records this as a ruling the owner makes,
not as an assumption.

`noexcept` is added in the last wave only. Until a module's callees are converted, a `noexcept`
on it turns a recoverable `Error` into `std::terminate`.

### 2.11 The eager surface

The eager specification draws the caller's loop with `Graph::ValueId` and no status
(`docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md:37-56`) and says the names are proposals (`:29`).
Its sentence "No layer, rule or builder method changes for the eager path" (`:60-61`) is false
under this design and needs an edit. The surface freezes in this shape:

```cpp
class Session final
{
    public:
        [[nodiscard]] static Result<std::unique_ptr<Session>> make(Runtime& runtime,
                                                                   const SessionOptions& options) noexcept;
        [[nodiscard]] Result<Step> begin_step() noexcept;
};

class Step final
{
    public:
        [[nodiscard]] Graph::Traced input(const SushiBLAS::Tensor& tensor) noexcept;
        [[nodiscard]] const Graph::GraphBuilder& builder() const noexcept;
        [[nodiscard]] Status backward(const Graph::Traced& loss) noexcept;
        [[nodiscard]] Status apply(Optim::Optimizer& optimizer) noexcept;
        [[nodiscard]] Result<double> read_scalar(const Graph::Traced& value) noexcept;
};
```

A failure in the trace surfaces at `backward`, `apply` or `read_scalar`. Section 3.6 shows the
loop.

### 2.12 State after a failure

A failure that returns leaves state behind that a thrown exception used to abandon with the
program.

| State | What holds it | Rule |
| --- | --- | --- |
| A half-traced graph | `Graph` | A graph with `has_refusal()` is refused by `compile` and `differentiate`. The caller discards it |
| Parameters declared into that graph | `Parameter::id_`; `declare` returns the cached id for any builder (`include/SushiAI/nn/module.hpp:84`) | `NN::forget_declarations(model)` clears the ids of every parameter of a model, so the model can be traced into a new graph. Section 9, D9 |
| Tasks recorded before a lowering failed | the engine's task graph | The seam drains them (section 4.3) |
| Tensors after a failed run | `TensorPool` | Unspecified. With the optimizer in the plan, parameters may be partly updated. `DeviceFailed` and `OutOfMemory` from `run()` mean the caller reloads a checkpoint |

The second row is also open for a run with no failure: with one graph per eager step
(`docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md:73`), the second step receives an id issued by
the first step's graph. The eager design owns that case.

### 2.13 Naming

**Mapping rule.**

| Kind | Rule | Examples |
| --- | --- | --- |
| Constant (`constexpr`, `static const`) | `K_` plus the present spelling | `NO_VALUE` to `K_NO_VALUE` (`include/SushiAI/graph/ir.hpp:48`); `MAX_RANK` to `K_MAX_RANK` (`core/common.hpp:52`); `Ops::MATMUL` to `Ops::K_MATMUL`; `KNOWN_DTYPES` to `K_KNOWN_DTYPES` (`src/io/checkpoint_format.cpp:45`) |
| Enumerator | split at underscores, capitalise each part, lower-case the rest | `ParameterRole::WEIGHT` to `Weight` (`include/SushiAI/nn/module.hpp:54`); `GEMM_BIAS_RELU` to `GemmBiasRelu`; `NHWC` to `Nhwc`; `Kind::NUL` to `Null` by hand |
| Interface | drop the `I` | `ILayer` to `Layer`, `IObjective`, `IFusionSelector`, `IPrecisionPolicy`, `IStepRecorder`; `nn/ilayer.hpp` moves with its type |
| Unsized enum | add `: std::uint8_t` | `Optim::UpdateKernel`, `Train::OptimizerPlacement` |
| Unscoped enum used as an index | `K_` constants of type `std::size_t` | the two `Scalar` enums in `src/optim/adamw.cpp` and `src/optim/sgd.cpp` |
| Macro | `SA_` | the new macros of this design |

**Names that do not change.** The wave renames identifier tokens and never the inside of a string
literal. That protects:

- the `sushiai.*` op strings in `include/SushiAI/core/ops.hpp`;
- JSON keys, type names and the document version keys;
- the parameter names `"weight"` and `"bias"` (`src/nn/layers.cpp:60`, `:69`,
  `src/nn/conv2d.cpp:95`, `:101`), which checkpoints store;
- the checkpoint magic;
- the name tables tests match, `dtype_name` among them (`include/SushiAI/core/common.hpp:83-89`);
- op names printed in capitals inside messages (`src/graph/builder.cpp:161`), which
  `tests/functional/unit/test_diagnostics.cpp:101` matches.

The wave never reorders an enumerator or changes its value. The checkpoint reads a `DType` as
its numeric value (`src/io/checkpoint_format.cpp:59`, `:210`), and `DType` is an alias of
`SushiBLAS::Core::DataType` (`include/SushiAI/core/common.hpp:46`), so a reorder in SushiBLAS
changes the meaning of every checkpoint file. This pass adds one `static_assert` per enumerator
value in `checkpoint_format.cpp`.

**Names SushiAI cannot rename.** `DType::FLOAT32`, `ExecutionMode::FORCE_REFERENCE` and the other
sibling enumerators stay in capitals, so `Lifetime::Activation` sits beside `DType::FLOAT32` in
one expression. "One naming form" (`docs/design/PROGRAMME.md:46`) then holds for names SushiAI
declares and not for names it aliases. Section 9, D11 gives the two ways to close that.

**Renaming with no compiler in the loop.** A tokenising tool, `tools/naming/rename_identifiers.py`,
reads a reviewed mapping file of `qualified old name -> new name`. Its lexer knows comments,
string literals, character literals and raw strings. It rewrites identifier tokens, and inside
comments only after `@ref`, `@c` and `@see`. Its safeguards:

1. It refuses to write when a new name already exists as a token anywhere in the tree.
2. It extracts every string literal before and after and requires the two multisets to be equal.
3. It requires zero old tokens left and, per file, as many new tokens as there were old ones.
4. It applies the inverse mapping to its own output and requires the original tree byte for byte.
5. A commit of the wave is the tool's output and nothing else, so a second run reproduces it.

Enumerators go first, in the qualified form `Enum::NAME` only, with the enum bodies edited from
the same mapping. Constants go second; by then a bare `RELU` or `GEMM_BIAS` can only be an op
constant. Interfaces and the file move go third. The enum underlying types and the two unscoped
enums go last, by hand. Bare occurrences the tool cannot resolve by qualifier are printed as a
list for a person to clear.

`build/compile_commands.json` exists. The Builds ruling permits single-file syntax checks
(`docs/design/PROGRAMME.md:30`). If the owner allows them for this wave, `clang -fsyntax-only`
over every translation unit after each commit is the strongest check on the rename and on every
changed signature. The safeguards above stand without it.

## 3. Before and after

### 3.1 `GraphBuilder::matmul`

Before (`src/graph/builder.cpp:155-176`, checks abridged):

```cpp
ValueId GraphBuilder::matmul(ValueId a, ValueId b, bool transpose_a, bool transpose_b) const
{
    const Value& va = graph_->value(a);
    const Value& vb = graph_->value(b);

    SA_THROW_IF(va.shape.rank() != 2 || vb.shape.rank() != 2,
                "MATMUL needs two rank-2 operands; got {} and {}",
                va.shape.to_string(), vb.shape.to_string());
    ...
```

After:

```cpp
Traced GraphBuilder::matmul(const Traced& left, const Traced& right,
                            bool transpose_a, bool transpose_b) const noexcept
{
    SA_TRY_ASSIGN(const ValueId a, left);
    SA_TRY_ASSIGN(const ValueId b, right);

    const Value& va = graph_->value(a);
    const Value& vb = graph_->value(b);

    SA_REFUSE_IF(*this, va.shape.rank() != 2 || vb.shape.rank() != 2,
                 "MATMUL needs two rank-2 operands; got {} and {}",
                 va.shape.to_string(), vb.shape.to_string());
    ...
```

The message literal does not change.

### 3.2 `Linear::forward`

Before (`src/nn/layers.cpp:45-74`):

```cpp
Graph::ValueId Linear::forward(const Graph::GraphBuilder& builder, Graph::ValueId x)
{
    require_positive_features(*this);

    const Shape& input_shape = builder.graph().value(x).shape;
    SA_THROW_IF(input_shape.rank() != 2,
                "Linear needs a rank-2 input; got {}", input_shape.to_string());
    SA_THROW_IF(input_shape[1] != in_features,
                "Linear({}, {}) was given an input of {}; its inner extent must be {}",
                in_features, out_features, input_shape.to_string(), in_features);

    const DType dtype = builder.graph().value(x).dtype;

    const Graph::ValueId w = weight.declare(builder, Shape{out_features, in_features},
                                            ParameterRole::WEIGHT,
                                            in_features, out_features, "weight", dtype);

    Graph::ValueId y = builder.matmul(x, w, false, true);

    if (use_bias)
    {
        const Graph::ValueId b = bias.declare(builder, Shape{out_features},
                                              ParameterRole::BIAS,
                                              in_features, out_features, "bias", dtype);
        y = builder.bias_add(y, b);
    }

    return y;
}
```

After:

```cpp
Graph::Traced Linear::forward(const Graph::GraphBuilder& builder,
                              const Graph::Traced& input) noexcept
{
    SA_TRY_ASSIGN(const Graph::ValueId x, input);
    SA_TRY(builder.observe(check_features(*this)));

    const Shape& input_shape = builder.graph().value(x).shape;
    SA_REFUSE_IF(builder, input_shape.rank() != 2,
                 "Linear needs a rank-2 input; got {}", input_shape.to_string());
    SA_REFUSE_IF(builder, input_shape[1] != in_features,
                 "Linear({}, {}) was given an input of {}; its inner extent must be {}",
                 in_features, out_features, input_shape.to_string(), in_features);

    const DType dtype = builder.graph().value(x).dtype;

    const Graph::ValueId w = weight.declare(builder, Shape{out_features, in_features},
                                            ParameterRole::Weight,
                                            in_features, out_features, "weight", dtype);

    Graph::Traced y = builder.matmul(x, w, false, true);

    if (use_bias)
    {
        const Graph::ValueId b = bias.declare(builder, Shape{out_features},
                                              ParameterRole::Bias,
                                              in_features, out_features, "bias", dtype);
        y = builder.bias_add(y, b);
    }

    return y;
}
```

What changes: the signature, one added first line, the helper call, the macro on the two checks
and the type of `y`. The emitting half keeps its text. `check_features` returns `Status` and
serves `output_shape` as well.

### 3.3 `Conv2d`

Before (`src/nn/conv2d.cpp:46-71`, `:74-105`):

```cpp
[[nodiscard]] Graph::SpatialExtents fit_window(const Conv2d& layer, const Shape& sample)
{
    SA_THROW_IF(layer.in_channels <= 0 || layer.out_channels <= 0, ...);
    SA_THROW_IF(sample.rank() != 3 || sample[2] != layer.in_channels, ...);

    Graph::SpatialExtents extents;
    try
    {
        extents = Graph::window_extents(sample[0], sample[1], layer.geometry);
    }
    catch (const Error& error)
    {
        SA_THROW_IF(true,
                    "Conv2d({}, {}) needs a legal window over its per-sample shape; "
                    "got {}: {}",
                    layer.in_channels, layer.out_channels, sample.to_string(),
                    std::string_view(error.what()));
    }

    return extents;
}

Graph::ValueId Conv2d::forward(const Graph::GraphBuilder& builder, Graph::ValueId x)
{
    const Shape image = builder.graph().value(x).shape;
    const DType dtype = builder.graph().value(x).dtype;

    SA_THROW_IF(image.rank() != 4, ...);
    SA_THROW_IF(builder.graph().value(x).layout != Graph::TensorLayout::NHWC, ...);
    (void)fit_window(*this, Shape{image[1], image[2], image[3]});
    ...
    return builder.conv2d(x, w, b, geometry);
}
```

After:

```cpp
[[nodiscard]] Result<Graph::SpatialExtents> fit_window(const Conv2d& layer,
                                                       const Shape& sample) noexcept
{
    SA_FAIL_IF(layer.in_channels <= 0 || layer.out_channels <= 0,
               StatusCode::InvalidDescription, ...);
    SA_FAIL_IF(sample.rank() != 3 || sample[2] != layer.in_channels,
               StatusCode::InvalidDescription, ...);

    Result<Graph::SpatialExtents> extents =
        Graph::window_extents(sample[0], sample[1], layer.geometry);
    if (!extents.has_value())
        return SA_ANNOTATE(extents.status(),
                           "Conv2d({}, {}) needs a legal window over its per-sample shape; got {}",
                           layer.in_channels, layer.out_channels, sample.to_string());

    return extents;
}

Graph::Traced Conv2d::forward(const Graph::GraphBuilder& builder,
                              const Graph::Traced& input) noexcept
{
    SA_TRY_ASSIGN(const Graph::ValueId x, input);

    const Shape image = builder.graph().value(x).shape;
    const DType dtype = builder.graph().value(x).dtype;

    SA_REFUSE_IF(builder, image.rank() != 4, ...);
    SA_REFUSE_IF(builder, builder.graph().value(x).layout != Graph::TensorLayout::Nhwc, ...);
    SA_TRY(builder.observe(fit_window(*this, Shape{image[1], image[2], image[3]})));
    ...
    return builder.conv2d(x, w, b, geometry);
}

Result<Shape> Conv2d::output_shape(const Shape& input) const noexcept
{
    SA_TRY_ASSIGN(const Graph::SpatialExtents out, fit_window(*this, input));
    return Shape{out.height, out.width, out_channels};
}
```

`window_extents` (`src/graph/builder.cpp:60-86`) is checked in one place and annotated, so the
`try` block goes and no twin of the check exists. The annotated text is the text the catch block
builds today, so test needles hold.

The cast at `src/nn/conv2d.cpp:87` needs a hand edit. `(void)fit_window(...)` is legal today
because the function throws; left as it is after the conversion, it drops a failure. Four more
casts of this kind exist: `src/nn/avg_pool2d.cpp:78`, `max_pool2d.cpp:78`, `upsample2d.cpp:78`
and `src/optim/optimizer.cpp:142`.

### 3.4 The registry's `adapt`

Before (`src/nn/layer_registry.cpp:210-224`):

```cpp
template<typename L>
std::unique_ptr<ILayer> adapt(L layer, const Shape& input, std::string_view context)
{
    Shape output;
    try
    {
        output = layer.output_shape(input);
    }
    catch (const Error& error)
    {
        SA_THROW_IF(true, "{}: {}", context, std::string_view(error.what()));
    }

    return std::make_unique<LayerAdapter<L>>(std::move(layer), output);
}
```

After:

```cpp
template<typename L>
Result<std::unique_ptr<Layer>> adapt(L layer, const Shape& input, std::string_view context) noexcept
{
    const Result<Shape> output = layer.output_shape(input);
    if (!output.has_value())
        return SA_ANNOTATE(output.status(), "{}", context);

    std::unique_ptr<Layer> adapted =
        std::make_unique<LayerAdapter<L>>(std::move(layer), output.value());
    return adapted;
}
```

### 3.5 A gradient rule

Before (`src/autograd/rules/linalg.cpp:40-55`):

```cpp
const Graph::GraphBuilder& builder = tape.builder();
const Graph::ValueId a = node.inputs.at(0);
const Graph::ValueId b = node.inputs.at(1);
...
if (tape.input_requires_grad(0))
{
    Graph::ValueId da;
    if (!ta && !tb)     da = builder.matmul(dc, b, false, true);
    else if (ta && !tb) da = builder.matmul(b, dc, false, true);
    else if (!ta)       da = builder.matmul(dc, b, false, false);
    else                da = builder.matmul(b, dc, true, true);
    tape.contribute_input(0, da);
}
```

After:

```cpp
SA_FATAL_IF(node.inputs.size() != 2,
            "MATMUL takes two operands; this node has {}", node.inputs.size());

const Graph::GraphBuilder& builder = tape.builder();
const Graph::ValueId a = node.inputs[0];
const Graph::ValueId b = node.inputs[1];
...
if (tape.input_requires_grad(0))
{
    Graph::Traced da;
    if (!ta && !tb)     da = builder.matmul(dc, b, false, true);
    else if (ta && !tb) da = builder.matmul(b, dc, false, true);
    else if (!ta)       da = builder.matmul(dc, b, false, false);
    else                da = builder.matmul(b, dc, true, true);
    tape.contribute_input(0, da);
}
```

### 3.6 The eager loop

Before (`docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md:37-56`, abridged):

```cpp
Eager::Step step = session.begin_step();
Graph::ValueId x = step.input(batch.inputs);
Graph::ValueId h = encoder.forward(step.builder(), x);
...
Graph::ValueId loss = objective.loss(step.builder(), h, step.input(batch.targets));
step.backward(loss);
step.apply(optimizer);
const double value = step.read_scalar(loss);
```

After:

```cpp
SA_TRY_ASSIGN(Eager::Step step, session.begin_step());
Graph::Traced x = step.input(batch.inputs);
Graph::Traced h = encoder.forward(step.builder(), x);
...
Graph::Traced loss = objective.loss(step.builder(), h, step.input(batch.targets));
SA_TRY(step.backward(loss));
SA_TRY(step.apply(optimizer));
SA_TRY_ASSIGN(const double value, step.read_scalar(loss));
```

The trace lines change only in their type name.

### 3.7 An app

Before (`apps/train.cpp:56-58`, `:171-177`):

```cpp
int main(int argc, char** argv)
{
    try
    {
        ...
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sushiai train failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
```

After:

```cpp
int main(int argc, char** argv)
{
    const sa::Status status = run(argc, argv);
    if (status.is_ok())
        return EXIT_SUCCESS;

    std::cerr << "sushiai train failed: " << status.message() << "\n"
              << "  kind: " << sa::status_code_name(status.code()) << "\n";
    return EXIT_FAILURE;
}
```

- The first line is the line printed today. The second line is new.
- Exit codes stay 0 and 1. A fatal check ends the process through `std::abort()` after it prints
  `sushiai: fatal: <message> (<file>:<line>)`.
- The `sa` CLI passes the program's exit code through unchanged: `cli/sushiai/cli.py:223` raises
  `typer.Exit` with the value `run_app` returns, and `run_app` returns the child's code
  (`cli/sushiai/services/project.py:524-547`). The child inherits the CLI's streams. The CLI
  inspects neither the code's value nor the text.
- The CLI's tests assert a non-zero code and a substring of the combined output
  (`cli/tests/test_train.py:90-92`, `:101-103`); both hold.
- CTest pins the `compiles:` line by regular expression and not by exit code
  (`tests/CMakeLists.txt:129`, `:145`, `:179`, `:194`); that line does not move.
- `examples/demo_mlp.cpp:161-213` catches `std::logic_error` from the `std::sto*` family; it
  moves to `std::from_chars`.

## 4. What crosses the seam

### 4.1 What the siblings throw after their v0.1.0

| Source | Exception | Where stated |
| --- | --- | --- |
| SushiBLAS, record time | `std::invalid_argument` for every refusal, a device-only family under `FORCE_REFERENCE` and a device-only USM operand among them | `sushiblas/docs/agent/2026_10_07_V010/REPORT.md`, heading `EXCEPTIONS`; a device or mode refusal is `SushiBLAS::UnsupportedExecution`, derived from it |
| SushiBLAS, tensor creation | `std::runtime_error` | `sushiblas/docs/agent/2026_10_07_V010/REPORT.md`, heading `EXCEPTIONS` |
| SushiBLAS, inside a task | `std::runtime_error` for a device that cannot host the work | `sushiblas/docs/agent/2026_10_07_V010/REPORT.md`, heading `EXCEPTIONS`, "Execute time" |
| SushiRuntime | `Core::SushiException` carrying an `Errc` | `sushiruntime/include/SushiRuntime/core/error.hpp:46-57`, `:100-126` |
| SushiRuntime, a run | the first exception a task threw | `sushiruntime/include/SushiRuntime/execution/execution_engine.hpp:211`, `:346` |

`std::invalid_argument` therefore covers two cases the seam cannot tell apart: a user who chose
`FORCE_REFERENCE` for a model with a device-only family, and a lowering that recorded a wrong
operand. SushiBLAS plans no status surface, so the conversion is SushiAI's job.

### 4.2 The seam unit

`include/SushiAI/core/seam.hpp` and `src/core/seam.cpp`. The header is public because two public
headers call the engine inline (`include/SushiAI/nn/init.hpp:194`,
`include/SushiAI/io/checkpoint.hpp:113`).

```cpp
namespace SushiAI::Detail
{
    /** @brief Turns the exception being handled into a Status; call only inside a catch block. */
    [[nodiscard]] Status status_of_current_exception(std::string_view what) noexcept;

    /** @brief Runs @p record; on a throw drains the pending tasks and returns the failure. */
    template<typename Record>
    [[nodiscard]] Status record_guarded(SushiBLAS::Engine& engine, std::string_view what,
                                        Record&& record) noexcept;

    [[nodiscard]] Result<SushiRuntime::Execution::CompiledGraph>
        compile_recorded(SushiBLAS::Engine& engine) noexcept;
    [[nodiscard]] Status run_recorded(SushiBLAS::Engine& engine) noexcept;   // execute, then wait
    [[nodiscard]] Status run_plan(SushiBLAS::Engine& engine,
                                  SushiRuntime::Execution::CompiledGraph& plan) noexcept;
    [[nodiscard]] Result<SushiRuntime::sushi_ptr<SushiBLAS::Storage>>
        allocate_storage(...) noexcept;
}
```

Call sites:

| Site | Today | After |
| --- | --- | --- |
| `src/graph/compiled_step.cpp:39-47` | `lowering.record(graph)`, then the trailing recorder | one `record_guarded` around the per-node dispatch of the lowering tables, one around the trailing recorder |
| `src/graph/compiled_step.cpp:50` | `engine.get_graph().compile()` | `compile_recorded` |
| `src/graph/compiled_step.cpp:69` | `execute(plan_)` | `run_plan` |
| `src/graph/lowering.cpp:196`, `src/optim/optimizer.cpp:149`, `src/train/trainer.cpp:261`, `src/data/dataset.cpp:226`, `:255`, `include/SushiAI/nn/init.hpp:194`, `include/SushiAI/io/checkpoint.hpp:113` | `engine.execute()` | `run_recorded` |
| `src/core/allocation.cpp:90-102`, `:156-168` | `try` around `make_sushi`, rethrown as `Error` | `allocate_storage` |

Mapping, in catch order:

| Caught | Becomes |
| --- | --- |
| `SushiException` with `allocation_failed` | `OutOfMemory` |
| `SushiException` with `kernel_failed` or `device_lost` | `DeviceFailed` |
| `SushiException` with `unsupported` | `Unsupported` |
| `SushiException` with `invalid_graph`, `compile_failed`, `invalid_access` or `dangling_resource` | `SA_FATAL` with the runtime's text: SushiAI built the plan wrongly |
| `std::bad_alloc` | `OutOfMemory` |
| `std::invalid_argument` at record time | `Unsupported`, annotated with the op name and the node id |
| `std::exception` at an allocation site | `OutOfMemory` with `what()` |
| any other `std::exception`, or `...` | `DeviceFailed` with `what()` where there is one |

The stage 1 lowering table entry is declared `Status (*)(...)` from the start, so the 57 sites of
`src/graph/lowering.cpp` are written once.

The eager path calls `DynamicGraph::try_run`, which returns SushiRuntime's own `Result`; the seam
converts its `Error` by the same `Errc` rows without a catch.

After the wave `try` and `catch` appear in four files: `core/status.hpp`, `core/seam.hpp`,
`src/core/seam.cpp` and `src/config/value.cpp`, where the block around nlohmann's parser stays as
the third-party rule requires (`src/config/value.cpp:317-326`) and returns `InvalidFile`.

### 4.3 A lowering that fails part way

A lowering that fails after recording some tasks leaves them in the engine's task graph, and the
next `compile()` picks them up. Exceptions hide this today because the program usually ends.

`TaskGraph::take_nodes()` hands off the built nodes without finalising a plan
(`sushiruntime/include/SushiRuntime/graph/task_graph.hpp:234-238`). `record_guarded` calls it on
a failure and drops the nodes. After v0.1.0 the recorder is constructed over a `TaskGraph` the
caller supplies (`sushiblas/docs/agent/2026_10_07_V010/SPEC.md`, heading `DESIGN_RECORDER`), so the graph to drain is one SushiAI holds.
Whether dropping the nodes also releases what the recorder retained for them is listed in
section 10.

### 4.4 Requests to SushiBLAS v0.1.0

SushiBLAS v0.1.0 is being designed in the same pass, so these go to its design now:

1. A distinct exception type, or a predicate on the recorder, for a refusal caused by the
   execution mode. `Unsupported` then stops covering lowering defects.
2. A statement that `DataType`'s enumerator values are a file-format contract.
3. The enumerator rename of section 2.13, if the owner chooses that route in D11.

## 5. Order of work

A wave ends with the whole tree in a state that compiles. Between waves the tree is never half
converted across a module boundary: a caller in a module not yet converted reaches a converted
function through `Detail::bridge(result)`, which throws the old `Error`. A checker prints the
bridge count per module and the last wave requires zero.

| Wave | Content | Files, disjoint between agents inside the wave | Leaves compiling |
| --- | --- | --- | --- |
| 0 | Owner decisions. The per-site class table for all `SA_THROW_IF` lines. The `Shape{` list. The rename mapping file. The tools and their tests | this folder's `PLAN.md`; `tools/naming/`, `tools/contract/`, `tools/tests/` | unchanged; no source edit |
| 1 | Foundation, additive | `include/SushiAI/core/status.hpp`, `result.hpp`, `check.hpp`, `seam.hpp`; `src/core/status.cpp`, `check.cpp`, `seam.cpp`; `include/SushiAI/graph/traced.hpp`; `tests/common/test_status.hpp`, `test_environment.hpp`; three new unit test files | the whole tree, still under the old contract |
| 2 | Naming, at a freeze | the whole tree, by the tool; one commit per class | the whole tree after each commit |
| 3 | Fatal class: `SA_THROW_IF` to `SA_FATAL_IF` at sites the table marks fatal; `.at(` to `[]`; the tests of those sites | one agent per module: `include/SushiAI/<module>`, `src/<module>` and the test files the table assigns to it | the whole tree after each module; no signature changes |
| 4 | Leaf modules to statuses | `core` (shape, allocation), then `config`, `imaging`, `detect`, `io`, `data`; each with its tests | the whole tree after each module, through the bridge |
| 5 | The trace surface, as one wave | `graph` host units (`ir.hpp`, `builder`, the passes, the memory plan); `autograd`; `nn`; `train/objective`; their tests | at the end of the wave only |
| 6 | The engine lane, together with stage 1 | the lowering tables, `TensorPool`, `CompiledStep`; `optim`; `nn/init`; `train`; `eager` | the whole tree after each module |
| 7 | Apps and examples | `apps/`, `examples/` | the whole tree |
| 8 | Removal: `Error`, `SA_THROW_IF`, the bridge, `test_throws.hpp`. `noexcept` on every public declaration. `@throws` to `@return` | the whole tree, by script and checker | the whole tree; the checkers gate on zero |

Notes on the order:

- **Wave 1 goes before the naming wave.** New code is written under the old contract today;
  `src/imaging` and `src/detect` grew by more than 100 sites after the inventory. Once the macros
  exist, the orchestrator adds one line to every dispatch that writes library code: new failure
  sites use the new macros. A ratchet checker holds a per-file baseline of `SA_THROW_IF` counts
  and fails when a count grows, which catches sites added after the class table was written.
- **Wave 1 adds three `.cpp` files.** `src/CMakeLists.txt` lists sources by name (`:15-25`), and
  build configuration is the owner's to edit. The wave hands over the lines to add.
- **The freeze for wave 2.** The rename needs a tree in which no stream holds uncommitted edits
  under `include/`, `src/`, `tests/`, `apps/` or `examples/`. This design cannot obtain that. The
  orchestrator stops dispatching to those paths, every stream commits, `git status --short` shows
  nothing under them, and the rename runs as the only dispatch. Today that listing shows 37
  entries.
- **Wave 5 does not compile between its parts.** The emitter signatures, the rules, the layers
  and the objective change type together. Agents take disjoint file sets and the wave is one
  consistent state at its end.
- **The graph module spans waves 5 and 6.** Its host units convert before the lowering, which
  stage 1 rewrites. Until wave 6 the lowering reaches converted graph functions through the
  bridge, inside one module. The alternative holds the host units back until stage 1 lands.
- **`forward` is edited once** only if the multi-input form is settled before wave 5.

## 6. What a script does and what needs judgement

Every script is a new file under `tools/` and needs the owner's approval before wave 0 ends. Each
has tests under `tools/tests/`, in the shape of `tools/tests/test_check_docs_layout.py`, that run
it over small fixture trees. A script's output is not proof while the script is unreviewed.

**A script does:**

| Work | Tool |
| --- | --- |
| The rename, with its five safeguards | `tools/naming/rename_identifiers.py` |
| `SA_THROW_IF` to `SA_FATAL_IF` at sites the class table marks fatal: a one-token edit | `tools/contract/convert_sites.py` |
| `SA_THROW_IF(c, fmt...)` to `SA_FAIL_IF(c, code, fmt...)` or `SA_REFUSE_IF(builder, c, fmt...)`, with the code from the table | the same |
| `.at(i)` to `[i]` on the listed lines | the same |
| `SA_EXPECT_ERROR` lines to `SA_EXPECT_FAILURE` or `SA_EXPECT_FATAL`, by matching each needle against the library's format strings and reading the site's class | `tools/contract/convert_tests.py` |
| Single-line `const ValueId y = builder.f(...)` in tests to `SA_ASSERT_OK_AND_ASSIGN` | the same |
| `@throws` lines to `@return` text in Doxygen blocks (113 lines today) | the same |
| `noexcept` on public declarations in wave 8 | the same |
| Listing: every `(void)` cast on a call, every call site of a changed signature, every bare `Shape{` with non-literal extents | `tools/contract/list_sites.py` |
| Checking: no `throw`, `try` or `catch` outside the four allowed files; no `SA_FAIL_IF` or `SA_TRY` in a function not declared to return `Status`, `Result` or `Traced`; no `SA_FAIL_IF` in a function that takes a `GraphBuilder`; no `SA_TRY_ASSIGN` as an unbraced body; no bare-statement call of a status-returning function; the bridge count; the `SA_THROW_IF` ratchet | `tools/contract/check_error_contract.py` |
| Checking the naming rule afterwards | `tools/naming/check_naming.py` |

**Judgement is needed for:**

- the class of each of the 510 lines, the 115 unclassified ones first;
- the status code of each status site;
- every return type and how its failure travels to the caller;
- the factory of each constructor in section 2.6;
- each of the five `(void)` casts;
- the four dummy returns;
- `CompiledStep`, `Trainer`, `Scope` and the seam mapping;
- the fourteen retyped locals in the rules, and each arity check that guards an index;
- multi-line test statements, and which fatal-class tests are worth a death test;
- the hand list of bare identifiers the rename tool could not resolve.

## 7. The tests that pin it

`tests/common/test_status.hpp` replaces `test_throws.hpp`:

```cpp
namespace SushiAI::Test
{
    ::testing::AssertionResult status_names(const Status& status, StatusCode expected,
                                            std::initializer_list<std::string_view> needles);

    /** @brief Joins @p needles with ".*" and escapes punctuation, for a death test. */
    std::string death_pattern(std::initializer_list<std::string_view> needles);
}

#define SA_EXPECT_FAILURE(expression, code, ...)      // Status, Result or Traced: the code and every needle
#define SA_EXPECT_OK(expression)                      // prints the message on failure
#define SA_ASSERT_OK_AND_ASSIGN(declaration, expression)  // fails this test with the message
#define SA_EXPECT_FATAL(statement, ...)               // EXPECT_DEATH on death_pattern
```

A status-class test, before (`tests/functional/unit/test_diagnostics.cpp:100-101`) and after:

```cpp
SA_EXPECT_ERROR((void)builder.matmul(vector, matrix),
                "MATMUL needs two rank-2 operands", "[5]", "[5, 3]");

SA_EXPECT_FAILURE(builder.matmul(vector, matrix), StatusCode::InvalidDescription,
                  "MATMUL needs two rank-2 operands", "[5]", "[5, 3]");
```

A fatal-class test, before (`tests/functional/unit/test_diagnostics.cpp:67`) and after:

```cpp
SA_EXPECT_ERROR((void)Shape({4, 0}), "Shape extents must be positive", "got 0");

SA_EXPECT_FATAL((void)Shape({4, 0}), "Shape extents must be positive", "got 0");
```

A passing path, before (`tests/functional/unit/test_ir.cpp:94`) and after:

```cpp
const ValueId y = builder.matmul(x, w, false, true);

SA_ASSERT_OK_AND_ASSIGN(const ValueId y, builder.matmul(x, w, false, true));
```

`.value()` is not used on a passing path in tests: a failure there aborts the whole binary, where
`SA_ASSERT_OK_AND_ASSIGN` fails one test and prints the message.

**Death tests.** The test binary links `GTest::gtest` without gmock
(`tests/CMakeLists.txt:97-102`), so a death test takes one regular expression. `death_pattern`
joins the needles with `.*` in the order they appear in the format string and escapes
punctuation. `error_names` matches needles in any order today
(`tests/common/test_throws.hpp:89`); the conversion script reorders them from the format string.

**The shared environment.** `TestEnvironment::SetUp` constructs the process-wide
`RuntimeContext` before the first test (`tests/common/test_environment.hpp:45-48`). A death test
on Windows starts the test binary again. `SetUp` therefore stops constructing: `context()`
builds the context on first use, so a child that tests `Shape`, the IR or the scope trace never
initialises a device.

**What each mechanism is pinned by:**

| Claim | Test |
| --- | --- |
| Success allocates nothing | `test_status.cpp`: a default `Status` holds a null pointer; copying it touches no heap |
| The failure path survives a throwing formatter | `test_status.cpp`: `from_message` with a lambda that throws returns `OutOfMemory` and the literal message |
| `annotated` yields "outer: inner" and keeps the code | `test_status.cpp` |
| `Result::value()` on a failure is fatal with the failure's message | `test_result.cpp`, a death test |
| A failed operand passes through an emitter and emits no node | `test_traced.cpp`: the node count does not change |
| A dropped failed emitter result is caught | `test_traced.cpp`: after a refused emitter, `CompiledStep::compile` and `differentiate` return `InvalidDescription` |
| No public function throws | `tests/functional/unit/test_noexcept_surface.cpp`: one `static_assert(noexcept(...))` per public function, added in wave 8 |
| Each seam row | `test_seam.cpp`: a callable that throws each exception type yields the mapped code |
| A failed recording leaves no task | `test_seam.cpp`: after `record_guarded` fails, `take_nodes()` on the graph is empty |
| A mode refusal is `Unsupported` | an integration test that records a device-only family under `FORCE_REFERENCE` |
| The checkpoint's dtype values | `static_assert`s in `src/io/checkpoint_format.cpp` |
| An app prints the same first line and returns 1 | `cli/tests/test_train.py:81-103`, unchanged |
| No `throw`, `try` or `catch` outside the allowed files | `check_error_contract.py`, with its own tests |
| The rename moved no string literal and reverses byte for byte | `rename_identifiers.py`'s safeguards 2 and 4, with its own tests |

## 8. Costs

Measured figures are counts of matching lines on 2026-10-07. Estimates are marked.

**Lines touched.**

| Item | Figure |
| --- | --- |
| Check sites | 510 lines, each edited once in wave 3, 4, 5 or 6 |
| Split of those between status and fatal | unknown until wave 0; the three earlier estimates put the fatal share between 165 and 315 of roughly 400 |
| Throw expectations in tests | 439 lines in the test sources, plus 25 no-throw lines |
| Test lines that call the builder, `forward` or `output_shape` | 1,243 in 37 files, by a regular expression that over-counts |
| Emitter calls in `include/` and `src/` | 81 lines, by the same kind of expression |
| `Shape{` in `include/` and `src/` | 35 lines to read; 759 in tests, untouched unless the extents come from data |
| `@throws` lines | 113 |
| Public signatures that gain a return type or `noexcept` | not counted; the inventory's header scan found 593 declarations, 199 already `noexcept`, before `imaging` and `detect` existed |
| Naming | about 1,700 identifier tokens in code and about 400 in documents by the inventory's counts, which predate `imaging` and `detect` and were not recounted |
| Tree size | 25,297 lines under `include/` and `src/`, 26,149 under `tests/`, 1,217 under `apps/` and `examples/` |

**Readability lost.**

- A layer's `forward` opens with `SA_TRY_ASSIGN(const Graph::ValueId x, input);`.
- Config and io code carries one `SA_TRY_ASSIGN` per read. A chain such as
  `input.at("features", context).as_int(...)` (`src/config/model_config.cpp:64`) splits into two
  statements.
- Four macros hide a `return`: `SA_FAIL_IF`, `SA_REFUSE_IF`, `SA_TRY`, `SA_TRY_ASSIGN`.
- Trace-surface code has two failure macros, and which one applies depends on whether the
  function takes a builder.
- `CompiledStep`, `Trainer` and the optimizers live behind `std::unique_ptr`.
- `Ops::K_GEMM_BIAS_GELU` and `TensorLayout::Nhwc` read worse than what they replace.
- The rule signature does not show that a rule can end a differentiation.

**Readability gained.**

- A signature states whether a call can fail.
- The five `try` blocks that only re-word a message go (four layers and the registry), the two
  in `allocation.cpp` move into the seam, and the four `main` functions lose theirs. Four dummy
  returns go.
- `Scope` cannot end the process without a message.
- One spelling per kind of name; `LINEAR_KEYS` no longer sits beside `K_LEAKY_RELU_KEYS`
  (`src/nn/layer_registry.cpp:56`, `:59`).

**Behaviour that changes.**

- A fatal-class failure ends the process where it was a catchable exception. A long-running host
  loses the process on an out-of-range id.
- Host heap exhaustion ends the process.
- A test that provokes a refused emitter and then compiles the same graph fails at the compile.
  It needs a second graph.
- Each fatal-class test costs one process start on Windows. The number of such tests and the
  start-up time are not measured.

**What breaks for a consumer.** Every source-level use: each `catch (const SushiAI::Error&)`,
the return type of every emitter and of `forward`, every converted constructor, the op constant
names, the enumerators of the library's enums, five interface names and one header path. No tag
exists (`docs/design/PROGRAMME.md:27`), so the consumers are the in-tree apps, examples and
tests, and no deprecation alias is proposed. File formats and the first line the apps print do
not change.

## 9. Decisions for the owner

| # | Question | Options | Recommendation |
| --- | --- | --- | --- |
| D1 | Is a wrong model written in C++ a status or a broken precondition? | (a) Status in JSON and in C++: content is a status, handle, order and state are fatal. (b) Emitters are preconditions and abort; each check reachable from data gets a status twin | (a). Under (b) a batch of the wrong size ends an eager run, and the emitter signatures change later under a frozen surface |
| D2 | What happens on host heap exhaustion? | (a) Public functions are `noexcept`; `std::bad_alloc` from SushiAI's own containers terminates. (b) No `noexcept`; a `try` in every public function | (a), recorded as a ruling |
| D3 | The always-on fatal macro | (a) `SA_FATAL_IF`, same polarity as `SA_THROW_IF`. (b) `SA_VERIFY` with assert polarity, each condition wrapped as `!(...)` by script | (a) |
| D4 | The shape of a failing emitter | (a) `Graph::Traced` as operand and result. (b) `Result<ValueId>` with one `SA_TRY_ASSIGN` per call, which forbids nested calls. (c) A handle whose message lives in the `Graph` | (a) |
| D5 | May `Graph` carry a refusal flag? | (a) One `bool`, read by `compile` and `differentiate`. (b) Nothing; rely on `[[nodiscard]]` and the text checker | (a). It is the only run-time catch for a dropped failure while builds are off |
| D6 | The gradient-rule signature | (a) `void (*)(const GradientTape&) noexcept`; a failed contribution returns from `differentiate()`. (b) `Status (*)(const GradientTape&)`, to the letter of the ruling | (a). This is a signed departure from "every operation that can fail returns a status" |
| D7 | Fatal-class tests | (a) Death tests, with the lazy `TestEnvironment`. (b) A test handler that throws, which forbids `noexcept` and lets an exception cross the surface in the test binary. (c) Drop them | (a), after the owner measures one death test's start-up |
| D8 | `Eager::Scope` and a refused name | (a) Fatal in the constructor, a published `is_legal_scope_name`, `open` and `close` public. (b) An inert `Scope` and a `seal()` that returns `Result`. (c) `open` and `close` private to `Scope`, which breaks 15 calls in `tests/functional/unit/test_scope_trace.cpp` | (a) |
| D9 | Who clears parameter ids after a failed trace? | (a) A free function `NN::forget_declarations(model)`. (b) `Parameter` records the graph it was declared in. (c) Left to the eager design | (a) now; (b) is the eager design's to judge |
| D10 | The eager surface | Freeze `Session`, `Step` and `Scope` with the signatures of section 2.11, or with `ValueId` and no status as the eager specification draws them | The signatures of section 2.11, and the multi-input `forward` settled before wave 5 |
| D11 | "One naming form" and sibling enumerators | (a) Ask SushiBLAS v0.1.0 to rename `DataType`, `Layout` and `ExecutionMode` enumerators. (b) Reword the stage's exit criterion to names SushiAI declares | (a) while its v0.1.0 is still on paper; otherwise (b) |
| D12 | Naming exceptions | The traits `Module` and `ModuleContainer` (`include/SushiAI/nn/module.hpp:286`, `:294`): `K_IS_MODULE` or exempt. `Nhwc` or `NHWC`. `Nonlinearity::Relu` beside the class `NN::ReLU`. `nn/layer.hpp` beside `nn/layers.hpp`, or `layers.hpp` split by type. `SUSHIAI_VERSION_*` kept or moved to `SA_` | Exempt the traits; `Nhwc`; `Relu`; split `layers.hpp`; keep `SUSHIAI_VERSION_*`, which matches `SUSHIBLAS_VERSION_STRING` (`apps/train.cpp:114`) |
| D13 | `CompiledStep` and `Trainer` | (a) Factories returning `Result<std::unique_ptr<T>>`. (b) Make both movable and return `Result<T>` | (a) |
| D14 | Exit codes | (a) 0 and 1. (b) A third code for description errors | (a) |
| D15 | Deletions | `SushiAI::Error`, `SA_THROW_IF`, `tests/common/test_throws.hpp`, the `nullptr` return of `try_load_mnist` | Delete all four in wave 8 |
| D16 | Permissions this work needs | New files under `tools/`. Source lines in `src/CMakeLists.txt` and `tests/CMakeLists.txt`. Unused-result warnings as errors. Per-file `clang -fsyntax-only` for these waves | Grant all four |
| D17 | Requests to SushiBLAS v0.1.0 | The three of section 4.4 | Send them |

## 10. Unverified

- **The class of every site.** No per-site table exists, and at least 115 lines have never been
  classified. Every cost that depends on the split is open until wave 0.
- **The tree moves.** Counts changed between two scans. The figures are those of one scan.
- **Death tests.** That a death-test child runs the global environment's `SetUp`, the regular
  expression dialect of the GTest version in use, and the start-up time of one child were not
  checked. Nothing was run.
- **`std::abort()` on Windows.** Whether the C runtime shows a dialog or reports a fault, and
  which exit code the process returns, was not checked. The `platform-portability` skill covers
  it.
- **Draining a failed recording.** `take_nodes()` was read in the header only. Whether dropping
  the nodes releases the recorder's retained storage and scratch buffers was not traced.
- **`execute`'s exceptions.** The header documents the rethrow of a task's exception on
  `complete()` (`execution_engine.hpp:211`, `:346`); `execute` (`:177`, `:328`) carries no
  `@throws` line. The inventory states that `execute` rethrows as well.
- **`DynamicGraph::try_run`** was not read; its shape is taken from the inventory.
- **`CompiledGraph` in a `Result`.** `plan_` is assigned from `compile()`
  (`src/graph/compiled_step.cpp:50`), so the type is move-assignable; its move constructor was
  not read.
- **A `noexcept` function-pointer type in the `constexpr` rule tables** is valid C++17; it was
  not checked against this toolchain.
- **Module order.** The order of waves 4 to 6 follows the includes read here.
  `tools/layering/check_layering.py` holds the authoritative order and was not extracted.
- **`sa test`.** The CLI's `run`, `demo` and `train` paths were read; the path that runs CTest
  was not.
- **The headers of `imaging` and `detect`** were not read; only their `SA_THROW_IF` lines were.
- **The naming counts** are the inventory's and were not recounted.
- **The second file-sourced `Shape`** at `include/SushiAI/io/checkpoint.hpp:96` was found by
  search; the checks before it were not read.
- **Buffers and gradients.** `GraphBuilder::parameter` sets `requires_grad` on everything it
  declares (`src/graph/builder.cpp:144`) and `Differentiator::contribute` does not test it
  (`src/autograd/differentiator.cpp:59-75`). A fatal check there was proposed; no one has read
  every rule against it, so it is left to stage 3.
- **Nothing here was compiled.** Every declaration in this document is unchecked text.
