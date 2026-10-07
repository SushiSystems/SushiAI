# Error contract and naming wave: plan

**Status:** In progress. Waves 0 and 1; written and syntax-checked only.

The design is [`SPEC.md`](SPEC.md). This file records what each wave put on disk, where it left
the design, and what the next wave has to do to existing files.

## Wave 1: foundation

Wave 1 adds ten files and changes none. The tree stays under the old contract: nothing includes
the new headers except their own three test files. No file here was compiled to an object, linked
or run. Each `.cpp` passed `clang++ -fsyntax-only` with the flags of the build's own compile
command (`-Wall -Wextra -Werror -pedantic`, `-DNDEBUG`); `test_status.cpp` passed a second time
with `-UNDEBUG`, which is the only check the debug form of `SA_ASSERT` has had.

| File | Holds |
| --- | --- |
| `include/SushiAI/core/check.hpp`, `src/core/check.cpp` | The fatal family and its handler |
| `include/SushiAI/core/status.hpp`, `src/core/status.cpp` | `StatusCode`, `Status`, `SA_FAIL_IF`, `SA_ANNOTATE`, `SA_TRY` |
| `include/SushiAI/core/result.hpp` | `Result<T>`, `Result<T&>`, `SA_TRY_ASSIGN` |
| `include/SushiAI/graph/traced.hpp` | `Graph::Traced`, `SA_REFUSE_IF` |
| `tests/common/test_status.hpp` | The assertions that replace `test_throws.hpp` in wave 8 |
| `tests/functional/unit/test_status.cpp`, `test_result.cpp`, `test_traced.cpp` | Suites `Unit_Status`, `Unit_Result`, `Unit_Traced` |

The include graph is `check.hpp` ← `status.hpp` ← `result.hpp`, and `traced.hpp` includes
`check.hpp`, `status.hpp` and `graph/ir.hpp`. None of them includes `core/common.hpp` directly.

### Declarations on disk

`core/check.hpp`:

```cpp
namespace SushiAI
{
    using FatalHandler = void (*)(const char* file, int line, std::string_view message) noexcept;
    FatalHandler set_fatal_handler(FatalHandler handler) noexcept;   // null installs the default

    namespace Detail
    {
        inline constexpr std::string_view K_UNCOMPOSED_FATAL_MESSAGE = "...";
        void report_fatal(const char* file, int line, std::string_view message) noexcept;
        [[noreturn]] void fail_fast(const char* file, int line, std::string_view message) noexcept;
        template<typename Compose>
        [[noreturn]] void fail_fast_composed(const char* file, int line, Compose&& compose) noexcept;
    }
}

#define SA_FATAL_IF(condition, ...)   // every build; polarity of SA_THROW_IF
#define SA_FATAL(...)                 // every build
#define SA_ASSERT(condition)          // debug builds; not evaluated under NDEBUG
```

The default handler writes `sushiai: fatal: <message> (<file>:<line>)` to stderr. `fail_fast`
calls the installed handler, then `std::fflush(nullptr)`, then `std::abort()`. A handler that
returns does not keep the process alive.

`core/status.hpp`:

```cpp
namespace SushiAI
{
    enum class StatusCode : std::uint8_t
    {
        Ok, InvalidDescription, FileNotFound, InvalidFile,
        Incompatible, Unsupported, OutOfMemory, DeviceFailed,
    };

    [[nodiscard]] constexpr const char* status_code_name(StatusCode code) noexcept;

    namespace Detail
    {
        struct Failure;
        [[nodiscard]] const Failure* allocate_failure(StatusCode code, std::string&& message) noexcept;
        [[nodiscard]] const Failure* exhausted_failure() noexcept;
        void retain_failure(const Failure* failure) noexcept;
        void release_failure(const Failure* failure) noexcept;
        [[nodiscard]] std::size_t live_failure_count() noexcept;
    }

    class [[nodiscard]] Status final
    {
        public:
            Status() noexcept = default;
            Status(const Status& other) noexcept;
            Status(Status&& other) noexcept;
            Status& operator=(const Status& other) noexcept;
            Status& operator=(Status&& other) noexcept;
            ~Status();

            template<typename Compose>
            [[nodiscard]] static Status from_message(StatusCode code, Compose&& compose) noexcept;
            template<typename Compose>
            [[nodiscard]] Status annotated(Compose&& compose) const noexcept;

            [[nodiscard]] bool is_ok() const noexcept;
            [[nodiscard]] StatusCode code() const noexcept;
            [[nodiscard]] std::string_view message() const noexcept;

        private:
            explicit Status(const Detail::Failure* failure) noexcept;
            const Detail::Failure* failure_ = nullptr;
    };

    namespace Detail
    {
        [[nodiscard]] inline const Status& status_of(const Status& status) noexcept;
    }
}

#define SA_FAIL_IF(condition, code, ...)
#define SA_ANNOTATE(status, ...)
#define SA_TRY(expression)            // copies the Status out of status_of((expression))
```

`SA_TRY` holds the status by value: `const Status sa_try_status_(Detail::status_of((expression)));`.
The copy is taken inside the full expression, so `SA_TRY(load(path).status())`, a reference into a
temporary, is safe. A success copies a null pointer. A failure costs one atomic increment more
than a reference would. `status_of` is called once, so an overload that returns a `Status` by
value (the seam's, for SushiRuntime's `Result`) builds its message once.

`core/result.hpp`:

```cpp
namespace SushiAI
{
    template<typename T>
    class [[nodiscard]] Result final
    {
        public:
            Result(T value) noexcept(std::is_nothrow_move_constructible<T>::value);
            Result(Status failure) noexcept;                 // fatal when failure.is_ok()

            [[nodiscard]] bool has_value() const noexcept;
            [[nodiscard]] const Status& status() const noexcept;
            [[nodiscard]] T& value() & noexcept;             // fatal when no value is held
            [[nodiscard]] const T& value() const& noexcept;
            [[nodiscard]] T&& value() && noexcept;

        private:
            void require_value() const noexcept;
            std::optional<T> value_;
            Status status_;
    };

    template<typename T>
    class [[nodiscard]] Result<T&> final
    {
        public:
            Result(T& value) noexcept;
            Result(std::remove_const_t<T>&& value) = delete;
            Result(Status failure) noexcept;                 // fatal when failure.is_ok()

            [[nodiscard]] bool has_value() const noexcept;
            [[nodiscard]] const Status& status() const noexcept;
            [[nodiscard]] T& value() const noexcept;         // fatal when no reference is held

        private:
            T* target_ = nullptr;
            Status status_;
    };

    namespace Detail
    {
        template<typename T>
        [[nodiscard]] const Status& status_of(const Result<T>& result) noexcept;
    }
}

#define SA_DETAIL_JOIN_TOKENS(left, right)
#define SA_DETAIL_JOIN(left, right)
#define SA_DETAIL_TRY_ASSIGN(outcome, declaration, expression)
#define SA_TRY_ASSIGN(declaration, expression)
```

`SA_TRY_ASSIGN` binds `auto&& outcome = (expression);`, since it has to keep the object to move
the value out. Its expression is an lvalue or a prvalue. A reference into a temporary, such as
`first(make()).second()`, dangles there, and the macro's `@param` says so.
`SA_ASSERT_OK_AND_ASSIGN` in `test_status.hpp` has the same shape and the same limit.

`graph/traced.hpp`:

```cpp
namespace SushiAI
{
    namespace Graph
    {
        class [[nodiscard]] Traced final
        {
            public:
                Traced() noexcept = default;              // holds NO_VALUE
                Traced(ValueId id) noexcept;              // implicit
                Traced(Status failure) noexcept;          // implicit; fatal when failure.is_ok()

                [[nodiscard]] bool has_value() const noexcept;
                [[nodiscard]] const Status& status() const noexcept;
                [[nodiscard]] ValueId value() const noexcept;   // fatal when a failure is held

            private:
                ValueId id_ = NO_VALUE;
                Status status_;
        };
    }

    namespace Detail
    {
        [[nodiscard]] inline const Status& status_of(const Graph::Traced& traced) noexcept;
    }
}

#define SA_REFUSE_IF(builder, condition, ...)
```

`tests/common/test_status.hpp`:

```cpp
namespace SushiAI
{
    namespace Test
    {
        ::testing::AssertionResult status_is_ok(const Status& status);
        ::testing::AssertionResult status_names(const Status& status, StatusCode expected,
                                                std::initializer_list<std::string_view> needles);
        template<typename... Needles>
        ::testing::AssertionResult failure_names(const Status& status, StatusCode expected,
                                                 const Needles&... needles);
        std::string death_pattern(std::initializer_list<std::string_view> needles);
    }
}

#define SA_EXPECT_FAILURE(expression, ...)                 // ... is the code, then the needles
#define SA_EXPECT_OK(expression)
#define SA_ASSERT_OK_AND_ASSIGN(declaration, expression)
#define SA_EXPECT_FATAL(statement, ...)                    // ... is one needle or more
```

### Departures from the SPEC

**Status.**

- A moved-from `Status` keeps its failure. The move constructor and move assignment share the
  block as a copy does, at the cost of one atomic increment on the failure path. With a move that
  empties its source, a moved-from failure reads as success, and a moved-from failed `Result`
  would hold neither a value nor a status.
- The copy and move members, the destructor and `is_ok()` are defined in the header. On success
  each is a null test, with no call into the library.
- `Detail::Failure` holds a `std::string` for a heap block and a `std::string_view` that points
  at it or at a literal. The static `OutOfMemory` block views a literal, so building it allocates
  nothing at the moment the heap has just refused.
- `from_message` with `StatusCode::Ok` is fatal. The SPEC does not say.
- `annotated` returns the status unchanged when the prefix cannot be composed or the block cannot
  be allocated, so the original code and message survive. On a success it returns success and
  does not call `compose`. The SPEC does not say.
- `Detail` gained `allocate_failure`, `exhausted_failure`, `retain_failure`, `release_failure`
  and `live_failure_count`. The first four are what the header's templates and inline members
  call. The last counts heap failure blocks alive and exists for the tests: it is how they see
  that a success path built no block and that the last copy frees one.
- `Detail::status_of` has one overload per type (`Status`, `Result<T>`, `Traced`), each in its
  type's header. `SA_TRY`, `SA_TRY_ASSIGN` and the test macros read through it, so a fourth
  status-carrying type adds an overload and edits no macro.

**Result.**

- Two `static_assert`s refuse `Result<Status>` and a `Result` of a non-object type.
- `Result<T&>` deletes construction from a temporary, which would dangle, and has one `value()`
  that is `const`: the reference form does not own what it refers to.
- `return std::make_unique<Derived>(...)` from a function returning
  `Result<std::unique_ptr<Base>>` does not compile, since it needs two user conversions. The
  SPEC's own example (section 3.4) names a typed local first. No converting constructor was
  added.

**The fatal family.**

- The SPEC declares `fail_fast` only. The dispatch asks for a replaceable handler, so
  `FatalHandler`, `set_fatal_handler` and `Detail::report_fatal` exist. The handler reports;
  `fail_fast` stops the process whatever the handler does. A test installs a recorder and calls
  `report_fatal` to see the report in-process, and a death test covers the stop.
- `Detail::fail_fast_composed` formats inside a `try`. A throwing formatter reports
  `K_UNCOMPOSED_FATAL_MESSAGE` and does not reach `std::terminate` without a line of text.
- The fatal macros format through a `[&]` lambda, so they cannot be used at namespace scope.

**Traced.**

- `K_NO_VALUE` does not exist until wave 2; the header uses `Graph::NO_VALUE`, and the rename
  tool has to cover it.
- `has_value()` reports that no failure is held, so a default `Traced` (an absent operand)
  answers true and `value()` returns `NO_VALUE`. `SA_TRY_ASSIGN` over an absent operand binds
  `NO_VALUE` and continues, which `conv2d` needs for its optional bias.
- `SA_REFUSE_IF` is in `traced.hpp` for this wave only. It is a builder-side macro: it calls
  `refuse(Status)` on its first argument and does not mention `Traced`. `builder.hpp` could not
  be edited in wave 1, so the macro has no home of its own yet; the next wave moves it (see the
  table below). Its code is fixed at `InvalidDescription`. `test_traced.cpp` drives it with a
  stand-in builder, since `GraphBuilder::refuse` does not exist.

**Tests.**

- `SA_EXPECT_FAILURE(expression, ...)` takes the code as the first variadic argument. With the
  code as a named parameter, a call with no needle is a zero-argument variadic macro, which
  `-pedantic -Werror` rejects in C++17.
- `death_pattern` escapes the regular-expression metacharacters only, which both POSIX and
  GoogleTest's own syntax accept. Its needles are ordered; `status_names` takes them in any order.
- The fatal family's tests are in `test_status.cpp` under `Unit_Status`. No `test_check.cpp` was
  in this wave's file list.
- No test hooks the global heap. "Success allocates nothing" is observed as: the block count of
  `live_failure_count()` does not move, the composing lambda and the format arguments are not
  evaluated, and `static_assert`s pin the sizes (`Status` is one pointer, `Result<int>` and
  `Traced` are two) and the `noexcept` constructors. An allocation from some other source inside
  these types would go unseen; none exists in the code as written.
- Ten statements are death tests in a release build, eleven in a debug build. D7's condition,
  the start-up time of one death-test child, has not been measured.
- `stop_when` returns the `__LINE__` of its own `SA_FATAL_IF`, and the death test matches
  `test_status.cpp:<that line>)`, so the line the macro passes and the line the default writer
  prints are both pinned.
- `SA_TRY` over a reference into a temporary is tested in `Unit_Result`. The test would pass by
  luck against a macro that dangled; a sanitizer run is what would catch a regression there.
- The two `.cpp` files include their own header first, then a blank line, then the standard
  headers, as `cpp-code-style` and `src/graph/builder.cpp` do. `src/core/allocation.cpp` and
  `src/graph/fusion.cpp` list their own header last; the tree holds both orders.

**Not written in this wave.**

- `core/seam.hpp`, `src/core/seam.cpp` and `test_seam.cpp`: they wait for the list of exception
  types SushiBLAS throws after its v0.1.0.
- The lazy `TestEnvironment` (SPEC section 7). Until it lands, every death-test child constructs
  the shared `RuntimeContext`.
- `Graph::mark_refused`, `Graph::has_refusal`, `GraphBuilder::refuse` and `observe`,
  `Detail::bridge`, and the two `test_traced.cpp` rows of SPEC section 7 that need a real graph.
- The licence box of the ten files. `check_source_comments.py` reports `rule_license_block` on
  each and nothing else.

### Edits the next wave makes to existing files

| File | Edit |
| --- | --- |
| `src/CMakeLists.txt` | Add `core/check.cpp` and `core/status.cpp` to `SUSHIAI_SOURCES`, after `core/allocation.cpp` (line 21) |
| `tests/CMakeLists.txt` | Add `functional/unit/test_status.cpp`, `functional/unit/test_result.cpp` and `functional/unit/test_traced.cpp` to `sushiai_tests` |
| `include/SushiAI/SushiAI.h` | Add `#include <SushiAI/core/check.hpp>`, `<SushiAI/core/status.hpp>`, `<SushiAI/core/result.hpp>` and `<SushiAI/graph/traced.hpp>` |
| The ten new files | Run `tools/licensing/write_license_block.py` |
| `include/SushiAI/graph/ir.hpp`, class `Graph` | Add `void mark_refused() noexcept`, `[[nodiscard]] bool has_refusal() const noexcept` and the private member `bool refused_ = false;` beside `values_` and `nodes_` (lines 440-441) |
| `include/SushiAI/graph/builder.hpp`, class `GraphBuilder` | Include `core/status.hpp`, `core/result.hpp` and `graph/traced.hpp`. Add `[[nodiscard]] Status refuse(Status failure) const noexcept` (calls `graph_->mark_refused()`, returns the failure), `[[nodiscard]] Status observe(Status outcome) const noexcept` and `template<typename T> [[nodiscard]] Result<T> observe(Result<T> outcome) const noexcept` (each marks when the outcome is a failure). Move `SA_REFUSE_IF` here from `traced.hpp`, beside `refuse`, with the `SushiBLAS/core/format.hpp` include it needs |
| `include/SushiAI/graph/traced.hpp` | Drop `SA_REFUSE_IF` and the `SushiBLAS/core/format.hpp` include once the macro is in `builder.hpp` |
| `src/graph/compiled_step.cpp`, `src/autograd/differentiator.cpp` | Return `InvalidDescription` for a graph whose `has_refusal()` is true (waves 5 and 6) |
| `tests/common/test_environment.hpp` | Build the context on first use, so a death-test child that tests a host unit starts no device |
| `tests/common/README.md` | A table row and a section for `test_status.hpp` |
| `tests/functional/unit/test_traced.cpp` | Once the two graph edits land: the real-graph rows of SPEC section 7, `NO_VALUE` to `K_NO_VALUE` with wave 2, and the `SA_REFUSE_IF` tests either include `graph/builder.hpp` for the macro or move to the builder's test file |

## Wave 0: the class table and the tools

Wave 0 adds nine files under `tools/` and edits `tools/README.md`. No file under `include/`,
`src/`, `tests/`, `apps/` or `examples/` changed. `apply_rename.py` was not run on the tree, in
either mode; its tests run it on fixture trees in a temporary directory.

A review of the wave returned 18 findings. The section below is the state after them; what each
changed is said where it applies.

| File | Holds |
| --- | --- |
| `tools/contract/site_classes.tsv` | 588 rows, one per failure site of `include/`, `src/`, `apps/` and `examples/` (549 when the wave closed) |
| `tools/contract/check_contract.py`, `README.md` | The ratchet, the two coverage rules, the table-shape rule and four notes |
| `tools/naming/build_rename_map.py`, `rename_map.tsv` | The scanner and the map it wrote, 324 rows |
| `tools/naming/apply_rename.py`, `README.md` | The token-level rename, with a dry run, a kind filter and the freeze check |
| `tools/tests/test_check_contract.py`, `test_naming.py` | 35 and 54 `unittest` cases on fixture trees |

### The class table

Seven classifiers read the code and returned 526 rows for `include/` and `src/`. The table was
assembled from their rows and checked against the tree by `check_contract.py`'s own scanner: it
found 526 sites there, 524 `SA_THROW_IF(` calls and the 2 `throw` statements of
`src/core/allocation.cpp`, in 78 files, and every row's file and line named one of them.
`grep -rn "SA_THROW_IF(" include src` prints 529 lines in 79 files: the 524 calls, the `#define`
in `include/SushiAI/core/common.hpp:128`, and four comments (`src/config/value.cpp:243`, `:326`,
`include/SushiAI/config/registry.hpp:65`, `src/train/objective.cpp:68`). No row was dropped and
none was invented.

No classifier read `apps/` or `examples/`. The scanner finds 23 `SA_THROW_IF(` calls there: 3 in
`apps/inference.cpp`, 2 in `apps/train.cpp` and 18 in `examples/demo_mlp.cpp`. Each has a row
whose cause and mechanism are `unclassified`, so the ratchet covers the three files and wave 7
has a list to class. The `function` cell of those rows was read from the file by hand.

`tests/` is outside the table and outside the ratchet. It holds 4 `SA_THROW_IF(` calls, 3 in
`tests/common/reference_fixture.hpp` and 1 in
`tests/functional/integration/test_resilience.cpp`; `convert_tests.py` of SPEC section 6 owns
them. A new `SA_THROW_IF` in a test file is therefore not caught by this checker.

The `message_head` cells come from the scanner, not from the classifiers' text, so that the
checker and the table cut the head the same way. Eight heads were compared by eye with the
classifiers' and agreed, among them one that ends in a space (`src/imaging/affine.cpp:66`) and
one with an escaped quote (`src/eager/scope_trace.cpp:72`). The other 518 were not compared.

**Files whose counts disagreed.** None. The line set of every file equals the classifiers'.

**Summaries that disagree with their own rows.** Three classifiers reported a split that their
rows do not add up to. The table follows the rows.

| Group | Reported | Rows give |
| --- | --- | --- |
| core, config, io | 46 status, 20 fatal | 48 status, 18 fatal |
| graph, host units | 92 status, 26 fatal (4 invariant) | 93 status, 25 fatal (3 invariant) |
| imaging | 63 status (15 outside), 26 fatal | 66 status (14 outside), 23 fatal |

The lowering classifier names an arity family of 28 sites and lists 27 lines.

**Against SPEC 1.1.** The SPEC counted 510 matching lines; the tree now holds 529. Detect grew
from 30 to 42, imaging from 85 to 89, autograd from 24 to 26 and graph from 176 to 177. The other
modules are as the SPEC counted them, once its comment lines are taken out.

**Rows a review changed.** Sixteen rows left the class their classifier gave them:

- Thirteen imaging rows went from precondition and fatal to user-content and status:
  `src/imaging/affine.cpp:66`, `:70`, `:73`, `:77`, `:210`, `:213`, `flip.cpp:59`,
  `letterbox.cpp:120`, `:123`, `mosaic.cpp:125`, `:134`, `:142`, `:219`. Each check reads matrix
  values, extents or window geometry, which SPEC 2.1 names as content, and the rule is applied
  without asking who built the plan. `is_valid_size` is now a status at both of its call sites.
- `src/nn/layer_stack.cpp:64` and `:71` went from fatal to status, the class `:46` already had.
  The three run `layers_.empty()` on one object, and D1 gives one check one class. A status from
  `output_features()` and from the `noexcept` `output_shape()` of SPEC 2.8 is not a one-token
  edit: the check has to move to where the stack is built.

**The table after the model-graph work and the rulings.** The paragraphs above describe the
table as wave 0 wrote it, with 549 rows. The model-graph work then changed the tree, and the
engineer ruled on three questions. The table was brought in line with both on 2026-10-07 and
holds 588 rows; the totals below are that table's.

- 41 rows were added, one per `SA_THROW_IF` the model-graph work wrote into a file that existed
  before it, where a new line keeps the file's contract: 24 in `src/config/model_config.cpp`, 6 in
  `src/graph/builder.cpp` (`batchnorm_fold`), 4 in `include/SushiAI/graph/ir.hpp`
  (`bind_state_leaf`), 2 in `include/SushiAI/graph/step_recorder.hpp`, 2 in
  `src/nn/layer_registry.cpp` and 1 each in `include/SushiAI/nn/layers.hpp`,
  `include/SushiAI/nn/module.hpp` and `src/autograd/differentiator.cpp`. 35 are content and a
  status, one of them, the gradient of a buffer, under `handle_or_content`. The 6 of `Graph` and
  `StepRecorderList` are preconditions and fatal.
- 2 rows were removed. The id-preservation checks of `src/graph/cast_insertion.cpp` and
  `src/graph/fusion.cpp` are no longer in the tree.
- 122 line hints in 7 files were set to the lines the scanner finds now. The three older rows of
  `model_config.cpp` also took the functions their sites moved to.
- The scanner finds no `SA_THROW_IF` and no `throw` in a file the model-graph work created.
- `hand_built_graph` left 76 rows. 28 checks in `GraphBuilder` emitters stay statuses. 35 fatal
  rows stay fatal. 11 rows went from status to fatal: `Graph::prove_layout_uniformity`'s check on
  a result's layout (`ir.hpp:495`), which still carries `layout_proof`, and ten rank, shape and
  element-type checks of `src/graph/lowering.cpp` (`:466`, `:496`, `:510`, `:515`, `:819`, `:864`,
  `:882`, `:1162`, `:1246`, `:1314`). An emitter or the fusion pass holds each of the ten already,
  so they fire on a SushiAI defect or on a graph built by hand, and such a graph takes the
  structure's contract. `src/autograd/differentiator.cpp:114` and `:256` stay statuses: the
  builder alone reaches both, so the ruling does not apply to them.
- `hand_built_plan` left 13 rows, all now precondition and fatal. `apply_affine:213` reads
  `plan.output`, so `is_valid_size` is fatal there and a status in `src/imaging/image.cpp`.
  `plan_mosaic:219` reads the centre of a `MosaicDraw`; the ruling names plans, and the row
  treats the draw that `draw_mosaic` returns the same way.
- `empty_stack` left 3 rows, all now precondition and fatal. The ruling puts the refusal where
  the stack is built, as a status, and says the accessors of a stack that exists cannot fail. It
  gives these three lines no mechanism of their own. The rows read what is left of the check as
  the caller's contract; the wave that converts `LayerStack` may delete them instead, and it adds
  the status to `build_layer_stack`. `src/config/model_config.cpp:487` already refuses a document
  with no layers.
- `tools/contract/README.md` says a new `SA_THROW_IF` is not answered by a row. These 41 sites
  were given rows because the model-graph work's contract let an existing file keep its macro.
  The ratchet counts from this table on.

**Totals.**

| Cause | Sites | | Mechanism | Sites |
| --- | --- | --- | --- | --- |
| user-content | 351 | | status | 393 |
| outside | 42 | | fatal | 172 |
| precondition | 112 | | unclassified | 23 |
| invariant | 60 | | | |
| unclassified | 23 | | | |

The library holds 565 sites in 81 files: 563 `SA_THROW_IF(` calls and the 2 `throw` statements
of `src/core/allocation.cpp`.

Four rows carry a mechanism their cause does not predict, each by a line of the SPEC:
`src/eager/scope_trace.cpp:71` and `:72` are content made fatal by D8(a);
`src/train/trainer.cpp:184` is a handle check and `src/optim/step_scalars.cpp:40` an invariant,
both returned as statuses because SPEC 2.6 lists them under `make`.

| Module | Sites | user-content | outside | precondition | invariant | Status | Fatal | In frozen files | With an open question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| autograd | 27 | 16 | 0 | 6 | 5 | 16 | 11 | 0 | 7 |
| config | 45 | 41 | 3 | 1 | 0 | 44 | 1 | 0 | 3 |
| core | 15 | 3 | 2 | 8 | 2 | 5 | 10 | 0 | 11 |
| data | 18 | 13 | 2 | 3 | 0 | 15 | 3 | 18 | 7 |
| detect | 42 | 42 | 0 | 0 | 0 | 42 | 0 | 0 | 16 |
| eager | 16 | 2 | 0 | 12 | 2 | 0 | 16 | 0 | 6 |
| graph, `lowering.cpp` and `arena.cpp` | 59 | 1 | 0 | 14 | 44 | 1 | 58 | 57 | 6 |
| graph, host units | 128 | 98 | 0 | 28 | 2 | 98 | 30 | 0 | 10 |
| imaging | 89 | 52 | 14 | 23 | 0 | 66 | 23 | 0 | 33 |
| io | 30 | 2 | 21 | 5 | 2 | 23 | 7 | 0 | 25 |
| nn | 48 | 43 | 0 | 5 | 0 | 43 | 5 | 4 | 4 |
| optim | 32 | 23 | 0 | 6 | 3 | 24 | 8 | 32 | 13 |
| train | 16 | 15 | 0 | 1 | 0 | 16 | 0 | 16 | 7 |
| Library | 565 | 351 | 42 | 112 | 60 | 393 | 172 | 127 | 148 |
| apps | 5 | unclassified | | | | | | 0 | 0 |
| examples | 18 | unclassified | | | | | | 0 | 0 |
| Total | 588 | | | | | | | 127 | 148 |

### Waves 3 to 7 in numbers

A fatal site in a file that is not frozen converts in wave 3. A fatal site in a frozen file
converts with its file. Status sites follow the module order of SPEC section 5.

| Module | Wave 3, fatal | Of those, held by a question | Wave 4, leaf statuses | Wave 5, trace surface | Wave 6, engine lane | Wave 7 |
| --- | --- | --- | --- | --- | --- | --- |
| autograd | 11 | 0 | | 16 | | |
| config | 1 | 1 | 44 | | | |
| core | 10 | 6 | 5 | | | |
| data | | | 18 (15 status, 3 fatal; `src/data/` is frozen today) | | | |
| detect | | | 42 | | | |
| eager | 16 | 2 | | | | |
| graph, `arena.cpp` | 2 | 1 | | | | |
| graph, `lowering.cpp` | | | | | 57 (1 status, 56 fatal) | |
| graph, host units | 30 | 4 | | 98 | | |
| imaging | 23 | 1 | 66 | | | |
| io | 7 | 1 | 23 | | | |
| nn | 5 | 0 | | 39 | 4 (`init.cpp`) | |
| optim | | | | | 32 (24 status, 8 fatal) | |
| train | | | | 4 (`objective.cpp`; `src/train/` is frozen today) | 12 (`trainer.cpp`) | |
| apps | | | | | | 5 |
| examples | | | | | | 18 |
| Total | 105 | 16 | 198 | 157 | 105 | 23 |

Wave 3 is 105 edits in 8 modules. 89 of them can be converted today; 16 carry an open question
that could change their mechanism and wait for its answer. Not all 105 are one-token edits: the
13 imaging rows and the 3 `LayerStack` rows that the rulings made fatal also move a refusal to
where a pipeline or a stack is built. 67 further fatal sites wait for their frozen files, 6 of
them behind such a question. Eager is finished by wave 3: none of its 16 sites is a status.

### Doubtful sites

Each entry is a doubt a classifier raised. The row keeps the class the classifier chose, except
the sixteen rows named above. Most doubts reduce to a few questions for the owner or the
engineer, so they are grouped by the question that decides them.

Each question has a key, and the `open_question` cell of every row it touches holds that key. A
row with two doubts holds both keys, joined by `;`. 148 rows carry a key, under 52 keys. A key
that starts with `label_` questions the cause label, the status code, the `function` cell or how
the row was read, and leaves the mechanism settled: 60 rows carry only such keys. The other 88
rows, 22 of them fatal, wait for an answer before a script converts them. The table after this
one gives the row count of each key. `check_contract.py`
prints one note per key with its row count. The cell is emptied by hand when the question is
answered.

| Key | Sites | Row says | Decided by |
| --- | --- | --- | --- |
| `shape_make_coverage` | `include/SushiAI/core/shape.hpp:56`, `:61`, `:73`, `:78` | fatal, by SPEC 2.6 | Whether every data-sourced `Shape{` goes through `Shape::make`. The list of the 35 `Shape{` lines decides it and was not produced in this wave. Known callers with data: `src/io/checkpoint_format.cpp:226`, `include/SushiAI/io/checkpoint.hpp:96` |
| `layout_proof` | `include/SushiAI/graph/ir.hpp:489`, `:495` | `:489` status, `:495` fatal | `prove_layout_uniformity` also runs inside `insert_casts` as a proof of the pass's own output. Either one status-returning proof and the pass turns a failure into `SA_FATAL`, or both checks are fatal outright. The `hand_built_graph` ruling made `:495` fatal; operands of two layouts reach `:489` through the builder, so it stays a status |
| `bind_state_leaf_return` | `include/SushiAI/graph/ir.hpp:409`, `:413` | fatal | Whether `Graph::bind_state_leaf` returns `Status` for a state or a value that is already bound, as the model-graph SPEC declares it. SPEC 2.1 reads both as object state inside `Graph`, which is fatal. The function is `void` today, and its callers `Buffer::read` and `Parameter::declare_described` return a `ValueId` |
| `cast_into_return` | `src/graph/builder.cpp:492`, `:495` | status | `cast_into` returns `void` and its one caller is the cast pass. It needs a `Status` return, and the pass decides what a failure is |
| `op_choice` | `src/graph/builder.cpp:421` | fatal | Whether the choice of an `OpID` is part of the description. The function returns `Traced` either way |
| `branch_list_source` | `src/graph/builder.cpp:878` | status | Whether a branch list can arrive from a model document (stage 3 multi-input layers) |
| `precision_policy_extension` | `src/graph/cast_insertion.cpp:91` | status | Whether a user-written precision policy is a supported extension point. If not, it is an invariant once `MixedPrecisionPolicy::make` checks its type |
| `plan_alignment_source` | `src/graph/memory_plan.cpp:248` | status | Who fills `PlanOptions::alignment`; `lowering.cpp` and `compiled_step.cpp` were not read for it |
| `bound_tensor_rule` | `src/graph/lowering.cpp:726` | status | A rule for a bound tensor that disagrees with its `Value`; `TensorPool::bind` checks none today |
| `bound_tensor_layout` | `src/graph/lowering.cpp:938`, `:1251`, `:1324` | fatal | Whether the layout of a bound tensor counts as content. Which lowerings can produce a strided result was not traced |
| `selector_source` | `src/graph/lowering.cpp:363` | fatal | Whether a fusion selector is ever configured from a document |
| `foreign_plan` | `src/graph/arena.cpp:54` | fatal | Whether a pool may adopt a plan built from another graph |
| `raw_span_overload` | `src/core/allocation.cpp:50`, `:51` | status | Whether the raw-span `allocate_tensor` overload stays public. If only `Shape` reaches it, `:50` is fatal |
| `platform_size` | `src/core/allocation.cpp:113` | status | The supported platform list: on a 64-bit `size_t` it is unreachable after `:51` |
| `like_source` | `src/core/allocation.cpp:134` | fatal | How `src/optim/optimizer.cpp:156` obtains `like`; not read past the call line |
| `zero_parameter_count` | `src/core/allocation.cpp:152` | fatal | Whether `LossScaler::make` refuses a zero parameter count before `src/optim/loss_scaler.cpp:138` |
| `label_cause_public_surface` | `src/core/allocation.cpp:180`, `:189`; `src/io/checkpoint_format.cpp:136`, `:164`, `:189`; `include/SushiAI/io/checkpoint.hpp:76`; `src/optim/loss_scaler.cpp:157`; `src/autograd/differentiator.cpp:82`; `src/eager/scope_partitioner.cpp:67`, `:89`, `:93`, `:203` | fatal | The cause label only (precondition or invariant): whether `view_tensor`, the checkpoint reader and writer, `TensorPool` rebinding, `record_screens`, `Provenance` and `Segment` stay public |
| `value_object_shape` | `src/config/value.cpp:175` | fatal | Whether the two-vector shape of `Value::object` is the call contract or part of a document built in C++. A single vector of pairs removes the check |
| `label_status_code` | `src/config/value.cpp:310`, `:324`; `src/data/dataset.cpp:322` | status | The status code only: `InvalidFile`, `InvalidDescription` or `Incompatible` |
| `tensor_count_limit` | `src/io/checkpoint_format.cpp:124` | status | The value of `MAX_TENSOR_COUNT` against a realistic model |
| `writer_public` | `src/io/checkpoint_format.cpp:141` | status | Whether `CheckpointWriter` stays public outside `save_checkpoint` |
| `format_rank_limit` | `src/io/checkpoint_format.cpp:144` | fatal | Whether the format's rank limit stays the same constant as `Shape`'s |
| `label_cause_outside_or_content` | `include/SushiAI/io/checkpoint.hpp:157`, `:160`, `:166`, `:170`, `:181`; `src/data/dataset.cpp:305`, `:308`, `:311`, `:342`, `:358`; `src/imaging/dataset_partners.cpp:41`; `src/imaging/recent_partners.cpp:43`; `src/imaging/pipeline.cpp:180`, `:191`, `:253`; `src/imaging/rectangle.cpp:86`; `src/imaging/raw_cache.cpp:156`, `:193` | status | The cause label only: `outside` or `user-content`. The four causes have no slot for SPEC 2.3's `Incompatible` |
| `label_read_without_parse` | `src/io/checkpoint_format.cpp:172-239`, `include/SushiAI/io/checkpoint.hpp:76-181` | as listed | Classified by reading only; `checkpoint.hpp` calls the engine and does not parse today |
| `gradcheck_surface` | `src/autograd/gradcheck.cpp:94`, `:140`, `:141` | status | Whether gradcheck is public surface or a test utility whose misuse may abort. A status changes both return types to `Result<GradCheckResult>` |
| `handle_or_content` | `src/autograd/differentiator.cpp:138`, `:193` | status | Read as a wrong handle each would be fatal; the id is valid, so SPEC 2.1 points to content. `:138` is a request for the gradient of a buffer, `:193` a secondary result passed as the loss |
| `saved_statistics_grad` | `src/autograd/rules/norm.cpp:37` | status | Whether a value computed from batch norm's saved statistics can carry `requires_grad` through the builder; `Graph::add_node`'s propagation was not read |
| `init_fans` | `src/nn/init.cpp:51`, `:58`, `:68`, `:79` | status | Through `Detail::apply_initializer` the fans are already checked, so there they are invariants. The functions return `void` and the file is frozen. The Xavier condition is `fan_in + fan_out <= 0`, so one negative fan passes it |
| `make_by_spec_2_6` | `src/optim/step_scalars.cpp:40`; `src/train/trainer.cpp:184` | status, by SPEC 2.6 | Whether `StepScalars::make` exists for this check or only for the allocation; whether an unset `StepSpec` field is a bad handle or an incomplete description |
| `loss_scaler_surface` | `src/optim/loss_scaler.cpp:124`, `:127`, `:133` | status | Whether users construct a `LossScaler` over a tensor of their own, and whether a model with no parameters is refused earlier |
| `separate_kernel_source` | `src/optim/optimizer.cpp:62`, `:69`; `src/train/trainer.cpp:132` | status | Whether the `SEPARATE` kernel with loss scaling, or with `IN_PLAN`, can be reached from a JSON document. A status changes `set_loss_scaler` from `void` to `Status` |
| `slot_validation_home` | `src/optim/optimizer.cpp:42`, `:45`, `:49`, `:52` | status | Where the shared slot validation lives once constructors cannot fail; SPEC 2.6 gives `make` to `AdamW` and `SGD` only |
| `options_accessor` | `src/optim/sgd.cpp:98`, `:103` | fatal | Whether the mutable `options()` accessor stays. A checked setter would move the content check there |
| `declare_return` | `src/train/objective.cpp:40` | status | Whether `IObjective::declare` returns `Status`; if it stays `void`, `add` has no path for one |
| `policy_mismatch_source` | `src/train/trainer.cpp:156` | status | Whether a JSON-described model can reach `Trainer` with a policy its graph was not built under |
| `dtype_check_once` | `src/train/trainer.cpp:206`, `:217`; `src/train/objective.cpp:97` | status | Whether `Trainer::make` checks the loss and logits dtype once; the per-step checks then become invariants |
| `label_check_order` | `src/data/dataset.cpp:287` | status | Not the class: the check sits after the `mnist_available()` return, so a bad batch size with absent files reads as a missing dataset |
| `empty_step` | `src/eager/scope_partitioner.cpp:80` | fatal | Whether an eager step that traces no node can reach the partitioner; `Session` and `Step` do not exist yet |
| `constant_like_origin` | `src/eager/scope_partitioner.cpp:176` | fatal | Whether `constant_like` emits a `LEAF` and which origin an accumulation node gets; `Differentiator::contribute` and `GraphBuilder::constant_like` were not traced |
| none | `src/eager/scope_trace.cpp:71`, `:72` | fatal, cause user-content | Decided by D8(a); listed because the cause and the mechanism disagree |
| `output_span` | `src/detect/anchors.cpp:58`, `:83`; `average_precision.cpp:385`; `box_coding.cpp:54`; `box_format.cpp:44`; `detections_topk.cpp:64`; `iou.cpp:89`, `:159`; `level_rows.cpp:75` | status | Whether a caller-allocated output span is content. The caller can compute each size first, which is SPEC 2.1's argument for fatal; the answer sets how many `void` signatures change |
| `assignment_source` | `src/detect/target_leaves.cpp:51`, `:60` | status | Whether an `Assignment` must come from `assign()` |
| `top_k_public` | `src/detect/top_k.cpp:45` | status | Whether `top_k.hpp` stays public; its two callers already bound `k` |
| `assigner_make` | `src/detect/assigner.cpp:98`, `:101`; `assigner_config.cpp:63` | status | `TaskAlignedAssigner` is not in SPEC 2.6's list of types that get `make`; the same treatment is assumed |
| `image_index_field` | `src/detect/ground_truth.cpp:60` | status | An image index is a field of a label row here, not an id SushiAI issued |
| `image_view_handle` | `src/imaging/image.cpp:101`, `:104` | status | Whether an `ImageView` is a handle. `check_source` holds two fatal checks beside them, so one function would return a status and abort |
| `image_constructor` | `src/imaging/image.cpp:49`, `:51` | status | The `Image` constructor cannot return a status; the classifier did not read SPEC 2.6 for what it becomes |
| `label_set_source` | `src/imaging/label_set.cpp:38` | fatal | Whether a user builds a `LabelSet` from three lists of their own; `label_set.hpp` was not read |
| `resampled_count_zero` | `src/imaging/polygon.cpp:192` | status | Whether `resampled_point_count` can return 0 |
| `epoch_source` | `src/imaging/pipeline.cpp:199` | fatal | Whether an epoch can arrive from a resumed checkpoint or a configuration file |
| `sample_source_contract` | `src/imaging/pipeline.cpp:83`, `:85` | status | Whether the `SampleSource` contract promises an image for every index below `size()` |
| `check_layout_two_paths` | `src/imaging/raw_cache.cpp:87`, `:89`, `:91`, `:97` | status | `check_layout` serves the read path and the write path; on the write path `:91` fires only on a programmer's misuse |
| `label_function_from_message` | `src/imaging/affine.cpp:166-213`, `flip.cpp:40-59`, `gray.cpp:39`, `polygon.cpp:192`, `resize.cpp:59`, `:69` | as listed | The `function` cell is taken from the message prefix; the bodies were not opened |

Open questions that remain, with the rows each key holds:

| Rows | Keys |
| --- | --- |
| 20 | `label_read_without_parse` |
| 18 | `label_cause_outside_or_content` |
| 13 | `label_function_from_message` |
| 12 | `label_cause_public_surface` |
| 9 | `output_span` |
| 4 | `check_layout_two_paths`, `init_fans`, `shape_make_coverage`, `slot_validation_home` |
| 3 | `assigner_make`, `bound_tensor_layout`, `dtype_check_once`, `gradcheck_surface`, `label_status_code`, `loss_scaler_surface`, `separate_kernel_source` |
| 2 | `assignment_source`, `bind_state_leaf_return`, `cast_into_return`, `handle_or_content`, `image_constructor`, `image_view_handle`, `layout_proof`, `make_by_spec_2_6`, `options_accessor`, `raw_span_overload`, `sample_source_contract` |
| 1 | `bound_tensor_rule`, `branch_list_source`, `constant_like_origin`, `declare_return`, `empty_step`, `epoch_source`, `foreign_plan`, `format_rank_limit`, `image_index_field`, `label_check_order`, `label_set_source`, `like_source`, `op_choice`, `plan_alignment_source`, `platform_size`, `policy_mismatch_source`, `precision_policy_extension`, `resampled_count_zero`, `saved_statistics_grad`, `selector_source`, `tensor_count_limit`, `top_k_public`, `value_object_shape`, `writer_public`, `zero_parameter_count` |

The counts sum to 156 over 148 rows, because eight rows carry two keys. `hand_built_graph` (76
rows), `hand_built_plan` (13) and `empty_stack` (3) are answered, and no row carries them.

### The rename map

`build_rename_map.py` ran in the base interpreter (Python 3.13) and wrote 324 rows. The tree
held wave 1's ten new files at that moment, so their macros are in the map.

| Kind | Status | Rows | Code tokens | Markdown mentions |
| --- | --- | --- | --- | --- |
| constant | rename | 148 | 1,747 | 1,568 |
| constant | by_hand | 7 | 146 | 206 |
| constant | excluded | 2 | 43 | 69 |
| enumerator | rename | 48 | 537 | 79 |
| interface | rename | 4 | 46 | 49 |
| interface | by_hand | 1 | 60 | 49 |
| enum_underlying | by_hand | 4 | 58 | 13 |
| unscoped_enum | by_hand | 2 | 4 | 4 |
| macro | excluded | 26 | 1,199 | 212 |
| type_name | excluded | 2 | 95 | 100 |
| sibling_enumerator | sibling | 10 | 336 | 20 |
| sibling_name | sibling | 6 | 10 | 41 |
| string_spelling | excluded | 64 | 291 | 0 |

The tool would rewrite 2,330 identifier tokens for 200 names: 537 in the enumerator run, 1,747 in
the constant run and 46 in the interface run. The Markdown counts are whole-word counts and
over-count names that are also English words (`ALL`, `SCALE`, `CONSTANT`).

What the map shows that the SPEC did not:

- `ILayer` cannot become `Layer` by the tool: `Layer` is already a token 18 times. It is a
  `by_hand` row, as is the move of `include/SushiAI/nn/ilayer.hpp`.
- `CLASSES` and `OVERFLOWING_SCALE` exist in tests both with and without `K_`, so the `K_` name
  is taken. `EPSILON` and `MOMENTUM` are test constants and also enumerators of the unscoped
  `Scalar` enums. All four are `by_hand`.
- Four enums have no underlying type, not two: `UpdateKernel`, `OptimizerPlacement`, and
  `OptimizerKind` and `ActivationKind` in `examples/demo_mlp.cpp`.
- 36 of the 148 constants are declared under `tests/`, `examples/` or `apps/`; the wave renames
  them with the library's.
- The classes `SGD` and `GELU` have the names of the enumerators `OptimizerKind::SGD` and
  `ActivationKind::GELU` of `examples/demo_mlp.cpp`. Each has a `type_name` row, which tells the
  tool that a bare `SGD` is the class.
- The two `Scalar` enums have one row each, `src/optim/adamw.cpp` and `src/optim/sgd.cpp`, and
  each row lists its own file's enumerators.

**What the tool cannot resolve today.** The builder prints the tokens that would still spell a
renamed enumerator after a run of the whole map and that no qualifier explains. It prints 11,
all in `tests/functional/unit/test_checkpoint_format.cpp`: `WEIGHT` at lines 52, 61, 115 and 133
and `BIAS` at lines 53, 63, 123, 207, 208, 211 and 219. They are two namespace-scope constants
declared with plain `const` (`const std::vector<float> WEIGHT{...}`), which the scanner does not
list because it reads `constexpr` and `static const` only, and they share their spelling with
`ParameterRole::WEIGHT` and `ParameterRole::BIAS`. `apply_rename.py` refuses the enumerator run
while they stand. They are a hand edit of wave 2, made and committed before the tool runs:
`WEIGHT` to `K_WEIGHT` and `BIAS` to `K_BIAS` in that one file. The builder was not taught plain
`const`, because telling a namespace-scope constant from a local or a parameter needs scopes the
scanner does not parse.

**Ten new names the first map had wrong.** `upper_snake` split after a digit, so the first map
wrote `K_AVGPOOL2_D`, `K_BATCHNORM2_D`, `K_MAXPOOL2_D`, `K_UPSAMPLE2_D`, their four `_BACKWARD`
forms, `K_COL2_IM` and `K_IM2_COL`. It now splits after a lower-case letter only, and the map
reads `K_AVGPOOL2D`, `K_COL2IM`, `K_IM2COL` and so on, which is "`K_` plus the present spelling".

**The order of the wave.** SPEC section 5 asks for one commit per class and 2.13 fixes the order.
The three runs, each on a clean tree and each followed by its commit:

```
python tools/naming/apply_rename.py --kind enumerator
python tools/naming/apply_rename.py --kind constant
python tools/naming/apply_rename.py --kind interface
```

The same map serves all three. A run applies the rows of its kind and reads every other row for
the names it must leave alone. No row of kind `macro` is marked `rename`, so there is no fourth
run.

### Where wave 0 left the SPEC

- **The tool names.** The SPEC names `rename_identifiers.py`, `check_error_contract.py`,
  `convert_sites.py`, `convert_tests.py`, `list_sites.py` and `check_naming.py`. This wave wrote
  `apply_rename.py`, `build_rename_map.py` and `check_contract.py`, as its task named them. The
  three conversion scripts, the site lister and the naming checker are not written.
- **Safeguards of the rename.** `apply_rename.py` has safeguards 1 and 2 of SPEC 2.13 and the
  freeze check. Of safeguard 3 it has the first half: it refuses when a token still spells an
  applied enumerator and no qualifier, enum body or map row explains it, and when a second pass
  over its own output would rename anything. It does not compare, per file, the number of new
  tokens with the number of old ones. It does not apply the inverse map to its output
  (safeguard 4).
- **What the survivor check does not see.** A name another row of the map lists as an
  unqualified old name is taken as explained. So `K::SGD`, where `K` is an alias of
  `OptimizerKind`, would pass, because `SGD` is also a class. The check also covers enumerators
  only: a constant left behind an enum qualifier the map knows is not reported. A sibling name
  written as `SushiBLAS::Holder<T>::NAME` loses its protection, since the chain's root is read as
  `T`. A text search of the five code roots finds no alias of any of the 13 renamed enums, no
  `using enum` and no `>::NAME` under a `SushiBLAS::` chain. No compiler confirmed it.
- **Comments.** The rename rewrites a comment only after `@ref`, `@c` and `@see`, as SPEC 2.13
  says, and prints the count of other mentions per file.
- **The match key.** The checker matches a site and a row on file, kind and message head. The
  `function` column is carried and not checked, because the checker does not parse C++ scopes.
- **A fourth failing rule.** `rule_table_shape` fails on a missing or malformed table. Without it
  a broken table would read as an empty one.
- **A ninth column.** `open_question` is not in the task's column list. It marks the rows whose
  class waits for an answer, so that a conversion script can skip them.
- **Wrapper macros.** The scanner blanks the `#define SA_THROW_IF` directive and no other. An
  `SA_THROW_IF(` in the body of another `#define` is a site of that file and counts against the
  ratchet. A `throw` in a `#define` body is a site too; the scanner finds none in the tree.
- **`string_spelling` rows.** A name that a JSON file, a checkpoint or a CLI flag spells is
  protected by the tool never rewriting a literal. The map lists those strings as excluded rows
  and still renames the identifier beside them, which SPEC 2.13 requires for `ParameterRole` and
  the op constants.
- **The lexer and the map's schema have no home of their own.** `apply_rename.py` holds the C++
  tokeniser, the map's columns, its reader and its writer, beside the rewriter and the freeze.
  `build_rename_map.py` imports the first four; `check_contract.py` carries a second, smaller
  masking function. Two files would separate them: `tools/naming/rename_map.py` for the columns,
  `RenameEntry`, the reader and the writer, and `tools/common/cpp_text.py` for one tokeniser that
  all three tools use. Both are outside this wave's file list. Inside the list, the writer moved
  next to the reader, and `rewrite()` now asks two trackers, `QualifierChain` and `EnumScope`,
  where a token stands.
- **No `@typing.final`.** `python-code-style` marks a class that is not built for inheritance
  `@typing.final`. No checker under `tools/documentation/` or `tools/common/` does, so the new
  classes follow the neighbours.
- **No `format_version`.** Both tables are plain TSV under a header row, and their readers refuse
  any other header. The `file-formats` rule asks for a version field, which a header-row TSV has
  no place for.
- **Not produced.** The list of `Shape{` lines with non-literal extents, which SPEC 2.6 and
  section 5 assign to wave 0.

## Engineer's rulings on the open questions

The owner left the remaining decisions to the engineer on 2026-10-07. These three settle the
largest groups of rows in `tools/contract/site_classes.tsv`. The table carries them: each of
the 92 rows has its mechanism, an empty key and a reason that names its ruling. "The table after
the model-graph work and the rulings" under Wave 0 lists what changed.

- **`hand_built_graph`.** `Graph` is the structure and `GraphBuilder` is where a description
  enters. A check in `Graph::add_value`, `add_node` or `insert_node` reads ids and the
  single-assignment and ordering rules, which are handle and structure checks: fatal. A check in
  a `GraphBuilder` emitter reads rank, shape, element type or layout, which is content: status,
  under ruling D1. Code that builds a `Graph` by hand takes the structure's contract.
- **`hand_built_plan`.** An imaging plan is produced by its `plan_` function and consumed by its
  `apply_` function. A plan that does not fit the image it is applied to is misuse of that pair:
  fatal. The settings a user fills are content and are refused once, when a pipeline is
  constructed, as a status. `apply_` keeps returning nothing.
- **`empty_stack`.** A `LayerStack` with no layer is refused where the stack is built, as a
  status from the builder. `output_shape()` and `output_features()` on a stack that exists
  cannot fail.

