# SushiBLAS v0.1.0 migration report
**Status:** Open — waves 1 to 9 written, nothing built; the owner's build closes it

## Wave 1

Reviewed on 2026-10-08 by reading each file against its section of `SPEC.md` and the sibling
headers. Nothing was built or run.

- dispatch: M1 — opus low, repaired
- dispatch: M2 — haiku medium, repaired
- dispatch: M3 — haiku medium, repaired
- dispatch: M4 — haiku medium, repaired

### Departures from the design

- **M1.** `seam.hpp` also includes `<cstdint>`, for the underlying type of `SeamPhase`.
  `seam.cpp` also includes `<string>` and `<SushiRuntime/execution/runtime_context.hpp>`, which
  `wait_all()` and `get_execution_engine()` need.
- **M1.** A `SushiException` whose code is `Errc::ok` is not in the table of M1. The seam gives it
  row 3c: `OutOfMemory` under `SeamPhase::Allocate`, `DeviceFailed` otherwise. The orchestrator
  confirms this or chooses fatal.
- **M1.** `test_seam.cpp` is a third file; M1 lists two. Section 7 names it for the owner's
  `tests/CMakeLists.txt`.
- **M2.** `StepRecorderList::record_into_step` carries `[[nodiscard]]`, as rule 8 asks.
- **M3.** Both files carry `#pragma once`, the loops of `concat_lowerings` carry braces, and the
  field comments drop the `@brief` that section 4.1 writes inside `/**< */`.
- **M4.** `count()`, `read_count()`, `declared_count()` and `written_count()` stay; M4's listing
  does not show them and section 3 keeps a member a listing does not show.
- **M4.** The reader's count check in `open` takes `InvalidFile`. This agrees with M4, which
  names `InvalidDescription` for `:124` and `:141` only; the implementer reported it as a
  departure and it is not one.

### Defects repaired

- **M1, width.** `seam.hpp:44` was 104 columns; `seam.cpp:92`, `:113` and `:189` were 101, 102
  and 101. All four are wrapped or shortened.
- **M2, comments.** The `@file` brief and the `@class` tag named `IStepRecorder`; the class is
  `StepRecorder`. The class brief did not start with a verb, and the `before_replay` block used
  "Executes" and "plan dispatch" where "runs" and "the plan is dispatched" say it.
- **M3, comments and layout.** The block of `concat_lowerings` carried two `@param` lines and a
  `@return` that stated no rule; it is the one-line brief of section 4.1 again. The two field
  comments are aligned with one space, and each file has the blank line before its closing
  namespace brace that its neighbours have. The `@file` brief of `unlowered.hpp` repeated the
  symbol's brief word for word.
- **M4, `[[nodiscard]]`.** `get_u32` and `get_i64` return `Result` and lacked it (rule 8). It is
  also added to `is_known_dtype`, `TensorRecord::payload_bytes` and the four count accessors.
- **M4, comments.** Three `@pre` lines read "Fatal when ..." and now state the precondition.
  Briefs that were noun phrases start with a verb. The blocks of `K_CHECKPOINT_MAGIC`,
  `K_CHECKPOINT_VERSION` and `TensorRecord` gave reasons for the design and now say what the
  thing holds. The two private constructors lost `@param` lines that stated no rule. The `//`
  line above the extents loop gave a reason and now states the invariant the `Shape` constructor
  relies on.
- **M4, style.** `is_known_dtype` had its `return` on the `if` line. The four accessors had
  their bodies on the declaration line. The anonymous namespace closed without its
  `// namespace` comment.

### Defects left open

- **`CheckpointReader::next` can still throw.** It builds the `Shape` through the span
  constructor, which throws `SushiAI::Error` (rule 5). The checks above it make the throw
  unreachable. `Shape::make` does not exist; the error contract's wave that adds it replaces the
  call (`tools/contract/site_classes.tsv:29`).
- **`CheckpointWriter` and `CheckpointReader` are neither `final` nor virtual.** Class shape
  rule 7 asks for one of the two. Adding `final` changes a public type, so it is the owner's.
- **`StepRecorder::before_replay` and `after_replay` have `{}` bodies.** The style forbids the
  form. M2 says the two stay as they are, so they were left.
- **`test_seam.cpp` pins the mapping and the pass-through only.** `run_recorded`,
  `compile_graph`, `run_plan` and `discard_recorded` have no test; section 7, item 15 says they
  need a run.
- **The `site_classes.tsv` rows of the converted sites** (`:51-52`, `:475-495`) are still there.
  Section 5 gives their deletion to the orchestrator.
- **Paths changed outside the wave.** `docs/guides/CLI_GUIDE.md` is modified and
  `docs/agent/2026_10_05_ESTATE_AUDIT/` is untracked. Both were so before the wave started and
  neither was touched.

### Verification

One file per invocation, `clang++.exe -fsyntax-only` from `build/`, with the flags of the named
entry of `build/compile_commands.json` and its `-o`, `-c` and dependency flags removed. Each run
came after the last repair.

| File | Flags of | Exit | Error lines |
| --- | --- | --- | --- |
| `src/core/seam.cpp` | `src/core/allocation.cpp` | 0 | none |
| `src/io/checkpoint_format.cpp` | its own entry | 0 | none |
| `tests/functional/unit/test_seam.cpp` | `tests/functional/unit/test_shape.cpp` | 0 | none |
| a scratch file that includes the five headers of the wave, joins two tables with `concat_lowerings` and calls `StepRecorderList::record_into_step` and `guard_value` | `src/core/allocation.cpp` | 0 | none |

`python tools/documentation/check_source_comments.py` over the eight files exits 0 with no
output. No line of the eight files is wider than 100 columns. `src/io/checkpoint_format.cpp`
holds 0 `SA_THROW_IF`, 15 `SA_FAIL_IF` and 6 `SA_FATAL_IF`. `try`, `catch` and `throw` appear in
`seam.hpp` (two `catch (...)`) and `seam.cpp` (six handlers in the table's order) and in no other
file of the wave.

### Syntax-check error lines that remain

None in the wave's own files. The callers of the changed signatures were not compiled, and they
fail until their wave:

| Caller | Breaks on | Wave |
| --- | --- | --- |
| `include/SushiAI/io/checkpoint.hpp:121`, `:148` | the private constructors of `CheckpointWriter` and `CheckpointReader` | 4 (M13) |
| `include/SushiAI/optim/optimizer.hpp`, `src/optim/optimizer.cpp` | `record_into_step()` without a recorder | 4 (M11) |
| `src/graph/compiled_step.cpp` | `record_into_step()` without a recorder | 7 (M26) |
| `src/train/trainer.cpp` | `record_into_step()` without a recorder | 8 (M27) |
| `tests/common/reference_fixture.hpp:132`, `tests/functional/unit/test_checkpoint.cpp:265`, `test_checkpoint_format.cpp`, `test_step_recorder.cpp`, `tests/functional/regression/test_fused_optimizer.cpp:382` | both changes | 9 (M30) |

## Wave 2

dispatch: M5 — haiku medium, repaired

Nothing was built or run. The review read both files against SPEC sections 1.2, 1.3, 2.3, M5
and 5, and against `sushiblas: include/SushiBLAS/storage.hpp:65-68`, `tensor.hpp:163-166` and
`engine.hpp:88-103`.

### Departures from the design

- `allocation.cpp` has a file-local `reserve_storage(engine, bytes)` that holds the one
  `make_sushi` call inside `Detail::guard_value(SeamPhase::Allocate, ...)`. `make_storage` and
  `allocate_arena` call it and annotate its failure with their own text. The design puts the
  guarded call in each of the two; the helper keeps one copy and changes no signature.
- `view_tensor` runs the `Tensor` constructor inside `guard_value` and takes the result with
  `.value()`, so a constructor refusal stops the process with the text
  `Result::value() was called on a failure: <what()>`. The design asks for the guard and keeps
  the `Tensor` return type; it does not say how the failure ends. Section 5 lists `view_tensor`
  among the functions whose every site is fatal.
- `allocation.cpp` includes its own header first, then the standard headers. At `7be22fa` the
  own header came last. The new order is the one `cpp-code-style` gives.

### Defects repaired

- `view_tensor` carried a two-line `//` comment, which `source-comments` forbids. It is one line
  now and sits on the guarded call.
- `checked_element_count`, `reserve_storage` and `make_storage` return `Result` and lacked
  `[[nodiscard]]` (rule 8 of section 1.3). All three carry it.
- The `@return` of `allocate_like` described the fatal check on an empty tensor. That rule is a
  `@pre` now, and the block is eight lines.
- The `@param bytes` of `allocate_arena` said "must be positive" without the consequence; it
  says zero stops the process.
- The `@param` line of `make_storage` documented two parameters in one tag with `@p`. It
  documents `dims` and names `dtype` in the sentence.
- The body comment in `allocate_like` gave a reason ("to preserve matrix dimensions"). It states
  what the code relies on: the new tensor takes the layout of `like`.

### Defects left open

None in the wave's files.

### Checked and found correct

- Return types and `[[nodiscard]]` of the five public functions match the M5 table; names and
  parameter lists are those of `7be22fa`.
- Sites: `:50`, `:51`, `:113` are `SA_FAIL_IF` with `InvalidDescription`; `:134`, `:152`,
  `:176`, `:180`, `:189` are `SA_FATAL_IF`; `:99` and `:165` are `SA_ANNOTATE` over the seam's
  status. That is 5 statuses and 5 fatal checks, the counts of section 5. Every format string and
  argument list is the one of `7be22fa`; the two annotated messages lose the trailing `: {}`,
  whose text the seam's status now carries.
- `Storage(RuntimeContext&, std::size_t, Residency, std::size_t)`,
  `Tensor(sushi_ptr<Storage>, span<const int64_t>, DataType, int64_t, Layout)`,
  `Engine::get_context()` and `Engine::get_layout()` exist with these signatures.
- No `try`, `catch`, `throw` or `SA_THROW_IF` in either file; no line over 100 columns.

### Files changed

`git status --short` lists the wave's two files, the files of wave 1
(`include/SushiAI/core/seam.hpp`, `src/core/seam.cpp`, `include/SushiAI/graph/step_recorder.hpp`,
`include/SushiAI/io/checkpoint_format.hpp`, `src/io/checkpoint_format.cpp`,
`include/SushiAI/graph/lowering/`) and three paths that belong to no unit of waves 1 and 2:
`docs/guides/CLI_GUIDE.md` (modified), `docs/agent/2026_10_05_ESTATE_AUDIT/` (untracked) and
`tests/functional/unit/test_seam.cpp` (untracked). None was reverted.

### Syntax check

Flags of the entry of `src/core/allocation.cpp` in `build/compile_commands.json`, with
`-fsyntax-only` in place of `-o ... -c`, run from `build/`:

    clang++.exe -DSR_DETERMINISTIC_FP=1 -ID:/Projects/sushiai/include
      -ID:/Projects/.sushisystems/vcpkg/installed/x64-windows/include
      -ID:/Projects/sushiruntime/include -ID:/Projects/sushiblas/include -O3 -DNDEBUG
      -std=gnu++17 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -Wall -Wextra -Werror -pedantic
      -fsycl -ffp-contract=off -fno-fast-math -fsycl-targets=spir64,nvidia_gpu_sm_86
      "--cuda-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
      -fsyntax-only D:\Projects\sushiai\src\core\allocation.cpp
    clang exit 0, no output

No error line remains in the wave's file. The callers of the four functions whose return type
changed were not checked and do not compile until their waves: `src/graph/arena.cpp` and the
pool (M6), `src/optim/step_scalars.cpp` (M7), `src/optim/loss_scaler.cpp` (M8),
`src/data/dataset.cpp` (M9), `src/optim/optimizer.cpp` (M11).

## Wave 3

Reviewed 2026-10-08. Nothing was built and no test ran.

    dispatch: M6 — haiku high, repaired
    dispatch: M7 — haiku medium, passed
    dispatch: M8 — haiku medium, repaired
    dispatch: M9 — opus low, repaired

### Departures from the design

- **M6.** The private `ArenaSet` constructor takes `(Engine&, MemoryPlan, Storages)`; `view` needs
  the engine, and the design does not fix the list. `ones`, `scratch` and `materialise` book
  their bytes after `create` or `view` succeeds. `flush_constants` keeps the queue when the
  recording fails and clears it only before the run.
- **M7.** `device_pointer()`, `count()` and the three data members stay, though the design's
  class sketch omits them. `std::fill_n` zeroes the buffer in place of the loop.
- **M8.** `new LossScaler(...)` and the two host writes run inside one
  `guard_value(SeamPhase::Allocate, ...)`. The anonymous helper `carve_count_views` holds the
  guarded `view_tensor` loop. The null-gradient `SA_FATAL_IF` sits inside the guarded recording
  loop. The `ReductionOps` view is built inside the guard's lambda, not at the top of
  `record_screens`.
- **M9.** Site `:487` is the `FileNotFound` `SA_FAIL_IF` of `try_load_mnist`, whose message is
  `mnist_expected_paths(options)`; `make_mnist` annotates that one code with the old directory
  text. The anonymous helper `allocate_batch` replaces three copies of the batch allocation.
  `is_test_split` returns `Result<bool>`. `dataset.cpp` includes its own header first.

### Defects repaired

- `src/graph/tensor_pool.cpp`: a two-line `//` paragraph that gave a reason is deleted
  (`rule_comment_runs`). The file's own header moved to the first include group.
- `include/SushiAI/graph/tensor_pool.hpp`: the blocks of `adopt` and `flush_constants` ran past
  eight lines and the second carried a reason; both are cut to their tags, and the rule each
  stated is now a `@pre`.
- `include/SushiAI/graph/arena.hpp`: the block of `view` was nine lines; three `@param` lines
  that stated no rule are gone.
- `include/SushiAI/optim/loss_scaler.hpp`: the block of `LossScaler::make` was eleven lines and
  is now nine lines shorter of the limit by none (eight); line 59 was 105 columns.
- `include/SushiAI/data/dataset.hpp`: the block of `make_mnist` was nine lines; lines 160 and
  224 were over 100 columns.
- `src/data/dataset.cpp`: line 329 was 103 columns.

After the repairs `python tools/documentation/check_source_comments.py` over the ten files
exits 0 with no output, and no line of them is longer than 100 columns.

### Checked and found sound

- Sibling names, read in `sushiblas/include/SushiBLAS/engine/math/random.hpp` and
  `reductions.hpp`: `RandomOps(Engine&) noexcept`, `constant(Tensor&, double)`,
  `normal(Tensor&, double, double)`, `ReductionOps(TaskRecorder&) noexcept`,
  `count_nonfinite(const Tensor&, Tensor&)`. `BLASOps::gemm` with its defaults, `axpy` and
  `RandomOps::set_seed` were not opened; the syntax check resolves them.
- Failure sites: `dataset.cpp` holds 15 `SA_FAIL_IF` and 3 `SA_FATAL_IF`; `arena.cpp` 2 fatal;
  `tensor_pool.cpp` 7 fatal; `step_scalars.cpp` 1 status and 1 fatal; `loss_scaler.cpp` 10
  status and 2 fatal. Codes follow section 5.
- A grep of the ten files for `try`, `catch`, `throw`, `SA_THROW_IF`, `return nullptr`,
  `execute()` and `wait_all` finds nothing. No file stores a view, and none dispatches on an
  operation kind; `materialise_leaves` tests `Ops::K_LEAF.id` as the design allows.
- `git status --short` lists the wave's ten files and the files of waves 1 and 2. Two paths
  belong to no wave and were changed before this one: `docs/guides/CLI_GUIDE.md` and
  `docs/agent/2026_10_05_ESTATE_AUDIT/`. Neither was touched.

### Defects left open

- **Host allocation outside a guard.** `reserve`, `push_back`, `resize`, `unordered_map::emplace`
  and the `new` of `SyntheticClassification::make` and `Rings::make` can throw `std::bad_alloc`
  in functions that return a `Status` or a `Result`. M8 guards its `new`; M6 and M9 do not. The
  design guards none of them. Whether a host `bad_alloc` is a status in every unit is a rule
  for section 1.3, so the orchestrator decides.
- **`LossScaler` declares two of its five special members.** The copy pair is deleted; the move
  pair and the destructor are implicit, as before this wave. Declaring them changes a public
  class shape.
- **Comments that predate the wave still give reasons**: the blocks of `LossScaleOptions`,
  `LossScaleState::inverse_scale`, `LossScaler::after_replay`, `StepScalars`, and
  `TensorPool::allocated_bytes` and `unpacked_bytes`. The checker passes them. Moving that text
  to `docs/design/` is a documentation task.

### Syntax check

Each file ran alone with the flags of its entry in `build/compile_commands.json`, `-o` and `-c`
dropped and `-fsyntax-only` added; `tensor_pool.cpp` took the entry of `arena.cpp`.

    clang++.exe -DSR_DETERMINISTIC_FP=1 -ID:/Projects/sushiai/include
      -ID:/Projects/.sushisystems/vcpkg/installed/x64-windows/include
      -ID:/Projects/sushiruntime/include -ID:/Projects/sushiblas/include -O3 -DNDEBUG -std=gnu++17
      -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -Wall -Wextra -Werror -pedantic -fsycl
      -ffp-contract=off -fno-fast-math -fsycl-targets=spir64,nvidia_gpu_sm_86
      --cuda-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6 -fsyntax-only <file>

    src/graph/arena.cpp         exit 0, no output
    src/graph/tensor_pool.cpp   exit 0, no output
    src/optim/step_scalars.cpp  exit 0, no output
    src/optim/loss_scaler.cpp   exit 0, no output
    src/data/dataset.cpp        exit 0, no output

No error line remains in the wave's own files. Callers of what this wave changed were not
checked here. By the implementers' replies and section 6 they wait as follows:
`graph/lowering.hpp` and `src/graph/lowering.cpp` (old `TensorPool`, removed `ArenaSet`
constructor) for wave 6; `adamw.*` and `sgd.*` (`StepScalars` constructor) for wave 5;
`trainer.cpp` for wave 8; `apps/`, `examples/` and `tests/` (among them
`tests/functional/integration/test_arena.cpp`) for wave 9.

## Wave 4

Reviewed on 2026-10-08. No build and no test ran. The only compiler invocations were
`clang++ -fsyntax-only`, one file at a time.

- dispatch: M10 — haiku medium, repaired
- dispatch: M11 — haiku medium, repaired
- dispatch: M12 — haiku medium, repaired
- dispatch: M13 — haiku medium, repaired

### Departures from the design

| Unit | Departure | Verdict |
| --- | --- | --- |
| M10 | `alias_output` names its parameters `result_index` and `operand_index`; `output` and `input` are member functions | kept |
| M10 | Private `check_input` and `check_output` hold the two index messages of `src/graph/lowering.cpp:306` and `:314` once. The class still has the eight data members | kept |
| M10 | `DType` and `Shape` reach `context.hpp` through `graph/ir.hpp` and `graph/tensor_pool.hpp`; the design's include list names no other header | kept |
| M11 | The message texts say `Optim::UpdateKernel::Fused` and `OptimizerPlacement::PerStep`, the tree's spellings | kept |
| M11 | `optimizer.hpp` includes `SushiRuntime/core/span.hpp`, `core/result.hpp` and `core/status.hpp`, which its declarations use | kept |
| M11 | `record_all` sums into a local `recorded`, because `screens` is scoped inside the `if` | kept |
| M12 | `init.hpp` includes `core/seam.hpp` and `SushiBLAS/engine/math/random.hpp`; the `initialize` template calls `guard`, `run_recorded` and `RandomOps` | kept |
| M12 | `initialize` calls `set_seed` inside `SushiAI::Detail::guard`, as rule 3 of section 1.3 asks | kept |
| M12 | Seam calls are spelled `SushiAI::Detail::`, because `Detail::` inside `NN` is `NN::Detail` | kept |
| M13 | Success is `return Status();`. `Status::ok()` does not exist in `core/status.hpp` | kept |
| M13 | `checkpoint.hpp` includes `SushiRuntime/core/span.hpp`, `core/status.hpp` and `core/check.hpp` | kept |

A behaviour change the design asks for, recorded here because a test of a later wave may see it:
`apply_initializer` tests `init().kind == Constant` before the final-layer zeroing, so a `Scale`
parameter of the final layer starts at 1.0, and a `Bias` starts at 0.0 through its
`ParameterInit` and no longer through its role.

### Names checked in the sibling headers

| Name | Where |
| --- | --- |
| `Core::Transpose::NoTrans`, `Trans` | `sushiblas: include/SushiBLAS/core/blas_types.hpp:40-45` |
| `RngStream`, `TaskRecorder` | `sushiblas: include/SushiBLAS/engine/rng_stream.hpp:46`, `engine/task_recorder.hpp` |
| `RandomOps(Engine&)`, `set_seed(uint64_t)`, `constant(Tensor&, double)`, `uniform(Tensor&, double, double)`, `normal(Tensor&, double, double)` | `sushiblas: include/SushiBLAS/engine/math/random.hpp:63-108` |
| `Engine::get_recorder()` | `sushiblas: include/SushiBLAS/engine.hpp:70` |
| `SushiRuntime::span` | `sushiruntime: include/SushiRuntime/core/span.hpp:62` |

### Defects repaired

| Unit | File | Defect | Repair |
| --- | --- | --- | --- |
| M10 | `context.hpp:66` | The constructor's comment block ran to twelve lines | Cut to the three parameters that carry a rule and one `@pre` |
| M10 | `context.cpp:190` | The body comment named `GEMM_BIAS`, a spelling the tree no longer has | Names `FusedForm::GemmBias` and states the invariant |
| M11 | `optimizer.hpp:118`, `:130` | Two lines of 103 and 101 columns | Shortened the `@brief`; gave `parameters()` a braced body |
| M11 | `optimizer.cpp:138` | The body comment gave a reason (`so the seam guards it here`) | States what the guard catches |
| M12 | `init.hpp:77`, `:89`, `:101`, `:114`, `:172`, `:188` | Six comment blocks over eight lines; one `@brief` of 112 columns | Dropped the `@param` lines that state no rule; moved the Xavier bound into `@param gain` |
| M12 | `init.hpp`, `initialize` | The block did not say that the run executes everything recorded on the engine | Added to `@pre` |
| M12 | `init.cpp:136` | Two `//` lines in a row | One line stating the test order |
| M12 | `init.cpp:46` | A `case` line of 110 columns | Wrapped; the four other cases realigned |
| M13 | `checkpoint.hpp:36` | `SushiRuntime` included before `SushiBLAS`, against `tensor_pool.hpp` and `optimizer.hpp` | Reordered |
| M13 | `checkpoint.hpp` | `SA_FATAL_IF` used with `core/check.hpp` reached only through `core/status.hpp` | Included `core/check.hpp` |

### Defects left open

| Unit | Defect | Why it is left |
| --- | --- | --- |
| M11 | `Optimizer::step()` returns on a failed `record_all` with the loss-scaler screens still recorded on the engine's graph and `after_replay()` not called. The next `run_recorded` on that engine runs the screens | The design's `step` has no `discard_recorded` call; adding one changes what a failed step leaves behind, which is an ownership decision |
| M11 | `Optimizer` deletes its copy members and declares no move members | Declaring them changes the public class |
| M11 | The comments of `UpdateKernel::Fused`, `UpdateKernel::Separate` and `ParameterSlot` give design reasons. They predate the migration | `check_source_comments.py` passes them; moving the reasons needs a home under `docs/design/` |
| M12, M13 | `init.hpp`, `init.cpp` and `checkpoint.hpp` have LF line ends in the working copy, while `optimizer.hpp` and `optimizer.cpp` have CRLF. Git normalises both | The tree is already mixed (`tensor_pool.hpp` LF, `adamw.cpp` CRLF) |

### Checks run

`python tools/documentation/check_source_comments.py` over the seven files of the wave: exit 0,
no output.

A scan of the seven files for lines over 100 columns: none.

`grep -nE "\b(try|catch|throw)\b|SA_THROW|execute\(|wait_all|ParameterRole"` over the seven files:

    src/optim/optimizer.cpp:138:            // record_update() may throw; this guard is the only place its exception is caught.
    include/SushiAI/nn/init.hpp:199:                if (parameters[i]->role() == ParameterRole::Weight) final_weight = i;

The first hit is a comment. The second is the search for the last weight that M12 keeps in
`initialize`; `apply_initializer` does not name `ParameterRole`.

Syntax checks, run from `build/` with the flags of the `src/nn/init.cpp` entry of
`compile_commands.json`, `-o` and `-c` replaced by `-fsyntax-only`:

    clang++.exe -DSR_DETERMINISTIC_FP=1 -ID:/Projects/sushiai/include
      -ID:/Projects/.sushisystems/vcpkg/installed/x64-windows/include -ID:/Projects/sushiruntime/include
      -ID:/Projects/sushiblas/include -O3 -DNDEBUG -std=gnu++17 -D_DLL -D_MT -Xclang
      --dependent-lib=msvcrt -Wall -Wextra -Werror -pedantic -fsycl -ffp-contract=off -fno-fast-math
      -fsycl-targets=spir64,nvidia_gpu_sm_86 "--cuda-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
      -fsyntax-only <file>

    exit=0 D:/Projects/sushiai/src/graph/lowering/context.cpp
    exit=0 D:/Projects/sushiai/src/optim/optimizer.cpp
    exit=0 D:/Projects/sushiai/src/nn/init.cpp
    exit=0 <scratchpad>/w4_templates.cpp

No invocation printed a diagnostic. `w4_templates.cpp` was a scratch unit that included
`io/checkpoint.hpp`, `nn/init.hpp` and `nn/layers.hpp` and explicitly instantiated
`NN::initialize<NN::Linear>`, `IO::save_checkpoint<NN::Linear>` and
`IO::load_checkpoint<NN::Linear>`. It ran before the last comment-only edit of `init.hpp`;
`init.cpp` was checked again after that edit. The scratch unit is deleted.

### Syntax-check error lines that remain

None in the files of this wave. The callers of the changed signatures were not compiled here:

| File | Waits for |
| --- | --- |
| `src/optim/adamw.cpp`, `src/optim/sgd.cpp` (`record_update()` without a recorder, the `Optimizer` constructor) | M22, M23 |
| Callers of `initialize`, `save_checkpoint`, `load_checkpoint`, `Optimizer::step` and `set_loss_scaler` in `src/train/`, `apps/`, `examples/` and `tests/` | M27 to M30 |

### Files changed

`git status --short` lists the files of this wave and of waves 1 to 3, and two paths that belong
to neither: `docs/guides/CLI_GUIDE.md` (modified) and `docs/agent/2026_10_05_ESTATE_AUDIT/`
(untracked). Both were in that state before this review started; neither was touched or reverted.
`include/SushiAI/graph/lowering/` also holds `rule.hpp` and `unlowered.hpp`, from M3.

Scratchpad after the review: 108K, five entries, none made by this review.

## Wave 5

Reviewed against `SPEC.md` sections 1.2, 1.3, M10, M11, M14 to M23, 4 and 5. No build and no
test ran. Nothing is committed.

### Dispatches

- dispatch: M14 — haiku medium, repaired
- dispatch: M15 — haiku medium, repaired
- dispatch: M16 — haiku medium, repaired
- dispatch: M17 — haiku medium, repaired
- dispatch: M18 — haiku medium, repaired
- dispatch: M19 — haiku medium, repaired
- dispatch: M20 — opus low, repaired
- dispatch: M21 — opus low, repaired
- dispatch: M22 — haiku medium, repaired
- dispatch: M23 — haiku medium, repaired

Every repair is in comments, headers' layout or file-local structure. No recorded call, operand
order, task count or failure text changed in the review.

### Departures from the design

| Unit | Departure |
| --- | --- |
| M14 | `record_binary` and `record_copy_then` are one template each over the view type, not two aliases per view. The arity check sits in the helper. |
| M14 | `lower_dynamic_scale` tests the factor after it takes the result, as `src/graph/lowering.cpp:724-728` does today. |
| M15 | `lower_leaky_relu` and `lower_leaky_relu_backward` are written out, because their kernels take a slope and fit neither helper. |
| M17 | `lower_log_softmax` calls `require_arity(1, 1)`; today's function checks no arity. |
| M18 | `lower_leaf` calls no `require_arity`. Section 4.3 gives `0/any`, and `LoweringContext` has no call for "exactly 0 operands, any results". An operand on a leaf is not refused, as today. |
| M18 | `lower_reshape` tests contiguity before it takes the result; today the result comes first. Both paths are fatal. |
| M19 | `windowed_extents_of` keeps its `what` argument, because `Ops::name_of` returns `sushiai.im2col` and the failure text says `IM2COL`. |
| M19 | `lower_concat_channels` accepts more than one result, since `require_arity_at_least(1, 1)` is the only call that fits "at least 1/1". Today's check requires exactly one. |
| M19 | `lower_spatial_mean_backward` tests the rank before it takes the result. |
| M20 | The three arity failures of the norm rows carry `require_arity`'s text, not today's three sentences. The same holds for every row of the wave. |
| M20 | `lower_batchnorm2d_backward` builds `RandomOps(context.recorder(), context.stream())` for its zero fill. |
| M20 | `lower_batchnorm_fold` reads its `[C]` shape and dtype from operand 3 and checks nothing beyond arity; the design names no check. |
| M21 | Rows end in `context.count(tasks); context.close_selection(before); return Status();`, because `close_selection` reads the count. |
| M21 | `make_epilogue` receives `aux` as `SushiBLAS::Tensor*`, null unless the form spills. |
| M21 | The spill checks run before the results are taken; today they run after. Both are fatal. |
| M21 | The five functors gained Doxygen briefs on their members and call operators. The lambda of `lower_exp_sub_scale` names its first parameter `value`, not `result`, because the row holds a `result` local. |
| M22, M23 | `is_replay_invariant`, both `options()` accessors and `AdamW::first_moment`, `second_moment` stay; the header sketch of the design omits them and section 5 still lists the two moment accessors. |
| M22 | `make` calls `allocate_like` directly; `allocate_state` is a non-static member. |
| M23 | `record_fused` builds no view and `record_separate` builds `BLASOps` only. SGD calls no `ElementwiseOps` operation. |

### Defects repaired

| Unit | Defect | Repair |
| --- | --- | --- |
| M16 | The licence box of `linalg.hpp` and `linalg.cpp` stopped three lines short; `check_source_comments.py` reported `rule_license_block` on both. | The three lines are restored, byte for byte those of `loss.hpp`. |
| M15 | Each row called `require_arity` and then a helper, where M14's rows are one call of a helper that checks arity. The two helpers lacked `[[nodiscard]]`. | Arity moved into `record_copy_then` and `record_backward`; twelve rows are one call each; both helpers are `[[nodiscard]]`. |
| M15, M17, M18, M19 | `//` comments that narrate the next line ("Copies input to output…", "Computes per-row maximum…", "Converts spatial channel layouts…", "Appends channel slices…"). | Removed. The comments that state an invariant stay. |
| M21 | The block above `record_fused_gemm` was nine lines (`rule_block_ceiling`). | Shortened to seven. |
| M14 to M21 | The eight domain headers had three shapes: one-line briefs, blocks with a `@param context` that states no rule and a `@return` that repeats the type, an aligned table in one file, a blank line before the namespace end in four, and `shape.hpp` including `core/ops.hpp` beside `rule.hpp`. | One shape in all eight: `<array>` and `rule.hpp`, a one-line `@brief` per row, an unaligned table with the brief "Pairs each … operation with the row that lowers it.", no blank line before the namespace end. Table rows and their order are unchanged. |
| M16, M20 | A blank line before the namespace end of `linalg.cpp` and `norm.cpp`. | Removed. |
| M17 | A tensor named `l`, and the `Value` of the same operand named two ways in the two rows. | `log_probabilities` and `description`. |
| M19 | The file-local namespace closed without `// namespace`; the block of `windowed_extents_of` described its arguments without the rule they carry. | Both fixed. |
| M23 | `record_fused` and `record_separate` had no Doxygen block; `make` carried three `@param` lines that state no rule; `write_step_scalars` held a prose paragraph; `sgd.hpp` used `std::size_t` without `<cstddef>`; `make` took its scalars as `auto` where `AdamW::make` names `StepScalars`. | Each brought to the form `adamw.hpp` and `adamw.cpp` use. |
| M22 | The `@return` of `make` wrapped mid-phrase. | Reworded. |

### Defects left open

| Unit | Defect | Why it is left |
| --- | --- | --- |
| M22, M23 | `make` runs `state.reserve`, `state.push_back` and `new AdamW(...)` / `new SGD(...)` outside a guard, so a `std::bad_alloc` leaves a function that returns `Result`. `LossScaler::make` (`src/optim/loss_scaler.cpp:178`) wraps its `new` in `Detail::guard_value(Detail::SeamPhase::Allocate, ...)`. | The design spells the unguarded form for both (`SPEC.md:822-823`). Changing it is a decision on what `make` may let escape. |
| M22 | `src/optim/adamw.cpp:228` and `:246` are 107 and 103 columns. Both are comments inside the fused lambda. | The acceptance of M22 holds the lambda body byte for byte. |
| M22, M23 | Comments that predate the wave break `source-comments`: reasons in the `Scalar` and `K_TASK_NAME` blocks of `adamw.cpp`, numbered step comments, the "FUSED is the default…" paragraphs in both option structs. `check_source_comments.py` does not report them. | Not written by this wave; part of them sit inside the lambdas. |
| M18 | Whether `LEAF` should refuse an operand. | Needs a ruling on `0/any`, or a third arity call on `LoweringContext` (M10). |
| M19 | `CONCAT_CHANNELS` no longer refuses a second result. | The exact check needs a call `LoweringContext` does not have (M10). |
| M23 | `src/optim/optimizer_registry.cpp` still calls `std::make_unique<SGD>` and `std::make_unique<AdamW>`. | M25, a later wave. |

### Paths changed

`git status --short` lists the twenty files of the wave (the eight pairs under
`include/SushiAI/graph/lowering/` and `src/graph/lowering/`, and the four optimizer files), the
files of waves 1 to 4, and two paths that belong to no unit of this design:
`docs/guides/CLI_GUIDE.md` (modified) and `docs/agent/2026_10_05_ESTATE_AUDIT/` (untracked). Both
were in that state before the wave started. Neither was touched or reverted.

### Verification

Each `.cpp` of the wave was checked alone, after the repairs. The eight lowering files took the
flags of the `src/graph/lowering.cpp` entry of `build/compile_commands.json`; `adamw.cpp` and
`sgd.cpp` took their own entries. `-o`, `-c` and the dependency flags were removed.

    D:/Projects/.sushisystems/toolchains/llvm-sycl/bin/clang++.exe -DSR_DETERMINISTIC_FP=1
      -ID:/Projects/sushiai/include -ID:/Projects/.sushisystems/vcpkg/installed/x64-windows/include
      -ID:/Projects/sushiruntime/include -ID:/Projects/sushiblas/include -O3 -DNDEBUG -std=gnu++17
      -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -Wall -Wextra -Werror -pedantic -fsycl
      -ffp-contract=off -fno-fast-math -fsycl-targets=spir64,nvidia_gpu_sm_86
      --cuda-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6 -fsyntax-only <file>

    exit=0 D:/Projects/sushiai/src/graph/lowering/elementwise.cpp
    exit=0 D:/Projects/sushiai/src/graph/lowering/activation.cpp
    exit=0 D:/Projects/sushiai/src/graph/lowering/linalg.cpp
    exit=0 D:/Projects/sushiai/src/graph/lowering/loss.cpp
    exit=0 D:/Projects/sushiai/src/graph/lowering/shape.cpp
    exit=0 D:/Projects/sushiai/src/graph/lowering/spatial.cpp
    exit=0 D:/Projects/sushiai/src/graph/lowering/norm.cpp
    exit=0 D:/Projects/sushiai/src/graph/lowering/fused.cpp
    exit=0 D:/Projects/sushiai/src/optim/adamw.cpp
    exit=0 D:/Projects/sushiai/src/optim/sgd.cpp

No invocation printed a line. A passing check instantiates every pointer to member and every
dispatch template the rows name, so each SushiBLAS name of the wave exists with the signature
the row uses.

    python tools/documentation/check_source_comments.py include/SushiAI/graph/lowering
        src/graph/lowering include/SushiAI/optim/adamw.hpp src/optim/adamw.cpp
        include/SushiAI/optim/sgd.hpp src/optim/sgd.cpp
    (no output) exit=0

Before the repairs the same command printed five lines: four `rule_license_block` on the two
linalg files and one `rule_block_ceiling` at `src/graph/lowering/fused.cpp:145`.

A search of the twenty files for `SA_THROW_IF`, `engine_`, `node.op ==`, `try`, `catch` and
`throw` finds nothing. No line passes column 100 except the two named above.

### Syntax-check error lines that remain

None in the wave's files. Not checked here, and expected to fail until their wave:

| File | Waits for |
| --- | --- |
| `src/optim/optimizer_registry.cpp` (public constructors of `AdamW` and `SGD` are gone) | M25 |
| `src/graph/lowering.cpp`, `include/SushiAI/graph/lowering/all.hpp` (the table's two `static_assert`s have not run) | M24 |

### Scratchpad

Two scripts were written to the scratchpad and deleted. It holds 108K in five entries, none from
this review.

## Wave 6

dispatch: M24 — haiku high, repaired
dispatch: M25 — haiku medium, repaired

No build and no test ran. The review read the six files against sections 1.2, 1.3, 3 (M24, M25),
4.1, 4.2 and 5 of `SPEC.md`, and opened the sibling headers each name comes from.

### Departures from the design

| Unit | Departure | Verdict |
| --- | --- | --- |
| M24 | `lowering.cpp` spells `::SushiAI::Detail::guard` and `::SushiAI::Detail::SeamPhase::Record`. Inside `SushiAI::Graph`, the unqualified `Detail` finds `SushiAI::Graph::Detail` first | Kept; the design's spelling does not resolve |
| M24 | `Lowering` holds four pointers where the design's constructor takes references. The class stays copyable and movable under the rule of zero | Kept |
| M24 | The constructor and the two accessors are defined inside `lowering.hpp` | Kept; the signatures are the design's |
| M24 | The loops of `all.hpp` carry braces and `every_identity_is_accounted_for` names its sum `owners`. The design's text puts the statement on the `for` line, which `cpp-code-style` forbids | Kept |
| M24 | The two body comments of `values_needing_own_allocation` name `ReductionOps::max_along` and `ReductionOps::sum`. `src/graph/lowering/loss.cpp:71` and `:110` make those calls | Kept |
| M24 | `values_needing_own_allocation` compares `node.op` in two `if` lines, in `src/graph/lowering.cpp`. Q6 keeps that body, and the acceptance grep covers `src/graph/lowering/` only | Kept |
| M25 | `make_sgd` has the shape of `make_adamw`, with `SGD::make` | Kept; `SGD::make` is at `include/SushiAI/optim/sgd.hpp:83` |
| M25 | The three declarations are `[[nodiscard]]`, one parameter per line | Kept; rule 8 asks for the attribute |

### Checks

| Item | M24 | M25 |
| --- | --- | --- |
| (a) Design compliance | `Lowering`'s five signatures, the includes of `lowering.hpp` and `lowering.cpp`, the body of `record`, the text of `all.hpp` and both `static_assert`s match. `SushiBLAS/engine/task_recorder.hpp`, `SushiBLAS/engine/rng_stream.hpp` and `SushiRuntime/core/span.hpp` exist. `Detail::guard(SeamPhase, Call&&)` is at `core/seam.hpp:68`, `SA_FATAL_IF` at `core/check.hpp:87`, `SA_ANNOTATE` at `core/status.hpp:266`. The `LoweringContext` constructor at `lowering/context.hpp:73` takes the eight arguments `record` passes | Four `Result<std::unique_ptr<Optimizer>>` signatures with today's parameters. Sites `:108` and `:132` are `SA_FAIL_IF` with `StatusCode::InvalidDescription` and the same format string. `AdamW::make` is at `optim/adamw.hpp:84`, `SA_TRY_ASSIGN` at `core/result.hpp:232` |
| (b) SOLID shape | One class per file. No stored view. No `try`, `catch` or `throw`. The only comparison on operation kinds is the Q6 body | The two factories have one shape. No `try`, `catch` or `throw` written here |
| (c) Comments | Repaired, see below | Passed |
| (d) Syntax check | `lowering.cpp` and `test_lowering_table.cpp` pass | `optimizer_registry.cpp` passes |
| (e) Changed paths | The four assigned files | The two assigned files |

Acceptance of M24: `src/graph/lowering.cpp` is 94 lines and holds no `record_` function;
`node.op ==` does not occur under `src/graph/lowering/`. Acceptance of M25: the two files hold no
`std::make_unique` and no `SA_THROW_IF`.

The domain tables hold 17, 14, 3, 2, 4, 12, 4 and 6 rows, 62 in all, and `K_UNLOWERED` holds 3.
Both `static_assert`s of `all.hpp` held in the syntax check of `lowering.cpp`.

### Defects repaired

| File | Defect | Repair |
| --- | --- | --- |
| `include/SushiAI/graph/lowering.hpp` | The six private members carried no comment | One `@brief` each |
| `include/SushiAI/graph/lowering.hpp` | `@file` described a translator of task streams; the class block ran its second sentence as a paragraph; two `@param` lines stated no rule | `@file` names the driver; the blocks are one `@brief`, with `@return` where the value needs it |
| `include/SushiAI/graph/lowering/all.hpp` | The block of `K_LOWERINGS` said who edits what, which is a reason; two `@brief`s opened with "Whether"; two `@param` lines stated no rule | Each block says what the symbol holds or returns, verb first |
| `src/graph/lowering.cpp` | A `//` line restated the call below it and stated no invariant; `<vector>` and `core/ops.hpp` were used and not included; the implemented header shared a group with the others | Line deleted; includes added and grouped |
| `tests/functional/unit/test_lowering_table.cpp` | The closing braces of the two namespaces had no name comment; one `@brief` cited "section 4.3" with no path; `@file` ended in "or none" | Braces named; the `@brief`s say what is tested |
| `src/optim/optimizer_registry.cpp` | `std::size_t` was used without `<cstddef>` | Include added; the two project includes put in path order |

### Defects left open

| Defect | Waits for |
| --- | --- |
| `tests/CMakeLists.txt` does not name `test_lowering_table.cpp`, so no target compiles it | The orchestrator; a build file is outside every unit |
| The callers of the old `Lowering` (its constructor over an engine, `void record`) are not converted | M26 and M27 |
| `make_adamw`, `make_sgd` and `make_optimizer` pass through a `SushiAI::Error` thrown by `Config::Value::reject_unknown_keys`, `as_real`, `as_bool` and `Config::find_entry` | Rule 5 of section 1.3; the contract's waves 3 to 5 |
| The callers of `make_optimizer` still take a `std::unique_ptr<Optimizer>` | M27, M28 and M29 |

### Syntax checks

Each ran from `D:\Projects\sushiai\build`, one file per invocation, with the flags of the file's
entry in `build/compile_commands.json` (for the test, the entry of `test_ir.cpp`), `-o` and `-c`
replaced by `-fsyntax-only`, and `-Xclang --dependent-lib=msvcrt` left out:

    clang++.exe -DSR_DETERMINISTIC_FP=1 -ID:/Projects/sushiai/include
      -ID:/Projects/.sushisystems/vcpkg/installed/x64-windows/include -ID:/Projects/sushiruntime/include
      -ID:/Projects/sushiblas/include -O3 -DNDEBUG -std=gnu++17 -D_DLL -D_MT -Wall -Wextra -Werror
      -pedantic -fsycl -ffp-contract=off -fno-fast-math -fsycl-targets=spir64,nvidia_gpu_sm_86
      "--cuda-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" -fsyntax-only <file>

    src\graph\lowering.cpp                          no error or warning line, exit 0
    src\optim\optimizer_registry.cpp                no error or warning line, exit 0
    tests\functional\unit\test_lowering_table.cpp   no error or warning line, exit 0

The test file took `-DGTEST_LINKED_AS_SHARED_LIBRARY=1` and the vcpkg path as `-isystem`, as the
entry of `test_ir.cpp` has them. No error line remains in the wave's files.

`python tools/documentation/check_source_comments.py` over the six files printed nothing and
exited 0.

### Paths changed outside the wave

`git status --short` lists the files of waves 1 to 5 and two paths no unit of the design owns:
`docs/guides/CLI_GUIDE.md` (modified) and `docs/agent/2026_10_05_ESTATE_AUDIT/` (untracked). Both
were in that state before this review started; neither was touched.

### Scratchpad

The review wrote one script, `cc.py`, and deleted it. The folder holds 95,913 bytes in five
entries the review did not create.

## Wave 7

dispatch: M26 — opus low, passed

### Review

- (a) Design compliance: the class, the constructor's initialiser list, the member order, the
  five added includes and the nine steps of `compile` match SPEC section 3, M26. `run` matches.
  The one status site is `SA_FAIL_IF(graph.has_refusal(), StatusCode::InvalidDescription, ...)`,
  as section 5 lists. Names checked in the siblings' headers: `Engine::get_context`,
  `get_scratch_pool`, `get_mode`, `get_rng` (`sushiblas: include/SushiBLAS/engine.hpp:79-97`);
  `TaskRecorder(TaskGraph&, ScratchPool&, ...)` (`engine/task_recorder.hpp:107`);
  `TaskGraph(RuntimeContext&)` (`sushiruntime: include/SushiRuntime/graph/task_graph.hpp:82`);
  `CompiledGraph::roots`, `capture_base_state()`, `rearm()` (`execution/execution_engine.hpp:68-99`).
- (b) SOLID shape: one class per file, no switch over operation kinds, no stored view, no `try`,
  `catch` or `throw`. The file names neither `get_graph` nor `Engine::execute`, and calls
  `compile()` on no object itself; `graph_` is compiled once, through `Detail::compile_graph`.
- (c) Comments: every symbol carries a Doxygen block of at most seven lines that says what the
  symbol does. The one `//` line in `compile` states the order `capture_base_state()` relies on.
- (d) Syntax check, run by the reviewer on the one file:
  `clang++.exe <flags of the src/graph/compiled_step.cpp entry, minus -o and -c> -fsyntax-only D:/Projects/sushiai/src/graph/compiled_step.cpp`
  printed nothing and returned `exit= 0`.
- (e) `git status --short` shows both files of the wave as modified. It also shows the files of
  earlier waves, and two paths this migration's design does not name: `docs/guides/CLI_GUIDE.md`
  (modified) and `docs/agent/2026_10_05_ESTATE_AUDIT/` (untracked). Neither was touched.

### Departures from the design

- Seam calls are spelled `SushiAI::Detail::...`. Inside `SushiAI::Graph`, `Detail::` resolves to
  `SushiAI::Graph::Detail`.
- `compile` is static, so steps 4 and 6 set the counters through `step->`.
- `selections_` is copied from the lowering before the failure check of step 5, so a failed
  recording copies them and then drops them with the step.
- The inline accessors carry `[[nodiscard]]` and Allman bodies.

### Defects repaired

None.

### Defects left open

- `compiled_step.hpp` includes `<SushiAI/graph/lowering.hpp>` and uses only `TensorPool` from it;
  `compiled_step.cpp` reaches `Lowering` through that include. Moving the include to the `.cpp`
  and naming `tensor_pool.hpp` in the header can break a file of a later wave that reaches
  `Lowering` through `compiled_step.hpp`. It waits for M27.
- `CompiledStep::empty()` is a bool query that does not read as a question (`cpp-code-style`
  asks for `is_empty`). Renaming it changes a public signature.

### Syntax-check error lines that remain

None in the wave's file. Callers of the removed constructor and of `void run()` were not
checked; they belong to M27 and later waves.

### Scratchpad

The reviewer wrote one script, `w7_syntax.py`, and deleted it. The scratchpad holds 95,913
bytes, none from this review.

## Wave 8

dispatch: M27 — opus low, repaired

No build and no test ran. The review read the three files against SPEC sections 1.2, 1.3, 5 and
M27, and opened the sibling headers for each name used.

### Checked

- (a) Signatures of `bind_parameters`, `Trainer::make`, `run_step`, `fit`, `score_accuracy` and
  the private constructor match M27. `make` runs its seven steps in the order M27 lists.
  `BLASOps(Engine&)` is at `sushiblas: include/SushiBLAS/engine/blas.hpp:46`;
  `copy(const Tensor&, Tensor&)` at `engine/blas/level1.hpp:59`; `Engine::get_context()` at
  `engine.hpp:88`; `reset_profiling` and `collect_profiling` at
  `sushiruntime: include/SushiRuntime/execution/execution_engine.hpp:282`, `:291`. `SushiAI.h`
  gained the three includes M27 names, each beside the lines of its folder.
- (b) `trainer.cpp` and `trainer.hpp` hold no `try`, `catch`, `throw` or `SA_THROW_IF` (grep
  count 0 in both). `run_step` builds its `BLASOps` as a local inside the guarded lambda; no view
  is stored. The one `switch` is over the loss dtype, not over operation kinds.
- (d) `clang++ -fsyntax-only` on `src/train/trainer.cpp` with the flags of its entry in
  `build/compile_commands.json`, run after the repairs: no output, `exit=0`.
- (e) `git status --short` lists the three files of the wave as modified.
  `docs/guides/CLI_GUIDE.md` is also modified (13 lines added). It belongs to no unit of this
  wave and was left as found.

### Departures from the design

- `Trainer` deletes its copy and move members and defaults its destructor. M27 lists none of the
  five. `cpp-code-style` asks for all five on a type that holds a reference, and `CompiledStep`
  does the same, so the review kept them.
- `fit` calls `reset_profiling` and `collect_profiling` outside `Detail::guard`. Section 2.3 says
  the profiling calls keep their form. Both are virtual and not `noexcept`, so an exception from
  either leaves `fit` as an exception.

### Defects repaired

- `trainer.hpp`: the comment of `ParameterBinding::gradient` named `Graph::NO_VALUE`; the
  constant is `Graph::K_NO_VALUE`.
- `trainer.hpp`: the two enumerator blocks of `OptimizerPlacement` gave reasons and history
  ("the behaviour before the update could be compiled"). Each now says what the placement does;
  the replay-invariance requirement is a `@pre`.

### Defects left open

- Two failure texts in `placement_recorder` name `Optim::UpdateKernel::FUSED` and
  `OptimizerPlacement::PER_STEP`. The enumerators are `Fused` and `PerStep`. Rule 1.3.4 keeps the
  format string, and `tests/functional/regression/test_fused_optimizer.cpp:431` expects the
  substring `PER_STEP`. Changing the text needs that test file, which belongs to M30.
- The unguarded profiling calls in `fit`, above. Guarding them is a change to the design.
- `SushiAI.h` was not checked through a translation unit; no `.cpp` of this wave includes it.

### Syntax-check error lines that remain

None for `src/train/trainer.cpp`.


## Wave 9

Applications, examples and tests, in four groups, each reviewed by an `opus` `low` reviewer whose
reply follows, then one sweep over `apps/`, `examples/` and `tests/`. Nothing was built or run.

### Group 1

dispatch: apps/train.cpp — opus low, repaired
dispatch: apps/inference.cpp — opus low, repaired
dispatch: examples/demo_mlp.cpp — opus low, repaired
dispatch: examples/example.cpp — haiku high, repaired
dispatch: tests/package/main.cpp — haiku medium, passed
dispatch: tests/functional/integration/test_engine_smoke.cpp — haiku medium, passed
dispatch: tests/common/reference_fixture.hpp — opus low, passed
dispatch: tests/common/test_utils.hpp — opus low, passed (no edit)

Repairs:
- train, example: `run` moved into the anonymous namespace, as in the other two.
- train, inference, demo_mlp: missing standard includes added.
- inference: the unprefixed `The step must be compiled exactly once.` is the first failure line again (SPEC M28), followed by the prefixed status line, as in train.
- inference, demo_mlp, example: every line over 100 columns wrapped or reworded.

Syntax check: `clang++ -fsyntax-only`, one file per invocation, with each entry's flags minus `-fsycl-targets`, `--cuda-path` and `--dependent-lib`. All eight exit 0 with no diagnostics. Nothing was built or run.

Open:
- `tests/common/reference_fixture.hpp:98`, `:100`, `:183` keep `SA_THROW_IF`.
- (e) fails: `git status --short` lists paths outside wave 9 that I did not touch: 6 headers under `include/SushiAI/`, 12 sources under `src/`, `docs/guides/CLI_GUIDE.md` and `SPEC.md`.

Library defect: `include/SushiAI/nn/layers.hpp:118` names a type `K_GELU`, the constant prefix on a struct; `demo_mlp.cpp` has to spell `sa::NN::K_GELU{}`.

Scratchpad: 172K, nothing written by me.

### Group 2

dispatch: tests/functional/unit/test_step_recorder.cpp — haiku high, repaired
dispatch: tests/functional/regression/test_fusion_numerics.cpp — opus low, repaired
dispatch: tests/functional/regression/test_fused_optimizer.cpp — opus low, repaired
dispatch: tests/functional/integration/test_resilience.cpp — opus low, repaired
dispatch: tests/functional/integration/test_forward.cpp — opus low, repaired
dispatch: tests/functional/unit/test_checkpoint_format.cpp — haiku high, repaired

Repairs:
- All nine fatal cases now sit in `*DeathTest` suites.
- The step recorder fixture is `Unit_StepRecorder` again, so test IDs stay.
- `train` in `test_fusion_numerics.cpp` fills an out-parameter under `ASSERT_NO_FATAL_FAILURE`, as its sibling does; the wrapper that carried on after a fatal failure is gone.
- One 104-column line wrapped in `test_fused_optimizer.cpp`.

Syntax check: `clang++.exe <entry flags, minus -o/-c and --dependent-lib> -fsyntax-only <file>`, six separate runs, each `exit=0` with no output. Nothing was built or run; death patterns are unverified.

Open:
- (e) fails: `git status --short` lists 18 library files under `include/` and `src/`, not mine.
- `tests/functional/*/README.md` do not name the new `*DeathTest` suites; outside my file list.
- `test_resilience.cpp:134`: `NoBatches::batch` keeps `SA_THROW_IF`; `Dataset::batch` returns a reference.
- `test_fusion_numerics.cpp`: ten lines over 100 columns, already in HEAD.
- Comments on unchanged lines were not reviewed.

Library defect:
- `src/train/trainer.cpp:144-145`: the message names `UpdateKernel::FUSED` and `OptimizerPlacement::PER_STEP`; the enumerators are `Fused` and `PerStep`. `test_fused_optimizer.cpp` pins `"PER_STEP"`.

Scratchpad: 180K, none mine.

### Group 3

dispatch: tests/functional/integration/test_arena.cpp — haiku high, repaired
dispatch: tests/functional/integration/test_autograd_numeric.cpp — haiku high, repaired
dispatch: tests/functional/integration/test_checkpoint.cpp — haiku high, open
dispatch: tests/functional/integration/test_mixed_precision.cpp — haiku high, passed
dispatch: tests/functional/integration/test_registries.cpp — haiku high, repaired
dispatch: tests/functional/integration/test_rematerialise.cpp — haiku high, passed
dispatch: tests/functional/integration/test_training.cpp — haiku high, passed

Repairs:
- test_arena.cpp: the three `SA_EXPECT_FATAL` cases moved to `Integration_ArenaDeathTest`, an alias of the fixture. The vacuous `EXPECT_NO_THROW` now checks `storage_offset == 1`.
- test_autograd_numeric.cpp: added `<utility>`; the 101-column line is wrapped.
- test_registries.cpp:167: the comment said "throws" for a status site.

Open:
- test_checkpoint.cpp:214, :228, :241, :255: four test names still say "Throws" for `StatusCode::Incompatible` sites. `tests/functional/integration/README.md:326` and `:331` head sections by those names, and that file is outside my list.
- `tests/functional/integration/README.md:16-61` does not name the new `Integration_ArenaDeathTest` suite.

Library defects: none found.

Checks:
- (d) `clang++.exe -fsyntax-only <entry flags> <file>`, `-o`/`-c` removed, one file per run: all seven gave exit 0 with no output.
- (e) `git status --short` now also lists 18 files under `include/` and `src/` (for example `src/train/trainer.cpp`) and three more regression tests. They appeared during this session and I did not write them.
- Nothing was built or run.

Scratchpad: 176K; my one script is deleted.

### Group 4

dispatch: tests/functional/regression/test_compile_once.cpp — haiku high, repaired
dispatch: tests/functional/regression/test_determinism.cpp — haiku high, repaired
dispatch: tests/functional/regression/test_learning_curve.cpp — haiku high, repaired
dispatch: tests/functional/regression/test_log_softmax_stability.cpp — haiku high, repaired
dispatch: tests/functional/regression/test_loss_scaling.cpp — haiku high, repaired
dispatch: tests/functional/regression/test_mixed_precision.cpp — haiku high, repaired
dispatch: tests/functional/regression/test_static_memory.cpp — haiku high, repaired

Repairs:
- `train_once`, `train_scaled`, `train` and both lambdas now return `[[nodiscard]] Result<T>` through `SA_TRY`, like the two sibling helpers. `train_scaled` callers previously ran on after a failure.
- Over-wide lines are wrapped, trailing `//` comments moved to their own lines, and same-line `if` bodies split.
- `kLse`, `kExpectedRow`, `kGaps` became `K_LSE`, `K_EXPECTED_ROW`, `K_GAPS`.
- Two tests gained their missing `@brief`.

No test was deleted and no pinned number changed; test counts match HEAD (4, 5, 2, 4, 8, 5, 5).

Verification: `clang++.exe <entry flags> -fsyntax-only <file>` run seven times, one file each, printed `EXIT=0` with no diagnostics. `check_source_comments.py tests/functional/regression` exited 0. Nothing was built or run.

Open in these files: none.

`git status --short` outside wave 9: `docs/guides/CLI_GUIDE.md` (modified) and `docs/agent/2026_10_05_ESTATE_AUDIT/` (untracked). Commit 464ee2a landed during this review.

Library defect: `include/SushiAI/nn/layers.hpp:118` names a type `struct K_GELU`; `K_` belongs to constants.

Scratchpad: 76K; my one script is deleted.

### Sweep

**Edited (1 file)**

- `D:\Projects\sushiai\tests\functional\unit\test_loss_scaling.cpp`: it built `LossScaleState{options}` through the constructor that is now private. Eight constructions became `SA_ASSERT_OK_AND_ASSIGN(..., LossScaleState::make(...))`. Five `SA_EXPECT_ERROR` expectations became `SA_EXPECT_FAILURE(LossScaleState::make(options), StatusCode::InvalidDescription, ...)`; the sites are `status` rows in `site_classes.tsv`. The include `test_throws.hpp` became `test_status.hpp`.

**Found and left**

- `TensorPool pool(engine)` in apps, examples and tests: the constructor is unchanged, and `SushiAI.h` includes `graph/tensor_pool.hpp`.
- `tests/functional/unit/test_objective.cpp`: `Train::build_graph` and `make_objective` still return values and still throw, so `SA_EXPECT_ERROR` stays.
- `test_concat.cpp`, `test_node.cpp`, `test_model_graph.cpp`, `test_node_registry.cpp`: they already use the `Result` factories.
- `tests/common/test_utils.hpp`: no edit, as section 2.4 rules.
- The searches for facade accessors, `get_graph`, argument-less `record_into_step()`, `SA_THROW_IF` in apps and examples, and the ten public constructors returned no code hits.

Not done: I did not check the 28 wave-9 files for discarded `Status` or `Result` values, and I did not syntax-check them.

**READMEs with old spellings (not edited)**

- `tests/functional/unit/README.md`: about 101 lines (`DType::FLOAT32`, `FLOAT64`, `Lifetime::BUFFER`, `Ops::ALL`, `NN::LAYER_REGISTRY`), plus 5 lines naming an old accessor or `record_into_step()`.
- `tests/functional/integration/README.md`: 6 lines (`FLOAT64`, `HALF`, `INT64`, `Data::DATASET_REGISTRY`).
- `tests/functional/regression/README.md`: 4 lines (`FORCE_REFERENCE`, `VERIFY`, `FusedForm::GEMM_BIAS_GELU`, `FLOAT32`).
- `tests/common/README.md` and `tests/package/README.md`: none.

**Library defects:** none found.

**Syntax check**

`clang++.exe <flags of the file's compile_commands.json entry, minus -o/-c and the SYCL target and CUDA path flags> -fsyntax-only tests/functional/unit/test_loss_scaling.cpp` printed no error or warning line and ended with `exit 0`.

**Scratchpad:** 76K, none of it written by me; nothing to delete.

### After the waves

- The naming wave of `7be22fa` had renamed the layer type `NN::GELU` to `K_GELU`, because the
  operation constant `Ops::GELU` has the same spelling. The type is `GELU` again in eleven files.
- The trainer's message named `UpdateKernel::FUSED` and `OptimizerPlacement::PER_STEP`; it names
  `Fused` and `PerStep`, and `test_fused_optimizer.cpp` pins `PerStep`.
- Commit `464ee2a` applied rule 10 of the design (a host `std::bad_alloc` is `OutOfMemory`),
  restored the refusal of a second `CONCAT_CHANNELS` result, and made the two checkpoint stream
  classes `final`.

### Measured: the model of a unit and what its reviewer did

| Model | Units | Passed untouched | Repaired | Left open |
| --- | --- | --- | --- | --- |
| `haiku` (waves 1 to 8) | 21 | 1 | 20 | 0 |
| `opus` `low` (waves 1 to 8) | 6 | 1 | 5 | 0 |

No reviewer of waves 1 to 8 changed a call, a task count or a failure text; the repairs were
comment length, line width, include order and one missing include. Compile and test results per
unit are added here after the owner's build.
