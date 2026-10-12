# Eager front end: report

dispatch: 2026-10-12, general-purpose, model opus, effort medium (set by the owner), task: implement what SPEC.md and PLAN.md leave open (`Session`, `Step`, `StepExecutor`, `RegionExecutor`, `RegionEmitter`, boundary storage matching, host cost measurement), no build.

The harness did not let the agent write this file, so the orchestrator placed the agent's report
here. The section "Build and test run" at the end is the orchestrator's.

## State

The units are written and every new file passed `clang++ -fsyntax-only` under the module's own
flags (`-Wall -Wextra -Werror -pedantic`). The agent built nothing and ran no test. Three things
are not done; see "Not done".

## PLAN tasks

| Task | State |
| --- | --- |
| 1 Provenance, 2 scope tracing, 3 partitioner, 4 structure key | Done before this dispatch. Not touched. |
| `RegionEmitter` | Done: `region_emitter.{hpp,cpp}`, `test_region_emitter.cpp`. |
| `StepExecutor` | Done: `step_executor.hpp`, an interface with `execute` and `discard`. |
| `RegionExecutor` | Done: `region_executor.{hpp,cpp}`, `test_region_executor.cpp`. |
| `Session`, `Step` | Done for a step that runs once: `session.{hpp,cpp}`, `step.{hpp,cpp}`, `test_session.cpp`. |
| Boundary storage matching | Done: `region_boundary`, `slot_ledger`, `region_table`, one test file each. |
| Host cost of an unchanged step and of renumbering | Delivered as `test_step_cost.cpp`, which prints the numbers. |
| Flush points | Done: a read, `apply()` and `finish()` each run the step. |
| Phases (SPEC section 8), debug flush after every node | Not done. |

Units the SPEC did not name and the work needed: `StorageID` with `StorageIDSource`,
`BoundaryMap` with `find_boundaries`, `RegionNames`, `SlotLedger`, `RegionTable`, `RegionProgram`
with `compile_region`, `StorageBank`.

## How it works

- A region is lowered from a graph of its own. `compile_region` copies the step's values with
  their ids, adds one `K_LEAF` per value the region reads from outside, then the region's nodes,
  and runs the cast pass, fusion, rematerialisation and the memory planner over it.
  `Graph::Lowering::record` lowers it whole. No hook in `modules/core/graph` was needed.
- `Region::add_emitter` does not exist in SushiRuntime. `Region::recorder()` hands out the
  `Recorder`, and `SushiBLAS::TaskRecorder` is built over it, as `CompiledStep` does over
  `Graph::recorder()`.
- The update is passed to `Step::apply` as a `Graph::StepRecorder`. `Optim::Optimizer` is one, so
  `eager` keeps its dependencies `core graph autograd`.

## Files

Added under `modules/core/eager/`:

- `include/SushiAI/eager/`: `storage_id.hpp`, `region_boundary.hpp`, `region_names.hpp`,
  `slot_ledger.hpp`, `region_table.hpp`, `region_program.hpp`, `storage_bank.hpp`,
  `region_emitter.hpp`, `step_executor.hpp`, `region_executor.hpp`, `session.hpp`, `step.hpp`
- `source/`: `region_boundary.cpp`, `region_names.cpp`, `slot_ledger.cpp`, `region_table.cpp`,
  `region_program.cpp`, `storage_bank.cpp`, `region_emitter.cpp`, `region_executor.cpp`,
  `session.cpp`, `step.cpp`, and the private header `tensor_shape.hpp`
- `tests/`: `test_region_boundary.cpp`, `test_slot_ledger.cpp`, `test_region_table.cpp`,
  `test_region_program.cpp` (host lane); `test_storage_bank.cpp`, `test_region_emitter.cpp`,
  `test_region_executor.cpp`, `test_session.cpp`, `test_step_cost.cpp` (need an engine); support
  headers `eager_test_support.hpp` and `eager_run_support.hpp`

Changed by the agent: `modules/core/eager/README.md`, `tests/unit/README.md`, the "Eager front
end" section of `docs/design/REMAINING_WORK.md`. Changed by the orchestrator:
`modules/core/eager/CMakeLists.txt` (ten sources, nine tests) and the umbrella header
`modules/layers/umbrella/include/SushiAI/SushiAI.h` (twelve includes).

Pages outside the agent's write list that the work made false:

- Root `README.md` describes `eager/` as "The host side of the eager front end: scope tracing
  and partitioning". The module now also runs a step.
- `docs/reference/CHANGELOG.md` has no entry for the work.
- `SPEC.md` names `Graph::add_emitter`, `Session(runtime, options)` and
  `Eager::Scope scope(step, "head")`. The code has `Region::recorder()`,
  `Session(engine, executor, partitioner, options)` and `Eager::Scope scope(step.scopes(), "head")`.

## Not done

1. **Phases and the debug flush.** A step runs once. A node traced after the run, or after
   `backward()`, makes the next read return `Unsupported`. Two decisions are missing, and one
   change outside the write list:
   - Which values of a closed phase stay readable for a later one. At the read, the step does
     not know what later nodes will read, and an unobserved value may be folded away or
     overwritten in an arena.
   - `Autograd::differentiate` calls `Graph::finalise()`, which propagates `requires_grad`
     through every node, so values of an earlier phase cannot be treated as constants. That
     needs a parameter on `differentiate` in `modules/core/autograd/source/differentiator.{hpp,cpp}`.
2. **The comparison with `Train::Trainer`** (SPEC section 10, first item). A test that names
   `Trainer` crosses modules and belongs under the root `tests/`. `test_session.cpp` compares
   with `Graph::CompiledStep` and a hand-written update.
3. **A device test of a gradient summed across regions.** `test_region_boundary.cpp` covers the
   buffer rule on the host. No test runs a model in which one value feeds two scopes.

Outside the SPEC's scope and not covered: buffer updates (running statistics), mixed precision,
loss scaling.

## Open design choices

Each is implemented one way so the code runs; each is the owner's to change.

1. **Constructor of `Session`.** It takes the engine, a `StepExecutor&` and a
   `RegionPartitioner&`; the SPEC wrote `Session(runtime, options)`. Recommendation: keep it.
2. **`Scope` opens on `step.scopes()`.** `Scope` takes a `ScopeTrace&`; taking a `Step&` would
   change an approved unit. Recommendation: keep it.
3. **When a step runs.** `apply()`, a read and `finish()` each run it. The destructor does not,
   because it cannot return a `Status`. Recommendation: keep it.
4. **The update is a `Graph::StepRecorder`.** The module cannot check
   `Optimizer::is_replay_invariant()` through that type; the precondition is stated on
   `Step::apply`. The alternative is a dependency on `optim`. Recommendation: keep the interface
   and let the `train`-level wrapper check it.
5. **The matching rule** (`SlotLedger`, `RegionTable`). An unchanged producer keeps its slots by
   position. A changed producer's buffer takes the slot that a reader with unchanged structure
   read under the same import place, if that slot had the same producer, extents and element
   type. Otherwise it takes the slot at the same position, otherwise a new one. A wrong match
   costs a rebuild and cannot run wrong kernels, because the storage id is in the reader's
   signature. Recommendation: keep it and judge it on emitter counts of real models.
6. **Gradient storage.** One tensor per trainable state, allocated at `bind_state`; the step's
   gradient value is placed in it directly. Two states whose gradient is one graph value make
   the step return `Unsupported`. Recommendation: keep direct placement.
7. **A parameter the step did not trace gets a zero gradient** before an update runs. Its
   optimizer state still moves under momentum and weight decay. Recommendation: keep the fill;
   decide separately whether an optimizer should skip.
8. **A renumbered region gets a new `TensorPool`** and keeps its compiled program.
   Recommendation: keep the new pool until it is checked that every lowering row can run twice
   into one pool.
9. **Rematerialisation runs per region**; `ProgramOptions::rematerialise` turns it off.
10. **Renumbering itself.** A backward region has key `2m - k`, so one new scope moves every
    backward key and the optimizer's. Counting backward keys down from a fixed large number
    would change `ScopePartitioner`, an approved unit. Recommendation: read the renumbering
    cost off `test_step_cost.cpp` first.
11. **`signature_of` scans every node once per region**, so host cost per step is nodes times
    regions.

## Evidence

### SOLID shape of each new unit

| Unit | One job | Reached through |
| --- | --- | --- |
| `StorageIDSource` | Hands out ids | Reference held by `RegionTable`, `SlotLedger`, `Session` |
| `find_boundaries` | Lists what each region imports and exports | Free function over `Graph`, `Partition` |
| `RegionNames` | Name and direction to one id | Owned by `RegionTable` |
| `SlotLedger` | Which slot a shared buffer gets | Three calls; knows no graph and no tensor |
| `RegionTable` | Keep, record, build or drop each region | `plan()` returns a `StepPlan` value |
| `compile_region` | One region as a graph the lowering takes | Free function; returns a `RegionProgram` value |
| `StorageBank` | Storage id to tensor | Owned by `Session`; passed to the executor by reference |
| `RegionEmitter` | Lowers one program into one recorder | Constructed over engine, program, pool |
| `StepExecutor` | Interface: run one planned step | `Session` holds it by reference |
| `RegionExecutor` | The executor over `DynamicGraph` | Implements `StepExecutor`; `final` |
| `Session` | Owns what lives across steps | Depends on the two interfaces, not their implementations |
| `Step` | One step's trace and its flush points | Created by `Session::begin_step` |

The host-lane units include no engine header. `Session` and `Step` are friends of each other,
because `Step` calls the session's private `run`, `find_state` and `tensor_of`.

### Humanizer

The agent loaded the `humanizer` skill with `generic_english.md` before writing prose, and
rewrote three README sentences with history wording on reread.

### Syntax checks

One process at a time, with the flags of `region_signature.cpp` (sources) and
`test_scope_trace.cpp` (tests) from `build/compile_commands.json`, cut at ` -o ` and followed by
`-fsyntax-only <file>`. All ten sources and all nine tests returned exit code 0.
`tools/documentation/check_source_comments.py modules/core/eager` printed nothing after two
findings were shortened.

### Scratchpad

The agent wrote small Python and text files to the session scratchpad and deleted them. It wrote
no binary, no copy and no build output.

## Build and test run

The orchestrator added the CMake and umbrella lines and built with `sa build` on 2026-10-12.
The first run of the nine new suites passed 63 of 66 tests. The three failures and their repairs:

- `Unit_RegionBoundary.ImportsAreListedOnceInReadOrder`: a defect of the test. It traced
  `add(add(b, a), add(a, b))` in one expression, and C++ leaves the order of the two arguments
  open. The test traces the two inner sums in two statements.
- `Unit_EagerSession.AScopeThatAppearsAndDisappearsRecordsTheRegionsItRenumbers` and
  `Unit_EagerStepCost.AScopeThatAppearsAndDisappears`: a defect of `RegionExecutor`. A region
  recorded again under another key keeps its compiled program, whose value ids are those of the
  step that built it, and the executor looked them up in the storage table of the current step.
  The step returned `InvalidDescription`. `LiveRegion` now holds the storage ids of its imports
  and exports (`SharedStorage`), and a region recorded again binds those.

After the repairs the nine suites pass 66 of 66. `Unit_EagerStepCost` printed, on a machine that
was also building another project, so the whole-step figures are not a measurement to keep:

| Case | Regions | Host stages per step |
| --- | --- | --- |
| Two layers, no scope | 1 | 22.9 us |
| Two layers, one scope per layer | 5 | 25.6 us |
| Fifty layers, no scope | 1 | 140.3 us |
| Fifty layers, one scope per layer | 101 | 397.7 us |
| Fifty layers, a scope appears | 101 | 878.1 us |
| Fifty layers, a scope disappears | 101 | 522.2 us |

`SPEC.md` still holds the three names listed under "Files"; it is the owner's approved text and
was not edited.
