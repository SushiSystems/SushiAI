# Eager phases: plan

**Status:** Open — plan written; the owner's decisions below are not yet ruled, and no design text or code is written. Serves RW-17.
**Designs:** GRAPH_CAPABILITIES

**Goal:** design the phases of an eager step, so that a step can be read in the middle and traced
on, and name every code change that design needs.

**Source:** `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md` section 8, and item 1 of "Not done" in
the `REPORT.md` beside it.

**RW-17, done when:** the design says which values of a closed phase stay readable, and how
`Autograd::differentiate` treats an earlier phase's values as constants while `Graph::finalise`
propagates `requires_grad`.

Line numbers below are those of the tree on 2026-11-24.

## 1. What the code does today

A step runs once. `Step::flush` (`modules/core/eager/source/step.cpp:246`) seals the scopes, hands
the whole trace to `Session::run` and sets the stage to `Ran`. A node traced after that, or after
`backward`, makes `require_no_late_node` (`step.cpp:235`) return `Unsupported` on the next read.

Inside one run, a value has one of two homes:

- **A `StorageBank` slot** that the session owns across steps. `Session::run` (`session.cpp:440`)
  marks as observed the values passed to `keep`, the value whose read started the run, the loss,
  each parameter gradient and each blend source. `find_boundaries` adds every value two regions
  share. Each of these gets a `StorageID`, and `Session::tensor_of` (`session.cpp:532`) returns it.
  State leaves and feeds also have slots, written by `place_leaves`.
- **A region's own arenas**, in the `Graph::TensorPool` that `RegionExecutor::LiveRegion` owns.
  The fusion pass may fold such a value into a fused node (a `matmul`, `bias_add`, `relu` chain
  becomes one `GEMMBias` kernel) so that it never exists, and the memory planner may pack another
  value into its bytes later in the same region.

`Session::tensor_of` already refuses the second kind with "pass it to Step::keep() or read it
before the step runs". The module README says the same at lines 479 to 481.

`Autograd::differentiate` (`modules/core/autograd/source/differentiator.cpp:323`) builds a
`Differentiator` whose `run` (line 196) calls `graph_.finalise()` and then walks every forward node
from the last to node 0 (line 220). `Graph::finalise` (`modules/core/graph/include/SushiAI/graph/ir.hpp:470`)
calls `propagate_requires_grad` (line 519), which marks the results of every node that reads a
value with `requires_grad`, from node 0 on. The backward rules ask `GradientTape::input_requires_grad`
(`tape.cpp:97`), which reads that flag on the graph; `rules/loss.cpp:114` reads it directly.

SushiRuntime's `DynamicGraph` (`sushiruntime/modules/surface/api/include/SushiRuntime/api/graph/dynamic_graph.hpp`)
runs every live region on each run and offers no way to run a subset. `RegionExecutor` owns one
(`region_executor.hpp:208`).

## 2. The design to write

The worker who writes the design text states the following as decisions, with the reasons given.

### 2.1 What a phase is

A flush that does not end the step closes a phase. The flushes that close a phase are a `read` or
`read_scalar` before `backward`. The flushes that end the step are `apply`, `finish`, and any read
after `backward`. Phase `i` holds the forward nodes traced between flush `i - 1` and flush `i`, as
one contiguous range of node ids; the last phase also holds every backward node.

A flush in the middle of an open scope ends the running segment, so no segment spans two phases.
The scope continues in the next phase as a new occurrence of the same name (`head#1`), and so as a
region of its own.

Each phase runs on a `DynamicGraph` of its own, because a runtime graph runs every live region.
Phase `i` of one step is matched with phase `i` of the step before by its ordinal. A loop that
reads at the same points on every step keeps every phase's regions; a loop whose read moves
rebuilds the phases after it.

### 2.2 Which values of a closed phase stay readable

**Rule:** a value of a closed phase stays readable, and usable as an operand by a later phase,
when it had a `StorageID` at the flush that closed its phase. Those are:

1. a state leaf or a feed;
2. a value passed to `Step::keep` before the flush;
3. the value whose read caused the flush;
4. a value two regions of that phase share (a boundary value);
5. a blend source.

Any other value of a closed phase is gone. A later node that reads it makes the next flush return
`Unsupported`, naming the value and `Step::keep`. A later `read` of it returns the same refusal.

**Why this rule.** At the flush the step cannot know what later nodes will read: the caller's code
has not run yet. Three alternatives were weighed and refused:

- *Keep every value of a closed phase.* Every value becomes observed, so no chain folds and the
  planner packs nothing. A phase's memory becomes the sum of its activations, and its kernels lose
  the fusion the compiled path has. That is the debug flush of RW-18, offered as an option, not the
  default.
- *Learn from the step before.* The session would remember which earlier values later phases read
  and mark them observed on the next step. The first step, and any step whose structure changed,
  would still fail, so the same code would fail or pass depending on history.
- *Run the closed phase again* with the missing value observed. That is safe only while every
  write of the phase can be repeated: an `ACCUMULATE` into a slot cannot. The value that already
  fed a host branch must also come out bit for bit the same, and `dropout` draws its mask from the
  state and the step index (`graph/dropout_mask.hpp`), so that holds only while the index counts
  steps and not runs. The design would rest on both staying true, and the first step would pay a
  rebuild and a second run of the phase. One `keep` call costs less.

The rule is the one a single-phase step already applies (README lines 479 to 481) extended to a
phase boundary, so a reader learns one rule. Its cost falls on the caller: a branch that reads a
gate and then traces on from the encoder's output must `keep` that output before the read.

The design text states the readable window too: a value of phase `i` stays readable until the
session's next step runs its phase `i`, because the slot is rewritten then.

### 2.3 How `differentiate` treats earlier phases as constants

SPEC section 8 decides the semantics: `backward` runs in the phase that produced the loss, over
values of that phase and of earlier phases taken as constants. What is wrong today, if the step
simply called `differentiate` over the whole graph:

- `propagate_requires_grad` sets `requires_grad` on every result downstream of a parameter, earlier
  phases included, so the walk differentiates the nodes of a closed phase.
- Their backward rules read values of that phase (the input of a `matmul`, the result of a `relu`)
  that the rule in 2.2 has dropped. The backward regions would import values no slot holds.
- A parameter used in a closed phase would get a gradient through a path the design says is
  constant.

**The smallest change** is a node id below which every computed value is a constant. Call it
`constant_before`; the name is the worker's to settle with the owner.

- `Graph::finalise(NodeID constant_before = 0)` passes it to `propagate_requires_grad`. For a node
  with an id below it that is not a `K_LEAF`, the propagation sets each result's `requires_grad`
  to false and goes on. A `K_LEAF` keeps what the builder gave it, so a parameter declared in phase
  0 and read directly in phase 1 still receives the gradient of that read. With the default 0 the
  sweep is the one today.
- `Autograd::differentiate` gains a trailing `Graph::NodeID constant_before = 0` after
  `provenance`. `Differentiator` stores it, `run` passes it to `finalise`, and the reverse walk
  stops at it (`offset > constant_before` in place of `offset > 0`). `require_seedable_root`
  refuses a loss whose producer lies below it, with `InvalidDescription` naming the phase.
- `Step::backward` passes the first node id of the current phase.

The flag has to be right on the graph, not only skipped by the walk: the rules read it through
`input_requires_grad`, and a phase-1 node whose only differentiable operand is a phase-0 activation
must not count as requiring a gradient, or the root check at `differentiator.cpp:241` would accept
a loss no parameter of the phase reaches.

A parameter that only a closed phase reads gets no gradient value. The session already fills such
a gradient with zeros before the update (README line 456), so the optimizer sees zero, and its
momentum and weight decay still move the parameter (REPORT, open choice 7). In the branch example
of SPEC section 2, the encoder trains on no step that reads in the middle. The design text says
so plainly; question O7 asks the owner to accept it for the first version.

Refused alternative: a separate `Graph` per phase, with an earlier phase's values declared as input
leaves of the next. `differentiate` would need no change, but a `ValueID` names a value of one
graph, so the caller would have to import each earlier value with a new call before using it. Every
layer's `forward(builder, x)` would then receive ids of the wrong graph unless the caller wraps
them. The node-id frontier keeps one graph and changes two functions.

### 2.4 What the session does per phase

- `ScopeTrace` gains a public `cut()`: it ends the running segment and returns the segments so far,
  and leaves the open scopes open. `seal()` stays the last call.
- The partitioner runs over the graph as it stands at the flush, with the segments so far, as it
  does today; `ScopePartitioner` needs no change, because for a phase with no backward node the
  provenance has `forward_node_count` equal to the node count and no origin. A new free function
  restricts the result to the phase: a node outside the phase's node range gets `K_NO_REGION`, and
  the regions that lose every node leave the list. An earlier phase's value read by the phase is
  then external, as a leaf is.
- `Session` keeps one `RegionTable` and one `StepPlan` per phase ordinal. `place_leaves` writes into
  `preset` the `StorageID` of every earlier-phase value the phase reads, taken from the plan of the
  phase that produced it, and refuses one with no slot (2.2).
- `tensor_of` looks the value up in the plan of the phase that produced it.
- The trailing work (copies, the caller's update, the blends) runs only in the last phase. The step
  counter, which is the `run_index` the preludes get and so the dropout index, moves once per step,
  after the last phase, never per phase.
- `recover` forgets every phase's table and discards every phase's runtime graph.
- `StepCost` sums the phases. Per-phase cost is RW-16's to ask for.

## 3. Decisions for the owner

Each is the owner's under `sushiskills` boundary 5. The plan recommends; the owner rules.

| Id | Question | Recommendation |
| --- | --- | --- |
| O1 | May `Autograd::differentiate` gain a trailing `Graph::NodeID constant_before = 0`? Source compatible; the exported symbol changes, as it did for `provenance` on 2026-10-07. | Yes. |
| O2 | May `Graph::finalise` gain `NodeID constant_before = 0`, so the flag is right on the graph that the rules read? | Yes. |
| O3 | Is the readable rule of 2.2 (only values with a slot at the flush; the rest refused with `Unsupported` naming `keep`) the one the eager front end promises? | Yes; keep-everything stays the RW-18 debug option. |
| O4 | Where does a phase's runtime graph live: `StepWork` gains `phase` and `last_phase` and `RegionExecutor` keeps one `DynamicGraph` per phase, or `Session` takes a factory of executors and keeps one executor per phase? | The `StepWork` fields. The session's constructor and the tests that read `RegionExecutor` counters stay; the factory is the cleaner split and costs a constructor change. SPEC section 8 placed the graphs in the session; the design text records the move. |
| O5 | May `ScopeTrace` gain a public `cut()`? | Yes; the alternative, `Step` as a friend of `ScopeTrace`, hides a second caller of a private member. |
| O6 | May the eager module gain the public header `SushiAI/eager/phase.hpp` with `PhaseCut` and `restrict_to_phase`? | Yes; both are host lane and get their own test file. |
| O7 | Does the first version accept that a parameter only a closed phase reads trains on nothing in that step (zero gradient, momentum still applied)? | Yes, as SPEC section 8 already says; the design text names the cost. |
| O8 | Which design topic holds the eager design? `sa design owner modules/core/eager` answers PROFILING, whose subject is profiling, and no eager topic exists. | Open `EAGER_FRONT_END` with `sa design new` over `modules/core/eager`, after PROFILING's Modules line drops the eager module, and move SPEC section 8's ruling into it. |

## 4. Tasks

Interfaces come before their callers. No task builds or tests. One build and one test run, both
through `sa` and both started by the owner, follow task 9. Each worker syntax-checks the C++ files
it touched, one process, through `compile_commands.json` with `clang -fsyntax-only`, and pastes the
output.

Tasks 2 to 8 start only after the owner rules O1 to O6. Task 1 starts after O3, O7 and O8.

| # | Task | Files | Acceptance criterion | Model, effort |
| --- | --- | --- | --- | --- |
| 1 | Write the design text of section 2 into the topic O8 names, with the phase table entry for phases and the owner's rulings | the topic O8 names; `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md` section 8 | The topic states the readable rule, the `constant_before` frontier with its reason, and the per-phase session; SPEC section 8 links to it in place of its "not supported" paragraph | orchestrator, `opus` medium (design is the orchestrator's to write) |
| 2 | `Graph::finalise(NodeID constant_before = 0)` and the frontier in `propagate_requires_grad` | `modules/core/graph/include/SushiAI/graph/ir.hpp`; `tests/core/graph/test_ir.cpp` | `test_ir.cpp` holds a case where a parameter's result below the frontier ends with `requires_grad` false and a parameter leaf below it keeps true, and the syntax check passes | `haiku` medium |
| 3 | `differentiate(..., constant_before)`, the shortened walk and the root refusal | `modules/core/autograd/include/SushiAI/autograd/backward.hpp`; `modules/core/autograd/source/differentiator.{hpp,cpp}`; `tests/core/autograd/test_backward.cpp` | `test_backward.cpp` holds a two-stretch graph where the gradient of a parameter read only below the frontier is `K_NO_VALUE`, one read on both sides gets only the upper read's gradient, no appended node has an origin below the frontier, and a loss below it is `InvalidDescription` | `haiku` high |
| 4 | `ScopeTrace::cut()` | `modules/core/eager/include/SushiAI/eager/scope_trace.hpp`; `modules/core/eager/source/scope_trace.cpp`; `tests/core/eager/test_scope_trace.cpp` | A test cuts inside an open scope and shows the next segment has the same name, occurrence plus one, and that `seal()` still covers every node once | `haiku` medium |
| 5 | `PhaseCut` and `restrict_to_phase` | new `modules/core/eager/include/SushiAI/eager/phase.hpp`, `modules/core/eager/source/phase.cpp`; `modules/core/eager/CMakeLists.txt`; new `tests/core/eager/test_phase.cpp` and `.md`; `tests/core/eager/CMakeLists.txt`; the umbrella header if the configure check lists eager headers | A test shows a restricted partition gives `K_NO_REGION` to every node outside the cut, keeps the keys inside it, and lists only regions that keep a node | `sonnet` low |
| 6 | `StepWork::phase` and `last_phase`; `RegionExecutor` and `InMemoryStepExecutor` keep one region set per phase and drop the phases above the last one | `modules/core/eager/include/SushiAI/eager/step_executor.hpp`, `region_executor.hpp`, `in_memory_step_executor.hpp`; `modules/core/eager/source/region_executor.cpp`; `tests/core/eager/test_region_executor.cpp` and `.md` | A test runs two phases in one step and shows each phase composes its own graph once, a second identical step composes nothing, and a step with one phase drops the second | `sonnet` medium |
| 7 | `Session` per phase: tables and plans by ordinal, earlier-phase storage in `preset`, the refusal of 2.2, `tensor_of` across phases, trailing work and the step counter in the last phase only | `modules/core/eager/include/SushiAI/eager/session.hpp`; `modules/core/eager/source/session.cpp` | The syntax check passes and every behaviour of 2.4 has a named function; the tests come with task 8, which drives the session through `Step` | `sonnet` medium |
| 8 | `Step` phases: a read before `backward` closes a phase and tracing goes on; `backward` passes the phase's first node; `keep` of a closed phase's value is `Unsupported` | `modules/core/eager/include/SushiAI/eager/step.hpp`; `modules/core/eager/source/step.cpp`; `tests/core/eager/test_session.cpp` and `.md` | `test_session.cpp` holds: a read, more tracing and a second read giving the bytes of the one-phase step; a later node reading an unkept earlier value refused with `Unsupported` naming `keep`; a kept one accepted; a settled two-phase loop allocating nothing after step 1; and the existing "node after `finish`" case still refused | `sonnet` medium |
| 9 | Pages that follow the code (section 5) | `modules/core/eager/README.md`; `modules/core/autograd/README.md`; `modules/core/graph/README.md` | Each line section 5 names is rewritten, and `check_docs_truth.py` output is pasted with no new failure | `haiku` medium |

Review after task 8: one `opus` low dispatch that checks SOLID shape and the humanizer register of
the changed headers and pages as named items, and one `haiku` high dispatch that checks the tasks
against this plan.

## 5. Pages that describe the touched files

| Page | What it needs |
| --- | --- |
| `modules/core/eager/README.md` | Lines 488 to 490: replace "not built" with the phases. "One step, in order" (lines 76 to 93): a step is one or more phases. `step.hpp` notes (lines 473 to 481): the readable rule across phases. `session.hpp` and `region_executor.hpp` notes: per-phase tables and graphs. Public entry points table: `phase.hpp`, `ScopeTrace::cut`, the `StepWork` fields |
| `modules/core/autograd/README.md` | Lines 49 to 54: the `constant_before` parameter, what it makes constant, and the refusal of a loss below it |
| `modules/core/graph/README.md` | Line 189 and the `finalise` description: the frontier, and that a leaf keeps its flag |
| `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md` | Section 8: the ruling moves to the topic; section 11 keeps "differentiation across a mid-step read" as not in this version |
| `docs/architecture/ARCHITECTURE.md` | Line 458 describes the module in one line and stays |
| The design topic of O8 | Written by task 1 |
| Backlog | RW-17 closes with task 1 and the owner's rulings; RW-18 loses its blocker. A new `work` entry for tasks 2 to 9 cites this folder. The orchestrator runs these through `sa backlog` |

## 6. Risks and what the build must confirm

SPEC section 8 says it is "the least verified part of the design. It was derived from reading
`dynamic_graph.hpp`; no prototype exists." What that leaves open, and what has to be shown:

1. **Two runtime graphs over one set of slots.** Phase 1's graph reads a slot phase 0's graph wrote.
   SushiRuntime orders regions inside one graph by key; between two graphs only the host order
   holds. The build must show phase 0's run has finished on the device before phase 1 starts, the
   way `Step::read` already waits before copying to the host. Task 8's bytes test is the check.
2. **`K_NO_REGION` on a computed node.** `nodes_by_region` in `region_boundary.cpp` (line 177) states as
   a precondition that every node that is not a `K_LEAF` has a key. The restricted partition breaks
   that wording. The code skips `K_NO_REGION` before the check, and `compile_region` gives a
   computed import `Lifetime::Input`, which is right for an earlier-phase value; the precondition
   text needs widening, and `RegionTable::plan` and `signature_of` must be shown to take such an
   import's storage from `preset`. Task 7's worker reads both before writing.
3. **An `ACCUMULATE` into an earlier-phase value** would add into a slot another phase owns.
   `buffers_of` accepts it because the value is computed. The design text refuses it in the first
   version, and task 7 adds the refusal.
4. **Renumbering.** A phase's forward keys follow the segment count of the phases before it, so a
   scope that appears in phase 0 re-records every later phase. RW-16's measurement covers it.
5. **Dropout and the step counter.** Task 8 must show a two-phase step and a one-phase step of the
   same model draw the same masks; that holds only if the counter moves once per step.
6. **The `noexcept` surface test** (`tests/layers/umbrella/test_SushiAI/compile_noexcept_surface.cpp`
   lines 247 and 1635) calls `differentiate` and `finalise` with the old arguments; with defaulted
   trailing parameters it compiles unchanged, and the build confirms it.
7. **Training parity.** The first item of SPEC section 10 compares the eager loop with `Trainer`.
   With phases, that comparison holds only for a loop that reads at the end; a mid-step read
   changes what trains (O7), and no test can compare it with the compiled path.
