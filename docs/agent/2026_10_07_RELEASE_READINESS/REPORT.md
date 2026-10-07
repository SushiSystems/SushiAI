# SushiAI 0.1.0 readiness

**Status:** Complete. Read-only assessment of 2026-10-07; four assessors, each blocker checked by a second agent. Nothing was built or run.

## Answer

No. The owner's condition was "after benchmarks, tests and practical use", and none of the three has happened for this tree: it has never been linked or run as a whole, no benchmark exists, and CI has failed on all 24 recorded runs. A tag also has no base to stand on, because SushiBLAS has no tag, its breaking v0.1.0 redesign has started landing, and SushiAI's roughly 130 facade call sites are not migrated.

## Engineering: not ready

- **No build of this tree.** `build/src/sushiai.lib` and `build/tests/sushiai_tests.exe` are dated 2026-08-09. Today's ninja run (15:17 to 15:29) compiled 42 of the 50 library sources and linked nothing. Since then six sources were edited and eight new `src/nn/*.cpp` files have no object.
- **No test of this tree.** All 41 test objects are from August. About 210 new unit tests in 13 files have never been compiled.
- **The last test run failed.** The newest ctest log (`build/Testing/Temporary/LastTest.log.tmpc010c`) is an interrupted run of the stale August SushiBLAS test binary: 110 of 111 failed with `No kernel named _ZTS...SushiBLAS...Kernels4SYCL...`. The 2026-09-16 log shows the same error on 53 of 54, and no document diagnoses it.
- **CI has never been green.** `gh run list --workflow CI` gives 24 failures and 0 successes. On HEAD d496e2b the functional job dies at "Check out the SushiRuntime sibling" with Not Found: both siblings are private, `gh secret list` is empty, and `ci.yml:94-100` still says they are public. No runner has ever compiled a SushiAI source file.
- **CI lanes are missing.** There is no Windows C++ lane and no `release.yml`.
- **A checker fails.** `check_source_comments.py` exits 1 with 26 findings on today's files.

## Product: not ready

- **What a user gets at HEAD.** An MLP classifier with cross-entropy, SGD or AdamW, fp32 or mixed precision, through the C++ API, `sa demo mlp` and JSON-driven `sa train` / `sa inference`, with SACP checkpoints.
- **What the working tree adds, unrun.** 13 layer rows, an `mnist` dataset row and `examples/mnist_cnn/`, whose README says "No run has been made with these files".
- **No device statement.** No entry point selects a device at run time. Every published number is CPU through OpenCL, and no GPU run is recorded.
- **Documents say things that are false today.**
  - `README.md:37-38` and `ARCHITECTURE.md:389` call lowering the only file that calls the engine. `ARCHITECTURE.md:23-36` already names `init`, `data` and `optim` as exceptions, but `trainer.cpp:259-262`, `compiled_step.cpp:50`, `io/checkpoint.hpp:113-114` and `fusion_selector.cpp:38-39` are acknowledged nowhere.
  - `TUTORIAL.md:6-7` claims every number was run against the tree.
  - Test counts read 690 in the tutorial, 582 in `ci.yml` and 723 macros in the sources.
- **Changelog gaps.** The conv layers, `GraphBuilder::conv2d`, the registry rows, the mnist row and `mnist_cnn` have no changelog entry. The eager and provenance entries exist.
- **Install is source-only in practice.** The installed-package route is documented, but no run of it is recorded.

## SOLID: not ready, on the gate; the shape findings are debt and decisions

The only release-blocking item in this lens is the checker failure: 25 missing licence blocks on untracked nn files and one over-long comment block at `nn/layer_registry.hpp:68`. The rest survived checking only as should-fix or owner decisions:

- **Lowering dispatch (audit C1).** `Lowering::record_node` is a hand-written chain of about 32 to 35 `if` arms over 41 to 42 ops in a 1379-line file. It sits in a `.cpp` under `src/`, so replacing it breaks no public signature.
- **TensorPool (C3).** It is declared in `graph/lowering.hpp:56`, so `nn/init.hpp`, `io/checkpoint.hpp` and `train/trainer.hpp` include the lowering pass to get the pool.
- **Error model (C4).** Exceptions are the model in practice, at about 400 `SA_THROW_IF` sites, against the cpp-code-style rule and with no design record.
- **Naming (C11).** Two conventions coexist, and two files mix them internally (`src/nn/layer_registry.cpp:56/:59`, `src/data/dataset.cpp:47/:53`).
- **Layer shape (C12).** Eight new layers have one header and one source each, while six old types share `nn/layers.hpp`.
- **Scope destructor.** `Scope::~Scope()` calls a `close()` that can throw (`eager/scope.hpp:57-60`, `scope_trace.cpp:84-85`), including `std::bad_alloc` with no misuse.

Three audit findings are closed in the tree: C2, C18, and C9 for committed files.

## Architecture: not ready to freeze

- **Internal layering holds.** The ten-module include graph has no cycle and no upward include, and the host/device split holds at 43 and 17 headers, with all of today's headers on the host side.
- **Nothing enforces it.** `check_layering.py` collects zero files (`K_TIER_ORDER` is empty and there is no `modules/` folder), and CI runs it as a passing step.
- **The SushiBLAS seam is moving.** `SushiBLAS::Engine&` appears in 15 public headers. `compiled_step.cpp:50` calls `engine.get_graph().compile()`, which the redesign removes; that changes who owns the compiled plan.
- **Nothing is pinned.** `cmake/BLAS.cmake` and `cmake/Runtime.cmake` take whatever is checked out, and CI takes the default branch.
- **The eager design depends on uncommitted work.** `Graph::add_emitter` exists only as an uncommitted change in the sushiruntime working tree.
- **Eager is half a feature.** Seven eager headers are in the umbrella and the install set, with unit tests but no executor and no production consumer.
- **The backlog omits the open work.** `REMAINING_WORK.md` has no entry for the migration, the eager executor or the CV items that today's specs say are listed there.

## What must close before 0.1.0

1. **Build and run the tree.**
   - Clears when: `sa build` completes on the committed tree, and `sa test --suite all` and `--suite package` pass with output pasted.
   - Waiting on: the owner's build.
2. **Explain the `No kernel named` failure.**
   - Clears when: the cause is named (stale binary, device selection or kernel bundle) and both suites pass on a stated device after a clean rebuild.
   - Waiting on: the owner's build, then diagnosis.
3. **Give the release a SushiBLAS to stand on.**
   - Clears when: either the migration lands against a tagged SushiBLAS v0.1.0, or SushiAI is tagged against a named, tagged pre-redesign SushiBLAS commit. In both cases CMake and CI pin exact tags for both siblings.
   - Waiting on: an owner decision between the two routes; the first route also waits on the SushiBLAS migration. SushiRuntime v0.4.0 must be tagged first if `add_emitter` is needed.
4. **Make CI able to compile.**
   - Clears when: a token that reads both private siblings is set (or the siblings are made public), the stale comment at `ci.yml:94-100` is fixed, and the functional job is green on the release commit.
   - Waiting on: an owner decision on the secret, then work that can be done now.
5. **Decide the Windows C++ lane and add `release.yml`.**
   - Clears when: the functional job runs on both platforms, or the owner records that a pasted local Windows run stands in for it; and a tag-triggered workflow exists.
   - Waiting on: an owner decision, then work that can be done now.
6. **Clear the checker.**
   - Clears when: `check_source_comments.py` exits 0 after licence blocks are added to the 25 files and one comment line is cut.
   - Waiting on: work that can be done now, once the nn agent has finished.
7. **Cover or hold back the convolutional work.**
   - Clears when: an integration test trains a conv model for a few steps on a device (forward, gradcheck, one compile), a smoke test runs `examples/mnist_cnn`, and changelog entries exist for the layers, `conv2d`, the registry rows, the mnist row and the example. Otherwise these stay out of 0.1.0.
   - Waiting on: the tests and entries can be written now; passing them waits on the owner's build.
8. **State the devices.**
   - Clears when: the release notes name each device the suite passed on, with one measured training run per device, and a documented way to pick the device.
   - Waiting on: the owner's build. The `0xC0000006` Windows/CUDA fault recorded in SushiBLAS's `SUSHIAI_CONSTRAINTS.md:47-51` is the risk to test first on the 3080 Ti.
9. **Correct the documents.**
   - Clears when: the "only file" sentence is fixed in README and ARCHITECTURE; the `TUTORIAL.md:6-7` claim and the three test counts are retaken; measurements carry a device and a date; ARCHITECTURE covers eager, provenance, `ILayer::output_shape` and the conv layers; and `KNOWN_ISSUES.md` lines 38 to 41 are rewritten.
   - Waiting on: the sentences can be fixed now; the numbers wait on the owner's build.
10. **Cut the release.**
    - Clears when: `## Unreleased` is renamed, the commit is `chore(release): v0.1.0`, and an annotated tag sits on it.
    - Waiting on: everything above.

Three decisions do not block under the release skill but cost a breaking 0.2.0 if taken after the tag:

- Whether exceptions are the public error contract (C4).
- Whether the seven eager headers stay in the umbrella and install set, or are marked experimental.
- Which naming form the stack uses (C11).

## What can ship as known debt

| Item | Where it is recorded or should be |
| --- | --- |
| No benchmark target; README fusion figure uses three replicates against the skill's five; `ARCHITECTURE.md:446` cites a microbenchmark with no source in the tree. | `KNOWN_ISSUES.md`. This ships as debt only if the owner drops his own "after benchmarks" condition or the README stops quoting numbers. |
| Device tests carry no device label, cannot report skipped, and the test log names no device. | `KNOWN_ISSUES.md` and `REMAINING_WORK.md`. |
| Lowering if-chain (C1) and TensorPool header (C3). | `REMAINING_WORK.md`, as part of the migration pass; it currently only points at the audit report. |
| Layer registry concentration, five copies of `as_span`, four registries in two file shapes (C8). | `REMAINING_WORK.md`. |
| Old layers sharing `nn/layers.hpp`, `LayerAdapter` in `ilayer.hpp` (C12). | `REMAINING_WORK.md`; cheaper before the tag because include paths become contract. |
| `Scope` destructor can throw. | `KNOWN_ISSUES.md`; fix with a misuse test. |
| Sibling includes in 29 public headers with no decision record in `docs/design` (C6). | `docs/design`; `INTEGRATION.md:9-34` already describes the dependency. |
| `BATCHNORM_UPDATE_STATS` declared with no emitter, while `ARCHITECTURE.md:293-297` says in the present tense that it is recorded into the step (C19). | `REMAINING_WORK.md`, plus a correction of that wording. |
| Layering checker checks nothing (C14). | `REMAINING_WORK.md:89` already has "Declare the tiers"; discovery also needs changing. |
| `pyflakes ... \|\| true` at `ci.yml:166`. | `KNOWN_ISSUES.md:39`, with the line number corrected from :184. |
| Doxyfile says "v1 (unreleased)"; the CLI version is tied to nothing. | `KNOWN_ISSUES.md:42` and `REMAINING_WORK.md:98`, both already present. |
| `LICENSE` and `NOTICE.md` are not installed; an installed tree lacks the `Required Notice:` line. | `KNOWN_ISSUES.md`; one install rule fixes it. |
| No logging in the library; sibling INFO logs cannot be turned down (C5). | `KNOWN_ISSUES.md`. |
| Narrow feature surface: one loss, two optimizers, no Dropout, LayerNorm, scheduler, shuffle, Python binding or ONNX. | README first screen and the release notes. |
| Long functions in `graph/`, CLI shape (C13, C17), 17 `SA_THROW_IF(true` sites, quoted includes in the umbrella. | `REMAINING_WORK.md`. |
| No `.gitattributes`; the runtime DLL is copied beside the test binary. | `REMAINING_WORK.md`. |
| The design map lists only `ROADMAP.md`. | `docs/design/README.md` and `REMAINING_WORK.md`. |

## What is solid

- The module include graph is acyclic with no upward include across ten modules, and today's work added no upward edge.
- The host/device split holds by transitive include closure; every header added today is host lane.
- Five regression tests pin determinism bit for bit (`test_determinism.cpp:163-283`); seeds are fixed, and no wall clock or unseeded randomness appears in `src`, `include` or `tests`.
- Silent-failure guarantees have their own regression tests: compile-once, static memory, argmax lifetime, log-softmax stability, fused optimizer, fusion numerics.
- Three of four checkers exit 0 on the working tree. The checkers job passes on Ubuntu and Windows, and the CLI job passes.
- The version is checked at configure time: `CMakeLists.txt:78-87` compares `version.hpp` with `project()`.
- Install and export rules, a package config with `find_dependency`, and an out-of-tree consumer under `tests/package` exist.
- Error reporting uses one exception type, with no swallowing catch found.
- The eager units take their dependencies as arguments, depend only on `graph`, `autograd` and `core`, and can be withdrawn without touching another module.
- The eight new layers have one shape, and `GraphBuilder::conv2d` composes existing emitters without adding an op.
- Audit findings C2 and C18 are fixed in the tree.
- The documents report their own unfavourable results: fusion makes the default demo slower, mixed precision is 4.5x slower on that device, and `mnist_cnn` was never run.

## What a 0.1.0 could honestly promise

The soonest tag is the pre-redesign route: SushiAI 0.1.0 against a tagged pre-redesign SushiBLAS commit, with the migration following as 0.2.0. It still needs items 1, 2, 4, 6, 9 and 10, plus the pin from item 3. The owner would also have to tag a SushiBLAS commit that predates the 30 unpushed redesign commits.

Scope that the evidence could support once the build is green:

- A C++17 static library, built from source beside its two siblings with the SushiStack compiler.
- MLP classification with cross-entropy, SGD and AdamW, in fp32 and mixed fp16/fp32.
- The C++ API, `sa demo mlp`, and JSON-driven `sa train` / `sa inference` with the `rings` dataset.
- SACP checkpoints.
- Run-to-run determinism on the one device the suite passed on.

The release notes would have to say that it does not:

- Include the convolutional layers and MNIST, unless item 7 passes; otherwise they are held back or marked untested.
- Include an eager front end. The seven headers are either left out or marked experimental with no executor.
- Offer a stable API. 0.2.0 follows with the SushiBLAS migration, and `SameMinorVersion` rejects 0.1 consumers.
- Run on any device other than those named. There is no GPU claim unless the suite passed on the 3080 Ti.
- Carry reproducible benchmarks.
- Ship binary packages, Python bindings or ONNX.
- Provide regression losses, Dropout, normalisation layers, schedulers or data shuffling.
- Have CI on Windows for C++, if the owner accepts a local run in its place.

Next release: v0.1.0 (first tag), about 80 entries under Unreleased. It is due by the 15-entry trigger and not proposable while CI is red.

## Not checked

- Nothing was built, compiled or run by any assessor or by me. I ran no command for this statement; it rests on the four reports and their skeptic checks. I wrote no file and no scratch, so rules 6 and 7 had nothing to check. The humanizer pass was applied to this text.
- Whether the 15:17 build stopped on an error or was interrupted, and who started it.
- The cause of the `No kernel named` failures, and which device a default `RuntimeContext` selects now.
- Whether the committed HEAD alone passes the source-comment checker.
- Correctness of today's layer, eager and provenance code, and the content of the new tests.
- The exact facade count. Counts ranged from 123 to 134 depending on the pattern; the SushiBLAS design says 131 in `src`.
- Whether SushiBLAS HEAD has already removed the facades in its 30 unpushed commits.
- GPU behaviour of any SushiAI test, and CPU/GPU numeric agreement.
- `install.sh`, `sushicore` 0.7.0 on PyPI, and `sa test --suite package` on any recent tree.
- Estate audit findings other than those named above.
- The YOLO26 spec beyond its first table, and section 9.4 of the BLAS design (`sushiblas/docs/agent/2026_10_07_V010/SPEC.md`, heading `DESIGN_BLAS`).
- The statement that the owner calls `ROADMAP.md` obsolete: the tree does not say so.
- The nn files and READMEs were being edited while they were read, so the checker counts and test counts can have moved.