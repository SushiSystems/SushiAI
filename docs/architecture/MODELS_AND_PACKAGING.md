# Model files and packaging

How a model becomes a file read by one driver, and how the library is packaged. Part of the architecture chapter; [`FUSION_AND_PRECISION.md`](FUSION_AND_PRECISION.md) holds §5.2 and §5.3.

## 5.4 A model can be a file, and the driver that reads one is written once

`NN::Sequential` decides its layer list at compile time: it holds a
`std::tuple<Ls...>`, its `fields()` is `static constexpr`, and the parameter
walk over it is a template walk. A JSON file decides that list at run time. The
two cannot be reconciled (one is a type, the other is data), so there is a
second, runtime-polymorphic path: `nn/layer.hpp`'s `Layer` and
`nn/layer_stack.hpp`'s `LayerStack`.

**In this architecture that path costs nothing.** `forward()` runs *once*, at
trace time, to emit IR; the graph is compiled once and replayed (§1). A
`std::vector<std::unique_ptr<Layer>>` therefore adds no per-step work at all.
The same design would be indefensible in an eager framework, where it would
mean a virtual call per layer per step: the graph-native architecture is what
pays for this feature.

### One walk, four cases

`for_each_parameter` (`nn/module.hpp`) runs `walk_state`, which tries four
cases in this order: a type's own `walk_state`; a type that enumerates its
parameters through `walk_parameters`; a container that recurses through
`visit_children`; a leaf that applies its `fields()` list. The
`walk_parameters` arm is what a runtime layer list needs, because
`fields()` is static and returns member pointers of one concrete type, and a
container walk over children that are all `Layer&` would find no parameters
*while reporting no error*.

The arm is strictly additive: `layers.hpp` asserts that neither `Linear` nor
`Sequential` takes it, and everything downstream is a template over that one
walk, so `NN::ModelState::create`, `Train::build_training_graph`, `NN::parameter_elements` and
the checkpoint I/O all consume a `LayerStack` without knowing a second kind of
model exists. A virtual cannot be a template, so the visitor is type-erased
through `NN::ParameterVisitor`, which *refers* to its callable rather than
owning it and therefore allocates nothing.

### Four registries, and who fills them

```
applications/programs/source/train.cpp  (no type names; never edited again)
   |
   +-- NN::NodeRegistry          layer and node rows: NN::register_layers, then
   |                             the vision and language registration
   +-- Data::DatasetRegistry     "rings" from Data::register_datasets; "mnist"
   |                             and the readers from the vision registration
   +-- Train::ObjectiveRegistry  "cross_entropy" and the other train rows, then
   |                             the vision and language rows
   +-- Optim::K_OPTIMIZER_REGISTRY "adamw", "sgd"
```

Three of the four are objects a program fills by calling a `register_*` function of each tier it
links, so a domain adds its rows without editing a lower tier. The optimizers are a file-scope
`constexpr` array of `{name, factory function pointer}`, read with `Config::find_entry`. In all
four `"type"` is the discriminator key. **Nothing self-registers**, for the reason §4 gives for
the backward-rule table: a row joins a registry only when a program calls the function that adds
it.

Two rules make growth local. A **layer's own factory** is the only thing that
knows which keys it accepts, so adding a layer type never edits the parser; and an
unknown `type` or key is an **error listing what is available**, so a typed
`"unitss": 256` cannot train silently against a default.

The layer seam carries a per-sample `Shape`, batch axis excluded.
`Layer::output_shape()` in `modules/layers/nn/include/SushiAI/nn/layer.hpp` is the virtual, and
a `LayerFactory` in `modules/layers/nn/include/SushiAI/nn/layer_registry.hpp` receives the shape the
layer before it reports, so a convolution, a pool and a flatten chain through
`build_layer_stack` with no width restated in the file. `GraphBuilder::conv2d`
in `modules/core/graph/include/SushiAI/graph/builder.hpp` owns the convolution's node sequence, a
reshape of the rank-4 weight, K_IM2COL, K_MATMUL, an optional K_BIAS_ADD and a reshape
back to NHWC, and `NN::Conv2d` in `modules/layers/nn/include/SushiAI/nn/conv2d.hpp` calls it and
does not restate it.

`Train::Objective` is the seam that keeps the classifier from being fixed:
`build_training_graph(module, objective, ...)` takes any `Objective`, and
`build_classifier` is a wrapper that passes it a `CrossEntropyObjective`.

### Checkpoints, split along the host/device lane

`io/checkpoint_format.hpp` is the `SACP` byte stream alone: a magic, a
version (2; version 1 still reads), the model-state records and the run-state
records of an optimizer, each a name, a dtype, a `Shape` and a payload, every scalar
encoded a byte at a time so the little-endian order is stated rather than
inherited. No tensors, no engine, so its whole malformed-file surface is a
device-free unit test. `CheckpointWriter::create` and `CheckpointReader::open`
return a `Result`, and a malformed record is a status. `io/checkpoint.hpp` is
the device half: `save_checkpoint` writes every entry of a `NN::ModelState`,
parameters and buffers alike, after the engine has run its recorded tasks, and
`load_checkpoint` fills a state from a file. `io/import_weights.hpp` holds
`import_weights`, which applies `RenameRule`s to record names before the lookup.
Each returns a `Status`. A parameter the file does not supply, one the model does
not have, a shape mismatch and a duplicated name are each refused by name.

Three properties are structural rather than checked beside the code:

- **The reader allocates nothing from a file-declared size.** `next()` returns
  a header, the caller compares it against a destination it already owns, and
  `read_payload()` copies into that destination.
- **Filled slots, not a record count.** A file with one duplicated name and one
  missing name has the right count, and the unfilled parameter would then read
  uninitialised device memory: `allocate_tensor` does not zero, so a forward
  pass over it produces numbers rather than crashing.
- **A `Shape`, not an element count.** `[256, 6]` cannot load into `[6, 256]`.

`LayerStack` names its parameters exactly as `Sequential` does, gaps included,
because `LayerAdapter` delegates to the same `for_each_parameter` walk. That is
what lets a checkpoint written by either load into the other, and
`Medium_Checkpoint.ASequentialCheckpointLoadsIntoAnEquivalentLayerStackAndBack`
proves the two paths describe one model **by the checkpoint itself**, rather
than through a second mechanism whose own correctness would then be the open
question.

---

## 6. Packaging

SushiAI is `find_package`-able from its very first commit, mirroring
SushiBLAS's SB-6 packaging shape exactly: `SushiAI::sushiai` (a `STATIC`
library, aliased in-tree and exported out-of-tree under the same name),
`install(EXPORT)` / `export(EXPORT)`, a generated `SushiAIConfig.cmake`
(`find_dependency(SushiRuntime)` and `find_dependency(SushiBLAS)`, each
guarded by a TARGET-guard so a consumer that already resolved either
dependency itself doesn't get a duplicate), `SameMinorVersion` package
version compatibility (pre-1.0), and a version-drift check between
`modules/core/version/include/SushiAI/version.hpp` and the CMake project version at configure
time.

`sushiai_apply_build_policy()` (`cmake/SushiAIBuildPolicy.cmake`) composes
warnings/architecture/sanitizer policy targets and pulls their flag *values*
onto the real target via `target_compile_options`/
`target_compile_definitions`/`target_link_options` +
`$<TARGET_PROPERTY:...>`, never `target_link_libraries`: the STATIC-library
export-set-safe pattern SushiBLAS's own SB-6 work established (see
SushiBLAS's `ARCHITECTURE.md §10`): linking a PRIVATE policy target, even
`BUILD_INTERFACE`-wrapped, records its name in the exported
`INTERFACE_LINK_LIBRARIES` and breaks `install(EXPORT)`/`export(EXPORT)`.

`tests/package/` is a real out-of-tree consumer: `find_package(SushiAI)`
against an install prefix, no access to this source tree: that runs a real
op through both transitively-resolved siblings.

It is built at **C++17**, which is the one place this repository deliberately
stops mirroring sushiblas's `tests/package/`. SushiBLAS keeps its consumer at
C++20 as a live assertion that a C++20 consumer still links against a C++17
library, and that assertion holds there because all of its span-taking entry
points are header-inline. SushiAI's are not (§1), so the same assertion here
would be false, and would pass only because `main.cpp` happens to call
nothing that takes a span. The file records the two mangled symbols that
prove it.

`SA_BUILD_TESTS` (default `ON`) guards `enable_testing()` and
`add_subdirectory(tests)` together, matching SushiRuntime's `SR_BUILD_TESTS`
shape exactly. It exists for the other consumption mode: a superproject that
`add_subdirectory`s SushiAI would otherwise be made to compile the suite,
have `gtest_discover_tests` enumerate it, and register 1803 tests it never
asked for into its own CTest run. SushiBLAS guards its suite the same way:
`SB_BUILD_TESTS` defaults to on only when SushiBLAS is the top-level project,
so its tests do not enter this tree. The default is `ON` rather than
SushiRuntime's `OFF` because `sa test` expects a test binary to exist and a
direct `cmake` build that silently produces none is the wart SushiRuntime's
own docs record against that choice.

CMake module files that define SushiAI-specific functions
(`SushiAIBuildPolicy.cmake`, `SushiAICompilerWarnings.cmake`,
`SushiAIOptimization.cmake`, `SushiAISanitizers.cmake`, `SushiAISyclTarget.cmake`)
are prefixed with the program's full name, because D3's sibling
`add_subdirectory` puts every checked-out repo's `cmake/` directory on one
shared `CMAKE_MODULE_PATH`: an unprefixed `SyclTarget.cmake` here would
shadow (or be shadowed by) SushiBLAS's own file the moment both are
`add_subdirectory`'d together, exactly the collision SushiBLAS's own SB-6
work found and fixed. `Vcpkg.cmake` and `Toolchain.cmake` stay unprefixed
because their logic is idempotent regardless of which sibling's copy loads
first.
