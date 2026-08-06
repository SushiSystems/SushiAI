# Config-driven training: `sushiai train` and `sushiai inference`

**Status:** design, approved 2026-08-07. Not yet implemented.

Moves a model's architecture out of the C++ type system and into a JSON file,
and adds two driver programs that consume it. The point is not convenience: it
is that `apps/train.cpp` must be written **once** and never edited again as the
library grows convolutions, detection heads or higher-order derivatives.

---

## 1. Scope, and what is deliberately not in it

### In

| Layer | What |
|---|---|
| library | `Config::Value` + JSON parsing (nlohmann-json linked `PRIVATE`), `ModelSpec` / `TrainSpec` / `DataSpec` |
| library | `NN::ILayer`, `NN::LayerStack`, `NN::LayerRegistry` — entries: `linear`, `gelu` |
| library | `Train::IObjective`, `Train::ObjectiveRegistry` — entry: `cross_entropy` |
| library | `Data::DatasetRegistry` — entry: `rings`; `Data::Rings` moved in from the example |
| library | `Optim::OptimizerRegistry` — entries: `adamw`, `sgd` (both already exist) |
| library | `IO::checkpoint_format` (host lane) + `IO::checkpoint` (device lane) |
| library | `nn/module.hpp`: one new detector, one new `if constexpr` arm |
| apps | `apps/train.cpp`, `apps/inference.cpp` — **contain no layer, loss, dataset or optimizer type name** |
| cli | `sushiai train`, `sushiai inference` subcommands |
| example | `examples/heavy_mlp/{model,train,eval}.json` — no `.cpp` files remain |

This is a vertical slice: every registry carries the **minimum** number of
entries, but every **seam** is the general one. Capability grows by adding table
rows, never by editing the driver. The slice was chosen over building the
foundations first in full knowledge of its one risk — a schema designed against
two layer types may need revision at the third — and the mitigation is §4's rule
that layer-specific fields are validated by the layer's own factory, so a schema
change stays local to the layer that caused it.

The work is large enough that the implementation plan will phase it: the
checkpoint format and the runtime layer stack are independent of each other and
of everything else, and both must land before the config layer has anything to
build.

### Out, and what each would actually need

These are named because the driver is being designed *for* them, and a design
that claims to enable something it cannot is worse than one that says where the
line is.

- **CNN.** SushiBLAS has no convolution or pooling kernel — `grep -rn
  "conv2d\|im2col\|winograd" include src` over that repository returns nothing.
  SushiAI has no `CONV2D`, `MAX_POOL2D` or `RESHAPE` IR node. The cheapest route
  is `im2col` + the existing GEMM, but `im2col`/`col2im` are new device kernels
  in **SushiBLAS**, not driver code. `Shape::MAX_RANK` is 6, so rank-4 tensors
  already fit; the container is not the blocker, the op set is.
- **YOLO.** Everything under CNN, plus route/concat, upsample, batched matmul in
  the IR (`src/graph/builder.cpp:83` rejects any operand that is not rank-2,
  although `SushiBLAS`'s `execute_gemm` already derives batch strides for rank >
  2 — see `SushiBLAS/engine/blas/dispatch.hpp:448`), a multi-part detection
  loss, anchor boxes, a COCO-format reader, NMS on the inference side, and mAP.
- **PINN.** Blocked on double-backward. `src/autograd/backward.cpp:362` holds
  fourteen rules and **none of them differentiates a gradient node**: there is no
  rule for `RELU_BACKWARD`, `GELU_BACKWARD`, `SIGMOID_BACKWARD`, `TANH_BACKWARD`
  or `DYNAMIC_SCALE`, so a second `differentiate()` over the transform's own
  output fails at the first activation gradient — loudly, through
  `has_backward_rule()`, not silently. The architecture already permits it:
  `include/SushiAI/graph/ir.hpp:44` argues that an IR-to-IR transform closed over
  its own output is what makes higher-order derivatives fall out without a second
  mechanism. What is missing is four or five table rows whose mathematics is
  elementary. Of the three, this is the nearest.

None of the three is work `apps/train.cpp` can do. The commitment this design
makes is narrower and checkable: **when those ops land, `apps/train.cpp` does not
change.**

---

## 2. The two commands

```
sushiai train      train.json
sushiai inference  model.json  model.ckpt  eval.json
```

`sushiai` is already a second entry point for the developer CLI
(`cli/pyproject.toml`, `[project.scripts]` binds both `sa` and `sushiai` to
`sushiai.cli:app`), so these are two new subcommands on an existing app rather
than a new program. They forward their arguments verbatim to the C++ binary, the
way `cli.py`'s `demo mlp` command already does.

Inference takes its data as a third positional argument. The architecture file
carries no data block — it describes a model — so without it there would be
nothing to run on, and an optional flag that always errors when omitted would be
a lie about optionality.

---

## 3. Architecture

```
library (new)
  include/SushiAI/config/value.hpp            Config::Value — a small typed tree
  include/SushiAI/config/model_config.hpp     ModelSpec, LayerSpec
  include/SushiAI/config/train_config.hpp     TrainSpec, DataSpec
  src/config/*.cpp                            <- nlohmann::json appears ONLY here
  include/SushiAI/nn/ilayer.hpp               runtime layer interface
  include/SushiAI/nn/layer_stack.hpp          holds ILayers, satisfies NN::Module
  include/SushiAI/nn/layer_registry.hpp       "linear" -> factory, constant table
  include/SushiAI/train/objective.hpp         IObjective + ObjectiveRegistry
  include/SushiAI/io/checkpoint_format.hpp    host lane: the file format alone
  include/SushiAI/io/checkpoint.hpp           device lane: module + pool binding

library (existing, changed)
  include/SushiAI/nn/module.hpp               +1 detector, +1 if constexpr arm
  include/SushiAI/data/dataset.hpp            +Data::Rings, +DatasetRegistry
  include/SushiAI/train/trainer.hpp           build_classifier becomes a wrapper

apps (new)
  apps/train.cpp
  apps/inference.cpp

example (shrinks to configuration)
  examples/heavy_mlp/model.json
  examples/heavy_mlp/train.json
  examples/heavy_mlp/eval.json
  examples/heavy_mlp/README.md
  (model.hpp, train.cpp, infer.cpp are deleted)
```

### Why the runtime path costs nothing

`NN::Sequential<Ls...>` decides its layer list at compile time: it holds a
`std::tuple<Ls...>`, its `fields()` is `static constexpr`, and
`for_each_parameter` is a template walk. A JSON file decides the list at run
time. The two cannot be reconciled; a second, runtime-polymorphic path is
required.

In an eager framework that would mean a virtual call per layer per step. Here it
does not. `forward()` runs **once**, at trace time, to emit IR; the graph is then
compiled once and replayed (`docs/ARCHITECTURE.md §1`). A
`std::vector<std::unique_ptr<ILayer>>` therefore costs exactly nothing in the
steady state. The graph-native architecture pays for this feature.

### The seam in `module.hpp`

`for_each_parameter` (`include/SushiAI/nn/module.hpp:332`) dispatches on two
cases today: a `ModuleContainer` recurses through `visit_children`, and a leaf
applies its `fields()` list. Neither can enumerate a runtime list — `fields()` is
static and returns member pointers of one concrete type.

One additive arm fixes it, using the same detector idiom the file already uses
twice:

```cpp
if      constexpr (Detail::HasParameterWalk<M>) module.walk_parameters(prefix, visitor);
else if constexpr (ModuleContainer<M>)          /* unchanged */
else                                            /* unchanged */
```

This is the only change to an existing library header's behaviour, and it is
strictly additive: no type that compiles today takes the new arm.

Everything downstream then works unchanged, because it is all
`template<typename M>` over that one walk — `NN::initialize`
(`include/SushiAI/nn/init.hpp:228`), `Train::build_classifier`
(`include/SushiAI/train/trainer.hpp:220`), `NN::parameter_elements`, and the new
checkpoint I/O.

### `LayerStack` and `Sequential` must name parameters identically

`Sequential` names its children by index including the gaps a parameterless
activation leaves, so a three-layer MLP is `0.weight`, `0.bias`, `2.weight`, ...
(`include/SushiAI/nn/layers.hpp:170`). `LayerStack` applies the same rule.

This is not cosmetic. It means a checkpoint written by a hand-written
`Sequential` model loads into a `LayerStack` built from the equivalent
`model.json`, and the reverse. The equivalence of the compile-time and runtime
paths is then provable by the checkpoint itself rather than by a separate
mechanism — see §7.

---

## 4. Four registries, and why they are constant tables

```
apps/train.cpp  (no type names; never edited again)
   |
   +-- NN::LayerRegistry          "linear", "gelu"     -> later "conv2d", "route"
   +-- Data::DatasetRegistry      "rings"              -> later "mnist", "coco"
   +-- Optim::OptimizerRegistry   "adamw", "sgd"
   +-- Train::ObjectiveRegistry   "cross_entropy"      -> later "yolo", "pinn_residual"
```

Each is a file-scope constant table of `{name, factory function pointer}`, in the
shape of `Ops::ALL` (`include/SushiAI/core/ops.hpp:272`) and the fusion
`CATALOG`.

**Nothing self-registers**, for the reason `docs/ARCHITECTURE.md §4` gives for
the backward-rule table: `sushiai` is a **static** library, so a translation unit
whose only job is to run a registration constructor resolves no symbol and is
dropped by the linker — silently, and only in a downstream consumer's build.
Adding a layer is one table row and one factory function.

`"type"` is the discriminator key in all four — one convention, not four.

An unknown `type` is an error whose message lists the table's contents, so the
diagnostic says what *is* available rather than only what is not.

### Layer-specific fields are validated by the factory, not the parser

The parser produces `LayerSpec { std::string type; Config::Value attributes; }`
and hands it to the factory named by `type`. The factory is the only thing that
knows which keys `linear` accepts, and it rejects any key it does not know.

This is the schema-side expression of "the driver knows no types": if the parser
validated layer fields, adding `conv2d` would mean editing the parser.

---

## 5. `IObjective`: breaking the classifier lock-in

`build_classifier` hardcodes `NN::CrossEntropyLoss`
(`include/SushiAI/train/trainer.hpp:242`), `StepSpec` carries `logits` and
`targets` fields, and `Data::Dataset` demands a pure-virtual `num_classes()`
(`include/SushiAI/data/dataset.hpp:112`). A PINN has no classes; YOLO has no
`[N, C]` logits. Without generalising this seam now, `apps/train.cpp` would be
rewritten at the first model that is not a classifier — which is exactly the
outcome this design exists to prevent.

```cpp
class IObjective
{
    public:
        virtual ~IObjective() = default;

        /** Declares whatever leaves the graph needs beyond the model's input. */
        virtual void declare(const Graph::GraphBuilder&, ObjectiveLeaves&) const = 0;

        /** Turns the model's output into the scalar that gets differentiated. */
        virtual Graph::ValueId loss(const Graph::GraphBuilder&,
                                    Graph::ValueId output,
                                    const ObjectiveLeaves&) const = 0;

        /** An optional per-step metric: accuracy, mAP, an L2 residual, or none. */
        virtual std::optional<Metric> score(const Data::Batch&,
                                            const SushiBLAS::Tensor& output) const = 0;
};
```

Two supporting types, both small and both in `train/objective.hpp`:

- `ObjectiveLeaves` — the graph values an objective declared, by role. For
  `cross_entropy` that is one `[N, C]` target leaf; for a PINN it would be the
  collocation points. It is what `StepSpec`'s `targets` field generalises into.
- `Metric { std::string_view name; double value; }` — one named number per step,
  so the driver can print `accuracy 92.40%` or `residual 3.1e-4` without knowing
  which it is. `std::nullopt` means the objective offers no metric, and the
  driver prints only the loss.

`CrossEntropyObjective` reproduces today's behaviour **operation for operation and
in the same order**, so the existing 286 tests are unaffected and any regression
is measurable rather than diffuse.

`build_classifier` is not deleted. It becomes a thin wrapper over
`build_graph(module, spec, objective, ...)` that supplies
`CrossEntropyObjective`, keeping every existing caller and test source-compatible.

`Dataset::num_classes()` stays for this slice — `rings` and the existing
`SyntheticClassification` both answer it meaningfully, and widening the data
abstraction is a separate change that deserves its own argument. It is recorded
here as known debt rather than quietly carried.

---

## 6. Configuration

### `Config::Value`, and why nlohmann-json stays `PRIVATE`

`Config::Value` is a small typed tree — double, int64, bool, string, array,
object. The parser converts `nlohmann::json` into it inside `src/config/*.cpp`,
and the name `nlohmann` appears in no public header.

Three things follow. The dependency links `PRIVATE`, so
`cmake/SushiAIConfig.cmake.in` needs no `find_dependency` and the out-of-tree
consumer in `tests/package/` never sees it. And replacing the parser later
changes no public header at all. This is the same containment
`SushiRuntime::span` and `SushiBLAS::Core::format` apply to `std::span` and
`std::format`.

A `vcpkg.json` manifest is added — the repository has a vcpkg toolchain hook
(`cmake/Vcpkg.cmake`) but no manifest today.

### `model.json` — architecture only

```json
{
  "sushiai_model": 1,
  "input": { "features": 6 },
  "layers": [
    { "type": "linear", "units": 256 },
    { "type": "gelu" },
    { "type": "linear", "units": 256 },
    { "type": "gelu" },
    { "type": "linear", "units": 128 },
    { "type": "gelu" },
    { "type": "linear", "units": 64 },
    { "type": "gelu" },
    { "type": "linear", "units": 4 }
  ]
}
```

- **`sushiai_model` is first and is a version.** Same reason the checkpoint
  carries one: a format change must be an error, not a misread.
- **`units` is the output width only; the input width is chained.**
  `input.features` seeds the chain and each layer's input is its predecessor's
  output, so a dimension mismatch between adjacent layers is not expressible.
- **An unknown key is an error, not an ignored one.** A typed `"unitss": 256`
  must not train silently against a default. This is the silent-wrong-answer
  class the rest of the codebase is built to refuse.

### `train.json` — references the model

```json
{
  "sushiai_train": 1,
  "model": "model.json",
  "objective": { "type": "cross_entropy" },
  "data":      { "type": "rings", "batch_size": 64, "batches": 64,
                 "classes": 4, "seed": 3735928559 },
  "optimizer": { "type": "adamw", "learning_rate": 3e-3, "weight_decay": 0.01 },
  "epochs": 50,
  "checkpoint": "heavy_mlp.ckpt"
}
```

The architecture is referenced rather than embedded, because inference reads the
same `model.json` and two copies of one architecture is two things that drift
apart silently.

**Relative paths resolve against the config file's own directory**, not the
process working directory, so `sushiai train examples/heavy_mlp/train.json` works
from the repository root.

### `eval.json` — inference data

```json
{
  "sushiai_data": 1,
  "data": { "type": "rings", "batch_size": 512, "batches": 1,
            "classes": 4, "seed": 3735928575 }
}
```

The seed differs from training's, so the evaluation set is genuinely unseen —
the property the current `examples/heavy_mlp/infer.cpp` already gets right and
documents.

---

## 7. The checkpoint format

```
magic     u32       "SACP"
version   u32       1
count     u32       tensor count                     (bounded: <= 4096)
--- count times ---
name_len  u32                                        (bounded: <= 256)
name      u8 * name_len
dtype     u32       SushiAI::DType
rank      u32                                        (bounded: <= MAX_RANK == 6)
dims      i64 * rank                                 (each > 0)
payload   u8 * (product(dims) * dtype_size(dtype))
```

Little-endian, stated rather than assumed. This closes the "No checkpointing, no
serialization" gap `docs/ARCHITECTURE.md §7` lists.

### Split along the host/device lane

- `io/checkpoint_format.hpp` — **host lane**. Reads and writes the byte stream
  over `TensorRecord { std::string name; DType dtype; Shape shape; }`. No engine,
  no SYCL, so it is testable against a `std::stringstream` with no device and
  belongs in the `unit` tier.
- `io/checkpoint.hpp` — **device lane**. The `template<typename M>` save/load that
  walks `for_each_parameter` and binds records to `TensorPool` tensors;
  `integration` tier.

### The current example's bugs, fixed structurally

`examples/heavy_mlp/model.hpp`'s loader has real defects. Each is closed by making
the wrong state unrepresentable rather than by adding a check beside it.

| Defect (current code) | Structural fix |
|---|---|
| `matched` counts records processed, not distinct parameters filled, so a file with a duplicated name and a missing one passes validation — and `infer.cpp` never calls `initialize()`, so the unfilled parameter reads **uninitialised device memory** (`allocate_tensor` does not zero) | `std::vector<bool> filled(targets.size())`; a second write to one index is an error that names the duplicate |
| Only the element count is compared, so `[256, 6]` loads happily into `[6, 256]` | `dtype` **and** `Shape` are recorded and compared |
| `std::vector<float> payload(elements)` sized from a file-declared `u32` — a corrupt file requests 16 GB | No intermediate buffer at all: once the shape check passes, the payload is read straight into the destination tensor. The file-driven allocation disappears by construction |
| `data_as<float>()` assumes FLOAT32 | dtype is recorded and checked; anything but FLOAT32 throws naming the dtype. The format already carries dtype, so widening later is additive |
| No version field | `version`, so a future format change is an error rather than a misread |

---

## 8. Verification

`sa test` runs CTest only (`cli/sushiai/services/project.py:310`); the pytest
suite under `cli/tests/` is executed by CI alone. Everything below is therefore a
CTest case, reachable from `sa test`.

**`unit` (device-free):**

- checkpoint format round trip over a `std::stringstream`
- bad magic, bad version, truncated header, truncated payload
- `name_len`, `rank`, `count` over bound; a zero or negative dim
- config: unknown layer type, unknown key, missing version field, relative-path
  resolution
- `LayerStack` parameter naming against `Sequential`'s for the same architecture

**`integration` (engine required):**

- save a trained model, load into a fresh pool, compare parameters bit for bit
- duplicate name in the file throws and names the duplicate
- a parameter the file does not supply throws and names it
- a shape mismatch throws with both shapes
- **the equivalence test:** a `Sequential<Linear, GELU, Linear>` and a
  `LayerStack` built from the equivalent `model.json` produce identical parameter
  names, and a checkpoint written by either loads into the other. The two paths
  are proved to describe one model by the checkpoint itself.
- two independent app smoke tests, in the shape of
  `Integration_Demo.MlpRunsAndReportsOneCompile`
  (`tests/CMakeLists.txt:107`): `train` over a tiny config asserting
  `compiles: 1` via `PASS_REGULAR_EXPRESSION`, and `inference` over a checkpoint.
  Two separate tests, not a chained round trip.

---

## 9. Consequences to record

- `examples/heavy_mlp/` loses all three `.cpp`/`.hpp` files and becomes three JSON
  files plus a README. `examples/CMakeLists.txt` returns to its two original
  targets.
- `heavy_mlp.ckpt` is added to `.gitignore`; it is currently untracked in the
  repository root.
- `docs/ARCHITECTURE.md` needs: the checkpoint gap struck from §7, a section on
  the runtime layer path, and its stale "153 tests" figure corrected.
- `docs/README.md` and `docs/CHANGELOG.md` gain the two commands.
- `Dataset::num_classes()` remains classification-shaped. Known debt, recorded in
  §5.
