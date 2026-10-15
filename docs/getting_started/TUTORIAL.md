# SushiAI tutorial

A walk from an empty checkout to a model that trains, and then to understanding
what the library actually did with it.

Every command in this file was run against the tree at the time of writing, and
every number quoted is from that output. Where a number was not measured, it
says so rather than guessing. If a command here does not do what this file says
it does, the file is wrong: please treat it as a bug.

> **Conventions.** `sa` is SushiAI's CLI; never invoke `cmake`, `ninja` or
> `ctest` directly ([CONTRIBUTING.md](../CONTRIBUTING.md)). Code samples use the
> short namespace aliases `sa::` and `sr::`, which `<SushiAI/SushiAI.hpp>` brings
> in, and `sb::`, which a program declares itself: `namespace sb = SushiBLAS;`.

---

## 1. What this library is, in one paragraph

SushiAI traces your model **once** into its own graph IR, differentiates that
IR into more IR, fuses it, plans every buffer's address, lowers the whole thing
to a SushiRuntime `TaskGraph` **once**, compiles it **once**, and then replays
it per batch. There is no tape rebuilt per step and no kernel launched from
Python. `compile_count() == 1` is not a goal, it is a guarantee with a test on
it, and every design decision in the repo defers to it.

That single fact explains most of what looks unusual here: why a learning rate
lives in device memory instead of a C++ variable, why fusion is a lookup in a
pre-compiled catalog rather than a code generator, and why the gradient of a
loss is an IR rewrite rather than a callback.

---

## 2. Getting it to build

You need the SushiStack toolchain and two sibling checkouts:

```
<workspace>/
  sushiruntime/     # the backbone
  sushiblas/        # the tensor + BLAS engine
  sushiai/          # you are here
```

```bash
hub install-cli sushiai   # puts `sa` and `sushiai` on your PATH
sa setup                 # provisions the toolchain (compiler, cmake, ninja, doxygen)
sa config                # shows what the CLI resolved, and where each value came from
```

`hub install` provisions several checkouts at once; `sa setup` covers this one.

If a sibling cannot be found, `sa build` fails immediately and names which one
and where it looked: it does not let CMake's `add_subdirectory` error surface
instead. `sa config` is the first thing to run when that happens.

```bash
sa build --type release
sa test --label all
```

On 2026-10-09 the suite held **1803 cases in 121 suites, all passing**. Every
one is SushiAI's own: SushiBLAS is still compiled from source via
`add_subdirectory`, but it builds its tests only when it is the top-level
project (`SB_BUILD_TESTS`), so none of them enter this tree.

`sa test` starts one process per case through CTest. Each module builds one test
program, `sushiai_<tier>_<module>_tests`, and `sa run` runs one of them in one process.

> **A warning about `--clean`.** It exists, and on most machines you should not
> reach for it. Because SushiBLAS is compiled from source here, a clean build of
> SushiAI is a clean build of two projects.

---

## 3. Your first run

```bash
sa demo mlp --epochs 1 --batches 4
```

```
Dataset  : MNIST not found (...); falling back to the seeded synthetic problem.
Schedule : 1 epochs x 4 batches = 4 steps, seed 942917386647760909
  epoch   1/1   loss 1.791759 -> 1.133352   accuracy 100.00%
compiles       : 1
steps          : 4
ops per step   : 25
loss           : 1.791759 -> 1.133352
final accuracy : 100.00%
```

Read those lines carefully, because each one is a claim the library is making
about itself:

| Line | What it is actually asserting |
|---|---|
| `compiles : 1` | One lowering, one compile, four replays. If this is ever 2, something re-entered the compiler between steps and the whole design premise is broken. |
| `ops per step : 25` | How many SushiBLAS calls the compiled plan consists of: also its node count. This is the number fusion moves. |
| `loss ... -> ...` | The first and last batch loss. `1.791759` is `ln 6`, which is exactly the cross-entropy of a uniform guess over 6 classes, so an untrained model starting anywhere else means the initialisation or the loss is wrong. |
| `accuracy` | Costs one host read of the logits per step, which is why it is optional. |

Drop MNIST's IDX files into `data/mnist/` and the same command trains on
784 → 10 instead, with no other change.

---

## 4. Your first program

The smallest complete training program. This is `applications/examples/source/example.cpp`
condensed; that file compiles and runs, so start by reading it.

```cpp
#include <SushiAI/SushiAI.hpp>
#include <SushiBLAS/SushiBLAS.hpp>

SA_TRY_ASSIGN(sr::Runtime runtime, sa::Detail::create_runtime());
sb::Engine engine(runtime);

// 1. A model. An aggregate list, not a registration ritual.
sa::NN::Sequential model{
    sa::NN::Linear{features, 128},
    sa::NN::ReLU{},
    sa::NN::Linear{128, classes},
};

// 2. Trace forward, add cross-entropy, differentiate, insert casts, fuse.
//    All IR-to-IR. No device work has happened yet.
SA_TRY_ASSIGN(sa::Train::TrainingGraph built,
              sa::Train::build_classifier(model, batch, features, classes));

// 3. Memory. The state owns the tensors; bind_parameters plans every offset and
//    binds the state's storage into the pool.
SA_TRY_ASSIGN(sa::NN::ModelState state, sa::NN::ModelState::create(engine, model));
sa::Graph::TensorPool pool(engine);
SA_TRY_ASSIGN(std::vector<sa::Optim::ParameterSlot> parameters,
              sa::Train::bind_parameters(built, state, pool));
SA_TRY(sa::NN::initialize(engine, state));

// 4. An optimizer over those exact parameters.
sa::Optim::AdamWOptions adam;
adam.learning_rate = 3e-3;
SA_TRY_ASSIGN(std::unique_ptr<sa::Optim::AdamW> optimizer,
              sa::Optim::AdamW::make(engine, std::move(parameters), adam));

// 5. The objective build_classifier traced; it feeds and measures every step.
const sa::Train::CrossEntropyObjective objective(batch, classes, sa::DType::Float32);

// 6. Trainer::make lowers and compiles. That is the only compile.
sa::Train::TrainerOptions options;
options.epochs = 8;
SA_TRY_ASSIGN(std::unique_ptr<sa::Train::Trainer> trainer,
              sa::Train::Trainer::make(engine, built, pool, objective, *optimizer, options));

SA_TRY_ASSIGN(const sa::Train::RunReport report, trainer->fit(*dataset));
std::cout << report.to_string();
```
Source: not compiled

The statements sit in a function that returns `sa::Status`, as `run()` does in
`applications/examples/source/example.cpp`. From step 3 on, every call allocates, compiles or runs,
so it returns a `Status` or a `Result<T>` (`core/status.hpp`,
`core/result.hpp`). `SA_TRY` returns a failed status to the caller, and
`SA_TRY_ASSIGN` does the same or binds the value. `AdamW`, `SGD`, `Trainer` and
the datasets are built by a static `make`, and `CompiledStep` by
`CompiledStep::compile`; each returns a `Result`. Steps 1 and 2 touch no device.

Three things worth pausing on:

**Step 2 does no device work.** `build_classifier` returns a `TrainingGraph` in
a `Result`. It holds IR, a `StepSpec` naming the four values the loop cares
about, the parameter bindings, and reports from the cast and fusion passes. You can inspect
and assert on all of it before a single kernel exists.

**`Trainer::make` is where compilation happens**, not `fit()`. A graph that
cannot be lowered therefore fails there, with a status that names the
operation, and not three steps into training.

**`ModelState::create` comes after the trace.** The walk yields only the
parameters a trace declared, so a model no trace has reached gives an empty
state. `bind_parameters` takes that state, so the pool's leaves and `initialize`
share one set of tensors.

---

## 5. What actually happened to your model

The pipeline, in order. Each stage is a separate header you can read on its own:

```
  your forward()          →  graph/builder.hpp     builds IR
  differentiate()         →  autograd/backward.hpp IR → IR
  insert_casts()          →  graph/cast_insertion.hpp  IR → IR
  fuse()                  →  graph/fusion.hpp      IR → IR
  MemoryPlan::build()     →  graph/memory_plan.hpp annotates IR
  Lowering::record()      →  graph/lowering.hpp    IR → SushiBLAS calls
  CompiledStep::compile() →  graph/compiled_step.hpp  once
  CompiledStep::run()     →  per batch
```

**How a node becomes SushiBLAS calls.** `Lowering::record` walks the nodes in
order and looks each operation up in `K_LOWERINGS` (`graph/lowering/all.hpp`),
the concatenation of eight tables: elementwise, linalg, activation, loss,
shape, spatial, norm and fused, each in its own header under `modules/core/graph/include/SushiAI/graph/lowering/`
and its own file under `modules/core/graph/source/lowering/`. The row receives a
`LoweringContext` for that one node, which holds the step's
`SushiBLAS::TaskRecorder` and the `TensorPool` (`graph/tensor_pool.hpp`); every
SushiBLAS call a row makes goes through a view built from that recorder. A
`static_assert` in `all.hpp` requires `K_LOWERINGS` and `K_UNLOWERED` together
to cover every operation. `CompiledStep` owns the `TaskGraph` and the
recorder.

**Why differentiation is IR→IR and not "emit backward kernels".** Four reasons,
argued in full at the top of `graph/ir.hpp`, but the shortest one is decisive: a
`TaskGraph` node carries byte ranges, and byte ranges cannot be known until the
memory planner has run, and the planner needs the *whole* forward-and-backward
graph to compute liveness over. Emitting device work during differentiation
would force the offsets to be decided before the information needed to decide
them exists.

**Why the order of the two IR→IR passes is fixed.** Casts go in after
differentiation (a backward node needs casting as much as a forward one, and
there are none to cast before the transform has run) and before fusion (a
conversion dropped into the middle of a `K_MATMUL → K_BIAS_ADD → K_RELU` chain would
break the chain). `graph/cast_insertion.hpp` argues both.

---

## 6. Writing your own layer

A layer is an aggregate struct with two members: a `static constexpr fields()`
listing its parameters, and a `forward()`.

```cpp
struct MyOffset
{
    int64_t width = 0;
    sa::NN::Parameter offset{};

    static constexpr auto fields()
    {
        return std::tuple{ sa::NN::field("offset", &MyOffset::offset) };
    }

    sa::Graph::ValueID forward(const sa::Graph::GraphBuilder& builder,
                               sa::Graph::ValueID x)
    {
        const sa::Graph::ValueID b = offset.declare(
            builder, sa::Shape{width}, sa::NN::ParameterRole::Bias,
            width, width, "offset");
        return builder.bias_add(x, b);
    }
};
```
Source: not compiled

`bias_add` and not `mul`, deliberately: `mul` goes through `require_same`, which
compares **shape and dtype**, so multiplying an `[N, width]` activation by a
`[width]` parameter throws rather than broadcasting. Broadcasting a vector over
rows is what `bias_add` is; there is no implicit broadcast anywhere in this
builder, and that is on purpose.

`ParameterRole` decides initialisation: `Weight` gets a fan-scaled random draw,
`Bias` always starts at zero. `Parameter::declare` is idempotent, so tracing a
model twice does not accumulate duplicates.

`fields()` is what makes a parameter discoverable: the generic walk
`for_each_parameter` is the same one `build_classifier` uses to decide what to
differentiate with respect to, and the walk `ModelState::create` runs. Two failure
modes, and they are not symmetric:

- **Listed but not wired into `forward()`**: caught, by name.
  `bind_parameters` returns `InvalidDescription`: `Parameter 'x' receives no
  gradient. It is not on any path from the loss, so no optimizer could ever
  move it` (`modules/layers/train/source/trainer.cpp:101`).
- **Wired into `forward()` but missing from `fields()`**: **not** caught. It is
  declared in the graph, gets no gradient, gets no optimizer slot, and silently
  never trains. No call reports it.

So the list is the thing to get right. Its virtue is not that omissions are
detected; it is that there is exactly one list, in the struct, next to the
member it names.

> The layer above is not just quoted here: it is compiled. `Small_Module`'s
> `TheTutorialsCustomLayerTracesAndIsWalkable` and
> `TheTutorialIsRightThatMulWouldNotBroadcast` hold the same code and assert
> both claims this section makes about it, so this sample cannot rot into
> something that does not build. Edit one, edit the other.

If your layer needs an operation the builder does not have, you are adding an
op, not a layer; see §10.

---

## 7. Fusion: what folds, and what refuses to

Fusion here is a **selector over a pre-compiled catalog**, not a code generator.
SYCL's runtime kernel-fusion extension was removed from intel-llvm, and every
distinct functor composition is a distinct template instantiation, so the
composition has to be chosen at C++ compile time. A runtime pass can only pick
among combinations that already exist in the binary.

Watch it work:

```bash
sa demo mlp --epochs 1 --batches 4 --activation gelu                        # 25 ops/step
sa demo mlp --epochs 1 --batches 4 --activation gelu --no-fuse gemm+bias+gelu # 27
sa demo mlp --epochs 1 --batches 4 --activation gelu --fusion off             # 35
```

All three print **`loss 1.791759 -> 1.124512`**. That identity across three
different kernel counts is the whole contract: *fusion moves no bits.* If those
three numbers ever diverge, a fused kernel is computing something its unfused
spelling does not, and that is a defect regardless of which one looks better.

### The trap that makes this interesting

A fused GEMM writes only its final result: the pre-activation is destroyed.
That is fine for ReLU, sigmoid and tanh, whose gradients are computable from the
**output**. ReLU is the clean case: `relu_backward` is `dy*(primal>0)` and
`relu(x)>0` exactly when `x>0`, so reading the output instead of the input is a
substitution, not an approximation: the two are bit-identical.

GELU is not. `gelu_backward` differentiates at its **input**. Fold it naively
and nothing throws, nothing goes `NaN`, the loss still falls, and the model
trains towards the wrong place.

So `K_GEMM_BIAS_GELU` spills the pre-activation to a second buffer on the way past
(cuBLASLt calls this `GELU_AUX_BIAS`), and the fusion report says so:

```
fused    gemm+bias+gelu -> value 6 (+ pre-activation spilled to value 5)
```

And the rule is enforced **at compile time**, not by review. Adding a catalog
entry that folds an input-gradient activation without a spill, or that spills
the wrong step of the chain: fails the build:

```
modules/core/graph/source/fusion.cpp:27: error: static assertion failed ...
A catalog entry folds an activation whose gradient differentiates at its input,
without spilling that tensor to an auxiliary output - or it spills the wrong
step of the chain. ...
```

Both of those failures were confirmed by deliberately introducing them. If you
add a catalog entry, expect to meet this assertion, and **do not delete it**:
relax it to distinguish your case, the way the GELU row did.

---

## 8. Mixed precision

```bash
sa demo mlp --epochs 1 --batches 4 --precision mixed
```

fp16 for the matrix products and the epilogues folded into them; **fp32 master
weights, fp32 gradients, fp32 reductions, fp32 optimizer update.** The demo
reports what that cost:

```
Precision: mixed; 10 cast(s) - 6 narrowing, 4 widening back to the master type
ops per step   : 35
loss           : 1.791759 -> 1.133352
```

35 against fp32's 25: the ten casts, exactly.

**Master weights are the reason this converges at all.** fp16's spacing at 1.0
is 9.8e-4, so `1.0 + 1e-4` is `1.0` in half precision: a weight update smaller
than that is lost entirely, and the weight stops moving while the loss still
falls. The parameters therefore stay fp32 for the whole run, and the fp16 tensor
a GEMM reads is a *derived* activation-lifetime copy, thrown away with the rest
of the step.

Which operations narrow is a policy (`graph/precision.hpp`), injected as an
interface, and each rule is argued there rather than tabulated. One is worth
repeating because it is forced rather than chosen: **`K_BIAS_ADD` and the foldable
activations must narrow with the GEMM.** A fused epilogue runs inside the GEMM's
kernel in the GEMM's element type, so if the IR said the bias computes in fp32,
fusing it into an fp16 GEMM would change the numbers, and fusion promises it
never does.

> **On this machine `--precision mixed` is a correctness switch, not a speed
> one.** There is no fp16 matrix unit here, so half arithmetic is emulated and
> the mixed run is *slower*. What the flag is for is running both arms from one
> binary and comparing loss curves. Any speed claim belongs to hardware with
> tensor cores, and none is made here.

GELU and mixed precision do not combine, and that is deliberate: GELU stays
fp32, so under a narrowing policy a `K_CAST` lands in the middle of the chain and
no catalog entry matches. You can see it:

```bash
sa demo mlp --epochs 1 --batches 4 --activation gelu --precision mixed
#  fused    gemm+bias -> value 5
#  fused    gemm+bias -> value 29        <- never gemm+bias+gelu
#  ops per step : 40
```

The reason is arithmetic: GELU's tanh form computes `x*x*x`, which overflows
fp16 at |x| ≈ 40.3, and loss scaling cannot rescue that, because it is a
forward activation and the scale applies to the backward seed.

---

## 9. Loss scaling

```bash
sa demo mlp --epochs 1 --batches 4 --precision mixed --loss-scale 65536
```

```
Scaling  : gradients scaled by 65536, halved on overflow, doubled every 2000 clean steps
ops per step   : 41
loss           : 1.791759 -> 1.346261
loss scale     : 32768 (steps overflowed: 1, backoffs: 1, growths: 0)
```

The backward seed is multiplied by a device-resident power of two, every
gradient is screened for infinities and NaNs *inside the same compiled plan*,
and an overflowed parameter's update is skipped by a branch in the optimizer
kernel.

**`S` must be a power of two**, and the CLI enforces it. Scaling by a power of
two is exact in binary floating point, so the unscale before the update recovers
the gradient with no rounding at all. You can see that exactness directly: run
the *fp32* arm with a scale and the loss is unchanged to the last printed digit:

```bash
sa demo mlp --epochs 1 --batches 4 --loss-scale 65536
#  ops per step : 31
#  loss         : 1.791759 -> 1.133352      <- identical to the unscaled run
#  loss scale   : 65536 (steps overflowed: 0, backoffs: 0, growths: 0)
```

Two honest notes:

- **This demo does not need loss scaling.** fp16 here neither stalls nor
  diverges without it; both arms converge to the same loss. The feature is
  justified by theory and by deeper models, not by anything measurable in this
  repo. The honest way to *see* it work is `--loss-scale 16777216`, which
  overflows fp16 on purpose.
- **The skip is currently per parameter, not per step.** A parameter whose own
  gradient stayed finite still updates on an overflowed step, where PyTorch
  would take no step at all. Nothing unsafe happens: no infinity or NaN reaches
  a parameter or a moment, and that is asserted bit-for-bit, but it differs,
  which is why the report says "overflowed" rather than "skipped".

### Why the scale lives in device memory

Because a per-step value baked into an IR node's `Attributes` becomes a
compile-time constant in the recorded task, and changing it would force a
recompile: destroying `compile_count() == 1`. So it is an operand, not an
attribute. `optim/step_scalars.hpp` is the general mechanism; the same one
carries AdamW's learning rate and bias corrections.

---

## 10. When something goes wrong

Most failures here are loud: SushiBLAS refuses a mixed-dtype GEMM by name, the
IR refuses a double-written value, `build_classifier` refuses a model whose
output shape does not match the loss. Those will tell you what to fix.

The ones worth knowing about are the quiet ones. Each of these has cost this
project real debugging time:

| Symptom | Cause |
|---|---|
| The loss falls, the model is worse than it should be | A fused activation whose gradient reads its input, folded without a spill (§7). |
| Two kernels that should be ordered are not | An operand declared from a raw pointer instead of `Tensor::region()`. A region converts implicitly from a bare pointer, and a bare pointer widens to the whole allocation, so two views of one arena become unrelated resources and **no edge is emitted**. |
| `compiles` is greater than 1 | A per-step value went into `Attributes`. |
| Accuracy is 100% and obviously false | You are reading a buffer as the wrong element type. This is now refused by name, but the shape of the bug recurs. |
| Weights stop moving while the loss still falls | fp16 weights without an fp32 master (§8). |

The unifying lesson: **a wrong answer with nothing to report it is the expensive
kind of bug**, and most of the assertions and `static_assert`s in this codebase
exist to convert one of those into a build failure, a refused status or a throw. When you find one
of them in your way, it is usually right.

---

## 11. Testing your change

```bash
sa test --label small         # one process, no device
sa test --label medium        # a CPU SYCL device, threads or files
sa test                       # both, as CI runs them
sa test --label all
sa test --filter 'Small_Precision.*'
sa run sushiai_core_autograd_tests -- --gtest_filter='Small_Precision.*'   # in one process
```

The sizes come from the GTest suite-name prefix and nothing else. Some tests hold
guarantees whose failure would otherwise be invisible: `compiles == 1`, byte-for-byte fused/unfused agreement,
the numerical stability of log-softmax.

The bar this project holds itself to, learned the hard way:

> **A test that would also pass without the change is not a test.**

Concretely, when you fix something, mutate your own fix and confirm the test
fails. Several times in this repo a first attempt at a test turned out to pass
against the broken code: an ordering test whose operands both sat at offset 0,
a fusion test that compared forward values where only gradients could tell the
difference. If your test cannot distinguish the two states, it is documentation,
not verification.

For gradient rules specifically, `autograd/gradcheck.hpp` is the strongest
oracle available: fp64 central differences against the analytic rule. It
requires `FLOAT64` for a stated reason and should not be weakened to accommodate
lower precision: a gradient rule is a statement about the mathematics, so
verifying it once in fp64 verifies it everywhere.

---

## 12. Where to go next

```bash
sa docs        # the generated API reference, into build/docs/api-site/html/
```

The prose docs, in the order they repay reading:

- [README.md](../../README.md): what the library is, and the measured numbers.
- [ARCHITECTURE.md](../architecture/ARCHITECTURE.md): the pipeline stage by stage.
- [CLI_GUIDE.md](../guides/CLI_GUIDE.md): every `sa` command in detail.
- [CONTRIBUTING.md](../CONTRIBUTING.md): the rules a change has to satisfy.
- [CHANGELOG.md](../reference/CHANGELOG.md): *why* the codebase looks like this. Longer than
  a changelog usually is, on purpose.

And the four headers that carry more reasoning than signature:

| Header | The question it answers |
|---|---|
| `graph/ir.hpp` | Why differentiation is IR→IR, and what must never go in `Attributes`. |
| `graph/fusion.hpp` | What may fold, and the compile-time proof that nothing else does. |
| `graph/precision.hpp` | Which operation computes in which element type, and why each. |
| `optim/step_scalars.hpp` | How a number changes between replays of a task recorded once. |
