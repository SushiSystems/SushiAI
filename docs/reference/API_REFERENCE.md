# SushiAI {#mainpage}

Graph-native deep learning on top of [SushiBLAS](https://github.com/SushiSystems/SushiBLAS) and
[SushiRuntime](https://github.com/SushiSystems/SushiRuntime). A model is traced once into
SushiAI's own IR, differentiated, fused, memory-planned, lowered to a `TaskGraph` **once** and
then replayed per batch. `compile_count() == 1` is a guarantee with a test on it, not an aspiration.

This is the generated API reference. It documents what the headers say; the reasoning behind the
design lives in the prose docs, and those are the better place to start:

- `docs/README.md` — what the library is and how to run the demo.
- `docs/ARCHITECTURE.md` — the pipeline, and why each stage is where it is.
- `docs/INTEGRATION.md` — consuming SushiAI from your own CMake project.
- `docs/CONTRIBUTING.md` — the build rules (`sa build`, never cmake directly).

## Where to start reading

| If you want to | Read |
|---|---|
| Define a model | `SushiAI::NN` — `nn/module.hpp`, `nn/layers.hpp` |
| Train one | `SushiAI::Train::Trainer` — `train/trainer.hpp` |
| Understand the IR every pass rewrites | `SushiAI::Graph::Graph` — `graph/ir.hpp` |
| Add a backward rule | `SushiAI::Autograd` — `autograd/backward.hpp` |
| Know what fuses, and what refuses to | `SushiAI::Graph::FusedForm` — `graph/fusion.hpp` |
| Run in fp16 with fp32 masters | `SushiAI::Graph::IPrecisionPolicy` — `graph/precision.hpp` |

## A caution about the reference

Doxygen lists every public symbol, which flattens a deliberate hierarchy. Three headers carry
warnings that matter more than their signatures do, and a reader who only skims the generated
pages will miss them:

- `graph/ir.hpp` — why a per-step value must never go in `Attributes`.
- `graph/fusion.hpp` — why an activation whose gradient reads its *input* cannot be folded away
  without an auxiliary buffer, and why removing that `static_assert` produces a model that trains
  and is silently wrong.
- `optim/step_scalars.hpp` — why a number that changes between replays lives in device memory.
