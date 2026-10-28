# SushiAI manual

What the library is and how to run it is in the root [`README.md`](../README.md). This tree is
the manual.

| Document | Holds |
| --- | --- |
| [Install](getting_started/INSTALL.md) | The two ways to install, standalone and with SushiHub, and what each needs |
| [Tutorial](getting_started/TUTORIAL.md) | An empty checkout to a trained model, then what each pipeline stage did with it |
| [How it works](getting_started/HOW_IT_WORKS.md) | One training program written against the library, step by step |
| [Architecture](architecture/ARCHITECTURE.md) | What SushiAI is, the sibling chain, the library target, the IR and autograd |
| [Lowering](architecture/LOWERING.md) | Memory planning, the lowering table, the compiled step and the optimizer inside it |
| [Fusion and precision](architecture/FUSION_AND_PRECISION.md) | Fusion over a catalog and mixed precision in the IR |
| [Models and packaging](architecture/MODELS_AND_PACKAGING.md) | The model file, its driver, the registries, checkpoints and packaging |
| [CLI guide](guides/CLI_GUIDE.md) | The `sa` commands that build, test, run and diagnose |
| [Running models](guides/RUNNING_MODELS.md) | `sushiai train`, `inference`, `import` and `tokenize` |
| [Profiling a run](guides/PROFILING.md) | `--profile` on `train` and `inference`, what a record holds and `sa profile` |
| [Demo](guides/DEMO.md) | What `sa demo mlp` prints and what each number means |
| [Integration guide](guides/INTEGRATION.md) | Consuming SushiAI from another CMake project |
| [Model file](guides/MODEL_FILE.md) | The JSON model description, version 2, and the upgrade from version 1 |
| [FAQ](guides/FAQ.md) | The questions a newcomer asks, each answered from this manual |
| [API reference](reference/API_REFERENCE.md) | The main page of the generated Doxygen site |
| [Glossary](reference/GLOSSARY.md) | The words these documents use |
| [Known issues](backlog/README.md) | Defects that are recorded and not yet fixed |
| [Changelog](reference/CHANGELOG.md) | What changed, by release |
| [Code of conduct](reference/CODE_OF_CONDUCT.md) | What is expected in project spaces |
| [Security policy](reference/SECURITY.md) | How to report a vulnerability |
| [Design map](design/README.md) | Which design document covers which topic |
| [Programme](design/PROGRAMME/README.md) | The target, the owner's rulings and the stages towards 0.1.0 |
| [Remaining work](backlog/README.md) | The backlog |
| [Contributing](CONTRIBUTING.md) | How a change is built, tested, documented and committed |
| [Documentation style guide](DOCUMENTATION_STYLE_GUIDE.md) | Where a document goes and how it is written |

READMEs beside the code: [`tools/README.md`](../tools/README.md) for the checkers, and
[`applications/examples/heavy_mlp/README.md`](../applications/examples/heavy_mlp/README.md) and
[`applications/examples/mnist_cnn/README.md`](../applications/examples/mnist_cnn/README.md) for the
two JSON-configured examples.

Agent work folders are under `agent/`. Finished material is under `archive/` and is not edited.
