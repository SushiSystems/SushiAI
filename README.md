# SushiAI

SushiAI is a C++17 library for training and running neural networks on SYCL devices, for
developers who build on [SushiBLAS](https://github.com/SushiSystems/SushiBLAS) and
[SushiRuntime](https://github.com/SushiSystems/SushiRuntime).

```cpp
        // A parameter is discoverable because the aggregate declares it.
        sa::NN::Sequential model{
            sa::NN::Linear{features, 128},
            sa::NN::ReLU{},
            sa::NN::Linear{128, classes},
        };

        // Tracing, the loss and differentiation are IR-to-IR; no device work happens yet.
        SA_TRY_ASSIGN(sa::Train::TrainingGraph built,
                      sa::Train::build_classifier(model, batch, features, classes));
        // ...
        SA_TRY_ASSIGN(const sa::Train::RunReport report, trainer->fit(*dataset));
```

Source: `applications/examples/source/example.cpp`

The model is traced into SushiAI's own graph, differentiated as a graph, compiled once and
replayed for every batch.

## Build

```bash
hub install-cli sushiai          # puts sa and sushiai on PATH
sa setup                         # provisions the toolchain and the C++ dependencies
sa build
sa test                          # small and medium tests
sa demo mlp                      # trains a small classifier end to end
```

[`docs/guides/CLI_GUIDE.md`](docs/guides/CLI_GUIDE.md) lists the requirements and every
command.

## Status

An MLP trains end to end, with memory planning, kernel fusion, mixed precision and gradient
loss scaling. A model can also be described as a JSON file and saved to a checkpoint.
[`docs/design/REMAINING_WORK.md`](docs/design/REMAINING_WORK.md) lists what does not work yet,
and [`docs/design/PROGRAMME.md`](docs/design/PROGRAMME.md) lists what is planned.

## Documentation

[`docs/README.md`](docs/README.md) indexes the manual. The
[tutorial](docs/getting_started/TUTORIAL.md) goes from an empty checkout to a model that trains.
Contributors read [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## Licence

SushiAI is free for non-commercial use under the PolyForm Noncommercial License 1.0.0; see
[LICENSE](LICENSE). Commercial use needs a licence from Sushi Systems; see
[COMMERCIAL.md](COMMERCIAL.md). Third-party components are listed in [NOTICE.md](NOTICE.md).
