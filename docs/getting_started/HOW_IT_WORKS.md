# How it works

One training program, written against the library, with what each step does. The [tutorial](TUTORIAL.md) walks the same path from an empty checkout.


```cpp
#include <SushiAI/SushiAI.hpp>
#include <SushiBLAS/SushiBLAS.hpp>

namespace sb = SushiBLAS;

sa::Status train()
{
    SA_TRY_ASSIGN(sr::Runtime runtime, sa::Detail::create_runtime());
    sb::Engine engine(runtime);

    // A model is an aggregate list, not a registration ritual.
    sa::NN::Sequential model{
        sa::NN::Linear{32, 128}, sa::NN::ReLU{}, sa::NN::Linear{128, 6},
    };

    // Trace, add cross-entropy, differentiate, insert casts, fuse: all IR-to-IR,
    // no device work yet. The trailing arguments default to float32, every
    // fusion and one element type everywhere; pass a pointer to a
    // sa::Graph::MixedPrecisionPolicy as `precision` for fp16 products.
    SA_TRY_ASSIGN(sa::Train::TrainingGraph built,
                  sa::Train::build_classifier(model, 32, 32, 6));

    SA_TRY_ASSIGN(sa::NN::ModelState state, sa::NN::ModelState::create(engine, model));
    sa::Graph::TensorPool pool(engine);
    SA_TRY_ASSIGN(std::vector<sa::Optim::ParameterSlot> parameters,
                  sa::Train::bind_parameters(built, state, pool));
    SA_TRY(sa::NN::initialize(engine, state));

    SA_TRY_ASSIGN(std::unique_ptr<sa::Optim::AdamW> optimizer,
                  sa::Optim::AdamW::make(engine, std::move(parameters)));

    // The objective build_classifier traced; it feeds and measures every step.
    const sa::Train::CrossEntropyObjective objective(32, 6, sa::DType::Float32);

    // Trainer::make lowers and compiles; it is the only compile of the run.
    SA_TRY_ASSIGN(std::unique_ptr<sa::Train::Trainer> trainer,
                  sa::Train::Trainer::make(engine, built, pool, objective, *optimizer));

    sa::Data::SyntheticOptions synthetic;
    synthetic.features = 32;
    synthetic.classes = 6;
    SA_TRY_ASSIGN(std::unique_ptr<sa::Data::SyntheticClassification> data,
                  sa::Data::SyntheticClassification::make(engine, synthetic));

    SA_TRY_ASSIGN(const sa::Train::RunReport report, trainer->fit(*data));
    std::cout << report.to_string();   // reports compiles: 1
    return sa::Status();
}
```
Source: not compiled

Every step that allocates, compiles or runs returns a `Result` or a `Status`;
`SA_TRY` and `SA_TRY_ASSIGN` return the failure to the caller.
`applications/examples/source/example.cpp` is this program with the dataset chosen at run time:
MNIST when its
files are present, synthetic data otherwise.

SushiAI writes no SYCL: no file here names a `sycl::` type or submits to a
queue, and every compute call goes through `SushiBLAS::sushiblas` (linked
`PUBLIC`). The rows under `modules/core/graph/source/lowering/` turn IR into those calls. The
optimizers do hand SushiBLAS a device functor for their fused
update: plain arithmetic that SushiBLAS's own elementwise seam inlines into
its own kernel. See
[ARCHITECTURE.md](../architecture/ARCHITECTURE.md) for the full picture and
[the programme](../design/PROGRAMME/README.md) for what comes next.
