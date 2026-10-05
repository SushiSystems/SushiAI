/****************************************************************/
/* example.cpp                                                  */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// A whole training run, end to end: declare a model, trace it into IR,
// differentiate the IR, compile the result exactly once, replay it per batch.
//
// Uses MNIST when the IDX files happen to be sitting in data/mnist, and a
// seeded synthetic problem otherwise - so the example runs anywhere, with no
// download and no fixture, and says which one it picked.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>

int main()
{
    try
    {
        sr::Execution::RuntimeContext context;
        sb::Engine engine(context);

        // --- Data -----------------------------------------------------------
        const int64_t batch = 32;

        sa::Data::MnistOptions mnist;
        mnist.batch_size = batch;
        mnist.max_samples = 2048;

        std::unique_ptr<sa::Data::Dataset> dataset = sa::Data::try_load_mnist(engine, mnist);

        if (dataset)
        {
            std::cout << "Dataset  : MNIST from " << mnist.directory.string() << "\n";
        }
        else
        {
            std::cout << "Dataset  : MNIST not found (" << sa::Data::mnist_expected_paths(mnist)
                      << ");\n           falling back to the seeded synthetic problem.\n";

            sa::Data::SyntheticOptions synthetic;
            synthetic.batch_size = batch;
            synthetic.batches = 16;
            synthetic.features = 32;
            synthetic.classes = 6;
            dataset = std::make_unique<sa::Data::SyntheticClassification>(engine, synthetic);
        }

        const int64_t features = dataset->input_dim();
        const int64_t classes = dataset->num_classes();

        // --- Model ------------------------------------------------------------
        // An aggregate list, not a registration ritual: a parameter is
        // discoverable because it is declared, not because something remembered
        // to register it.
        sa::NN::Sequential model{
            sa::NN::Linear{features, 128},
            sa::NN::ReLU{},
            sa::NN::Linear{128, classes},
        };

        // Trace the forward pass, add cross-entropy, differentiate. All of it is
        // an IR-to-IR transform; no device work happens yet.
        sa::Train::TrainingGraph built =
            sa::Train::build_classifier(model, batch, features, classes);

        // --- Memory and initialisation ------------------------------------------
        sa::Graph::TensorPool pool(engine);
        std::vector<sa::Optim::ParameterSlot> parameters = sa::Train::bind_parameters(built, pool);
        sa::NN::initialize(engine, model, pool);

        sa::Optim::AdamWOptions adam;
        adam.learning_rate = 3e-3;
        sa::Optim::AdamW optimizer(engine, parameters, adam);

        // --- Train ----------------------------------------------------------------
        // The trainer's constructor lowers and compiles the step. That is the
        // only compile of the whole run, and the report below has to agree.
        sa::Train::TrainerOptions options;
        options.epochs = 8;
        sa::Train::Trainer trainer(engine, built.graph, pool, built.spec, optimizer, options);

        std::cout << "SushiAI  : " << SushiAI::version_string()
                  << " on SushiBLAS " << SUSHIBLAS_VERSION_STRING << "\n"
                  << "Model    : " << features << " -> 128 -> " << classes
                  << "  (" << sa::NN::parameter_elements(model) << " learnable scalars)\n"
                  << "Batches  : " << dataset->batch_count() << " x " << batch
                  << " from '" << dataset->name() << "'\n"
                  << "Graph    : " << built.graph.node_count() << " IR nodes over "
                  << built.graph.value_count() << " values\n"
                  << "Buffers  : " << pool.tensor_count() << " tensors, "
                  << (pool.allocated_bytes() / 1024) << " KiB\n\n";

        const sa::Train::RunReport report = trainer.fit(*dataset);
        std::cout << report.to_string();

        if (report.compile_count != 1)
        {
            std::cerr << "\nThe step was compiled " << report.compile_count
                      << " times; it must be compiled exactly once.\n";
            return EXIT_FAILURE;
        }

        std::cout << "\nCompiled once, replayed " << report.steps << " times.\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "SushiAI example failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
