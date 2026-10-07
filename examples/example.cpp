/**************************************************************************/
/* example.cpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file example.cpp
 * @brief Runs one training run end to end: trace, differentiate, compile once, replay per batch.
 * @author Mustafa Garip
 * @see examples/README.md, "example.cpp".
 */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

/** @brief Runs the example and returns its exit code. */
int main()
{
    try
    {
        sr::Execution::RuntimeContext context;
        sb::Engine engine(context);

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

        // A parameter is discoverable because the aggregate declares it.
        sa::NN::Sequential model{
            sa::NN::Linear{features, 128},
            sa::NN::ReLU{},
            sa::NN::Linear{128, classes},
        };

        // Tracing, the loss and differentiation are IR-to-IR; no device work happens yet.
        sa::Train::TrainingGraph built =
            sa::Train::build_classifier(model, batch, features, classes);

        sa::Graph::TensorPool pool(engine);
        std::vector<sa::Optim::ParameterSlot> parameters = sa::Train::bind_parameters(built, pool);
        sa::NN::initialize(engine, model, pool);

        sa::Optim::AdamWOptions adam;
        adam.learning_rate = 3e-3;
        sa::Optim::AdamW optimizer(engine, parameters, adam);

        // The Trainer's constructor performs the only compile of the run.
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
