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
 * @see applications/examples/README.md, "example.cpp".
 */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/SushiAI.h>
#include <SushiAI/vision/datasets/mnist.hpp>
#include <SushiBLAS/SushiBLAS.h>

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

namespace
{
    /** @brief Runs the example and returns its status. */
    [[nodiscard]] sa::Status run([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
    {
        sr::Execution::RuntimeContext context;
        sb::Engine engine(context);

        const int64_t batch = 32;

        sa::Vision::Datasets::MnistOptions mnist;
        mnist.batch_size = batch;
        mnist.max_samples = 2048;

        std::unique_ptr<sa::Data::Dataset> dataset;
        sa::Result<std::unique_ptr<sa::Data::Dataset>> loaded =
            sa::Vision::Datasets::try_load_mnist(engine, mnist);

        if (loaded.has_value())
        {
            dataset = std::move(loaded.value());
            std::cout << "Dataset  : MNIST from " << mnist.directory.string() << "\n";
        }
        else if (loaded.status().code() == sa::StatusCode::FileNotFound)
        {
            std::cout << "Dataset  : MNIST not found ("
                      << sa::Vision::Datasets::mnist_expected_paths(mnist)
                      << ");\n           falling back to the seeded synthetic problem.\n";

            sa::Data::SyntheticOptions synthetic;
            synthetic.batch_size = batch;
            synthetic.batches = 16;
            synthetic.features = 32;
            synthetic.classes = 6;
            SA_TRY_ASSIGN(std::unique_ptr<sa::Data::SyntheticClassification> generated,
                          sa::Data::SyntheticClassification::make(engine, synthetic));
            dataset = std::move(generated);
        }
        else
        {
            return loaded.status();
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
        SA_TRY_ASSIGN(sa::Train::TrainingGraph built,
                      sa::Train::build_classifier(model, batch, features, classes));

        SA_TRY_ASSIGN(sa::NN::ModelState state, sa::NN::ModelState::create(engine, model));
        sa::Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<sa::Optim::ParameterSlot> parameters,
                      sa::Train::bind_parameters(built, state, pool));
        SA_TRY(sa::NN::initialize(engine, state));

        sa::Optim::AdamWOptions adam;
        adam.learning_rate = 3e-3;
        SA_TRY_ASSIGN(std::unique_ptr<sa::Optim::AdamW> optimizer,
                      sa::Optim::AdamW::make(engine, std::move(parameters), adam));

        // Trainer::make performs the only compile of the run.
        sa::Train::TrainerOptions options;
        options.epochs = 8;
        SA_TRY_ASSIGN(std::unique_ptr<sa::Train::Trainer> trainer,
                      sa::Train::Trainer::make(engine, built.graph, pool, built.spec, *optimizer,
                                               options));

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

        SA_TRY_ASSIGN(const sa::Train::RunReport report, trainer->fit(*dataset));
        std::cout << report.to_string();

        SA_FAIL_IF(report.compile_count != 1, sa::StatusCode::InvalidDescription,
                   "The step was compiled {} times; it must be compiled exactly once.",
                   report.compile_count);

        std::cout << "\nCompiled once, replayed " << report.steps << " times.\n";
        return sa::Status();
    }
} // namespace

/** @brief Runs the example and returns its exit code. */
int main(int argc, char** argv)
{
    try
    {
        const sa::Status status = run(argc, argv);
        if (!status.is_ok())
        {
            std::cerr << "SushiAI example failed: " << status.message() << "\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "SushiAI example failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
