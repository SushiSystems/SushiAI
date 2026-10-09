/**************************************************************************/
/* train.cpp                                                              */
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
 * @file train.cpp
 * @brief Runs `sushiai train`: trains the model a JSON configuration describes, by registry lookup.
 * @author Mustafa Garip
 * @see apps/README.md, "train.cpp".
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/format.hpp>
#include "registries.hpp"

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

namespace
{
    /** @brief Holds the text `--help` prints. */
    constexpr std::string_view K_USAGE =
        "sushiai train - train a model described by a JSON configuration.\n"
        "\n"
        "Usage: sushiai train <train.json>\n"
        "\n"
        "  train.json names the architecture file, the objective, the data, the\n"
        "  optimizer, the epoch count and where to write the checkpoint. Paths\n"
        "  inside it resolve against its own directory.\n";

    /**
     * @brief Trains the model the configuration at `argv[1]` describes.
     * @return Success, or the first failure a step of the run returned.
     */
    [[nodiscard]] sa::Status run(int argc, char** argv)
    {
        if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
        {
            std::cout << K_USAGE;
            return sa::Status();
        }

        SA_FAIL_IF(argc != 2, sa::StatusCode::InvalidDescription,
                   "usage: sushiai train <train.json>");

        const sa::Config::TrainSpec spec = sa::Config::load_train(argv[1]);
        const sa::Config::ModelSpec model_spec = sa::Config::load_model(spec.model_path);

        sr::Execution::RuntimeContext context;
        sb::Engine engine(context);

        // Everything below is selected by a string, not named by a type; see apps/README.md.

        SA_TRY_ASSIGN(const sa::Apps::Registries registries, sa::Apps::compose_registries());

        SA_TRY_ASSIGN(const std::unique_ptr<sa::Data::Dataset> dataset,
                      sa::Data::make_dataset(registries.datasets, engine, spec.data.type,
                                             spec.data.attributes, "the data block"));

        const sa::Shape sample_shape = dataset->sample_shape();
        SA_FAIL_IF(sample_shape != model_spec.input_shape,
                   sa::StatusCode::InvalidDescription,
                   "'{}' declares the input shape {}, but the '{}' data supplies {}",
                   spec.model_path.string(), model_spec.input_shape.to_string(),
                   dataset->name(), sample_shape.to_string());

        SA_TRY_ASSIGN(sa::NN::ModelGraph model, sa::NN::build_model(registries.nodes, model_spec));

        const std::unique_ptr<sa::Train::Objective> objective =
            sa::Train::make_objective(spec.objective_type, dataset->batch_size(),
                                      dataset->num_classes(), spec.objective_attributes,
                                      "the objective block");

        SA_TRY_ASSIGN(const sa::Shape batch_shape,
                      sa::Train::batch_shape(dataset->batch_size(), sample_shape));
        SA_TRY_ASSIGN(sa::Train::TrainingGraph built,
                      sa::Train::build_training_graph(model, *objective, batch_shape));

        // The trace above declared every parameter and buffer the state lists.
        SA_TRY_ASSIGN(sa::NN::ModelState state, sa::NN::ModelState::create(engine, model));

        sa::Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<sa::Optim::ParameterSlot> parameters,
                      sa::Train::bind_parameters(built, state, pool));
        SA_TRY(sa::NN::initialize(engine, state));

        SA_TRY_ASSIGN(const std::unique_ptr<sa::Optim::Optimizer> optimizer,
                      sa::Optim::make_optimizer(engine, spec.optimizer_type, parameters,
                                                spec.optimizer_attributes, "the optimizer block"));

        sa::Train::TrainerOptions options;
        options.epochs = 1;
        // The objective owns the metric, so the trainer's accuracy path is off; see apps/README.md.
        options.track_accuracy = false;

        SA_TRY_ASSIGN(std::vector<sa::Train::BufferUpdateSlot> updates,
                      sa::Train::bind_buffer_updates(built.updates, state, pool));

        SA_TRY_ASSIGN(const std::unique_ptr<sa::Train::Trainer> trainer,
                      sa::Train::Trainer::make(engine, built.graph, pool, built.spec, *optimizer,
                                               options, std::move(updates)));

        std::cout << "sushiai train\n"
                  << "----------------------------------------------------------------\n"
                  << "SushiAI    : " << SushiAI::version_string()
                  << " on SushiBLAS " << SUSHIBLAS_VERSION_STRING << "\n"
                  << SushiBLAS::Core::format("Config     : {}\n", argv[1])
                  << SushiBLAS::Core::format("Model      : {} ({} layers, {} learnable scalars)\n",
                                             spec.model_path.string(), model.size(),
                                             sa::NN::parameter_elements(model))
                  << SushiBLAS::Core::format("Objective  : {}\n", objective->name())
                  << SushiBLAS::Core::format("Data       : {} ({} x {})\n",
                                             dataset->name(), dataset->batch_count(),
                                             dataset->batch_size())
                  << SushiBLAS::Core::format("Optimizer  : {}\n", spec.optimizer_type)
                  << SushiBLAS::Core::format("Graph      : {} IR nodes over {} values\n",
                                             built.graph.node_count(), built.graph.value_count())
                  << SushiBLAS::Core::format("Schedule   : {} epochs\n\n", spec.epochs);

        // The loop calls run_step so the metric printed is the objective's own; see apps/README.md.
        for (int64_t epoch = 1; epoch <= spec.epochs; ++epoch)
        {
            double last_loss = 0.0;
            std::optional<sa::Train::Metric> last_metric;

            for (std::size_t b = 0; b < dataset->batch_count(); ++b)
            {
                const sa::Data::Batch& batch = dataset->batch(b);
                SA_TRY_ASSIGN(const sa::Train::StepMetrics metrics, trainer->run_step(batch));
                last_loss = metrics.loss;
                // spec.logits is unset when the model publishes several outputs.
                if (built.spec.logits != sa::Graph::K_NO_VALUE)
                    last_metric = objective->score(batch, pool.tensor(built.spec.logits));
            }

            if (epoch == 1 || epoch % 5 == 0 || epoch == spec.epochs)
            {
                std::cout << SushiBLAS::Core::format("  epoch {:>3}/{:<3} loss {:.6f}",
                                                     epoch, spec.epochs, last_loss);
                if (last_metric)
                    std::cout << SushiBLAS::Core::format(
                        "   {} {}", last_metric->name,
                        last_metric->is_fraction
                            ? SushiBLAS::Core::format("{:6.2f}%", last_metric->value * 100.0)
                            : SushiBLAS::Core::format("{:.6g}", last_metric->value));
                std::cout << "\n";
            }
        }

        // A CTest case pins this line, in the shape `sa demo mlp` prints it; see apps/README.md.
        std::cout << SushiBLAS::Core::format("\ncompiles: {}\n", trainer->compile_count());
        if (trainer->compile_count() != 1)
            std::cerr << "The step must be compiled exactly once.\n";

        SA_FAIL_IF(trainer->compile_count() != 1, sa::StatusCode::InvalidDescription,
                   "the step was compiled {} times", trainer->compile_count());

        if (!spec.checkpoint.empty())
        {
            SA_TRY(sa::IO::save_checkpoint(spec.checkpoint, state, engine));
            std::cout << SushiBLAS::Core::format("wrote {} ({} scalars)\n",
                                                 spec.checkpoint.string(),
                                                 sa::NN::parameter_elements(model));
        }

        return sa::Status();
    }
} // namespace

/** @brief Runs the training driver and returns its exit code. */
int main(int argc, char** argv)
{
    try
    {
        const sa::Status status = run(argc, argv);
        if (!status.is_ok())
        {
            std::cerr << "sushiai train failed: " << status.message() << "\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sushiai train failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
