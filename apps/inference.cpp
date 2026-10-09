/**************************************************************************/
/* inference.cpp                                                          */
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
 * @file inference.cpp
 * @brief Runs `sushiai inference`: evaluates a checkpoint over a dataset, forward pass only.
 * @author Mustafa Garip
 * @see apps/README.md, "inference.cpp".
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
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
        "sushiai inference - evaluate a trained checkpoint over a dataset.\n"
        "\n"
        "Usage: sushiai inference <model.json> <model.ckpt> <eval.json>\n"
        "\n"
        "  model.json  the architecture the checkpoint was trained under\n"
        "  model.ckpt  the checkpoint `sushiai train` wrote\n"
        "  eval.json   the data to evaluate on; the architecture file carries none\n";

    /** @brief Returns the index of the largest of @p count values. */
    int64_t argmax_row(const float* row, int64_t count)
    {
        int64_t best = 0;
        for (int64_t i = 1; i < count; ++i)
            if (row[i] > row[best]) best = i;
        return best;
    }

    /** @brief Evaluates the checkpoint the arguments name and prints the report. */
    [[nodiscard]] sa::Status run(int argc, char** argv)
    {
        if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
        {
            std::cout << K_USAGE;
            return sa::Status();
        }

        SA_FAIL_IF(argc != 4, sa::StatusCode::InvalidDescription,
                   "usage: sushiai inference <model.json> <model.ckpt> <eval.json>");

        const std::filesystem::path model_path = argv[1];
        const std::filesystem::path checkpoint_path = argv[2];
        const std::filesystem::path data_path = argv[3];

        const sa::Config::ModelSpec model_spec = sa::Config::load_model(model_path);
        const sa::Config::DataSpec data_spec = sa::Config::load_data(data_path);

        sr::Execution::RuntimeContext context;
        sb::Engine engine(context);

        SA_TRY_ASSIGN(const sa::Apps::Registries registries, sa::Apps::compose_registries());

        SA_TRY_ASSIGN(const std::unique_ptr<sa::Data::Dataset> dataset,
                      sa::Data::make_dataset(registries.datasets, engine, data_spec.type,
                                             data_spec.attributes, "the data block"));

        const sa::Shape sample_shape = dataset->sample_shape();
        SA_FAIL_IF(sample_shape != model_spec.input_shape,
                   sa::StatusCode::InvalidDescription,
                   "'{}' declares the input shape {}, but the '{}' data supplies {}",
                   model_path.string(), model_spec.input_shape.to_string(),
                   dataset->name(), sample_shape.to_string());

        // Only the forward pass is traced: no loss, no backward, no optimizer; see apps/README.md.
        SA_TRY_ASSIGN(sa::NN::ModelGraph model, sa::NN::build_model(registries.nodes, model_spec));
        const int64_t rows = dataset->batch_size();

        SA_TRY_ASSIGN(const sa::Shape batch_shape, sa::Train::batch_shape(rows, sample_shape));
        SA_TRY_ASSIGN(const sa::Train::TracedStep traced,
                      sa::Train::build_evaluation_graph(model, batch_shape));

        const std::vector<sa::NN::ModelOutputs::Entry>& published = traced.outputs.all();
        SA_FAIL_IF(published.size() != 1, sa::StatusCode::InvalidDescription,
                   "'{}' publishes {} outputs, but the report counts classes and reads one",
                   model_path.string(), published.size());

        SA_TRY_ASSIGN(sa::NN::ModelState state, sa::NN::ModelState::create(engine, model));
        SA_TRY(sa::IO::load_checkpoint(checkpoint_path, state));

        sa::Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(const std::unique_ptr<sa::Train::EvaluationStep> step,
                      sa::Train::EvaluationStep::make(engine, traced, state, pool));

        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const scores_tensor,
                      step->output(published.front().first));
        SA_FAIL_IF(scores_tensor->rank != 2, sa::StatusCode::InvalidDescription,
                   "'{}' publishes '{}' with rank {}, but the report counts classes and reads "
                   "[batch, classes]",
                   model_path.string(), published.front().first, scores_tensor->rank);

        const int64_t outputs = scores_tensor->shape[1];

        std::cout << "sushiai inference\n"
                  << "----------------------------------------------------------------\n"
                  << SushiBLAS::Core::format("Model      : {} ({} layers -> {} outputs)\n",
                                             model_path.string(), model.size(), outputs)
                  << SushiBLAS::Core::format("Checkpoint : {}\n", checkpoint_path.string())
                  << SushiBLAS::Core::format("Data       : {} ({} x {})\n\n",
                                             dataset->name(), dataset->batch_count(), rows);

        // Top-1 agreement needs an output width equal to the class count; see apps/README.md.
        const bool labelled = outputs == dataset->num_classes();

        std::vector<std::size_t> predicted_counts(static_cast<std::size_t>(outputs), 0);
        std::size_t samples = 0;
        std::size_t agreeing = 0;

        for (std::size_t b = 0; b < dataset->batch_count(); ++b)
        {
            const sa::Data::Batch& batch = dataset->batch(b);
            SA_TRY(step->run(batch.inputs));

            const float* scores = scores_tensor->data_as<float>();
            for (int64_t r = 0; r < rows; ++r)
            {
                const int64_t predicted = argmax_row(scores + r * outputs, outputs);
                ++predicted_counts[static_cast<std::size_t>(predicted)];
                if (labelled && predicted == batch.labels[static_cast<std::size_t>(r)])
                    ++agreeing;
                ++samples;
            }
        }

        std::cout << SushiBLAS::Core::format("Evaluated {} samples over {} batches.\n\n",
                                             samples, dataset->batch_count());
        std::cout << "  output  predicted\n";
        for (std::size_t i = 0; i < predicted_counts.size(); ++i)
            std::cout << SushiBLAS::Core::format("  {:>6}  {:>9} ({:5.2f}%)\n", i,
                                                 predicted_counts[i],
                                                 100.0 * static_cast<double>(predicted_counts[i])
                                                       / static_cast<double>(samples));

        if (labelled)
            std::cout << SushiBLAS::Core::format(
                "\ntop-1 agreement with the labels: {}/{} = {:.2f}%\n",
                agreeing, samples,
                100.0 * static_cast<double>(agreeing) / static_cast<double>(samples));
        else
            std::cout << SushiBLAS::Core::format(
                "\nThe model has {} outputs and the data {} classes, so top-1 agreement "
                "is not comparable and is not reported.\n",
                outputs, dataset->num_classes());

        std::cout << SushiBLAS::Core::format("\ncompiles: {}\n", step->step().compile_count());
        if (step->step().compile_count() != 1)
            std::cerr << "The step must be compiled exactly once.\n";

        SA_FAIL_IF(step->step().compile_count() != 1, sa::StatusCode::InvalidDescription,
                   "the step was compiled {} times", step->step().compile_count());

        return sa::Status();
    }
} // namespace

/** @brief Runs the inference driver and returns its exit code. */
int main(int argc, char** argv)
{
    try
    {
        const sa::Status status = run(argc, argv);
        if (!status.is_ok())
        {
            std::cerr << "sushiai inference failed: " << status.message() << "\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sushiai inference failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
