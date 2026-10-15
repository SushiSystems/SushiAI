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
 * @see applications/programs/README.md, "inference.cpp".
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
#include <SushiAI/core/seam.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/format.hpp>
#include "inference_run.hpp"
#include "registries.hpp"

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

namespace
{
    /** @brief Holds the text `--help` prints. */
    constexpr std::string_view K_USAGE =
        "sushiai inference - evaluate a trained checkpoint over a dataset.\n"
        "\n"
        "Usage: sushiai inference <model.json> <model.sushi> <eval.json>\n"
        "\n"
        "  model.json   the architecture the checkpoint was trained under\n"
        "  model.sushi  the weights `sushiai train` or `sushiai import` wrote\n"
        "  eval.json    the data to evaluate on; the architecture file carries none\n"
        "\n"
        "  A model file without a task section is reported as a classifier. One with\n"
        "  a task section is reported by the row of its task type; the data of a\n"
        "  `next_token_generation` model is a `prompt` block, and the report prints\n"
        "  the prompt and its continuation.\n";

    /** @brief Returns the index of the largest of @p count values. */
    int64_t argmax_row(const float* row, int64_t count)
    {
        int64_t best = 0;
        for (int64_t i = 1; i < count; ++i)
            if (row[i] > row[best]) best = i;
        return best;
    }

    /**
     * @brief Counts the predicted class of every sample and prints top-1 agreement.
     * @param output The name of the one output the model publishes.
     * @return Success, or the first failure of a read or a step; InvalidDescription for an
     *         output that is not `[batch, classes]` or a batch without its labels.
     */
    [[nodiscard]] sa::Status report_classes(sa::Apps::PreparedInference& run,
                                            std::string_view output)
    {
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const scores_tensor, run.step.output(output));
        SA_FAIL_IF(scores_tensor->rank != 2, sa::StatusCode::InvalidDescription,
                   "'{}' publishes '{}' with rank {}, but the report counts classes and reads "
                   "[batch, classes]",
                   run.model_path.string(), output, scores_tensor->rank);

        const sa::Data::Dataset& dataset = run.dataset;
        const int64_t rows = dataset.batch_size();
        const int64_t outputs = scores_tensor->shape[1];
        sa::Apps::print_inference_header(run, SushiBLAS::Core::format("{} outputs", outputs));

        // Top-1 agreement needs one output per class; see applications/programs/README.md.
        const sa::Data::TargetDescription* const described =
            sa::Data::find_target(dataset.targets(), sa::Data::K_ROLE_LABELS);
        const int64_t classes = described != nullptr ? described->classes : 0;
        const bool labelled = outputs == classes;

        std::vector<std::size_t> predicted_counts(static_cast<std::size_t>(outputs), 0);
        std::size_t samples = 0;
        std::size_t agreeing = 0;

        sa::Data::Batch batch;
        for (std::size_t b = 0; b < dataset.batch_count(); ++b)
        {
            SA_TRY(dataset.read(b, batch));
            SA_TRY(run.step.run(batch.inputs));

            const sa::Data::BatchTarget* const target =
                labelled ? sa::Data::find_target(batch.targets, sa::Data::K_ROLE_LABELS)
                         : nullptr;
            SA_FAIL_IF(labelled && target == nullptr, sa::StatusCode::InvalidDescription,
                       "batch {} of the '{}' data holds no '{}' role, which the data declares",
                       b, dataset.name(), sa::Data::K_ROLE_LABELS);
            const int32_t* const labels =
                target != nullptr ? target->values.data_as<int32_t>() : nullptr;

            const float* scores = scores_tensor->data_as<float>();
            for (int64_t r = 0; r < rows; ++r)
            {
                const int64_t predicted = argmax_row(scores + r * outputs, outputs);
                ++predicted_counts[static_cast<std::size_t>(predicted)];
                if (labelled && predicted == labels[r])
                    ++agreeing;
                ++samples;
            }
        }

        std::cout << SushiBLAS::Core::format("Evaluated {} samples over {} batches.\n\n",
                                             samples, dataset.batch_count());
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
                outputs, classes);

        return sa::Status();
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
                   "usage: sushiai inference <model.json> <model.sushi> <eval.json>");

        const std::filesystem::path model_path = argv[1];
        const std::filesystem::path checkpoint_path = argv[2];
        const std::filesystem::path data_path = argv[3];

        const sa::Config::ModelSpec model_spec = sa::Config::load_model(model_path);
        const sa::Config::DataSpec data_spec = sa::Config::load_data(data_path);

        SA_TRY_ASSIGN(sr::Runtime runtime, sa::Detail::create_runtime());
        sb::Engine engine(runtime);

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

        // Forward pass only: no loss, backward or optimizer; see applications/programs/README.md.
        SA_TRY_ASSIGN(sa::NN::ModelGraph model, sa::NN::build_model(registries.nodes, model_spec));
        const int64_t rows = dataset->batch_size();

        SA_TRY_ASSIGN(const sa::Shape batch_shape, sa::Train::batch_shape(rows, sample_shape));
        // The model file states the element type of the input leaf; int32 reads indices.
        sa::Train::GraphOptions graph_options;
        graph_options.dtype = model_spec.input_dtype;
        SA_TRY_ASSIGN(const sa::Train::TracedStep traced,
                      sa::Train::build_evaluation_graph(model, batch_shape, graph_options));

        // A model without a task is a classifier with one output; a task names its own.
        const bool tasked = !model_spec.task.is_null();
        const std::vector<sa::NN::ModelOutputs::Entry>& published = traced.outputs.all();
        SA_FAIL_IF(!tasked && published.size() != 1, sa::StatusCode::InvalidDescription,
                   "'{}' publishes {} outputs, but the report counts classes and reads one",
                   model_path.string(), published.size());

        SA_TRY_ASSIGN(sa::NN::ModelState state, sa::NN::ModelState::create(engine, model));
        SA_TRY(sa::IO::load_checkpoint(checkpoint_path, state));

        sa::Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(const std::unique_ptr<sa::Train::EvaluationStep> step,
                      sa::Train::EvaluationStep::make(engine, traced, state, pool));

        // The report is the row of the task type; see applications/programs/README.md.
        sa::Apps::PreparedInference prepared{
            model_path, checkpoint_path, model_spec, model.size(),
            *dataset, *step, published, engine, &model, &state,
        };
        if (tasked)
        {
            const std::string& task =
                model_spec.task.at("type", "the task section").as_string("the task type");
            SA_TRY(sa::Apps::report_inference(registries.inferences, task, prepared));
        }
        else
        {
            SA_TRY(report_classes(prepared, published.front().first));
        }

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
