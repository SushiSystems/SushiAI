/**************************************************************************/
/* inference_program.cpp                                                  */
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
 * @file inference_program.cpp
 * @brief Defines the assembly `sushiai inference` runs once it holds the parsed files.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "inference_program.hpp".
 */

#include "inference_program.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/config/data_fit.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include "inference_run.hpp"
#include "registries.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
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
        [[nodiscard]] Status report_classes(PreparedInference& run,
                                            std::string_view output)
        {
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const scores_tensor, run.step.output(output));
            SA_FAIL_IF(scores_tensor->rank != 2, StatusCode::InvalidDescription,
                       "'{}' publishes '{}' with rank {}, but the report counts classes and reads "
                       "[batch, classes]",
                       run.model_path.string(), output, scores_tensor->rank);

            const Data::Dataset& dataset = run.dataset;
            const int64_t rows = dataset.batch_size();
            const int64_t outputs = scores_tensor->shape[1];
            print_inference_header(run, SushiLog::format_text("{} outputs", outputs));

            // Top-1 agreement needs one output per class; see applications/programs/README.md.
            const Data::TargetDescription* const described =
                Data::find_target(dataset.targets(), Data::K_ROLE_LABELS);
            const int64_t classes = described != nullptr ? described->classes : 0;
            const bool labelled = outputs == classes;

            std::vector<std::size_t> predicted_counts(static_cast<std::size_t>(outputs), 0);
            std::size_t samples = 0;
            std::size_t agreeing = 0;

            Data::Batch batch;
            for (std::size_t b = 0; b < dataset.batch_count(); ++b)
            {
                SA_TRY(dataset.read(b, batch));
                SA_TRY(run.step.run(batch.inputs));

                const Data::BatchTarget* const target =
                    labelled ? Data::find_target(batch.targets, Data::K_ROLE_LABELS)
                             : nullptr;
                SA_FAIL_IF(labelled && target == nullptr, StatusCode::InvalidDescription,
                           "batch {} of the '{}' data holds no '{}' role, which the data declares",
                           b, dataset.name(), Data::K_ROLE_LABELS);
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

            std::cout << SushiLog::format_text("Evaluated {} samples over {} batches.\n\n",
                                                 samples, dataset.batch_count());
            std::cout << "  output  predicted\n";
            for (std::size_t i = 0; i < predicted_counts.size(); ++i)
            {
                const double share = 100.0 * static_cast<double>(predicted_counts[i])
                                     / static_cast<double>(samples);
                std::cout << SushiLog::format_text("  {}  {} ({}%)\n", SushiLog::right(i, 6),
                                                   SushiLog::right(predicted_counts[i], 9),
                                                   SushiLog::right(SushiLog::fixed(share, 2), 5));
            }

            if (labelled)
                std::cout << SushiLog::format_text(
                    "\ntop-1 agreement with the labels: {}/{} = {}%\n",
                    agreeing, samples,
                    SushiLog::fixed(
                        100.0 * static_cast<double>(agreeing) / static_cast<double>(samples), 2));
            else
                std::cout << SushiLog::format_text(
                    "\nThe model has {} outputs and the data {} classes, so top-1 agreement "
                    "is not comparable and is not reported.\n",
                    outputs, classes);

            return Status();
        }
        } // namespace

        Status run_inference_file(const std::filesystem::path& model_path,
                                  const std::filesystem::path& checkpoint_path,
                                  const Config::ModelSpec& model_spec,
                                  const Config::DataSpec& data_spec, SushiBLAS::Engine& engine)
        {
            SA_TRY_ASSIGN(const Registries registries, compose_registries());

            SA_TRY_ASSIGN(const Config::Value data_block,
                          Config::fit_data_to_model(data_spec.attributes, model_spec,
                                                        model_path.string()));
            SA_TRY_ASSIGN(const std::unique_ptr<Data::Dataset> dataset,
                          Data::make_dataset(registries.datasets, engine, data_spec.type,
                                                 data_block, "the data block"));

            const Shape sample_shape = dataset->sample_shape();
            SA_FAIL_IF(sample_shape != model_spec.input_shape,
                       StatusCode::InvalidDescription,
                       "'{}' declares the input shape {}, but the '{}' data supplies {}",
                       model_path.string(), model_spec.input_shape.to_string(),
                       dataset->name(), sample_shape.to_string());

            // Forward pass only: no loss, backward or optimizer; see the programs README.
            SA_TRY_ASSIGN(NN::ModelGraph model, NN::build_model(registries.nodes, model_spec));
            const int64_t rows = dataset->batch_size();

            SA_TRY_ASSIGN(const Shape batch_shape, Train::batch_shape(rows, sample_shape));
            // The model file states the element type of the input leaf; int32 reads indices.
            Train::GraphOptions graph_options;
            graph_options.dtype = model_spec.input_dtype;
            SA_TRY_ASSIGN(const Train::TracedStep traced,
                          Train::build_evaluation_graph(model, batch_shape, graph_options));

            // A model without a task is a classifier with one output; a task names its own.
            const bool tasked = !model_spec.task.is_null();
            const std::vector<NN::ModelOutputs::Entry>& published = traced.outputs.all();
            SA_FAIL_IF(!tasked && published.size() != 1, StatusCode::InvalidDescription,
                       "'{}' publishes {} outputs, but the report counts classes and reads one",
                       model_path.string(), published.size());

            SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
            SA_TRY(IO::load_checkpoint(checkpoint_path, state));

            Graph::TensorPool pool(engine);
            SA_TRY_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                          Train::EvaluationStep::make(engine, traced, state, pool));

            // The report is the row of the task type; see applications/programs/README.md.
            PreparedInference prepared{
                model_path, checkpoint_path, model_spec, model.size(),
                *dataset, *step, published, engine, &model, &state,
            };
            if (tasked)
            {
                const std::string& task =
                    model_spec.task.at("type", "the task section").as_string("the task type");
                SA_TRY(report_inference(registries.inferences, task, prepared));
            }
            else
            {
                SA_TRY(report_classes(prepared, published.front().first));
            }

            std::cout << SushiLog::format_text("\ncompiles: {}\n", step->step().compile_count());
            if (step->step().compile_count() != 1)
                std::cerr << "The step must be compiled exactly once.\n";

            SA_FAIL_IF(step->step().compile_count() != 1, StatusCode::InvalidDescription,
                       "the step was compiled {} times", step->step().compile_count());

            return Status();
        }
    } // namespace Apps
} // namespace SushiAI
