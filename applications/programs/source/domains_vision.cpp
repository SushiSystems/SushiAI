/**************************************************************************/
/* domains_vision.cpp                                                     */
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
 * @file domains_vision.cpp
 * @brief Adds the rows of vision to the app registries; holds an anchor model's run and report.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The run table".
 */

#include "domains.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/profile/timelines.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/head_decode.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_section.hpp>
#include <SushiAI/vision/detect/average_precision.hpp>
#include <SushiAI/vision/detect/batch_average_precision.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/free/head_contract.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/registration/registration.hpp>
#include <SushiAI/vision/training/anchor_bias_init.hpp>
#include <SushiAI/vision/training/anchor_objective.hpp>
#include <SushiAI/vision/training/anchor_objective_factory.hpp>
#include <SushiAI/vision/training/anchor_training_loop.hpp>
#include <SushiAI/vision/training/anchor_training_step.hpp>
#include <SushiAI/vision/training/free_batch_decoder.hpp>
#include <SushiAI/vision/training/free_bias_init.hpp>
#include <SushiAI/vision/training/free_objective.hpp>
#include <SushiAI/vision/training/free_objective_factory.hpp>
#include <SushiAI/vision/training/free_training_loop.hpp>
#include <SushiAI/vision/training/free_training_step.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>
#include <SushiAI/vision/training/truth_from_roles.hpp>

#include "epoch_loop.hpp"
#include "inference_run.hpp"
#include "training_run.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Prints the cost and the rate of the last update of an epoch. */
            void print_epoch(const Vision::Training::AnchorProgress& progress,
                             std::int64_t epochs)
            {
                std::cout << SushiLog::format_text(
                    "  epoch {}/{} loss {}   box {}   rate {}\n",
                    SushiLog::right(progress.epoch, 3), SushiLog::left(epochs, 3),
                    SushiLog::fixed(progress.cost.total, 6), SushiLog::fixed(progress.cost.box, 6),
                    SushiLog::general(progress.rate, 6));
            }

            /**
             * @brief Starts the objectness and class biases of every anchor head of @p run from
             *        the stride of its level, as Ultralytics' YOLOv5 does.
             * @return The failure of the model or task read, of the head lookup or of the write.
             */
            [[nodiscard]] Status start_anchor_biases(PreparedRun& run)
            {
                const Config::ModelSpec model = Config::load_model(run.spec.model_path);
                SA_TRY_ASSIGN(const Vision::Detect::Anchor::AnchorTask task,
                              Vision::Detect::Anchor::read_anchor_task(run.given.task));
                SA_TRY_ASSIGN(const std::vector<Vision::Training::AnchorBiasHead> heads,
                              Vision::Training::derive_anchor_bias_heads(
                                  model, run.given.input_sample, task, run.given.outputs));
                return Vision::Training::initialize_anchor_biases(task, heads, run.state);
            }

            /**
             * @brief Trains a model with anchor heads through AnchorTrainingStep, first starting
             *        the objectness and class biases when no checkpoint or weights are read.
             * @return The compile count of the training pass and where the run stands;
             *         InvalidDescription when the forward-only pass was not compiled exactly
             *         once; or the first failure of the objective, of the bias start, of the
             *         step, of the start or of run_anchor_training.
             */
            [[nodiscard]] Result<RunSummary> run_anchor(PreparedRun& run)
            {
                // The registry returns the interface; the step takes the type behind it.
                SA_TRY_ASSIGN(const Vision::Training::AnchorObjective objective,
                              Vision::Training::build_anchor_objective(
                                  run.given, run.spec.objective_attributes));

                if (run.spec.resume.empty() && run.spec.weights.empty())
                {
                    SA_TRY(start_anchor_biases(run));
                }

                SA_TRY_ASSIGN(
                    const std::unique_ptr<Vision::Training::AnchorTrainingStep> step,
                    Vision::Training::AnchorTrainingStep::make(
                        run.engine, objective, run.built, run.pool, run.optimizer,
                        std::move(run.updates)));

                SA_TRY(check_start_batch(run));
                SA_TRY(step->restore_run_count(static_cast<std::size_t>(run.start.step_runs)));

                const std::int64_t epochs = run.spec.epochs;
                const std::size_t batches = run.dataset.batch_count();

                Vision::Training::AnchorLoopOptions options;
                options.epochs = epochs;
                options.schedule = run.schedule;
                options.first_epoch = static_cast<std::int64_t>(run.start.epoch);
                options.first_batch = static_cast<std::size_t>(run.start.batch);
                options.on_update = [epochs, batches](
                                        const Vision::Training::AnchorProgress& progress)
                {
                    if (is_reported_epoch(progress.epoch, epochs)
                        && progress.batch + 1 == batches)
                        print_epoch(progress, epochs);
                };

                // The progress lines carry what the report holds, so only its status is read.
                SA_TRY(Vision::Training::run_anchor_training(*step, run.dataset, options)
                           .status());

                const std::size_t forward_compiles = step->training().lead()->compile_count();
                SA_FAIL_IF(forward_compiles != 1, StatusCode::InvalidDescription,
                           "the part of the step before its heads was compiled {} times",
                           forward_compiles);

                RunSummary summary;
                summary.compile_count = step->training().step().compile_count();
                summary.progress = finished_progress(run, step->run_count());
                return summary;
            }

            /** @brief Prints the loss and the terms of the last update of an anchor-free epoch. */
            void print_free_epoch(const Vision::Training::FreeProgress& progress,
                                  std::int64_t epochs)
            {
                std::cout << SushiLog::format_text(
                    "  epoch {}/{} loss {}   box {}   class {}   dfl {}   rate {}\n",
                    SushiLog::right(progress.epoch, 3), SushiLog::left(epochs, 3),
                    SushiLog::fixed(progress.terms.total, 6),
                    SushiLog::fixed(progress.terms.box, 6),
                    SushiLog::fixed(progress.terms.classes, 6),
                    SushiLog::fixed(progress.terms.distribution, 6),
                    SushiLog::general(progress.rate, 6));
            }

            /**
             * @brief Returns the dotted name the detect row's state hangs under, found as the
             *        one entry that ends in the first one-to-many box bias.
             * @return The prefix, empty for a detect row at the root, or InvalidDescription
             *         when no entry or more than one ends so.
             */
            [[nodiscard]] Result<std::string> find_free_head_prefix(const NN::ModelState& state)
            {
                const std::string suffix =
                    std::string(Vision::Detect::Free::K_DETECT_BOX_BRANCH) + ".0."
                    + std::to_string(Vision::Detect::Free::K_DETECT_LAST_CONV_INDEX) + ".bias";

                std::vector<std::string> found;
                for (const NN::StateEntry& entry : state.entries())
                {
                    const std::string& name = entry.name;
                    if (name == suffix)
                        found.emplace_back();
                    else if (name.size() > suffix.size() + 1
                             && name.compare(name.size() - suffix.size(), suffix.size(), suffix)
                                    == 0
                             && name[name.size() - suffix.size() - 1] == '.')
                        found.push_back(name.substr(0, name.size() - suffix.size() - 1));
                }

                SA_FAIL_IF(found.size() != 1, StatusCode::InvalidDescription,
                           "the model state holds {} entries ending in '{}'; an anchor-free "
                           "model holds one detect row",
                           found.size(), suffix);
                return found.front();
            }

            /**
             * @brief Trains an anchor-free detector through FreeTrainingStep, first starting
             *        the last biases from the strides when no checkpoint or weights are read.
             * @return The compile count and where the run stands; InvalidDescription when the
             *         part before the ports was not compiled once; or the first failure of the
             *         objective, the bias start, the step or run_free_training.
             */
            [[nodiscard]] Result<RunSummary> run_anchor_free(PreparedRun& run)
            {
                // The registry returns the interface; the step takes the type behind it.
                SA_TRY_ASSIGN(const Vision::Training::FreeObjective objective,
                              Vision::Training::build_free_objective(
                                  run.given, run.spec.objective_attributes));

                if (run.spec.resume.empty() && run.spec.weights.empty())
                {
                    SA_TRY_ASSIGN(const std::string prefix, find_free_head_prefix(run.state));
                    SA_TRY(Vision::Training::initialize_free_biases(objective.facts(), prefix,
                                                                    run.state));
                }

                SA_TRY_ASSIGN(
                    const std::unique_ptr<Vision::Training::FreeTrainingStep> step,
                    Vision::Training::FreeTrainingStep::make(
                        run.engine, objective, run.built, run.pool, run.optimizer,
                        std::move(run.updates)));

                SA_TRY(check_start_batch(run));
                SA_TRY(step->restore_run_count(static_cast<std::size_t>(run.start.step_runs)));

                const std::int64_t epochs = run.spec.epochs;
                const std::size_t batches = run.dataset.batch_count();

                Vision::Training::FreeLoopOptions options;
                options.epochs = epochs;
                options.schedule = run.schedule;
                options.first_epoch = static_cast<std::int64_t>(run.start.epoch);
                options.first_batch = static_cast<std::size_t>(run.start.batch);
                options.on_update = [epochs, batches](
                                        const Vision::Training::FreeProgress& progress)
                {
                    if (is_reported_epoch(progress.epoch, epochs)
                        && progress.batch + 1 == batches)
                        print_free_epoch(progress, epochs);
                };

                // The progress lines carry what the report holds, so only its status is read.
                SA_TRY(Vision::Training::run_free_training(*step, run.dataset, options).status());

                const std::size_t forward_compiles = step->training().lead()->compile_count();
                SA_FAIL_IF(forward_compiles != 1, StatusCode::InvalidDescription,
                           "the part of the step before its ports was compiled {} times",
                           forward_compiles);

                RunSummary summary;
                summary.compile_count = step->training().step().compile_count();
                summary.progress = finished_progress(run, step->run_count());
                return summary;
            }

            /** @brief Holds the score a detection must exceed to be printed. */
            constexpr float K_SCORE_THRESHOLD = 0.25f;

            /** @brief Holds the suppression measure above which the lower-scoring box goes. */
            constexpr float K_SUPPRESSION_THRESHOLD = 0.45f;

            /** @brief Holds the entries of a decoded box: centre x, centre y, width, height. */
            constexpr std::size_t K_BOX_ENTRIES = 4;

            /** @brief Holds one head of a task bound to the tensor the step publishes it in. */
            struct BoundHead
            {
                /** @brief The published tensor, `[B, H, W, A * (5 + C)]`; the pool owns it. */
                const SushiBLAS::Tensor* tensor = nullptr;

                /** @brief The batch and the grid of the head. */
                Vision::Detect::Anchor::HeadShape shape;

                /** @brief The rows one image of the head decodes to. */
                std::size_t rows = 0;
            };

            /**
             * @brief Binds every head of @p task to the output of @p run it names.
             * @return The heads in task order; the failure of validate_head or of the output
             *         lookup; InvalidDescription for an output that is not a head of the task.
             */
            [[nodiscard]] Result<std::vector<BoundHead>> bind_heads(
                const Vision::Detect::Anchor::AnchorTask& task, const PreparedInference& run)
            {
                std::vector<BoundHead> heads;
                heads.reserve(task.heads.size());
                for (std::size_t k = 0; k < task.heads.size(); ++k)
                {
                    const Vision::Detect::Anchor::AnchorTaskHead& head = task.heads[k];
                    const auto index = static_cast<std::int64_t>(k);
                    SA_TRY(Vision::Detect::Anchor::validate_head(head.options, task.layout,
                                                                 index));
                    SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor,
                                  run.step.output(head.output));

                    const std::int64_t channels = Vision::Detect::Anchor::count_head_channels(
                        head.options, task.layout, index);
                    SA_FAIL_IF(tensor->rank != 4 || tensor->shape[3] != channels,
                               StatusCode::InvalidDescription,
                               "'{}' publishes '{}' with rank {}, but head {} of its task reads "
                               "[batch, rows, columns, {}]",
                               run.model_path.string(), head.output, tensor->rank, k, channels);

                    BoundHead bound;
                    bound.tensor = tensor;
                    bound.shape = Vision::Detect::Anchor::HeadShape{
                        tensor->shape[0], tensor->shape[1], tensor->shape[2]};
                    bound.rows = static_cast<std::size_t>(
                        Vision::Detect::Anchor::count_head_rows(bound.shape, task.layout, index));
                    heads.push_back(bound);
                }

                return heads;
            }

            /**
             * @brief Decodes every head of one batch into `[B, rows, 4]` and `[B, rows, C]`.
             * @param boxes Receives the boxes in pixels of the network input, head after head
             *              within an image.
             * @param scores Receives objectness times class, same rows.
             * @pre Both vectors hold the rows of every head for every image of the batch.
             */
            void decode_batch(const Vision::Detect::Anchor::AnchorTask& task,
                              const std::vector<BoundHead>& heads, const Shape& input,
                              std::size_t rows, std::vector<float>& boxes,
                              std::vector<float>& scores)
            {
                const auto classes = static_cast<std::size_t>(task.num_classes);
                const float extents[] = {static_cast<float>(input[1]),
                                         static_cast<float>(input[0])};
                std::vector<float> head_boxes;
                std::vector<float> head_scores;
                std::size_t first_row = 0;
                for (std::size_t k = 0; k < heads.size(); ++k)
                {
                    const BoundHead& head = heads[k];
                    const auto batch = static_cast<std::size_t>(head.shape.batch);
                    head_boxes.resize(batch * head.rows * K_BOX_ENTRIES);
                    head_scores.resize(batch * head.rows * classes);
                    Vision::Detect::Anchor::decode_head(
                        SushiRuntime::span<const float>(
                            head.tensor->data_as<float>(),
                            static_cast<std::size_t>(head.tensor->num_elements)),
                        head.shape, task.heads[k].options, task.layout,
                        static_cast<std::int64_t>(k), head_boxes, head_scores);

                    for (std::size_t image = 0; image < batch; ++image)
                    {
                        const std::size_t to = image * rows + first_row;
                        const std::size_t from = image * head.rows;
                        for (std::size_t at = 0; at < head.rows * K_BOX_ENTRIES; ++at)
                            boxes[to * K_BOX_ENTRIES + at] =
                                head_boxes[from * K_BOX_ENTRIES + at] * extents[at % 2];
                        for (std::size_t at = 0; at < head.rows * classes; ++at)
                            scores[to * classes + at] = head_scores[from * classes + at];
                    }

                    first_row += head.rows;
                }
            }

            /**
             * @brief Returns how many leading rows of @p detections score above @p threshold.
             * @pre The rows are in descending score, as select_nms_detections writes them.
             */
            [[nodiscard]] std::size_t count_confident(const std::vector<float>& detections,
                                                      float threshold)
            {
                const std::size_t rows = detections.size() / Vision::Detect::K_DETECTION_VALUES;
                std::size_t count = 0;
                while (count < rows
                       && detections[count * Vision::Detect::K_DETECTION_VALUES
                                     + Vision::Detect::K_DETECTION_SCORE]
                              > threshold)
                    ++count;
                return count;
            }

            /** @brief Prints the first @p count rows of the detections of image @p image. */
            void print_detections(std::size_t image, const std::vector<float>& detections,
                                  std::size_t count)
            {
                std::cout << SushiLog::format_text("image {}: {} detections\n", image, count);
                for (std::size_t at = 0; at < count; ++at)
                {
                    const float* const row =
                        detections.data() + at * Vision::Detect::K_DETECTION_VALUES;
                    std::cout << SushiLog::format_text(
                        "  class {}  score {}  box {} {} {} {}\n",
                        SushiLog::right(
                            static_cast<std::int64_t>(row[Vision::Detect::K_DETECTION_CLASS]), 3),
                        SushiLog::fixed(row[Vision::Detect::K_DETECTION_SCORE], 4),
                        SushiLog::right(SushiLog::fixed(row[0], 2), 8),
                        SushiLog::right(SushiLog::fixed(row[1], 2), 8),
                        SushiLog::right(SushiLog::fixed(row[2], 2), 8),
                        SushiLog::right(SushiLog::fixed(row[3], 2), 8));
                }
            }

            /** @brief Prints the mean average precision at IoU 0.5 of the images in @p metric. */
            void print_mean_average_precision(const Vision::Detect::AveragePrecision& metric)
            {
                const Vision::Detect::MetricsSummary summary = metric.summarize();
                std::cout << SushiLog::format_text(
                    "Mean average precision at IoU 0.5 : {} over {} boxes of {} images, "
                    "detections scored above {}\n",
                    SushiLog::fixed(summary.map50, 4), summary.instances, summary.images,
                    Vision::Detect::K_VALIDATION_CONFIDENCE);
            }

            /**
             * @brief Decodes the heads of an anchor model over every batch and prints the
             *        detections that pass the score threshold and the suppression of the task,
             *        then the mean average precision against the labels of the split.
             * @return Success; the failure of the task section, of bind_heads, of a read, of
             *         a step or of the truth roles; InvalidDescription for an input that is
             *         not height, width, depth.
             */
            [[nodiscard]] Status report_anchor_detections(PreparedInference& run)
            {
                SA_TRY_ASSIGN(const Vision::Detect::Anchor::AnchorTask task,
                              Vision::Detect::Anchor::read_anchor_task(run.model.task));
                const Shape& input = run.model.input_shape;
                SA_FAIL_IF(input.rank() != 3, StatusCode::InvalidDescription,
                           "'{}' declares the input shape {}, but an anchor model reads "
                           "[height, width, channels]",
                           run.model_path.string(), input.to_string());

                SA_TRY_ASSIGN(const std::vector<BoundHead> heads, bind_heads(task, run));
                std::size_t rows = 0;
                for (const BoundHead& head : heads)
                    rows += head.rows;

                print_inference_header(
                    run, SushiLog::format_text("{} heads, {} classes", heads.size(),
                                                 task.num_classes));
                std::cout << SushiLog::format_text(
                    "Detections : score above {}, suppression above {}; boxes are x1 y1 x2 y2 "
                    "in pixels of the {} by {} input\n\n",
                    K_SCORE_THRESHOLD, K_SUPPRESSION_THRESHOLD, input[1], input[0]);

                // Suppression runs at the validation confidence; the printout cuts at the score.
                Vision::Detect::NMSConfig config;
                config.iou_threshold = K_SUPPRESSION_THRESHOLD;
                config.criterion = task.criterion;
                config.distance_exponent = task.distance_exponent;

                const auto batch_size = static_cast<std::size_t>(run.dataset.batch_size());
                const auto classes = static_cast<std::size_t>(task.num_classes);
                std::vector<float> boxes(batch_size * rows * K_BOX_ENTRIES);
                std::vector<float> scores(batch_size * rows * classes);
                std::vector<std::vector<float>> detections;
                std::size_t found = 0;

                const bool labelled =
                    Data::find_target(run.dataset.targets(), Data::K_ROLE_BOXES) != nullptr;
                Vision::Detect::AveragePrecision metric;
                Vision::Detect::GroundTruthBatch truth;

                Data::Batch batch;
                for (std::size_t b = 0; b < run.dataset.batch_count(); ++b)
                {
                    Profile::mark_period(Profile::Timeline::InferenceBatch);
                    {
                        SA_ZONE("data", "read");
                        SA_TRY(run.dataset.read(b, batch));
                    }
                    SA_TRY(run.step.run(batch.inputs));
                    {
                        SA_ZONE("eval", "decode");
                        decode_batch(task, heads, input, rows, boxes, scores);
                    }

                    {
                        SA_ZONE("eval", "nms");
                        Vision::Detect::select_nms_detections(
                            boxes, scores,
                            Vision::Detect::PredictionShape{
                                run.dataset.batch_size(), static_cast<std::int64_t>(rows),
                                task.num_classes},
                            config, detections);
                    }
                    for (std::size_t image = 0; image < detections.size(); ++image)
                    {
                        const std::size_t count = count_confident(detections[image], K_SCORE_THRESHOLD);
                        print_detections(b * batch_size + image, detections[image], count);
                        found += count;
                    }

                    if (labelled)
                    {
                        {
                            SA_ZONE("host", "feed");
                            SA_TRY(Vision::Training::truth_from_roles(batch, truth));
                        }
                        SA_ZONE("eval", "ap");
                        SA_TRY(Vision::Detect::add_batch(metric, detections, truth));
                    }
                }

                std::cout << SushiLog::format_text(
                    "\n{} detections over {} images.\n", found,
                    run.dataset.batch_count() * batch_size);
                if (labelled)
                    print_mean_average_precision(metric);
                else
                    std::cout << "Mean average precision : not computed, the data carries no "
                                 "boxes role\n";
                return Status();
            }

            /**
             * @brief Lists the shape of every output @p run publishes, as the step holds it.
             * @return The shapes in publishing order, or the failure of an output lookup.
             */
            [[nodiscard]] Result<std::vector<Train::OutputShape>> list_published_shapes(
                const PreparedInference& run)
            {
                std::vector<Train::OutputShape> shapes;
                shapes.reserve(run.outputs.size());
                for (const NN::ModelOutputs::Entry& output : run.outputs)
                {
                    SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor,
                                  run.step.output(output.first));
                    shapes.push_back(Train::OutputShape{
                        output.first,
                        Shape(SushiRuntime::span<const std::int64_t>(
                            tensor->shape.data(), static_cast<std::size_t>(tensor->rank)))});
                }

                return shapes;
            }

            /** @brief Returns the values of the tensor @p run publishes under @p name. */
            [[nodiscard]] Result<SushiRuntime::span<const float>> published_values(
                const PreparedInference& run, const std::string& name)
            {
                SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, run.step.output(name));
                return SushiRuntime::span<const float>(
                    tensor->data_as<float>(), static_cast<std::size_t>(tensor->num_elements));
            }

            /**
             * @brief Binds the one-to-many box and class outputs of every level of @p facts.
             * @return One entry per level, finest first, or the failure of an output lookup.
             */
            [[nodiscard]] Result<std::vector<Vision::Training::FreeLevelLogits>>
            bind_free_levels(const Vision::Training::FreeModelFacts& facts,
                             const PreparedInference& run)
            {
                std::vector<Vision::Training::FreeLevelLogits> levels;
                levels.reserve(facts.levels.size());
                for (const Vision::Training::FreeLevelFacts& level : facts.levels)
                {
                    Vision::Training::FreeLevelLogits bound;
                    SA_TRY_ASSIGN(bound.boxes, published_values(run, level.one_to_many.boxes));
                    SA_TRY_ASSIGN(bound.scores,
                                  published_values(run, level.one_to_many.scores));
                    levels.push_back(bound);
                }

                return levels;
            }

            /**
             * @brief Decodes the one-to-many outputs of an anchor-free model over every batch
             *        and prints the detections past the task's score and suppression, then
             *        the mean average precision against the labels of the split.
             * @return Success; the failure of the task section, of the facts, of a lookup, of
             *         a read, of a step or of the truth roles; InvalidDescription for an input
             *         that is not height, width, depth, or a batch the data does not read.
             */
            [[nodiscard]] Status report_free_detections(PreparedInference& run)
            {
                SA_TRY_ASSIGN(const Vision::Detect::Free::FreeTask task,
                              Vision::Detect::Free::read_free_task(run.model.task));
                const Shape& input = run.model.input_shape;
                SA_FAIL_IF(input.rank() != 3, StatusCode::InvalidDescription,
                           "'{}' declares the input shape {}, but an anchor-free model reads "
                           "[height, width, channels]",
                           run.model_path.string(), input.to_string());

                SA_TRY_ASSIGN(const std::vector<Train::OutputShape> shapes,
                              list_published_shapes(run));
                SA_TRY_ASSIGN(const Vision::Training::FreeModelFacts facts,
                              Vision::Training::derive_free_model_facts(input, shapes));
                SA_TRY_ASSIGN(const std::vector<Vision::Training::FreeLevelLogits> levels,
                              bind_free_levels(facts, run));

                const std::int64_t batch_size = run.dataset.batch_size();
                const std::size_t level_values = static_cast<std::size_t>(
                    facts.levels.front().height * facts.levels.front().width * facts.classes);
                const std::size_t published_batch = levels.front().scores.size() / level_values;
                SA_FAIL_IF(published_batch != static_cast<std::size_t>(batch_size),
                           StatusCode::InvalidDescription,
                           "'{}' publishes batches of {} images, but the data reads {}",
                           run.model_path.string(), published_batch, batch_size);

                SA_TRY_ASSIGN(Vision::Training::FreeBatchDecoder decoder,
                              Vision::Training::FreeBatchDecoder::make(facts, batch_size));
                const auto anchors = static_cast<std::size_t>(decoder.anchors());
                const auto images = static_cast<std::size_t>(batch_size);

                print_inference_header(
                    run, SushiLog::format_text("{} levels, {} classes, reg_max {}",
                                               facts.levels.size(), facts.classes,
                                               facts.reg_max));
                std::cout << SushiLog::format_text(
                    "Detections : score above {}, suppression above {}; boxes are x1 y1 x2 y2 "
                    "in pixels of the {} by {} input\n\n",
                    task.score_threshold, task.suppression_threshold, input[1], input[0]);

                // Suppression runs at the validation confidence; the printout cuts at the score.
                Vision::Detect::NMSConfig config;
                config.iou_threshold = task.suppression_threshold;
                config.label_mode = Vision::Detect::LabelMode::BestClass;
                config.criterion = task.criterion;
                config.distance_exponent = task.distance_exponent;

                std::vector<float> boxes(images * anchors * K_BOX_ENTRIES);
                std::vector<float> scores(images * anchors
                                          * static_cast<std::size_t>(facts.classes));
                std::vector<std::vector<float>> detections;
                std::size_t found = 0;

                const bool labelled =
                    Data::find_target(run.dataset.targets(), Data::K_ROLE_BOXES) != nullptr;
                Vision::Detect::AveragePrecision metric;
                Vision::Detect::GroundTruthBatch truth;

                Data::Batch batch;
                for (std::size_t b = 0; b < run.dataset.batch_count(); ++b)
                {
                    Profile::mark_period(Profile::Timeline::InferenceBatch);
                    {
                        SA_ZONE("data", "read");
                        SA_TRY(run.dataset.read(b, batch));
                    }
                    SA_TRY(run.step.run(batch.inputs));
                    {
                        SA_ZONE("eval", "decode");
                        SA_TRY(decoder.decode(levels, boxes, scores));
                    }

                    {
                        SA_ZONE("eval", "nms");
                        Vision::Detect::select_nms_detections(
                            boxes, scores,
                            Vision::Detect::PredictionShape{
                                batch_size, static_cast<std::int64_t>(anchors), facts.classes},
                            config, detections);
                    }
                    for (std::size_t image = 0; image < detections.size(); ++image)
                    {
                        const std::size_t count =
                            count_confident(detections[image], task.score_threshold);
                        print_detections(b * images + image, detections[image], count);
                        found += count;
                    }

                    if (labelled)
                    {
                        {
                            SA_ZONE("host", "feed");
                            SA_TRY(Vision::Training::truth_from_roles(batch, truth));
                        }
                        SA_ZONE("eval", "ap");
                        SA_TRY(Vision::Detect::add_batch(metric, detections, truth));
                    }
                }

                std::cout << SushiLog::format_text("\n{} detections over {} images.\n", found,
                                                   run.dataset.batch_count() * images);
                if (labelled)
                    print_mean_average_precision(metric);
                else
                    std::cout << "Mean average precision : not computed, the data carries no "
                                 "boxes role\n";
                return Status();
            }

        } // namespace

        Status register_vision_domain(Registries& registries)
        {
            SA_TRY(Vision::register_nodes(registries.nodes));
            SA_TRY(Vision::register_datasets(registries.datasets));
            SA_TRY(Vision::register_objectives(registries.objectives));
            SA_TRY(registries.runs.add(
                RunEntry{Vision::Detect::Anchor::K_ANCHOR_OBJECTIVE_TYPE, &run_anchor}));
            SA_TRY(registries.runs.add(
                RunEntry{Vision::Training::K_FREE_OBJECTIVE_TYPE, &run_anchor_free}));
            SA_TRY(registries.inferences.add(InferenceEntry{
                Vision::Detect::Anchor::K_ANCHOR_TASK_TYPE, &report_anchor_detections}));
            SA_TRY(registries.inferences.add(InferenceEntry{
                Vision::Detect::Free::K_FREE_TASK_TYPE, &report_free_detections}));
            SA_TRY(Vision::register_importers(registries.importers));

            return Status();
        }

    } // namespace Apps
} // namespace SushiAI
