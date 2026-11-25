/**************************************************************************/
/* detection_report_free.cpp                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file detection_report_free.cpp
 * @brief Implements the `anchor_free_detection` report row, decoded on the host.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The report table".
 */

#include "detection_reports.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiLog/text/text_format.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/training/free_batch_decoder.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>

#include "detection_report.hpp"
#include "host_readback.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
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
                const PreparedInference& run, const std::string& name, HostReadback& readback)
            {
                SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, run.step.output(name));
                return readback.bind(*tensor);
            }

            /**
             * @brief Binds the one-to-many box and class outputs of every level of @p facts.
             * @return One entry per level, finest first, or the failure of an output lookup.
             */
            [[nodiscard]] Result<std::vector<Vision::Training::FreeLevelLogits>>
            bind_free_levels(const Vision::Training::FreeModelFacts& facts,
                             const PreparedInference& run, HostReadback& readback)
            {
                std::vector<Vision::Training::FreeLevelLogits> levels;
                levels.reserve(facts.levels.size());
                for (const Vision::Training::FreeLevelFacts& level : facts.levels)
                {
                    Vision::Training::FreeLevelLogits bound;
                    SA_TRY_ASSIGN(bound.boxes,
                                  published_values(run, level.one_to_many.boxes, readback));
                    SA_TRY_ASSIGN(bound.scores,
                                  published_values(run, level.one_to_many.scores, readback));
                    levels.push_back(bound);
                }

                return levels;
            }

        } // namespace

        Status report_free_detections(PreparedInference& run)
        {
            SA_TRY_ASSIGN(const Vision::Detect::Free::FreeTask task,
                          Vision::Detect::Free::read_free_task(run.model.task));
            const Shape& input = run.model.input_shape;
            SA_TRY(check_input_rank(run, input, "an anchor-free model"));

            SA_TRY_ASSIGN(const std::vector<Train::OutputShape> shapes,
                          list_published_shapes(run));
            SA_TRY_ASSIGN(const Vision::Training::FreeModelFacts facts,
                          Vision::Training::derive_free_model_facts(input, shapes));
            HostReadback readback;
            SA_TRY_ASSIGN(const std::vector<Vision::Training::FreeLevelLogits> levels,
                          bind_free_levels(facts, run, readback));

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
            print_detection_legend(
                task.score_threshold, task.suppression_threshold,
                SushiLog::format_text("{} by {} input", input[1], input[0]));

            // Suppression runs at the validation confidence; the printout cuts at the score.
            Vision::Detect::NMSConfig config;
            config.iou_threshold = task.suppression_threshold;
            config.label_mode = Vision::Detect::LabelMode::BestClass;
            config.criterion = task.criterion;
            config.distance_exponent = task.distance_exponent;

            DetectionReportPlan plan;
            plan.images = images;
            plan.score_threshold = task.score_threshold;
            plan.confidence_floor = Vision::Detect::K_VALIDATION_CONFIDENCE;
            plan.map_to_source = false;

            std::vector<float> boxes(images * anchors * K_BOX_ENTRIES);
            std::vector<float> scores(images * anchors * static_cast<std::size_t>(facts.classes));

            return run_detection_report(
                run, plan,
                [&](std::vector<std::vector<float>>& detections) -> Status
                {
                    {
                        SA_ZONE("eval", "decode");
                        SA_TRY(readback.refresh());
                        SA_TRY(decoder.decode(levels, boxes, scores));
                    }

                    SA_ZONE("eval", "nms");
                    SA_TRY(Vision::Detect::select_nms_detections(
                        boxes, scores,
                        Vision::Detect::PredictionShape{
                            batch_size, static_cast<std::int64_t>(anchors), facts.classes},
                        config, detections));
                    return Status();
                });
        }

    } // namespace Apps
} // namespace SushiAI
