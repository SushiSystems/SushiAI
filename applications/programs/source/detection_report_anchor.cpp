/**************************************************************************/
/* detection_report_anchor.cpp                                            */
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
 * @file detection_report_anchor.cpp
 * @brief Implements the `anchor_detection` report row: heads decoded on the host.
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
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/head_decode.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>

#include "detection_report.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
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

        } // namespace

        Status report_anchor_detections(PreparedInference& run)
        {
            SA_TRY_ASSIGN(const Vision::Detect::Anchor::AnchorTask task,
                          Vision::Detect::Anchor::read_anchor_task(run.model.task));
            const Shape& input = run.model.input_shape;
            SA_TRY(check_input_rank(run, input, "an anchor model"));

            const float score_threshold =
                run.detection.score_threshold.value_or(K_SCORE_THRESHOLD);
            const float suppression_threshold =
                run.detection.suppression_threshold.value_or(K_SUPPRESSION_THRESHOLD);
            const float confidence_floor = run.detection.confidence_floor.value_or(
                Vision::Detect::K_VALIDATION_CONFIDENCE);
            SA_TRY(check_score_floor(run, score_threshold, confidence_floor));

            SA_TRY_ASSIGN(const std::vector<BoundHead> heads, bind_heads(task, run));
            std::size_t rows = 0;
            for (const BoundHead& head : heads)
                rows += head.rows;

            print_inference_header(
                run, SushiLog::format_text("{} heads, {} classes", heads.size(),
                                             task.num_classes));
            print_detection_legend(score_threshold, suppression_threshold,
                                   describe_box_frame(run, input));

            // Suppression runs at the validation confidence; the printout cuts at the score.
            Vision::Detect::NMSConfig config;
            config.iou_threshold = suppression_threshold;
            config.confidence_threshold = confidence_floor;
            config.criterion = task.criterion;
            config.distance_exponent = task.distance_exponent;

            DetectionReportPlan plan;
            plan.images = static_cast<std::size_t>(run.dataset.batch_size());
            plan.score_threshold = score_threshold;
            plan.confidence_floor = confidence_floor;
            plan.map_to_source = true;

            const auto classes = static_cast<std::size_t>(task.num_classes);
            std::vector<float> boxes(plan.images * rows * K_BOX_ENTRIES);
            std::vector<float> scores(plan.images * rows * classes);

            return run_detection_report(
                run, plan,
                [&](std::vector<std::vector<float>>& detections) -> Status
                {
                    {
                        SA_ZONE("eval", "decode");
                        decode_batch(task, heads, input, rows, boxes, scores);
                    }

                    SA_ZONE("eval", "nms");
                    SA_TRY(Vision::Detect::select_nms_detections(
                        boxes, scores,
                        Vision::Detect::PredictionShape{
                            run.dataset.batch_size(), static_cast<std::int64_t>(rows),
                            task.num_classes},
                        config, detections));
                    return Status();
                });
        }

    } // namespace Apps
} // namespace SushiAI
