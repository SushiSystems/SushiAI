/**************************************************************************/
/* detection_report_yolox.cpp                                             */
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
 * @file detection_report_yolox.cpp
 * @brief Implements the `yolox_detection` report row, decoded by select_yolox_detections.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The report table".
 */

#include "detection_reports.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiLog/text/text_format.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/yolox/yolox_decode.hpp>
#include <SushiAI/vision/detect/yolox/yolox_head_contract.hpp>
#include <SushiAI/vision/detect/yolox/yolox_task.hpp>

#include "detection_report.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Holds the rank of a YOLOX port: batch, rows, columns, channels. */
            constexpr std::int64_t K_YOLOX_PORT_RANK = 4;

            /** @brief Holds the ports one level of a YOLOX head publishes. */
            constexpr std::size_t K_YOLOX_PORTS_PER_LEVEL = 3;

            /**
             * @brief Returns the tensor @p run publishes under the port @p role of @p level.
             * @param channels The channel count the port must have, or zero for any.
             */
            [[nodiscard]] Result<const SushiBLAS::Tensor*> yolox_port(
                const PreparedInference& run, std::size_t level, std::string_view role,
                std::int64_t channels)
            {
                const std::string name = Vision::Detect::Yolox::yolox_port_name(level, role);
                SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, run.step.output(name));
                SA_FAIL_IF(tensor->rank != K_YOLOX_PORT_RANK
                               || (channels > 0 && tensor->shape[3] != channels),
                           StatusCode::InvalidDescription,
                           "'{}' publishes '{}' with rank {} and {} channels, but a YOLOX port "
                           "reads [batch, rows, columns, channels]",
                           run.model_path.string(), name, tensor->rank,
                           tensor->rank == K_YOLOX_PORT_RANK ? tensor->shape[3] : 0);
                return tensor;
            }

            /** @brief Returns the values of @p tensor as a span. */
            [[nodiscard]] SushiRuntime::span<const float> values_of(const SushiBLAS::Tensor& tensor)
            {
                return SushiRuntime::span<const float>(
                    tensor.data_as<float>(), static_cast<std::size_t>(tensor.num_elements));
            }

            /**
             * @brief Binds the three ports of every level the model publishes.
             * @param classes Receives the class count of the levels.
             * @param batch Receives the batch the levels publish.
             * @return The levels finest first; InvalidDescription for a port count that is not
             *         three per level, a grid that does not divide the input, or levels that
             *         disagree on the classes or the batch.
             */
            [[nodiscard]] Result<std::vector<Vision::Detect::Yolox::YoloxLevel>> bind_yolox_levels(
                const PreparedInference& run, const Shape& input, std::int64_t& classes,
                std::int64_t& batch)
            {
                SA_FAIL_IF(run.outputs.empty()
                               || run.outputs.size() % K_YOLOX_PORTS_PER_LEVEL != 0,
                           StatusCode::InvalidDescription,
                           "'{}' publishes {} outputs, but a YOLOX head publishes the classes, "
                           "boxes and objectness of each level",
                           run.model_path.string(), run.outputs.size());

                std::vector<Vision::Detect::Yolox::YoloxLevel> levels;
                const std::size_t count = run.outputs.size() / K_YOLOX_PORTS_PER_LEVEL;
                classes = 0;
                batch = 0;
                for (std::size_t level = 0; level < count; ++level)
                {
                    SA_TRY_ASSIGN(const SushiBLAS::Tensor* const scores,
                                  yolox_port(run, level, Vision::Detect::Yolox::K_YOLOX_CLASSES, 0));
                    SA_TRY_ASSIGN(const SushiBLAS::Tensor* const boxes,
                                  yolox_port(run, level, Vision::Detect::Yolox::K_YOLOX_BOXES,
                                             static_cast<std::int64_t>(
                                                 Vision::Detect::Yolox::K_YOLOX_BOX_CHANNELS)));
                    SA_TRY_ASSIGN(const SushiBLAS::Tensor* const objectness,
                                  yolox_port(run, level,
                                             Vision::Detect::Yolox::K_YOLOX_OBJECTNESS,
                                             static_cast<std::int64_t>(
                                                 Vision::Detect::Yolox::
                                                     K_YOLOX_OBJECTNESS_CHANNELS)));

                    const std::int64_t height = scores->shape[1];
                    const std::int64_t width = scores->shape[2];
                    SA_FAIL_IF(height < 1 || width < 1 || input[0] % height != 0
                                   || input[1] % width != 0
                                   || input[0] / height != input[1] / width,
                               StatusCode::InvalidDescription,
                               "'{}' level {} is {} by {} cells, which is no whole stride of the "
                               "{} by {} input",
                               run.model_path.string(), level, height, width, input[0], input[1]);
                    SA_FAIL_IF(level != 0 && (scores->shape[3] != classes
                                              || scores->shape[0] != batch),
                               StatusCode::InvalidDescription,
                               "'{}' level {} disagrees with level 0 on the batch or the classes",
                               run.model_path.string(), level);
                    classes = scores->shape[3];
                    batch = scores->shape[0];

                    Vision::Detect::Yolox::YoloxLevel bound;
                    bound.height = height;
                    bound.width = width;
                    bound.stride = static_cast<float>(input[0] / height);
                    bound.classes = values_of(*scores);
                    bound.boxes = values_of(*boxes);
                    bound.objectness = values_of(*objectness);
                    levels.push_back(bound);
                }

                return levels;
            }

        } // namespace

        Status report_yolox_detections(PreparedInference& run)
        {
            SA_TRY_ASSIGN(Vision::Detect::Yolox::YoloxTask task,
                          Vision::Detect::Yolox::read_yolox_task(run.model.task));
            const Shape& input = run.model.input_shape;
            SA_TRY(check_input_rank(run, input, "a YOLOX model"));

            const float score_threshold =
                run.detection.score_threshold.value_or(task.score_threshold);
            task.suppression.threshold =
                run.detection.suppression_threshold.value_or(task.suppression.threshold);
            const float confidence_floor = run.detection.confidence_floor.value_or(
                Vision::Detect::K_VALIDATION_CONFIDENCE);
            SA_TRY(check_score_floor(run, score_threshold, confidence_floor));

            std::int64_t classes = 0;
            std::int64_t batch_size = 0;
            SA_TRY_ASSIGN(const std::vector<Vision::Detect::Yolox::YoloxLevel> levels,
                          bind_yolox_levels(run, input, classes, batch_size));
            SA_FAIL_IF(batch_size != run.dataset.batch_size(), StatusCode::InvalidDescription,
                       "'{}' publishes batches of {} images, but the data reads {}",
                       run.model_path.string(), batch_size, run.dataset.batch_size());

            print_inference_header(
                run, SushiLog::format_text("{} levels, {} classes", levels.size(), classes));
            print_detection_legend(score_threshold, task.suppression.threshold,
                                   describe_box_frame(run, input));

            // Suppression runs at the validation confidence; the printout cuts at the score.
            Vision::Detect::Yolox::YoloxTask measured = task;
            measured.score_threshold = confidence_floor;

            DetectionReportPlan plan;
            plan.images = static_cast<std::size_t>(batch_size);
            plan.score_threshold = score_threshold;
            plan.confidence_floor = confidence_floor;
            plan.map_to_source = true;

            return run_detection_report(
                run, plan,
                [&](std::vector<std::vector<float>>& detections) -> Status
                {
                    SA_ZONE("eval", "nms");
                    SA_TRY(Vision::Detect::Yolox::select_yolox_detections(
                        SushiRuntime::span<const Vision::Detect::Yolox::YoloxLevel>(
                            levels.data(), levels.size()),
                        batch_size, classes, measured, detections));
                    return Status();
                });
        }

    } // namespace Apps
} // namespace SushiAI
