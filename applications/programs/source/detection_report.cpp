/**************************************************************************/
/* detection_report.cpp                                                   */
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
 * @file detection_report.cpp
 * @brief Implements the batch loop, the printing and the precision of the detection reports.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The report table".
 */

#include "detection_report.hpp"

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/profile/timelines.hpp>
#include <SushiAI/vision/datasets/detection_collator.hpp>
#include <SushiAI/vision/detect/average_precision.hpp>
#include <SushiAI/vision/detect/batch_average_precision.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/imaging/fit_transform.hpp>
#include <SushiAI/vision/training/truth_from_roles.hpp>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
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

            /**
             * @brief Returns how image @p image of @p batch sat in its source, or nothing.
             * @return Nothing when the batch has no readable fit role or the image has no source.
             */
            std::optional<Vision::Imaging::FitTransform> fit_of(const Data::Batch& batch,
                                                                std::size_t image)
            {
                const Data::BatchTarget* const role = Data::find_target(batch.targets,
                                                                        Data::K_ROLE_FIT);
                if (role == nullptr)
                    return std::nullopt;

                const SushiBLAS::Tensor& tensor = role->values;
                if (tensor.data_or_null() == nullptr || !tensor.is_contiguous()
                    || tensor.dtype != DType::Float32)
                    return std::nullopt;

                const auto* const values = static_cast<const float*>(tensor.data_or_null())
                    + image * static_cast<std::size_t>(Vision::Datasets::K_FIT_VALUES);
                if (values[0] < 1.0f || values[1] < 1.0f)
                    return std::nullopt;

                Vision::Imaging::FitTransform fit;
                fit.source.height = static_cast<std::int32_t>(values[0]);
                fit.source.width = static_cast<std::int32_t>(values[1]);
                fit.scale_x = values[2];
                fit.scale_y = values[3];
                fit.offset_x = values[4];
                fit.offset_y = values[5];
                return fit;
            }

            /**
             * @brief Prints the first @p count rows of the detections of image @p image.
             * @param fit When set, the boxes are mapped to pixels of the source image first.
             */
            void print_detections(std::size_t image, const std::vector<float>& detections,
                                  std::size_t count,
                                  const std::optional<Vision::Imaging::FitTransform>& fit)
            {
                std::cout << SushiLog::format_text("image {}: {} detections\n", image, count);
                for (std::size_t at = 0; at < count; ++at)
                {
                    const float* const row =
                        detections.data() + at * Vision::Detect::K_DETECTION_VALUES;
                    Vision::Imaging::PixelBox box{row[0], row[1], row[2], row[3]};
                    if (fit)
                        box = Vision::Imaging::box_in_source(*fit, box);
                    std::cout << SushiLog::format_text(
                        "  class {}  score {}  box {} {} {} {}\n",
                        SushiLog::right(
                            static_cast<std::int64_t>(row[Vision::Detect::K_DETECTION_CLASS]), 3),
                        SushiLog::fixed(row[Vision::Detect::K_DETECTION_SCORE], 4),
                        SushiLog::right(SushiLog::fixed(box.x1, 2), 8),
                        SushiLog::right(SushiLog::fixed(box.y1, 2), 8),
                        SushiLog::right(SushiLog::fixed(box.x2, 2), 8),
                        SushiLog::right(SushiLog::fixed(box.y2, 2), 8));
                }
            }

            /** @brief Prints the mean average precision at IoU 0.5 of the images in @p metric. */
            void print_mean_average_precision(const Vision::Detect::AveragePrecision& metric,
                                              float confidence_floor)
            {
                const Vision::Detect::MetricsSummary summary = metric.summarize();
                std::cout << SushiLog::format_text(
                    "Mean average precision at IoU 0.5 : {} over {} boxes of {} images, "
                    "detections scored above {}\n",
                    SushiLog::fixed(summary.map50, 4), summary.instances, summary.images,
                    confidence_floor);
            }

        } // namespace

        Status check_input_rank(const PreparedInference& run, const Shape& input,
                                std::string_view model_kind)
        {
            SA_FAIL_IF(input.rank() != 3, StatusCode::InvalidDescription,
                       "'{}' declares the input shape {}, but {} reads "
                       "[height, width, channels]",
                       run.model_path.string(), input.to_string(), model_kind);
            return Status();
        }

        Status check_score_floor(const PreparedInference& run, float score_threshold,
                                 float confidence_floor)
        {
            SA_FAIL_IF(score_threshold < confidence_floor, StatusCode::InvalidDescription,
                       "the score threshold {} of '{}' is below the confidence floor {}, "
                       "so a printed detection could already have been dropped",
                       score_threshold, run.model_path.string(), confidence_floor);
            return Status();
        }

        std::string describe_box_frame(const PreparedInference& run, const Shape& input)
        {
            const bool sourced =
                Data::find_target(run.dataset.targets(), Data::K_ROLE_FIT) != nullptr;
            return sourced ? std::string("source image of each file")
                           : SushiLog::format_text("{} by {} input", input[1], input[0]);
        }

        void print_detection_legend(float score_threshold, float suppression_threshold,
                                    const std::string& frame)
        {
            std::cout << SushiLog::format_text(
                "Detections : score above {}, suppression above {}; boxes are x1 y1 x2 y2 "
                "in pixels of the {}\n\n",
                score_threshold, suppression_threshold, frame);
        }

        Status run_detection_report(PreparedInference& run, const DetectionReportPlan& plan,
                                    const DecodeDetections& decode)
        {
            std::vector<std::vector<float>> detections;
            std::size_t found = 0;

            const bool labelled =
                run.comparison == Comparison::WithTruth
                && Data::find_target(run.dataset.targets(), Data::K_ROLE_BOXES) != nullptr;
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
                SA_TRY(run.step.run_batch(batch));
                SA_TRY(decode(detections));

                for (std::size_t image = 0; image < detections.size(); ++image)
                {
                    const std::size_t count =
                        count_confident(detections[image], plan.score_threshold);
                    print_detections(b * plan.images + image, detections[image], count,
                                     plan.map_to_source ? fit_of(batch, image) : std::nullopt);
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
                                               run.dataset.batch_count() * plan.images);
            if (labelled)
                print_mean_average_precision(metric, plan.confidence_floor);
            else
                std::cout << "Mean average precision : not computed, the data carries no "
                             "boxes to score against\n";
            return Status();
        }

    } // namespace Apps
} // namespace SushiAI
