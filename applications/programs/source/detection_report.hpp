/**************************************************************************/
/* detection_report.hpp                                                   */
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
 * @file detection_report.hpp
 * @brief Declares the batch loop every detection report of `sushiai inference` shares.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The report table".
 */

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>

#include "inference_run.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /** @brief Holds the score a detection must exceed to be printed, when nothing states one. */
        constexpr float K_SCORE_THRESHOLD = 0.25f;

        /** @brief Holds the suppression measure above which the lower-scoring box goes. */
        constexpr float K_SUPPRESSION_THRESHOLD = 0.45f;

        /** @brief Holds the entries of a decoded box: centre x, centre y, width, height. */
        constexpr std::size_t K_BOX_ENTRIES = 4;

        /**
         * @struct DetectionReportPlan
         * @brief Holds what one detection report states about its batches and its printout.
         */
        struct DetectionReportPlan
        {
            /** @brief The images of one batch. */
            std::size_t images = 0;

            /** @brief The score a detection must exceed to be printed. */
            float score_threshold = 0.0f;

            /** @brief The score a detection must exceed to count for the precision. */
            float confidence_floor = 0.0f;

            /** @brief Whether a box is printed in pixels of the source image of its file. */
            bool map_to_source = false;
        };

        /**
         * @brief Turns the batch the step just ran into the detections of each of its images.
         * @param detections Receives one row list per image, in descending score.
         * @return Success, or the failure of the decoding or of the suppression.
         */
        using DecodeDetections = std::function<Status(std::vector<std::vector<float>>& detections)>;

        /**
         * @brief Fails when the input of the model is not height, width, depth.
         * @param model_kind The kind of model with its article, as the message names it.
         * @return Success, or InvalidDescription.
         */
        [[nodiscard]] Status check_input_rank(const PreparedInference& run, const Shape& input,
                                              std::string_view model_kind);

        /**
         * @brief Fails when the score threshold is below the confidence floor.
         * @return Success, or InvalidDescription: a printed detection could already be dropped.
         */
        [[nodiscard]] Status check_score_floor(const PreparedInference& run, float score_threshold,
                                               float confidence_floor);

        /**
         * @brief Returns how the printed boxes are measured: the source image of each file when
         *        the data publishes the `fit` role, the network input otherwise.
         */
        [[nodiscard]] std::string describe_box_frame(const PreparedInference& run,
                                                     const Shape& input);

        /** @brief Prints the line that opens a detection listing: the thresholds and the frame. */
        void print_detection_legend(float score_threshold, float suppression_threshold,
                                    const std::string& frame);

        /**
         * @brief Runs the model over every batch, prints the detections past the score threshold
         *        of each image, then the total and the mean average precision.
         * @param decode Called once per batch after the step ran.
         * @return Success; the failure of a read, of the step, of @p decode or of the truth roles.
         */
        [[nodiscard]] Status run_detection_report(PreparedInference& run,
                                                  const DetectionReportPlan& plan,
                                                  const DecodeDetections& decode);

    } // namespace Apps
} // namespace SushiAI
