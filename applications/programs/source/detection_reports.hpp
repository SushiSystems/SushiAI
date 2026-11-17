/**************************************************************************/
/* detection_reports.hpp                                                  */
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
 * @file detection_reports.hpp
 * @brief Declares the detection report row of each detector family `sushiai inference` reports.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The report table".
 */

#pragma once

#include <SushiAI/failure/status.hpp>

#include "inference_run.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Decodes the heads of an anchor model over every batch, prints the detections
         *        that pass the score threshold and the suppression of the task, then the mean
         *        average precision against the labels of the split.
         * @return Success; the failure of the task section, of the head binding, of a read, of
         *         a step or of the truth roles; InvalidDescription for an input that is not
         *         height, width, depth.
         */
        [[nodiscard]] Status report_anchor_detections(PreparedInference& run);

        /**
         * @brief Decodes the one-to-many outputs of an anchor-free model over every batch,
         *        prints the detections past the task's score and suppression, then the mean
         *        average precision against the labels of the split.
         * @return Success; the failure of the task section, of the facts, of a lookup, of a
         *         read, of a step or of the truth roles; InvalidDescription for an input that
         *         is not height, width, depth, or a batch the data does not read.
         */
        [[nodiscard]] Status report_free_detections(PreparedInference& run);

        /**
         * @brief Decodes the three ports of every level of a YOLOX model over every batch,
         *        prints the detections that survive, then the mean average precision.
         * @return Success, or the first failure of the task section, a lookup, a read, a step
         *         or the decoding; InvalidDescription for an input that is not height, width,
         *         depth, ports that do not fit, or a score threshold below the confidence floor.
         */
        [[nodiscard]] Status report_yolox_detections(PreparedInference& run);

    } // namespace Apps
} // namespace SushiAI
