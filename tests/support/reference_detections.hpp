/**************************************************************************/
/* reference_detections.hpp                                               */
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
 * @file reference_detections.hpp
 * @brief Declares the detections a reference runtime made, how a file states them and how a test
 *        compares its own against them.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <support/tolerance.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Holds the entries of one reference detection: class, score, x1, y1, x2, y2. */
        inline constexpr std::size_t K_REFERENCE_VALUES = 6;

        /** @brief Holds the entries of a box: x1, y1, x2, y2. */
        inline constexpr std::size_t K_BOX_ENTRIES = 4;

        /**
         * @brief Holds the bound of a score and of a box corner as a fraction of the input side:
         *        2048 u, the row bound of the Darknet parity cases.
         */
        inline const double K_ROW_BOUND = 2048.0 * TestSupport::unit_roundoff<float>();

        /** @brief Holds one detection as both sides are compared, corners in pixels. */
        struct Detection
        {
            std::int64_t class_index = 0; /**< The class. */
            double score = 0.0;           /**< The score the detection ranks by. */
            double corners[4] = {};       /**< x1, y1, x2, y2. */
        };

        /** @brief Returns whether @p a sorts before @p b: by class, then by falling score. */
        [[nodiscard]] inline bool comes_first(const Detection& a, const Detection& b)
        {
            return a.class_index != b.class_index ? a.class_index < b.class_index
                                                  : a.score > b.score;
        }

        /**
         * @brief Reads the "detections" array of one image entry of a reference file.
         * @return The rows, or InvalidFile for a row that does not hold K_REFERENCE_VALUES values.
         */
        [[nodiscard]] inline Result<std::vector<Detection>> read_expected(
            const Config::Value& entry)
        {
            SA_TRY_ASSIGN(const Config::Value& rows, entry.at("detections", "reference"));
            SA_TRY_ASSIGN(const std::vector<Config::Value>& list, rows.as_array("detections"));
            std::vector<Detection> out;
            for (const Config::Value& row : list)
            {
                SA_TRY_ASSIGN(const std::vector<Config::Value>& values, row.as_array("detection"));
                SA_FAIL_IF(values.size() != K_REFERENCE_VALUES, StatusCode::InvalidFile,
                           "a reference detection holds {} values", values.size());
                Detection detection;
                SA_TRY_ASSIGN(const double index, values[0].as_real("class"));
                detection.class_index = static_cast<std::int64_t>(index);
                SA_TRY_ASSIGN(detection.score, values[1].as_real("score"));
                for (std::size_t k = 0; k < K_BOX_ENTRIES; ++k)
                {
                    SA_TRY_ASSIGN(detection.corners[k], values[k + 2].as_real("corner"));
                }
                out.push_back(detection);
            }
            return out;
        }

        /**
         * @brief Expects @p found to be @p expected, row by row, within K_ROW_BOUND.
         * @param side The input side the corner bound scales with.
         * @param image Names the image in a failure.
         */
        inline void expect_same_detections(std::vector<Detection> found,
                                           std::vector<Detection> expected, double side,
                                           const std::string& image)
        {
            std::sort(found.begin(), found.end(), comes_first);
            std::sort(expected.begin(), expected.end(), comes_first);
            ASSERT_EQ(found.size(), expected.size()) << image;
            for (std::size_t k = 0; k < found.size(); ++k)
            {
                EXPECT_EQ(found[k].class_index, expected[k].class_index) << image << " row " << k;
                EXPECT_NEAR(found[k].score, expected[k].score, K_ROW_BOUND)
                    << image << " row " << k;
                for (std::size_t c = 0; c < K_BOX_ENTRIES; ++c)
                {
                    EXPECT_NEAR(found[k].corners[c], expected[k].corners[c], K_ROW_BOUND * side)
                        << image << " row " << k << " corner " << c;
                }
            }
        }
    } // namespace Test
} // namespace SushiAI
