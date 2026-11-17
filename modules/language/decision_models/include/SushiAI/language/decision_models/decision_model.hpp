/**************************************************************************/
/* decision_model.hpp                                                     */
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
 * @file decision_model.hpp
 * @brief Defines the extents of the decision batches the language tests read and the bound of
 *        their calibration error.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

namespace SushiAI::Language::DecisionModels
{
    /**
     * @brief Holds the extents every batch of a decision set is padded to.
     *
     * The model itself is no extent of this module: applications/examples/decision_model/model.json
     * states it, and a session builds it from that file.
     */
    struct DecisionExtents
    {
        std::int64_t window = 0;  /**< T, the ids every text and every option is padded to. */
        std::int64_t options = 0; /**< K, the options every choice is padded to. */
        std::int64_t batch = 0;   /**< N, the records of a batch. */
    };

    /**
     * @brief Holds the extents the tests read: the window, the options and the batch that
     *        applications/examples/decision_model/train.json states for its data.
     */
    inline constexpr DecisionExtents K_TEST_EXTENTS{8, 4, 16};

    /** @brief Holds the expected calibration error the decision model promises on its test set. */
    inline constexpr double K_ECE_BOUND = 0.05;
} // namespace SushiAI::Language::DecisionModels
