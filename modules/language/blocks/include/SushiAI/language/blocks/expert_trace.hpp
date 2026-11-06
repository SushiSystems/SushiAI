/**************************************************************************/
/* expert_trace.hpp                                                       */
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
 * @file expert_trace.hpp
 * @brief Declares the values a block with experts traces and the names of its ports.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Names the port that holds the balance term of a block with experts. */
        inline constexpr std::string_view K_BALANCE_PORT = "balance";

        /** @brief Names the port that holds the share each expert has of a batch. */
        inline constexpr std::string_view K_USAGE_PORT = "usage";

        /** @brief Holds the three values a block with experts traces. */
        struct ExpertTrace
        {
            Graph::ValueID output = Graph::K_NO_VALUE;  /**< The [N, T, dim] result. */
            Graph::ValueID balance = Graph::K_NO_VALUE; /**< The [1, 1] spread of the usage. */
            Graph::ValueID usage = Graph::K_NO_VALUE;   /**< The [1, experts] mean gate. */
        };

    } // namespace Language::Blocks
} // namespace SushiAI
