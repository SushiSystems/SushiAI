/**************************************************************************/
/* unlowered.hpp                                                          */
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
 * @file unlowered.hpp
 * @brief Declares the list of operations that have no lowering row.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/ops/ops.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Lists the operations recorded beside the graph, which no node carries. */
        inline constexpr std::array<OpID, 4> K_UNLOWERED = {{
            Ops::K_ADAMW_UPDATE.id,
            Ops::K_SGD_UPDATE.id,
            Ops::K_CLIP_SCALE.id,
            Ops::K_BATCHNORM_UPDATE_STATS.id,
        }};

    } // namespace Graph
} // namespace SushiAI
