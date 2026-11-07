/**************************************************************************/
/* exchange_targets.hpp                                                   */
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
 * @file exchange_targets.hpp
 * @brief Declares the checks of a gradient exchange's targets that read shape and type.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <vector>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /**
         * @brief Checks that @p targets are one tensor per entry of @p layout, each of the
         *        entry's type, layout and shape.
         * @return Success, or InvalidDescription naming the count or the first target that
         *         differs.
         */
        [[nodiscard]] Status check_exchange_targets(const SushiBLAS::PackLayout& layout,
                                                    const std::vector<SushiBLAS::Tensor>& targets);

        /**
         * @brief Checks that no target has a type ordered_sum() cannot sum.
         * @return Success, or Unsupported naming the first uint32 target.
         */
        [[nodiscard]] Status check_exchange_dtypes(const std::vector<SushiBLAS::Tensor>& targets);

    } // namespace Distribution
} // namespace SushiAI
