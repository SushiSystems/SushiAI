/**************************************************************************/
/* exchange_targets.cpp                                                   */
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
 * @file exchange_targets.cpp
 * @brief Implements the checks of the tensors a gradient exchange sums into.
 * @author Mustafa Garip
 */

#include <SushiAI/distribution/exchange_targets.hpp>

#include <cstddef>
#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        Status check_exchange_targets(const SushiBLAS::PackLayout& layout,
                                      const std::vector<SushiBLAS::Tensor>& targets)
        {
            const std::vector<SushiBLAS::PackEntry>& entries = layout.entries();
            SA_FAIL_IF(targets.size() != entries.size(), StatusCode::InvalidDescription,
                       "The result pack holds {} tensors and {} targets were given",
                       entries.size(), targets.size());
            for (std::size_t index = 0; index < targets.size(); ++index)
            {
                SA_FAIL_IF(!SushiBLAS::PackLayout::matches(entries[index], targets[index]),
                           StatusCode::InvalidDescription,
                           "Target {} differs from its result entry in type, layout or shape",
                           index);
            }
            return Status();
        }

        Status check_exchange_dtypes(const std::vector<SushiBLAS::Tensor>& targets)
        {
            for (std::size_t index = 0; index < targets.size(); ++index)
            {
                SA_FAIL_IF(targets[index].dtype == SushiBLAS::Core::DataType::UInt32,
                           StatusCode::Unsupported,
                           "Target {} is uint32, which ordered_sum() does not sum", index);
            }
            return Status();
        }

    } // namespace Distribution
} // namespace SushiAI
