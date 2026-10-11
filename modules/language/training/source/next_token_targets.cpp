/**************************************************************************/
/* next_token_targets.cpp                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file next_token_targets.cpp
 * @brief Implements the fill of the next-token objective's target leaf.
 * @author Mustafa Garip
 */

#include <SushiAI/language/training/next_token_targets.hpp>

#include <cstdint>
#include <limits>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI::Language::Training
{
    Status fill_next_token_targets(const SushiBLAS::Tensor& labels, std::int64_t vocabulary,
                                   SushiBLAS::Tensor& targets)
    {
        SA_FAIL_IF(labels.dtype != DType::Int32 || targets.dtype != DType::Int32,
                   StatusCode::Incompatible,
                   "The batch's labels are {} and the step's targets are {}; both are read "
                   "as {}",
                   dtype_name(labels.dtype), dtype_name(targets.dtype),
                   dtype_name(DType::Int32));
        SA_FAIL_IF(labels.num_elements != targets.num_elements, StatusCode::Incompatible,
                   "The batch holds {} labels but the compiled step expects {}, one per "
                   "position",
                   labels.num_elements, targets.num_elements);

        const std::int64_t rows = targets.num_elements;
        constexpr std::int64_t K_LARGEST_INDEX = std::numeric_limits<std::int32_t>::max();
        SA_FAIL_IF(vocabulary < 1 || rows > K_LARGEST_INDEX / vocabulary,
                   StatusCode::Incompatible,
                   "{} positions over a vocabulary of {} cannot be indexed in {}", rows,
                   vocabulary, dtype_name(DType::Int32));

        const std::int32_t* const label = labels.data_as<std::int32_t>();
        for (std::int64_t row = 0; row < rows; ++row)
        {
            SA_FAIL_IF(label[row] < 0 || label[row] >= vocabulary, StatusCode::Incompatible,
                       "The label of position {} is {} but the vocabulary holds {} tokens",
                       row, label[row], vocabulary);
        }

        // Every label was checked, so a refused batch leaves the leaf as it was.
        std::int32_t* const target = targets.data_as<std::int32_t>();
        for (std::int64_t row = 0; row < rows; ++row)
            target[row] = static_cast<std::int32_t>(row * vocabulary + label[row]);

        return Status();
    }

} // namespace SushiAI::Language::Training
