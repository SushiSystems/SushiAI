/**************************************************************************/
/* pixel_targets.cpp                                                      */
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
 * @file pixel_targets.cpp
 * @brief Implements the fill of the two leaves of the per-pixel cross entropy.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/pixel_targets.hpp>

#include <limits>
#include <type_traits>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/vision/training/pixel_labels.hpp>

#include "host_reals.hpp"
#include "score_indices.hpp"

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the largest index K_GATHER_ROWS addresses a table row by. */
        constexpr std::int64_t K_LARGEST_INDEX = std::numeric_limits<std::int32_t>::max();
    } // namespace

    Status fill_pixel_targets(const SushiBLAS::Tensor& mask, std::int64_t classes,
                              std::int32_t ignore_index, SushiBLAS::Tensor& indices,
                              SushiBLAS::Tensor& weights)
    {
        const std::int64_t pixels = indices.num_elements;
        SA_FAIL_IF(indices.dtype != DType::Int32 || !Detail::holds_reals(weights),
                   StatusCode::Incompatible,
                   "The step's pixel indices are {} and its pixel weights are {}; they are "
                   "written as {} and as a real floating type",
                   dtype_name(indices.dtype), dtype_name(weights.dtype),
                   dtype_name(DType::Int32));
        SA_FAIL_IF(weights.num_elements != pixels, StatusCode::Incompatible,
                   "The step holds {} pixel indices but {} pixel weights; one of each per "
                   "pixel",
                   pixels, weights.num_elements);
        SA_FAIL_IF(classes < 1 || pixels > K_LARGEST_INDEX / classes, StatusCode::Incompatible,
                   "{} pixels over {} classes cannot be indexed in {}", pixels, classes,
                   dtype_name(DType::Int32));
        SA_TRY(validate_pixel_labels(mask, pixels, classes, ignore_index));

        const std::int32_t* const label = mask.data_as<std::int32_t>();
        std::int64_t kept = 0;
        for (std::int64_t pixel = 0; pixel < pixels; ++pixel)
            kept += label[pixel] != ignore_index ? 1 : 0;

        // Every label was checked, so a refused batch leaves both leaves as they were.
        Detail::write_score_indices(label, pixels, classes, ignore_index,
                                    indices.data_as<std::int32_t>());

        const double share = kept == 0 ? 0.0 : 1.0 / static_cast<double>(kept);
        Detail::write_reals(weights, [&](auto* const weight)
        {
            for (std::int64_t pixel = 0; pixel < pixels; ++pixel)
            {
                const double value = label[pixel] != ignore_index ? share : 0.0;
                weight[pixel] = static_cast<std::decay_t<decltype(*weight)>>(value);
            }
        });

        return Status();
    }

} // namespace SushiAI::Vision::Training
