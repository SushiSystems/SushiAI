/**************************************************************************/
/* pixel_targets.hpp                                                      */
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
 * @file pixel_targets.hpp
 * @brief Declares the fill of the two leaves of the per-pixel cross entropy.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Writes, for each of the R pixels, the index of its log-probability and its weight.
     *
     * A kept pixel r gets r * classes + mask[r] and 1 / kept; an ignored one r * classes and 0.
     * @param indices The INT32 [R] leaf; @p weights The float32 or float64 leaf of R elements.
     * @return Success, or Incompatible, with both leaves unchanged, for a refused label, a
     *         leaf that does not fit, or an R * classes that INT32 cannot index.
     */
    [[nodiscard]] Status fill_pixel_targets(const SushiBLAS::Tensor& mask, std::int64_t classes,
                                            std::int32_t ignore_index,
                                            SushiBLAS::Tensor& indices,
                                            SushiBLAS::Tensor& weights) noexcept;

} // namespace SushiAI::Vision::Training
