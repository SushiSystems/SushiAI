/**************************************************************************/
/* next_token_targets.hpp                                                 */
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
 * @file next_token_targets.hpp
 * @brief Declares the fill of the next-token objective's target leaf from a batch's labels.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI::Language::Training
{
    /**
     * @brief Writes, for each of the R labels, the index of its log-probability among R * V.
     * @param labels INT32 token ids of any shape holding R elements, each in [0, vocabulary).
     * @param vocabulary V, the last extent of the logits.
     * @param targets The INT32 [R] leaf; element r becomes r * vocabulary + labels[r].
     * @return Success, or Incompatible for another element type or count, a label outside
     *         the vocabulary, naming its row, or an R * V that INT32 cannot index.
     */
    [[nodiscard]] Status fill_next_token_targets(const SushiBLAS::Tensor& labels,
                                                 std::int64_t vocabulary,
                                                 SushiBLAS::Tensor& targets);

} // namespace SushiAI::Language::Training
