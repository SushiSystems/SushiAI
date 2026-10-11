/**************************************************************************/
/* step_leaf.hpp                                                          */
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
 * @file step_leaf.hpp
 * @brief Defines the checks of a leaf a cached generation feeds.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/train/evaluation_step.hpp>

namespace SushiAI::Language::Generation::Detail
{
    /** @brief Names the generator whose step these checks read, in a message. */
    inline constexpr std::string_view K_CACHED_GENERATOR = "CachedGenerator";

    /** @brief Names the rank of every leaf a run feeds. */
    inline constexpr std::int32_t K_LEAF_RANK = 2;

    /** @brief Stands for any positive count of columns in fitting_leaf(). */
    inline constexpr std::int64_t K_ANY_COLUMNS = 0;

    /** @brief Reports whether @p step is fed a leaf named @p name; no leaf has no name. */
    [[nodiscard]] inline bool is_fed(const Train::EvaluationStep& step,
                                     std::string_view name) noexcept
    {
        if (name.empty())
            return false;

        for (const std::string& fed : step.input_names())
        {
            if (fed == name)
                return true;
        }

        return false;
    }

    /**
     * @brief Returns the leaf @p name of @p step once it is known to be [1, columns].
     * @param columns The second extent, or K_ANY_COLUMNS for any positive one.
     * @return InvalidDescription for a leaf that is absent or does not fit.
     */
    [[nodiscard]] inline Result<const SushiBLAS::Tensor*> fitting_leaf(
        const Train::EvaluationStep& step, std::string_view name, std::int64_t columns,
        DType dtype)
    {
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const leaf, step.input(name));
        SA_FAIL_IF(leaf->rank != K_LEAF_RANK || leaf->shape[0] != 1 || leaf->shape[1] < 1 ||
                       (columns != K_ANY_COLUMNS && leaf->shape[1] != columns) ||
                       leaf->dtype != dtype,
                   StatusCode::InvalidDescription,
                   "{}: the leaf '{}' has rank {} and the element type {}; it reads a {} "
                   "leaf of [1, {}]",
                   K_CACHED_GENERATOR, name, leaf->rank, dtype_name(leaf->dtype),
                   dtype_name(dtype),
                   columns == K_ANY_COLUMNS ? std::string("n") : std::to_string(columns));

        return leaf;
    }

} // namespace SushiAI::Language::Generation::Detail
