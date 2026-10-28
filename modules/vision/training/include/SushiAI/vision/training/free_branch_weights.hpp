/**************************************************************************/
/* free_branch_weights.hpp                                                */
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
 * @file free_branch_weights.hpp
 * @brief Declares the weights that mix the one-to-many and one-to-one detection losses.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Holds the weight of the one-to-many loss at the first and the last epoch; the
     *        one-to-one loss takes the rest of 1.
     * @see docs/agent/2026_10_26_ONE_TO_ONE_LOSS/PLAN.md
     */
    struct FreeBranchWeights
    {
        float start = 0.8f; /**< One-to-many weight of the first epoch. */
        float end = 0.1f;   /**< One-to-many weight from the last epoch on. */
    };

    /** @brief Lists the keys read_free_branch_weights() reads, for a section's allowed keys. */
    inline constexpr std::array<std::string_view, 2> K_FREE_BRANCH_WEIGHT_KEYS = {
        "one_to_many_weight_start", "one_to_many_weight_end"};

    /**
     * @brief Checks that both weights are finite and lie in 0..1.
     * @return Success, or InvalidDescription naming the first weight that does not.
     */
    [[nodiscard]] Status validate_free_branch_weights(const FreeBranchWeights& weights) noexcept;

    /**
     * @brief Reads the branch weight keys of an objective section; other keys are left alone.
     * @param section The "objective" object of a training file.
     * @return The weights, a key left out keeping its default, or InvalidDescription naming
     *         the key that is no finite float32 number in 0..1.
     */
    [[nodiscard]] Result<FreeBranchWeights> read_free_branch_weights(const Config::Value& section) noexcept;

    /**
     * @brief Returns the one-to-many weight of epoch @p epoch of a run of @p epochs epochs.
     *
     * The weight moves in a straight line from @c start at epoch 0 to @c end at epoch
     * `epochs - 1`, and stays at @c end after it.
     * @param epoch The epoch index, from 0.
     * @param epochs The epochs of the run; a run of one epoch uses @c start.
     */
    [[nodiscard]] float one_to_many_weight(const FreeBranchWeights& weights, std::int64_t epoch,
                                           std::int64_t epochs) noexcept;

} // namespace SushiAI::Vision::Training
