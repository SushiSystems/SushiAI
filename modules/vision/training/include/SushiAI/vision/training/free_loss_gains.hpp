/**************************************************************************/
/* free_loss_gains.hpp                                                    */
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

/**
 * @file free_loss_gains.hpp
 * @brief Declares the gains of the anchor-free detection loss and their training-file keys.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <string_view>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds the factor of each term of the anchor-free detection loss. */
    struct FreeLossGains
    {
        float box_gain = 7.5f;          /**< Factor of the 1 - CIoU term. */
        float class_gain = 0.5f;        /**< Factor of the class cross-entropy term. */
        float distribution_gain = 1.5f; /**< Factor of the distribution focal term. */
        float l1_gain = 0.0f;           /**< Factor of the L1 distance term; 0 leaves it out. */
    };

    /** @brief Lists the keys read_free_loss_gains() reads, for a section's allowed keys. */
    inline constexpr std::array<std::string_view, 4> K_FREE_LOSS_GAIN_KEYS = {
        "box_gain", "class_gain", "distribution_gain", "l1_gain"};

    /**
     * @brief Checks that every gain of @p gains is finite and not negative.
     * @return Success, or InvalidDescription naming the first gain that is not.
     */
    [[nodiscard]] Status validate_free_loss_gains(const FreeLossGains& gains) noexcept;

    /**
     * @brief Reads the gain keys of an objective section; every other key is left to its caller.
     * @param section The "objective" object of a training file.
     * @return The gains, a key left out keeping its default, or InvalidDescription naming
     *         the key: a section that is no object, a value that is no finite float32 number,
     *         or a negative gain.
     */
    [[nodiscard]] Result<FreeLossGains> read_free_loss_gains(const Config::Value& section) noexcept;

} // namespace SushiAI::Vision::Training
