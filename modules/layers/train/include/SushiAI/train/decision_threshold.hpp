/**************************************************************************/
/* decision_threshold.hpp                                                 */
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
 * @file decision_threshold.hpp
 * @brief Declares the probability at which a logit reads as a yes, and its reader.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        /** @brief Names the key an objective section states its threshold under. */
        inline constexpr std::string_view K_THRESHOLD_KEY = "threshold";

        /** @brief Holds the threshold of an objective section that states none. */
        inline constexpr double K_DEFAULT_THRESHOLD = 0.5;

        /** @brief Returns InvalidDescription unless @p threshold is above 0 and below 1. */
        [[nodiscard]] Status validate_threshold(double threshold);

        /**
         * @brief Reads the optional K_THRESHOLD_KEY of an objective section.
         * @param attributes The keys of the objective; it may state the threshold alone.
         * @param context What to call the objective in a diagnostic.
         * @return The threshold, K_DEFAULT_THRESHOLD when the key is absent, or
         *         InvalidDescription for another key, a value that is no number, or what
         *         validate_threshold() refuses.
         */
        [[nodiscard]] Result<double> read_threshold(const Config::Value& attributes,
                                                    std::string_view context);

        /**
         * @brief Returns log(p / (1 - p)), the logit whose sigmoid is @p probability.
         * @pre validate_threshold() accepted @p probability.
         */
        [[nodiscard]] double logit_of(double probability) noexcept;

    } // namespace Train
} // namespace SushiAI
