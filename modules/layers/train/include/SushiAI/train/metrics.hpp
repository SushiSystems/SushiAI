/**************************************************************************/
/* metrics.hpp                                                            */
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
 * @file metrics.hpp
 * @brief Declares the named numbers an objective reports for a step.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        /** @brief Holds one named number a step produced. */
        struct Metric
        {
            /** @brief What to call it, e.g. "accuracy"; the text must outlive the metric. */
            std::string_view name;

            /** @brief Its value. */
            double value = 0.0;

            /** @brief Whether it reads as a percentage rather than a bare number. */
            bool is_fraction = false;
        };

        /** @brief Holds how many metrics one Metrics value keeps. */
        inline constexpr std::size_t K_METRIC_CAPACITY = 8;

        /** @brief Holds the metrics of one step, by name, in the order they were added. */
        class Metrics final
        {
            public:
                /**
                 * @brief Appends @p metric.
                 * @return InvalidDescription, with the set unchanged, for an empty name, a
                 *         name the set holds, or a set of K_METRIC_CAPACITY metrics.
                 */
                [[nodiscard]] Status add(const Metric& metric);

                /** @brief Removes every metric. */
                void clear() noexcept;

                /** @brief Returns the metric called @p name, or null when none is. */
                [[nodiscard]] const Metric* find(std::string_view name) const noexcept;

                /** @brief Returns the value of the metric called @p name, or @p fallback. */
                [[nodiscard]] double value_or(std::string_view name,
                                              double fallback) const noexcept;

                /**
                 * @brief Returns the metrics in the order they were added.
                 * @return A view the next add() or clear() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const Metric> all() const noexcept;

            private:
                /** @brief Holds the metrics; the first @c count_ are set. */
                std::array<Metric, K_METRIC_CAPACITY> entries_{};

                /** @brief Counts the metrics held. */
                std::size_t count_ = 0;
        };

        /**
         * @brief Returns the value of @p metric as text: a fraction as a percentage of two
         *        decimals, any other number in six significant digits.
         */
        [[nodiscard]] std::string describe_metric_value(const Metric& metric);

    } // namespace Train
} // namespace SushiAI
