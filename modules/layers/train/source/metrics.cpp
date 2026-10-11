/**************************************************************************/
/* metrics.cpp                                                            */
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
 * @file metrics.cpp
 * @brief Implements the named numbers an objective reports for a step.
 * @author Mustafa Garip
 */

#include <SushiAI/train/metrics.hpp>

#include <SushiBLAS/core/format.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Holds what a fraction is multiplied by to read as a percentage. */
            constexpr double K_PERCENT = 100.0;
        } // namespace

        Status Metrics::add(const Metric& metric)
        {
            SA_FAIL_IF(metric.name.empty(), StatusCode::InvalidDescription,
                       "A metric needs a non-empty name");
            SA_FAIL_IF(find(metric.name) != nullptr, StatusCode::InvalidDescription,
                       "The metric '{}' was reported twice for one step", metric.name);
            SA_FAIL_IF(count_ == K_METRIC_CAPACITY, StatusCode::InvalidDescription,
                       "A step reports at most {} metrics; '{}' is one more",
                       K_METRIC_CAPACITY, metric.name);

            entries_[count_] = metric;
            ++count_;
            return Status();
        }

        void Metrics::clear() noexcept
        {
            count_ = 0;
        }

        const Metric* Metrics::find(std::string_view name) const noexcept
        {
            for (std::size_t index = 0; index < count_; ++index)
            {
                if (entries_[index].name == name)
                    return &entries_[index];
            }

            return nullptr;
        }

        double Metrics::value_or(std::string_view name, double fallback) const noexcept
        {
            const Metric* const metric = find(name);

            return metric != nullptr ? metric->value : fallback;
        }

        SushiRuntime::span<const Metric> Metrics::all() const noexcept
        {
            return SushiRuntime::span<const Metric>(entries_.data(), count_);
        }

        std::string describe_metric_value(const Metric& metric)
        {
            if (metric.is_fraction)
                return SushiBLAS::Core::format("{:.2f}%", metric.value * K_PERCENT);

            return SushiBLAS::Core::format("{:.6g}", metric.value);
        }

    } // namespace Train
} // namespace SushiAI
