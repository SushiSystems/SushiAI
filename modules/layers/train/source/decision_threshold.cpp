/**************************************************************************/
/* decision_threshold.cpp                                                 */
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
 * @file decision_threshold.cpp
 * @brief Implements the reader of the probability at which a logit reads as a yes.
 * @author Mustafa Garip
 */

#include <SushiAI/train/decision_threshold.hpp>

#include <cmath>
#include <string>

namespace SushiAI
{
    namespace Train
    {
        Status validate_threshold(double threshold)
        {
            // The comparison is written so that a NaN is refused as well.
            SA_FAIL_IF(!(threshold > 0.0 && threshold < 1.0), StatusCode::InvalidDescription,
                       "A threshold is a probability above 0 and below 1; got {}", threshold);

            return Status();
        }

        Result<double> read_threshold(const Config::Value& attributes, std::string_view context)
        {
            for (const std::string& key : attributes.keys())
            {
                SA_FAIL_IF(key != K_THRESHOLD_KEY, StatusCode::InvalidDescription,
                           "{} states the key '{}', but the only key it reads is '{}'", context,
                           key, K_THRESHOLD_KEY);
            }

            if (!attributes.has(K_THRESHOLD_KEY))
                return K_DEFAULT_THRESHOLD;

            const Config::Value& stated = attributes.at(K_THRESHOLD_KEY, context);
            SA_FAIL_IF(stated.kind() != Config::Kind::Real && stated.kind() != Config::Kind::Int,
                       StatusCode::InvalidDescription,
                       "{}: '{}' is a number above 0 and below 1, but the file holds a {}",
                       context, K_THRESHOLD_KEY, Config::kind_name(stated.kind()));

            const double threshold = stated.as_real(context);
            SA_TRY(validate_threshold(threshold));

            return threshold;
        }

        double logit_of(double probability) noexcept
        {
            return std::log(probability / (1.0 - probability));
        }

    } // namespace Train
} // namespace SushiAI
