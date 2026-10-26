/**************************************************************************/
/* free_branch_weights.cpp                                                */
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
 * @file free_branch_weights.cpp
 * @brief Implements the weights that mix the two detection losses and their reader.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_branch_weights.hpp>

#include <algorithm>
#include <cmath>

#include <SushiAI/config/section_reader.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Names the section in a diagnostic; every refusal starts with it. */
        constexpr std::string_view K_SECTION = "objective";

        /** @brief Pairs one key with the weight it states. */
        struct WeightField
        {
            std::string_view key;              /**< The key in the section. */
            float FreeBranchWeights::*weight;  /**< The weight the key states. */
        };

        /** @brief Lists every weight, in the order of K_FREE_BRANCH_WEIGHT_KEYS. */
        constexpr WeightField K_FIELDS[] = {
            {K_FREE_BRANCH_WEIGHT_KEYS[0], &FreeBranchWeights::start},
            {K_FREE_BRANCH_WEIGHT_KEYS[1], &FreeBranchWeights::end}};
    } // namespace

    Status validate_free_branch_weights(const FreeBranchWeights& weights)
    {
        for (const WeightField& field : K_FIELDS)
        {
            const float weight = weights.*field.weight;
            SA_FAIL_IF(!std::isfinite(weight) || weight < 0.0f || weight > 1.0f,
                       StatusCode::InvalidDescription,
                       "{} is {}; a branch weight is a finite number in 0..1",
                       Config::member_of(K_SECTION, field.key), weight);
        }

        return Status();
    }

    Result<FreeBranchWeights> read_free_branch_weights(const Config::Value& section)
    {
        const Config::SectionReader reader(section, K_SECTION);
        SA_TRY(reader.check_object());

        FreeBranchWeights result;
        for (const WeightField& field : K_FIELDS)
        {
            if (!reader.has(field.key))
                continue;

            SA_TRY_ASSIGN(const Config::Value* const member, reader.find(field.key));
            SA_TRY_ASSIGN(result.*field.weight,
                          Config::read_float32(*member, reader.member(field.key)));
        }

        SA_TRY(validate_free_branch_weights(result));
        return result;
    }

    float one_to_many_weight(const FreeBranchWeights& weights, std::int64_t epoch,
                             std::int64_t epochs) noexcept
    {
        const double last = static_cast<double>(std::max<std::int64_t>(epochs - 1, 1));
        const double left = std::max(1.0 - static_cast<double>(epoch) / last, 0.0);
        return static_cast<float>(weights.end + (weights.start - weights.end) * left);
    }

} // namespace SushiAI::Vision::Training
