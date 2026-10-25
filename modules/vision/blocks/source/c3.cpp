/**************************************************************************/
/* c3.cpp                                                                 */
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
 * @file c3.cpp
 * @brief Implements the factory that builds a C3 node.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/c3.hpp>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "c3" row accepts. */
            constexpr std::string_view K_C3_KEYS[] = {
                "filters", "hidden", "repeats", "shortcut", Detail::K_NORM_EPSILON_KEY,
                Detail::K_NORM_MOMENTUM_KEY};

            /** @brief Holds the side of the first window of each bottleneck. */
            constexpr int64_t K_FIRST_KERNEL = 1;

            /** @brief Holds the side of the second window of each bottleneck. */
            constexpr int64_t K_SECOND_KERNEL = 3;

        } // namespace

        Result<std::unique_ptr<NN::Node>> make_c3_node(NN::ShapeSpan inputs,
                                                       const Config::Value& attributes,
                                                       std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_C3_KEYS), context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_feature_map(inputs, context));
            SA_TRY_ASSIGN(const int64_t filters,
                          Detail::read_positive_int(attributes, "filters", context));
            SA_TRY_ASSIGN(const int64_t hidden,
                          Detail::read_positive_int(attributes, "hidden", context));
            SA_TRY_ASSIGN(const int64_t repeats,
                          Detail::read_positive_int_or(attributes, "repeats", context,
                                                       Detail::K_DEFAULT_REPEATS));
            SA_TRY_ASSIGN(const bool shortcut,
                          Detail::read_bool_or(attributes, "shortcut", context,
                                               Detail::K_DEFAULT_SHORTCUT));
            SA_TRY_ASSIGN(const NN::NormOptions norm,
                          Detail::read_norm_options(attributes, context));

            return Detail::wrap_block(C3(input[2], filters, hidden, repeats, shortcut,
                                         K_FIRST_KERNEL, K_SECOND_KERNEL, norm),
                                      input);
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
