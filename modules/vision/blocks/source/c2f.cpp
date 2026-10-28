/**************************************************************************/
/* c2f.cpp                                                                */
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
 * @file c2f.cpp
 * @brief Implements the factory that builds a C2f node.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/c2f.hpp>

#include <utility>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/nn/module_list.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "c2f" row accepts. */
            constexpr std::string_view K_C2F_KEYS[] = {
                "filters", "hidden", "repeats", "shortcut", Detail::K_NORM_EPSILON_KEY,
                Detail::K_NORM_MOMENTUM_KEY};

        } // namespace

        Result<std::unique_ptr<NN::Node>> make_c2f_node(NN::ShapeSpan inputs,
                                                        const Config::Value& attributes,
                                                        std::string_view context) noexcept
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_C2F_KEYS), context));
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

            NN::ModuleList<Bottleneck> chain;
            for (int64_t index = 0; index < repeats; ++index)
                chain.push_back(Bottleneck(hidden, hidden, hidden, shortcut, norm));

            return Detail::wrap_block(C2f(input[2], filters, hidden, std::move(chain), norm),
                                      input);
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
