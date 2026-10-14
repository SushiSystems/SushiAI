/**************************************************************************/
/* layer_registry.hpp                                                     */
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
 * @file layer_registry.hpp
 * @brief Declares the registry that holds the rows register_layers adds, and the list of its types.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Builds a fresh registry that holds the rows register_layers adds. */
        [[nodiscard]] inline NN::NodeRegistry make_layer_registry()
        {
            NN::NodeRegistry registry;
            SA_EXPECT_OK(NN::register_layers(registry));
            return registry;
        }

        /** @brief Lists the type of every row of @p rows, in order. */
        template<typename Entry>
        [[nodiscard]] std::vector<std::string_view> types_of(SushiRuntime::span<const Entry> rows)
        {
            std::vector<std::string_view> types;
            for (const Entry& entry : rows)
                types.push_back(entry.type);

            return types;
        }

    } // namespace Test
} // namespace SushiAI
