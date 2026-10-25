/**************************************************************************/
/* c2f.hpp                                                                */
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
 * @file c2f.hpp
 * @brief Declares C2f, a split stem with a chain of bottlenecks that keep the hidden width.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/vision/blocks/bottleneck.hpp>
#include <SushiAI/vision/blocks/c3k2.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /**
         * @brief Splits a stem in two, chains Bottleneck(hidden, hidden, hidden) off the second
         *        half and joins both halves with every bottleneck result.
         */
        using C2f = C3k2<Bottleneck>;

        /**
         * @brief Builds a C2f node from "filters" and "hidden", with optional "repeats", 1 when
         *        absent, "shortcut", true when absent, "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape; C becomes the input width.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_c2f_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Vision::Blocks
} // namespace SushiAI
