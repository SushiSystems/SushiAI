/**************************************************************************/
/* c3.hpp                                                                 */
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
 * @file c3.hpp
 * @brief Declares C3, a split block of 1x1 and 3x3 bottlenecks, and its node factory.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/vision/blocks/c3k.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /**
         * @brief Narrows the input twice, chains Bottleneck(hidden, hidden, hidden) off one half
         *        with a 1x1 then a 3x3 window, and joins both halves.
         */
        using C3 = C3k;

        /**
         * @brief Builds a C3 node from "filters" and "hidden", with optional "repeats", 1 when
         *        absent, "shortcut", true when absent, "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape; C becomes the input width.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_c3_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

    } // namespace Vision::Blocks
} // namespace SushiAI
