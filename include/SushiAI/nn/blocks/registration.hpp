/**************************************************************************/
/* registration.hpp                                                       */
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
 * @file registration.hpp
 * @brief Declares register_blocks, which adds the five block node types to a node registry.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief The registry of node and layer rows; declared in node_registry.hpp. */
        class NodeRegistry;

        /**
         * @brief Adds the five block node types to @p registry, in the order conv, bottleneck,
         *        c3k, c3k2 and sppf.
         * @param registry The registry that receives the five rows.
         * @return Success, or the first failure of NodeRegistry::add().
         */
        [[nodiscard]] Status register_blocks(NodeRegistry& registry);

    } // namespace NN
} // namespace SushiAI
