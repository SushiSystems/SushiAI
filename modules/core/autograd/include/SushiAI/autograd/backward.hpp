/**************************************************************************/
/* backward.hpp                                                           */
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
 * @file backward.hpp
 * @brief Declares reverse-mode differentiation as an IR-to-IR transform.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/autograd/tape.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @brief Returns whether an operation can be differentiated.
         * @param op The operation to look up.
         * @return True when a backward rule exists for it.
         */
        bool has_backward_rule(OpID op) noexcept;

        /**
         * @brief Extends graph with reverse-mode gradients of loss.
         * @param loss One-element result of an operation that has a backward rule.
         * @param seed_factor One-element value multiplied into the root gradient, or NO_VALUE.
         * @param provenance Filled with the origin of every appended node when not null.
         * @return Gradient ValueID per wrt entry, Graph::NO_VALUE if unreachable, or
         *         InvalidDescription; see modules/core/autograd/README.md.
         */
        [[nodiscard]] Result<std::vector<Graph::ValueID>> differentiate(
            Graph::Graph& graph, Graph::ValueID loss, SushiRuntime::span<const Graph::ValueID> wrt,
            Graph::ValueID seed_factor = Graph::K_NO_VALUE, Provenance* provenance = nullptr) noexcept;

    } // namespace Autograd
} // namespace SushiAI
