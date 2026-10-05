/****************************************************************/
/* backward.hpp                                                 */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file backward.hpp
 * @brief Declares reverse-mode differentiation as an IR-to-IR transform.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/autograd/tape.hpp>

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
         * @param loss Scalar value being differentiated; must have one element.
         * @param seed_factor Optional scaling scalar, or Graph::NO_VALUE.
         * @return Gradient ValueId per wrt entry, or Graph::NO_VALUE if unreachable.
         */
        std::vector<Graph::ValueId> differentiate(Graph::Graph& graph,
                                                  Graph::ValueId loss,
                                                  SushiRuntime::span<const Graph::ValueId> wrt,
                                                  Graph::ValueId seed_factor = Graph::NO_VALUE);

    } // namespace Autograd
} // namespace SushiAI
