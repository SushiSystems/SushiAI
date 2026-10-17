/**************************************************************************/
/* rule.hpp                                                               */
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
 * @file rule.hpp
 * @brief Declares the lowering row type and the compile-time concatenation of row tables.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <vector>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>

namespace SushiAI
{
    namespace Graph
    {
        class LoweringContext;

        /** @brief Records the SushiBLAS work of one node and reports a refusal as a status. */
        using LoweringRule = Status (*)(const LoweringContext&);

        /**
         * @brief Appends the scratch buffers of one node that its row writes on every run
         *        before it reads them; a buffer that keeps its bytes between runs is left out.
         */
        using ScratchRule = void (*)(const Graph&, NodeID, std::vector<ScratchRequest>&);

        /** @brief Pairs an operation with the function that lowers it and its scratch rule. */
        struct LoweringEntry
        {
            OpID op;                       /**< The operation. */
            LoweringRule rule;             /**< What records it. */
            ScratchRule scratch = nullptr; /**< What it asks the plan to place, or nullptr. */
        };

        /** @brief Joins two lowering tables into one, @p a first. */
        template<std::size_t N, std::size_t M>
        constexpr std::array<LoweringEntry, N + M> concat_lowerings(
            const std::array<LoweringEntry, N>& a, const std::array<LoweringEntry, M>& b)
        {
            std::array<LoweringEntry, N + M> out{};
            for (std::size_t i = 0; i < N; ++i)
            {
                out[i] = a[i];
            }

            for (std::size_t i = 0; i < M; ++i)
            {
                out[N + i] = b[i];
            }

            return out;
        }

    } // namespace Graph
} // namespace SushiAI
