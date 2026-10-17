/**************************************************************************/
/* all.hpp                                                                */
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
 * @file all.hpp
 * @brief Joins the eight domain tables into the one lowering table and proves it complete.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/graph/lowering/rule.hpp>
#include <SushiAI/graph/lowering/unlowered.hpp>
#include <SushiAI/graph/lowering/elementwise.hpp>
#include <SushiAI/graph/lowering/activation.hpp>
#include <SushiAI/graph/lowering/linalg.hpp>
#include <SushiAI/graph/lowering/loss.hpp>
#include <SushiAI/graph/lowering/shape.hpp>
#include <SushiAI/graph/lowering/spatial.hpp>
#include <SushiAI/graph/lowering/norm.hpp>
#include <SushiAI/graph/lowering/fused.hpp>
#include <SushiAI/graph/lowering/random.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Holds every lowering row, the nine domain tables joined in the order of the
         *        includes above.
         */
        inline constexpr auto K_LOWERINGS = concat_lowerings(
            concat_lowerings(
                concat_lowerings(K_ELEMENTWISE_LOWERINGS, K_ACTIVATION_LOWERINGS),
                concat_lowerings(K_LINALG_LOWERINGS, K_LOSS_LOWERINGS)),
            concat_lowerings(
                concat_lowerings(K_SHAPE_LOWERINGS, K_SPATIAL_LOWERINGS),
                concat_lowerings(concat_lowerings(K_NORM_LOWERINGS, K_FUSED_LOWERINGS),
                                 K_RANDOM_LOWERINGS)));

        /** @brief Returns the rule of the first row that names @p op, or nullptr when none does. */
        [[nodiscard]] constexpr LoweringRule find_lowering(OpID op) noexcept
        {
            for (const LoweringEntry& entry : K_LOWERINGS)
            {
                if (entry.op == op)
                {
                    return entry.rule;
                }
            }

            return nullptr;
        }

        /** @brief Returns the scratch rule of the first row that names @p op, or nullptr. */
        [[nodiscard]] constexpr ScratchRule find_scratch_rule(OpID op) noexcept
        {
            for (const LoweringEntry& entry : K_LOWERINGS)
            {
                if (entry.op == op)
                {
                    return entry.scratch;
                }
            }

            return nullptr;
        }

        /** @brief Returns whether K_UNLOWERED lists @p op. */
        [[nodiscard]] constexpr bool is_unlowered(OpID op) noexcept
        {
            for (const OpID listed : K_UNLOWERED)
            {
                if (listed == op)
                {
                    return true;
                }
            }

            return false;
        }

        /** @brief Counts the rows of K_LOWERINGS that name @p op. */
        [[nodiscard]] constexpr std::size_t lowering_row_count(OpID op) noexcept
        {
            std::size_t rows = 0;
            for (const LoweringEntry& entry : K_LOWERINGS)
            {
                if (entry.op == op)
                {
                    ++rows;
                }
            }

            return rows;
        }

        /**
         * @brief Returns whether every identity of @p table has exactly one row or is listed in
         *        K_UNLOWERED, and never both.
         */
        [[nodiscard]] constexpr bool every_identity_is_accounted_for(
            SushiRuntime::span<const Ops::OpName> table) noexcept
        {
            for (const Ops::OpName& entry : table)
            {
                const std::size_t owners = lowering_row_count(entry.id)
                                         + (is_unlowered(entry.id) ? 1u : 0u);
                if (owners != 1u)
                {
                    return false;
                }
            }

            return true;
        }

        static_assert(every_identity_is_accounted_for(
                          SushiRuntime::span<const Ops::OpName>(Ops::K_ALL.data(),
                                                                Ops::K_ALL.size())),
                      "An identity of ops/ops.hpp has no lowering row, has two, or has a row and "
                      "is also listed in unlowered.hpp. A graph that held it would stop at "
                      "Lowering::record with no hint at build time.");

        static_assert(K_LOWERINGS.size() + K_UNLOWERED.size() == Ops::K_ALL.size(),
                      "The lowering tables name an operation that ops/ops.hpp does not declare.");

    } // namespace Graph
} // namespace SushiAI
