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
 * @brief Combines all domain backward rule tables into a single lookup table.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/autograd/rule.hpp>
#include <SushiAI/autograd/rules/elementwise.hpp>
#include <SushiAI/autograd/rules/linalg.hpp>
#include <SushiAI/autograd/rules/activation.hpp>
#include <SushiAI/autograd/rules/loss.hpp>
#include <SushiAI/autograd/rules/shape.hpp>
#include <SushiAI/autograd/rules/spatial.hpp>
#include <SushiAI/autograd/rules/norm.hpp>
#include <SushiAI/autograd/rules/random.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @brief Every backward rule, as one flat compile-time table.
         *
         * Adding an operation to a domain changes only that domain's file;
         * adding a domain is one line here.
         */
        inline constexpr auto K_RULES = concat_rules(
            concat_rules(
                concat_rules(K_ELEMENTWISE_RULES, K_LINALG_RULES),
                concat_rules(K_ACTIVATION_RULES, K_LOSS_RULES)),
            concat_rules(
                concat_rules(K_SHAPE_RULES, K_SPATIAL_RULES),
                concat_rules(K_NORM_RULES, K_RANDOM_RULES)));

        /** @brief Whether @p op has a rule in @ref K_RULES. */
        constexpr bool has_rule(OpID op) noexcept
        {
            for (const RuleEntry& e : K_RULES)
                if (e.op == op) return true;
            return false;
        }

        /**
         * @brief Verifies that every differentiable operation possesses a backward rule.
         * @param table Operation metadata table to inspect.
         * @return True when all differentiable operations have registered rules.
         */
        constexpr bool every_differentiable_op_has_a_rule(
            SushiRuntime::span<const Ops::OpName> table) noexcept
        {
            for (const Ops::OpName& entry : table)
                if (entry.differentiable && !has_rule(entry.id)) return false;
            return true;
        }

        static_assert(every_differentiable_op_has_a_rule(
                          SushiRuntime::span<const Ops::OpName>(Ops::K_ALL.data(),
                                                                Ops::K_ALL.size())),
                      "An operation is declared differentiable in core/ops.hpp but has no "
                      "backward rule. Declaring an identity and forgetting its rule is "
                      "otherwise silent: the graph builds, differentiate() skips the node, "
                      "and the model trains with a missing gradient.");

        /**
         * @brief Verifies that no gradient barrier possesses a backward rule.
         * @param table Operation metadata table to inspect.
         * @return True when every operation flagged blocks_gradient is absent from RULES.
         */
        constexpr bool no_gradient_barrier_has_a_rule(
            SushiRuntime::span<const Ops::OpName> table) noexcept
        {
            for (const Ops::OpName& entry : table)
            {
                if (entry.blocks_gradient && has_rule(entry.id))
                    return false;
            }
            return true;
        }

        static_assert(no_gradient_barrier_has_a_rule(
                          SushiRuntime::span<const Ops::OpName>(Ops::K_ALL.data(),
                                                                Ops::K_ALL.size())),
                      "An operation is declared a gradient barrier in core/ops.hpp and also has "
                      "a backward rule. The reverse walk steps over a barrier, so the rule "
                      "would never run. Remove the row, or declare the operation differentiable.");
    }
}
