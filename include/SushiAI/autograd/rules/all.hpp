/****************************************************************/
/* all.hpp                                                      */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
        inline constexpr auto RULES = concat_rules(
            concat_rules(
                concat_rules(ELEMENTWISE_RULES, LINALG_RULES),
                concat_rules(ACTIVATION_RULES, LOSS_RULES)),
            concat_rules(
                concat_rules(SHAPE_RULES, SPATIAL_RULES),
                NORM_RULES));

        /** @brief Whether @p op has a rule in @ref RULES. */
        constexpr bool has_rule(OpID op) noexcept
        {
            for (const RuleEntry& e : RULES)
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
                          SushiRuntime::span<const Ops::OpName>(Ops::ALL.data(),
                                                                Ops::ALL.size())),
                      "An operation is declared differentiable in core/ops.hpp but has no "
                      "backward rule. Declaring an identity and forgetting its rule is "
                      "otherwise silent: the graph builds, differentiate() skips the node, "
                      "and the model trains with a missing gradient.");
    }
}
