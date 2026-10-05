/****************************************************************/
/* rule.hpp                                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file rule.hpp
 * @brief Declares backward rule definitions and compile-time table concatenation.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/autograd/tape.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /** @brief What every backward rule looks like. */
        using BackwardRule = void (*)(const GradientTape&);

        /**
         * @struct RuleEntry
         * @brief One operation's backward rule, paired with its identity.
         */
        struct RuleEntry
        {
            OpID op;             /**< @brief The forward operation. */
            BackwardRule rule;   /**< @brief What differentiates it. */
        };

        /**
         * @brief Concatenates two rule tables at compile time.
         * @param a The left rule table.
         * @param b The right rule table.
         * @return Combined array of rule entries.
         */
        template<std::size_t N, std::size_t M>
        constexpr std::array<RuleEntry, N + M> concat_rules(
            const std::array<RuleEntry, N>& a, const std::array<RuleEntry, M>& b)
        {
            std::array<RuleEntry, N + M> out{};
            for (std::size_t i = 0; i < N; ++i) out[i] = a[i];
            for (std::size_t i = 0; i < M; ++i) out[N + i] = b[i];
            return out;
        }
    }
}
