/**************************************************************************/
/* rule.hpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

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
