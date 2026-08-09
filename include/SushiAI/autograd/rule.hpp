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

#pragma once

// The vocabulary every domain's backward rules and rule table are written in.
//
// A rules/*.cpp includes only this header (plus tape.hpp, which it drags in):
// it never needs differentiate() or has_backward_rule(), so it never needs
// backward.hpp. That keeps a rule file's dependency surface to exactly what a
// rule is - a function of a GradientTape - and the array type that domain
// rule tables are built from.

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
         *
         * The mechanism behind @c rules/all.hpp. Nothing self-registers:
         * @c sushiai is a static library, so a translation unit whose only job
         * is to run a registration constructor resolves no symbol and the
         * linker drops it - silently, and only in a downstream consumer's
         * build, which is the worst possible place to discover a missing
         * gradient. Taking the address of a function declared in a domain
         * header is an address constant expression, so the concatenated table
         * stays @c constexpr and lookup stays a compile-time walk.
         *
         * @tparam N The left table's size.
         * @tparam M The right table's size.
         * @param a The left table.
         * @param b The right table.
         * @return Their concatenation.
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
