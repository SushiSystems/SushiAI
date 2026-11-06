/**************************************************************************/
/* in_memory_precision_policy.hpp                                         */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file in_memory_precision_policy.hpp
 * @brief Declares InMemoryPrecisionPolicy, a PrecisionPolicy that answers from a script.
 * @author Mustafa Garip
 * @see modules/core/graph/README.md
 */

#pragma once

#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/ops/ops.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class InMemoryPrecisionPolicy
         * @brief Answers each query from a table of scripted choices and journals the queries;
         *        touches no device, file, thread or clock.
         *
         * An unscripted operation keeps its declared type with the reason "declared". Reading
         * queries() must not overlap a call of choose().
         */
        class InMemoryPrecisionPolicy final : public PrecisionPolicy
        {
            public:
                /**
                 * @brief Scripts @p choice for every query that names @p op.
                 * @param op The operation.
                 * @param choice The answer; its reason must be a static string.
                 * @return A second script for the same @p op replaces the first.
                 */
                void script(OpID op, PrecisionChoice choice)
                {
                    for (Scripted& entry : script_)
                    {
                        if (entry.op == op)
                        {
                            entry.choice = choice;
                            return;
                        }
                    }

                    script_.push_back(Scripted{op, choice});
                }

                /** @brief Journals @p query; returns its scripted choice or its declared type. */
                PrecisionChoice choose(const PrecisionQuery& query) const override
                {
                    queries_.push_back(query);
                    for (const Scripted& entry : script_)
                    {
                        if (entry.op == query.op)
                            return entry.choice;
                    }

                    return PrecisionChoice{query.declared, "declared"};
                }

                /** @brief Returns "in-memory". */
                std::string_view name() const noexcept override
                {
                    return "in-memory";
                }

                /** @brief Returns every query received, first received first. */
                [[nodiscard]] const std::vector<PrecisionQuery>& queries() const noexcept
                {
                    return queries_;
                }

            private:
                /// @brief Pairs an operation with the choice scripted for it.
                struct Scripted
                {
                    /// @brief The operation the choice answers for.
                    OpID op;

                    /// @brief The scripted answer.
                    PrecisionChoice choice;
                };

                /// @brief Holds one scripted choice per operation.
                std::vector<Scripted> script_;

                /// @brief Holds the queries choose() received, in order.
                mutable std::vector<PrecisionQuery> queries_;
        };

    } // namespace Graph
} // namespace SushiAI
