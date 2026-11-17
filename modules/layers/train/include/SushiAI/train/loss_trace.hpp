/**************************************************************************/
/* loss_trace.hpp                                                       */
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
 * @file loss_trace.hpp
 * @brief Declares LossTrace, what Objective::loss() built that a later hook reads.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @class LossTrace
         * @brief Holds the loss value of an objective and the values it chose to keep, which
         *        Objective::declare_metric_values() reads.
         *
         * A composite records the trace of each term with add_term(), in order. An evaluation
         * graph has no loss and passes the default trace, which holds nothing.
         */
        class LossTrace final
        {
            public:
                /** @brief Builds the trace of no loss: no value and no term. */
                LossTrace() noexcept = default;

                /** @brief Builds the trace of the loss @p loss with no kept value. */
                explicit LossTrace(Graph::ValueID loss) noexcept : loss_(loss)
                {

                }

                /** @brief Returns the loss value; Graph::K_NO_VALUE for the default trace. */
                [[nodiscard]] Graph::ValueID loss() const noexcept
                {
                    return loss_;
                }

                /**
                 * @brief Records that the objective kept @p id under @p name.
                 * @pre @p name was not recorded yet; has() tells. A second record stops the
                 *      process.
                 */
                void add(std::string_view name, Graph::ValueID id) noexcept;

                /** @brief Returns true when a value was recorded under @p name. */
                [[nodiscard]] bool has(std::string_view name) const noexcept;

                /**
                 * @brief Returns the value recorded under @p name.
                 * @pre has(@p name); anything else stops the process, listing the names held.
                 */
                [[nodiscard]] Graph::ValueID at(std::string_view name) const noexcept;

                /** @brief Returns every recorded value, in the order it was recorded. */
                [[nodiscard]] const std::vector<std::pair<std::string, Graph::ValueID>>& all()
                    const noexcept
                {
                    return values_;
                }

                /** @brief Appends the trace of the next term of a composite. */
                void add_term(LossTrace term) noexcept;

                /**
                 * @brief Returns the trace of term @p index.
                 * @return The recorded trace, or a trace that holds nothing when fewer terms
                 *         were recorded, as an evaluation graph passes none.
                 */
                [[nodiscard]] const LossTrace& term(std::size_t index) const noexcept;

            private:
                /// @brief Holds the loss value.
                Graph::ValueID loss_ = Graph::K_NO_VALUE;

                /// @brief Holds the kept values with their names, in recording order.
                std::vector<std::pair<std::string, Graph::ValueID>> values_;

                /// @brief Holds the trace of each term of a composite, in order.
                std::vector<LossTrace> terms_;
        };

    } // namespace Train
} // namespace SushiAI
