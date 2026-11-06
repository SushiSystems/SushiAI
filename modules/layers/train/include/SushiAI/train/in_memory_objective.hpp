/**************************************************************************/
/* in_memory_objective.hpp                                                */
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
 * @file in_memory_objective.hpp
 * @brief Declares InMemoryObjective, an Objective that declares and answers from a script.
 * @author Mustafa Garip
 * @see modules/layers/train/README.md
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/objective.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @class InMemoryObjective
         * @brief Declares the leaves and the batch roles it was given and returns a scripted loss
         *        value; touches no device, file, thread or clock.
         *
         * It emits no node: loss() returns the scripted value id. feed() and measure() keep the
         * defaults of Objective.
         */
        class InMemoryObjective final : public Objective
        {
            public:
                /**
                 * @brief Makes declare() add leaf @p id under @p role.
                 * @param role What the leaf is for; a role scripted twice stops the process in
                 *             declare(), as ObjectiveLeaves::add states.
                 * @param id The value to record.
                 */
                void script_leaf(std::string role, Graph::ValueID id)
                {
                    leaves_.emplace_back(std::move(role), id);
                }

                /**
                 * @brief Makes declare() require the batch role @p role.
                 * @param role The role, its per-sample shape and its element type.
                 */
                void script_required(Data::TargetDescription role)
                {
                    required_.push_back(std::move(role));
                }

                /**
                 * @brief Makes every later loss() return @p id.
                 * @param id The loss value.
                 */
                void script_loss(Graph::ValueID id) noexcept
                {
                    loss_ = id;
                    loss_failure_ = Status();
                }

                /**
                 * @brief Makes every later loss() return @p failure.
                 * @param failure A failure; a success restores the scripted loss value.
                 */
                void script_loss_failure(Status failure) noexcept
                {
                    loss_failure_ = std::move(failure);
                }

                /** @brief Adds the scripted leaves and roles to @p leaves and counts the call. */
                void declare(const Graph::GraphBuilder&, ObjectiveLeaves& leaves) const override
                {
                    ++declare_count_;
                    for (const auto& leaf : leaves_)
                        leaves.add(leaf.first, leaf.second);

                    for (const Data::TargetDescription& role : required_)
                        leaves.require(role);
                }

                /**
                 * @brief Counts the call and returns the scripted failure or loss value.
                 * @return InvalidDescription when no loss was scripted.
                 */
                [[nodiscard]] Result<Graph::ValueID> loss(
                    const Graph::GraphBuilder&, const Graph::Graph&, const NN::ModelOutputs&,
                    const ObjectiveLeaves&) const override
                {
                    ++loss_count_;
                    if (!loss_failure_.is_ok())
                        return loss_failure_;

                    SA_FAIL_IF(loss_ == Graph::K_NO_VALUE, StatusCode::InvalidDescription,
                               "InMemoryObjective::loss: no loss value was scripted");
                    return loss_;
                }

                /** @brief Returns "in-memory". */
                std::string_view name() const noexcept override
                {
                    return "in-memory";
                }

                /** @brief Returns how many times declare() ran. */
                [[nodiscard]] std::size_t declare_count() const noexcept
                {
                    return declare_count_;
                }

                /** @brief Returns how many times loss() ran. */
                [[nodiscard]] std::size_t loss_count() const noexcept
                {
                    return loss_count_;
                }

            private:
                /// @brief Holds the leaves declare() adds, in order.
                std::vector<std::pair<std::string, Graph::ValueID>> leaves_;

                /// @brief Holds the roles declare() requires, in order.
                std::vector<Data::TargetDescription> required_;

                /// @brief Holds the value loss() returns.
                Graph::ValueID loss_ = Graph::K_NO_VALUE;

                /// @brief Holds the failure loss() returns; a success when none is scripted.
                Status loss_failure_;

                /// @brief Counts the calls of declare().
                mutable std::size_t declare_count_ = 0;

                /// @brief Counts the calls of loss().
                mutable std::size_t loss_count_ = 0;
        };

    } // namespace Train
} // namespace SushiAI
