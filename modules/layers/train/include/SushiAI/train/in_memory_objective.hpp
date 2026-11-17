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
 * @brief Declares InMemoryObjective, an Objective that answers from a script.
 * @author Mustafa Garip
 * @see modules/layers/train/README.md
 */

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/step_view.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @class InMemoryObjective
         * @brief Declares, feeds, measures and answers from a script; touches no device, file,
         *        thread or clock.
         *
         * Whatever is not scripted keeps the default of Objective. The stated metric names
         * point into the object, so it can be neither copied nor moved.
         */
        class InMemoryObjective final : public Objective
        {
            public:
                /** @brief Emits a loss in place of a scripted value id. */
                using LossEmitter = std::function<Result<Graph::ValueID>(
                    const Graph::GraphBuilder&, const Graph::Graph&, const NN::ModelOutputs&,
                    const ObjectiveLeaves&)>;

                /** @brief Makes an objective with nothing scripted. */
                InMemoryObjective() = default;

                /** @brief Deleted: the stated views point into the object. */
                InMemoryObjective(const InMemoryObjective&) = delete;

                /** @brief Deleted: the stated views point into the object. */
                InMemoryObjective(InMemoryObjective&&) = delete;

                /** @brief Deleted: the stated views point into the object. */
                InMemoryObjective& operator=(const InMemoryObjective&) = delete;

                /** @brief Deleted: the stated views point into the object. */
                InMemoryObjective& operator=(InMemoryObjective&&) = delete;

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
                 * @brief Makes declare() add an input of the builder under @p role.
                 * @param role What the leaf is for; scripted once, as script_leaf() states.
                 * @param shape The extents of the leaf.
                 * @param dtype The element type of the leaf.
                 */
                void script_input_leaf(std::string role, Shape shape,
                                       DType dtype = DType::Float32)
                {
                    inputs_.push_back(InputLeaf{std::move(role), std::move(shape), dtype});
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

                /**
                 * @brief Makes every later loss() call @p emitter and return its result.
                 * @param emitter Builds the loss in the graph; a scripted failure still wins.
                 */
                void script_loss_emitter(LossEmitter emitter)
                {
                    emitter_ = std::move(emitter);
                }

                /**
                 * @brief Makes feed() fill every element of the leaf @p role with @p value.
                 *
                 * A leaf no call names is left as it was, which is how a test skips one.
                 * @param role The leaf; it must be declared and bound when feed() runs.
                 * @param value The value to write, converted to the element type of the leaf.
                 */
                void script_feed_fill(std::string role, double value)
                {
                    fills_.emplace_back(std::move(role), value);
                }

                /**
                 * @brief Makes every later feed() return @p failure after its fills.
                 * @param failure A failure; a success restores the plain return.
                 */
                void script_feed_failure(Status failure) noexcept
                {
                    feed_failure_ = std::move(failure);
                }

                /**
                 * @brief Sets what is_fed_by_batch() returns.
                 * @param is_fed True to let a sum take the objective as a term.
                 */
                void script_fed_by_batch(bool is_fed) noexcept
                {
                    is_fed_by_batch_ = is_fed;
                }

                /**
                 * @brief Makes measure() append the metric @p name with @p value.
                 * @param name The metric's name; a name scripted twice makes measure() fail, as
                 *             Metrics::add states.
                 * @param value The value to append.
                 */
                void script_metric(std::string name, double value)
                {
                    metrics_.emplace_back(std::move(name), value);
                }

                /**
                 * @brief Sets the names metric_names() returns, which need not be the ones
                 *        measure() appends.
                 * @param names The names to state, in order; none by default.
                 */
                void script_stated_names(std::vector<std::string> names)
                {
                    stated_names_ = std::move(names);
                    stated_views_.assign(stated_names_.begin(), stated_names_.end());
                }

                /** @brief Adds the scripted leaves and roles to @p leaves and counts the call. */
                void declare(const Graph::GraphBuilder& builder,
                             ObjectiveLeaves& leaves) const override;

                /**
                 * @brief Counts the call and returns the scripted failure, the emitted loss or
                 *        the scripted loss value.
                 * @return InvalidDescription when no loss was scripted.
                 */
                [[nodiscard]] Result<LossTrace> loss(
                    const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                    const NN::ModelOutputs& outputs,
                    const ObjectiveLeaves& leaves) const override;

                /**
                 * @brief Counts the call and writes the scripted fills.
                 * @return Unsupported, as the base answers, when no fill and no failure is
                 *         scripted; else the failure of a leaf that is not declared, not bound or
                 *         not float32, float64 or int32; else the scripted failure or success.
                 */
                [[nodiscard]] Status feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                                          Graph::TensorPool& pool) const override;

                /** @brief Returns the scripted answer, false by default. */
                [[nodiscard]] bool is_fed_by_batch() const noexcept override
                {
                    return is_fed_by_batch_;
                }

                /**
                 * @brief Counts the call and appends the scripted metrics, in order.
                 * @return The first failure of Metrics::add, or success.
                 */
                [[nodiscard]] Status measure(const StepView& view,
                                             Metrics& metrics) const override;

                /** @brief Returns the scripted names, none by default. */
                [[nodiscard]] SushiRuntime::span<const std::string_view> metric_names()
                    const noexcept override
                {
                    return SushiRuntime::span<const std::string_view>(stated_views_.data(),
                                                                      stated_views_.size());
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

                /** @brief Returns how many times feed() ran. */
                [[nodiscard]] std::size_t feed_count() const noexcept
                {
                    return feed_count_;
                }

                /** @brief Returns how many times measure() ran. */
                [[nodiscard]] std::size_t measure_count() const noexcept
                {
                    return measure_count_;
                }

            private:
                /** @brief Holds an input leaf declare() asks the builder for. */
                struct InputLeaf
                {
                    std::string role; /**< What the leaf is for. */
                    Shape shape;      /**< Its extents. */
                    DType dtype;      /**< Its element type. */
                };

                /// @brief Holds the leaves declare() adds by id, in order.
                std::vector<std::pair<std::string, Graph::ValueID>> leaves_;

                /// @brief Holds the leaves declare() asks the builder for, in order.
                std::vector<InputLeaf> inputs_;

                /// @brief Holds the roles declare() requires, in order.
                std::vector<Data::TargetDescription> required_;

                /// @brief Holds the value loss() returns.
                Graph::ValueID loss_ = Graph::K_NO_VALUE;

                /// @brief Holds the failure loss() returns; a success when none is scripted.
                Status loss_failure_;

                /// @brief Holds the emitter loss() calls; empty when none is scripted.
                LossEmitter emitter_;

                /// @brief Holds the leaves feed() fills and the value of each, in order.
                std::vector<std::pair<std::string, double>> fills_;

                /// @brief Holds the failure feed() returns; a success when none is scripted.
                Status feed_failure_;

                /// @brief Holds what is_fed_by_batch() returns.
                bool is_fed_by_batch_ = false;

                /// @brief Holds the metrics measure() appends, in order.
                std::vector<std::pair<std::string, double>> metrics_;

                /// @brief Holds the text of the names metric_names() states.
                std::vector<std::string> stated_names_;

                /// @brief Holds the views metric_names() returns, over stated_names_.
                std::vector<std::string_view> stated_views_;

                /// @brief Counts the calls of declare().
                mutable std::size_t declare_count_ = 0;

                /// @brief Counts the calls of loss().
                mutable std::size_t loss_count_ = 0;

                /// @brief Counts the calls of feed().
                mutable std::size_t feed_count_ = 0;

                /// @brief Counts the calls of measure().
                mutable std::size_t measure_count_ = 0;
        };

    } // namespace Train
} // namespace SushiAI
