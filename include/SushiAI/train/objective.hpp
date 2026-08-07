/**************************************************************************/
/* objective.hpp                                                          */
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

/**
 * @file objective.hpp
 * @brief What turns a model's output into the scalar that gets differentiated.
 *
 * @section why Breaking the classifier lock-in, before it costs anything
 *
 * @c build_classifier used to hardcode @c NN::CrossEntropyLoss, and
 * @c Data::Dataset still demands a pure-virtual @c num_classes(). A PINN has
 * no classes; a detector has no [N, C] logits. Without generalising this seam,
 * a config-driven driver would be rewritten at the first model that is not a
 * classifier - which is exactly the outcome the config-driven design exists to
 * prevent.
 *
 * So the loss is a table row like the layers, the datasets and the optimizers,
 * and @c apps/train.cpp names no loss type at all.
 *
 * @section identical Why cross-entropy is reproduced rather than rewritten
 *
 * @ref CrossEntropyObjective emits the same operations in the same order as
 * the code it replaced - @c nll_loss(log_softmax(logits), targets) - so the
 * whole existing suite is the regression test for this change. If any
 * operation moved, @c Regression_LearningCurve, @c Regression_FusionNumerics
 * and @c Regression_Determinism would each say so, rather than the difference
 * being diffuse and unattributable.
 *
 * @section debt What this slice does not yet do
 *
 * @c Train::Trainer still feeds @c StepSpec::targets per step, so an objective
 * that declared some other leaf has nothing writing it. Wiring the trainer to
 * feed arbitrary declared leaves is a separate change with its own argument;
 * @ref ObjectiveLeaves is the shape it will use. Recorded here rather than
 * quietly carried - see docs/ARCHITECTURE.md, alongside
 * @c Dataset::num_classes().
 *
 * Host lane except for @ref IObjective::score, which reads an output tensor.
 */

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @struct Metric
         * @brief One named number a step produced, whatever it means.
         *
         * So a driver can print "accuracy 92.40%" or "residual 3.1e-4" without
         * knowing which it is - which is the difference between a driver that
         * survives a non-classifier and one that does not.
         */
        struct Metric
        {
            /** @brief What to call it, e.g. "accuracy". */
            std::string_view name;

            /** @brief Its value. */
            double value = 0.0;

            /** @brief Whether it reads as a percentage rather than a bare number. */
            bool is_fraction = false;
        };

        /**
         * @class ObjectiveLeaves
         * @brief The graph values an objective declared, by role.
         *
         * For cross-entropy that is one [N, C] target leaf; for a PINN it would
         * be the collocation points. It is what @c StepSpec::targets
         * generalises into.
         */
        class ObjectiveLeaves
        {
            public:
                /**
                 * @brief Records that @p role is @p id.
                 * @param role What the leaf is for, e.g. "targets".
                 * @param id The declared value.
                 * @throws Error If @p role was already declared.
                 */
                void add(std::string_view role, Graph::ValueId id);

                /**
                 * @brief Whether @p role was declared.
                 * @param role What the leaf is for.
                 * @return True when it was.
                 */
                bool has(std::string_view role) const noexcept;

                /**
                 * @brief The value declared for @p role.
                 * @param role What the leaf is for.
                 * @return Its value id.
                 * @throws Error If @p role was not declared; lists what was.
                 */
                Graph::ValueId at(std::string_view role) const;

                /** @brief Every declared leaf, in declaration order. */
                const std::vector<std::pair<std::string, Graph::ValueId>>& all() const noexcept
                {
                    return leaves_;
                }

            private:
                std::vector<std::pair<std::string, Graph::ValueId>> leaves_;
        };

        /**
         * @class IObjective
         * @brief What a model is being trained to minimise.
         */
        class IObjective
        {
            public:
                virtual ~IObjective() = default;

                /**
                 * @brief Declares whatever leaves the graph needs beyond the model's input.
                 * @param builder Where the leaves go.
                 * @param leaves Where to record them, by role.
                 */
                virtual void declare(const Graph::GraphBuilder& builder,
                                     ObjectiveLeaves& leaves) const = 0;

                /**
                 * @brief Turns the model's output into the scalar that gets differentiated.
                 * @param builder Where the nodes go.
                 * @param graph The graph, for checking the output's shape.
                 * @param output The model's output value.
                 * @param leaves What @ref declare recorded.
                 * @return The 1-element loss.
                 * @throws Error If @p output is not the shape this objective needs.
                 */
                virtual Graph::ValueId loss(const Graph::GraphBuilder& builder,
                                            const Graph::Graph& graph,
                                            Graph::ValueId output,
                                            const ObjectiveLeaves& leaves) const = 0;

                /**
                 * @brief An optional per-step metric: accuracy, mAP, an L2 residual, or none.
                 * @param batch The batch that was just stepped.
                 * @param output The model's output tensor.
                 * @return The metric, or std::nullopt when this objective offers none.
                 */
                virtual std::optional<Metric> score(const Data::Batch& batch,
                                                    const SushiBLAS::Tensor& output) const = 0;

                /** @brief A short name for a run report. */
                virtual std::string_view name() const noexcept = 0;
        };

        /**
         * @class CrossEntropyObjective
         * @brief Multi-class cross-entropy over one-hot targets, and accuracy.
         *
         * Emits @c nll_loss(log_softmax(output), targets) - the same two
         * operations in the same order as the code this replaced, so the
         * existing suite is the regression test.
         */
        class CrossEntropyObjective final : public IObjective
        {
            public:
                /**
                 * @brief Constructs the objective for a fixed batch and class count.
                 * @param batch Samples per step.
                 * @param classes How many classes.
                 * @param dtype The element type the target leaf carries.
                 */
                CrossEntropyObjective(int64_t batch, int64_t classes,
                                      DType dtype = DType::FLOAT32) noexcept
                    : batch_(batch), classes_(classes), dtype_(dtype) {}

                void declare(const Graph::GraphBuilder& builder,
                             ObjectiveLeaves& leaves) const override;

                Graph::ValueId loss(const Graph::GraphBuilder& builder,
                                    const Graph::Graph& graph,
                                    Graph::ValueId output,
                                    const ObjectiveLeaves& leaves) const override;

                std::optional<Metric> score(const Data::Batch& batch,
                                            const SushiBLAS::Tensor& output) const override;

                std::string_view name() const noexcept override { return "cross_entropy"; }

            private:
                int64_t batch_ = 0;
                int64_t classes_ = 0;
                DType dtype_ = DType::FLOAT32;
        };

        // --- The registry a configuration file selects through -----------------

        /**
         * @brief Builds one objective from its configuration attributes.
         * @param batch Samples per step.
         * @param classes How many classes the dataset spans.
         * @param attributes The objective's own keys; the factory rejects any it does not know.
         * @param context What to call this objective in a diagnostic.
         * @return The objective.
         */
        using ObjectiveFactory = std::unique_ptr<IObjective> (*)(int64_t batch, int64_t classes,
                                                                 const Config::Value& attributes,
                                                                 std::string_view context);

        /**
         * @struct ObjectiveEntry
         * @brief One row of @ref OBJECTIVE_REGISTRY.
         */
        struct ObjectiveEntry
        {
            /** @brief The "type" a configuration file spells. */
            std::string_view type;

            /** @brief What builds it. */
            ObjectiveFactory make;
        };

        /**
         * @brief Builds a @ref CrossEntropyObjective; it takes no keys at all.
         * @param batch Samples per step.
         * @param classes How many classes.
         * @param attributes The objective's keys; any key whatsoever is an error.
         * @param context What to call it in a diagnostic.
         * @return The objective.
         * @throws Error If any key is present.
         */
        std::unique_ptr<IObjective> make_cross_entropy(int64_t batch, int64_t classes,
                                                       const Config::Value& attributes,
                                                       std::string_view context);

        /**
         * @brief Every objective a configuration file may name.
         *
         * A file-scope constant table; nothing self-registers. See
         * config/registry.hpp for why that is forced rather than preferred.
         */
        inline constexpr std::array<ObjectiveEntry, 1> OBJECTIVE_REGISTRY = {
            ObjectiveEntry{"cross_entropy", &make_cross_entropy},
        };

        /**
         * @brief Builds the objective named @p type.
         * @param type The registry key.
         * @param batch Samples per step.
         * @param classes How many classes the dataset spans.
         * @param attributes The objective's own keys.
         * @param context What to call it in a diagnostic.
         * @return The objective.
         * @throws Error If @p type is unknown; the message lists every type
         *         this build has.
         */
        std::unique_ptr<IObjective> make_objective(std::string_view type,
                                                   int64_t batch, int64_t classes,
                                                   const Config::Value& attributes,
                                                   std::string_view context);

    } // namespace Train
} // namespace SushiAI
