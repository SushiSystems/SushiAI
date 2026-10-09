/**************************************************************************/
/* objective.hpp                                                          */
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
 * @file objective.hpp
 * @brief Defines training objectives, metrics, and registry interfaces.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>

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
         * @class Objective
         * @brief What a model is being trained to minimise.
         */
        class Objective
        {
            public:
                virtual ~Objective() = default;

                /**
                 * @brief Declares whatever leaves the graph needs beyond the model's input.
                 * @param builder Where the leaves go.
                 * @param leaves Where to record them, by role.
                 */
                virtual void declare(const Graph::GraphBuilder& builder,
                                     ObjectiveLeaves& leaves) const = 0;

                /**
                 * @brief Computes the scalar loss from the model's named outputs.
                 * @param builder Where the nodes go.
                 * @param graph The graph, for checking shapes.
                 * @param outputs The model's outputs; the objective names the ones it reads.
                 * @param leaves Declared auxiliary leaves.
                 * @return The scalar loss value identifier.
                 */
                virtual Graph::ValueId loss(const Graph::GraphBuilder& builder,
                                            const Graph::Graph& graph,
                                            const NN::ModelOutputs& outputs,
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
        class CrossEntropyObjective final : public Objective
        {
            public:
                /**
                 * @brief Constructs the objective for a fixed batch and class count.
                 * @param batch Samples per step.
                 * @param classes How many classes.
                 * @param dtype The element type the target leaf carries.
                 */
                CrossEntropyObjective(int64_t batch, int64_t classes,
                                      DType dtype = DType::Float32) noexcept
                    : batch_(batch), classes_(classes), dtype_(dtype) {}

                void declare(const Graph::GraphBuilder& builder,
                             ObjectiveLeaves& leaves) const override;

                Graph::ValueId loss(const Graph::GraphBuilder& builder,
                                    const Graph::Graph& graph,
                                    const NN::ModelOutputs& outputs,
                                    const ObjectiveLeaves& leaves) const override;

                std::optional<Metric> score(const Data::Batch& batch,
                                            const SushiBLAS::Tensor& output) const override;

                std::string_view name() const noexcept override { return "cross_entropy"; }

            private:
                int64_t batch_ = 0;
                int64_t classes_ = 0;
                DType dtype_ = DType::Float32;
        };

        /**
         * @brief Builds one objective from its configuration attributes.
         * @param batch Samples per step.
         * @param classes How many classes the dataset spans.
         * @param attributes The objective's own keys; the factory rejects any it does not know.
         * @param context What to call this objective in a diagnostic.
         * @return The objective.
         */
        using ObjectiveFactory = std::unique_ptr<Objective> (*)(int64_t batch, int64_t classes,
                                                                 const Config::Value& attributes,
                                                                 std::string_view context);

        /**
         * @struct ObjectiveEntry
         * @brief One row of @ref K_OBJECTIVE_REGISTRY.
         */
        struct ObjectiveEntry
        {
            /** @brief The "type" a configuration file spells. */
            std::string_view type;

            /** @brief What builds it. */
            ObjectiveFactory make;
        };

        /**
         * @brief Builds a cross-entropy objective from attributes.
         * @param batch Samples per step.
         * @param classes Number of target classes.
         * @param attributes Objective attribute values.
         * @param context Diagnostic context string.
         * @return The constructed objective instance.
         */
        std::unique_ptr<Objective> make_cross_entropy(int64_t batch, int64_t classes,
                                                       const Config::Value& attributes,
                                                       std::string_view context);

        /**
         * @brief Every objective a configuration file may name.
         *
         * A file-scope constant table; nothing self-registers. See
         * config/registry.hpp for why that is forced rather than preferred.
         */
        inline constexpr std::array<ObjectiveEntry, 1> K_OBJECTIVE_REGISTRY = {
            ObjectiveEntry{"cross_entropy", &make_cross_entropy},
        };

        /**
         * @brief Builds the objective named by @p type.
         * @param type Registry key; @p batch Samples per step.
         * @param classes Target classes; @p attributes Configuration.
         * @param context Diagnostic context string.
         * @return The constructed objective instance.
         */
        std::unique_ptr<Objective> make_objective(std::string_view type,
                                                   int64_t batch, int64_t classes,
                                                   const Config::Value& attributes,
                                                   std::string_view context);

    } // namespace Train
} // namespace SushiAI
