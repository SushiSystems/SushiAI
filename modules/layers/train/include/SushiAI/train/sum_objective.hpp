/**************************************************************************/
/* sum_objective.hpp                                                      */
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
 * @file sum_objective.hpp
 * @brief Declares the composite objective: the weighted sum of the losses of its terms.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>

namespace SushiAI
{
    namespace Train
    {
        /** @brief Holds the "type" a training file states for the composite objective. */
        inline constexpr std::string_view K_SUM_TYPE = "sum";

        /** @brief Names the key that lists the terms of a composite objective. */
        inline constexpr std::string_view K_SUM_TERMS_KEY = "terms";

        /** @brief Names the key of a term that states the type of its objective. */
        inline constexpr std::string_view K_SUM_TERM_TYPE_KEY = "type";

        /** @brief Names the key of a term that states the weight of its loss; 1 when absent. */
        inline constexpr std::string_view K_SUM_TERM_WEIGHT_KEY = "weight";

        /** @brief Pairs one objective with the weight its loss takes in the sum. */
        struct SumTerm
        {
            std::unique_ptr<Objective> objective; /**< The term; the composite owns it. */
            double weight = 1.0;                  /**< Its weight; finite and not below zero. */
        };

        /**
         * @brief Computes the sum over its terms of weight * loss, in the graph.
         *
         * Term i declares its leaves under leaf_prefix(i), so two terms may use one role
         * name. README.md states how the metrics of the terms merge. A term of weight 0 adds
         * nothing to the loss and its gradient is 0, but it is fed and measured as any other,
         * so it may stand for its metrics alone.
         */
        class SumObjective final : public Objective
        {
            public:
                /**
                 * @brief Builds the composite over @p terms, in the order given.
                 * @return The objective, or InvalidDescription for no term, a weight that
                 *         is not finite or is below zero, every weight being 0, or a term its
                 *         own step feeds, named by its position and its name. A null term
                 *         stops the process.
                 */
                [[nodiscard]] static Result<std::unique_ptr<SumObjective>> make(
                    std::vector<SumTerm> terms) noexcept;

                /** @brief Returns the prefix of the leaves of term @p index: "term<index>.". */
                [[nodiscard]] static std::string leaf_prefix(std::size_t index) noexcept;

                /** @brief Declares every term's leaves under its prefix, and its roles. */
                void declare(const Graph::GraphBuilder& builder,
                             ObjectiveLeaves& leaves) const override;

                /**
                 * @brief Emits each term's loss, scaled by its weight, and their sum.
                 * @return The sum, with the trace of each term, in order, for
                 *         declare_metric_values().
                 * @pre Every term's loss is one element of one element type; anything else
                 *      stops the process.
                 */
                [[nodiscard]] Result<LossTrace> loss(
                    const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                    const NN::ModelOutputs& outputs, const ObjectiveLeaves& leaves) const override;

                /**
                 * @brief Declares the metric values of every term, in order, each over its own
                 *        trace; of two values of one name the first term's is kept.
                 */
                [[nodiscard]] Result<NN::ModelOutputs> declare_metric_values(
                    const Graph::GraphBuilder& builder, const NN::ModelOutputs& outputs,
                    const LossTrace& trace) const override;

                /** @brief Feeds every term from @p batch, in order; the first refusal ends it. */
                [[nodiscard]] Status feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                                          Graph::TensorPool& pool) const override;

                /** @brief Returns true: make() refused every term that is not. */
                [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

                /**
                 * @brief Appends the metrics of every term, in order; of two metrics of one
                 *        name the first term's is kept.
                 *
                 * A term whose metric_names() are all appended by the terms before it is not
                 * measured, so a metric two terms share is computed once; a term that states
                 * no names, or one name not yet appended, is measured whole.
                 */
                [[nodiscard]] Status measure(const StepView& step,
                                             Metrics& metrics) const override;

                /**
                 * @brief Returns the names of every term once, in the order the terms give
                 *        them; empty when a term states none, since the sum then cannot.
                 */
                [[nodiscard]] SushiRuntime::span<const std::string_view> metric_names()
                    const noexcept override;

                /** @brief Returns "sum". */
                std::string_view name() const noexcept override { return K_SUM_TYPE; }

                /** @brief Returns the terms, in the order make() was given them. */
                [[nodiscard]] SushiRuntime::span<const SumTerm> terms() const noexcept;

            private:
                /** @brief Stores what make() has checked. */
                SumObjective(std::vector<SumTerm> terms,
                             std::vector<std::string_view> names) noexcept;

                /// @brief The terms, in order.
                std::vector<SumTerm> terms_;

                /// @brief The metric names of every term, once; empty when a term states none.
                std::vector<std::string_view> names_;
        };

        /**
         * @brief Builds a composite from the "terms" of @p attributes: each states "type",
         *        may state "weight", and hands its other keys to the factory of its type.
         * @param given Its registry holds the row of every term's type.
         * @param context What to call this objective in a diagnostic.
         * @return The objective, or InvalidDescription for a context with no registry, a
         *         key or a term that does not parse, or what a term's factory or make refuses.
         */
        [[nodiscard]] Result<std::unique_ptr<Objective>> make_sum_objective(
            const ObjectiveContext& given, const Config::Value& attributes,
            std::string_view context) noexcept;

    } // namespace Train
} // namespace SushiAI
