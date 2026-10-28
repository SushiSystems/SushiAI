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
            double weight = 1.0;                  /**< Its weight; finite and above zero. */
        };

        /**
         * @brief Computes the sum over its terms of weight * loss, in the graph.
         *
         * Term i declares its leaves under leaf_prefix(i), so two terms may use one role
         * name. README.md states how the metrics of the terms merge.
         */
        class SumObjective final : public Objective
        {
            public:
                /**
                 * @brief Builds the composite over @p terms, in the order given.
                 * @return The objective, or InvalidDescription for no term, a weight that
                 *         is not finite and above zero, or a term its own step feeds, named
                 *         by its position and its name. A null term stops the process.
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
                 * @pre Every term's loss is one element of one element type; anything else
                 *      stops the process.
                 */
                [[nodiscard]] Result<Graph::ValueID> loss(
                    const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                    const NN::ModelOutputs& outputs, const ObjectiveLeaves& leaves) const override;

                /** @brief Feeds every term from @p batch, in order; the first refusal ends it. */
                [[nodiscard]] Status feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                                          Graph::TensorPool& pool) const override;

                /** @brief Returns true: make() refused every term that is not. */
                [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

                /**
                 * @brief Appends the metrics of every term, in order; of two metrics of one
                 *        name the first term's is kept.
                 */
                [[nodiscard]] Status measure(const StepView& step,
                                             Metrics& metrics) const override;

                /** @brief Returns "sum". */
                std::string_view name() const noexcept override { return K_SUM_TYPE; }

                /** @brief Returns the terms, in the order make() was given them. */
                [[nodiscard]] SushiRuntime::span<const SumTerm> terms() const noexcept;

            private:
                /** @brief Stores what make() has checked. */
                explicit SumObjective(std::vector<SumTerm> terms) noexcept;

                /// @brief The terms, in order.
                std::vector<SumTerm> terms_;
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
