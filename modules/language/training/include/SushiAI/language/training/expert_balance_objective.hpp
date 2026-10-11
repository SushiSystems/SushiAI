/**************************************************************************/
/* expert_balance_objective.hpp                                           */
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
 * @file expert_balance_objective.hpp
 * @brief Declares the objective that adds the balance terms of the blocks with experts and
 *        reports how the experts are used.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/step_view.hpp>

namespace SushiAI::Language::Training
{
    /** @brief Holds the "type" a training file states for the balance of the experts. */
    inline constexpr std::string_view K_EXPERT_BALANCE_OBJECTIVE_TYPE = "expert_balance";

    /** @brief Holds what the name of a published balance term begins with. */
    inline constexpr std::string_view K_BALANCE_OUTPUT_PREFIX = "balance";

    /** @brief Holds what the name of a published usage of experts begins with. */
    inline constexpr std::string_view K_USAGE_OUTPUT_PREFIX = "usage";

    /** @brief Names the metric that holds the smallest share an expert has of a batch. */
    inline constexpr std::string_view K_EXPERT_USAGE_MIN_METRIC = "expert_usage_min";

    /** @brief Names the metric that holds the largest share an expert has of a batch. */
    inline constexpr std::string_view K_EXPERT_USAGE_MAX_METRIC = "expert_usage_max";

    /** @brief Names one published usage and how many experts it holds a share for. */
    struct ExpertUsageOutput
    {
        std::string name;         /**< Name the model publishes the usage under. */
        std::int64_t experts = 0; /**< Elements of the output; at least 1. */
    };

    /**
     * @brief Computes the mean of the published balance terms, and reports the smallest
     *        and the largest share of the published usages.
     *
     * It declares no leaf and reads no role of a batch.
     */
    class ExpertBalanceObjective final : public Train::Objective
    {
        public:
            /**
             * @brief Builds the objective over the outputs named.
             * @param balances Names of the one-element outputs the loss is the mean of.
             * @param usages The outputs measure() reads.
             * @return The objective, or InvalidDescription for no balance, no usage, an
             *         empty name or a usage of no expert.
             */
            [[nodiscard]] static Result<std::unique_ptr<ExpertBalanceObjective>> make(
                std::vector<std::string> balances, std::vector<ExpertUsageOutput> usages);

            /** @brief Declares nothing: the objective reads the model's outputs alone. */
            void declare(const Graph::GraphBuilder& builder,
                         Train::ObjectiveLeaves& leaves) const override;

            /**
             * @brief Emits the mean of the balance outputs, as one element.
             * @pre @p outputs holds every name make() was given as one element of one
             *      element type; anything else stops the process.
             */
            Graph::ValueID loss(const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                                const NN::ModelOutputs& outputs,
                                const Train::ObjectiveLeaves& leaves) const override;

            /** @brief Returns success: the objective has no leaf to fill. */
            [[nodiscard]] Status feed(const Data::Batch& batch,
                                      const Train::ObjectiveLeaves& leaves,
                                      Graph::TensorPool& pool) const override;

            /** @brief Returns true: no step of its own computes a target. */
            [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

            /**
             * @brief Appends K_EXPERT_USAGE_MIN_METRIC and K_EXPERT_USAGE_MAX_METRIC, the
             *        extremes over every usage output of the step.
             * @return InvalidDescription for a usage the step did not publish or the host
             *         cannot read; Incompatible for another element count.
             */
            [[nodiscard]] Status measure(const Train::StepView& step,
                                         Train::Metrics& metrics) const override;

            /** @brief Returns "expert_balance". */
            std::string_view name() const noexcept override
            {
                return K_EXPERT_BALANCE_OBJECTIVE_TYPE;
            }

            /** @brief Returns the names of the balance outputs, in the order given. */
            [[nodiscard]] const std::vector<std::string>& balances() const noexcept
            {
                return balances_;
            }

            /** @brief Returns the usage outputs, in the order given. */
            [[nodiscard]] const std::vector<ExpertUsageOutput>& usages() const noexcept
            {
                return usages_;
            }

        private:
            /** @brief Stores what make() has checked. */
            ExpertBalanceObjective(std::vector<std::string> balances,
                                   std::vector<ExpertUsageOutput> usages) noexcept;

            /// @brief The names of the balance outputs.
            std::vector<std::string> balances_;

            /// @brief The usage outputs.
            std::vector<ExpertUsageOutput> usages_;
    };

    /**
     * @brief Builds the objective over every output of @p given whose name begins with
     *        K_BALANCE_OUTPUT_PREFIX or with K_USAGE_OUTPUT_PREFIX.
     * @param attributes The objective's own keys; it has none.
     * @param context What to call this objective in a diagnostic.
     * @return The objective, or InvalidDescription for a key, for a model that publishes
     *         no such output, for a balance of several elements or an empty usage.
     */
    [[nodiscard]] Result<std::unique_ptr<Train::Objective>> make_expert_balance_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes,
        std::string_view context);

} // namespace SushiAI::Language::Training
