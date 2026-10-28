/**************************************************************************/
/* next_token_objective.hpp                                               */
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
 * @file next_token_objective.hpp
 * @brief Declares the next-token objective: mean cross entropy over every position.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
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
    /** @brief Holds the "type" a training file states for the next-token objective. */
    inline constexpr std::string_view K_NEXT_TOKEN_OBJECTIVE_TYPE = "next_token";

    /** @brief Names the key that states which published output holds the logits. */
    inline constexpr std::string_view K_NEXT_TOKEN_OUTPUT_KEY = "output";

    /** @brief Holds the role of the leaf fill_next_token_targets() writes. */
    inline constexpr std::string_view K_NEXT_TOKEN_TARGETS_LEAF = "targets";

    /**
     * @brief Computes the mean cross entropy of [N, T, V] logits against [N, T] token ids.
     *
     * The loss reads one log-probability per position through K_GATHER_ROWS, so its target
     * leaf holds N * T int32 indices and no one-hot row.
     */
    class NextTokenObjective final : public Train::Objective
    {
        public:
            /**
             * @brief Builds the objective for @p batch rows of @p length positions.
             * @param output Names the published output that holds the logits; empty reads
             *        the one output of a model that publishes no other.
             * @return The objective, or InvalidDescription for an extent below 1 or for
             *         batch * length * vocabulary above what INT32 indexes.
             */
            [[nodiscard]] static Result<std::unique_ptr<NextTokenObjective>> make(
                std::int64_t batch, std::int64_t length, std::int64_t vocabulary,
                std::string output = {});

            /** @brief Declares the INT32 [N * T] "targets" leaf and requires the labels role. */
            void declare(const Graph::GraphBuilder& builder,
                         Train::ObjectiveLeaves& leaves) const override;

            /**
             * @brief Emits the mean, over the N * T positions, of minus the log-probability
             *        of each position's label.
             * @pre @p outputs holds the [N, T, V] value make() was told of, alone or under
             *      the name it was given; anything else stops the process.
             */
            [[nodiscard]] Result<Graph::ValueID> loss(
                const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                const NN::ModelOutputs& outputs,
                const Train::ObjectiveLeaves& leaves) const override;

            /**
             * @brief Writes the index of every label with fill_next_token_targets().
             * @return InvalidDescription for a batch with no labels role, or what the
             *         fill refuses.
             */
            [[nodiscard]] Status feed(const Data::Batch& batch,
                                      const Train::ObjectiveLeaves& leaves,
                                      Graph::TensorPool& pool) const override;

            /** @brief Returns true: a batch alone fills the leaves. */
            [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

            /** @brief Appends "perplexity", the exponential of the loss of the step. */
            [[nodiscard]] Status measure(const Train::StepView& step,
                                         Train::Metrics& metrics) const override;

            /** @brief Returns "next_token". */
            std::string_view name() const noexcept override
            {
                return K_NEXT_TOKEN_OBJECTIVE_TYPE;
            }

            /** @brief Returns N, the rows of a batch. */
            [[nodiscard]] std::int64_t batch() const noexcept { return batch_; }

            /** @brief Returns T, the positions of a row. */
            [[nodiscard]] std::int64_t length() const noexcept { return length_; }

            /** @brief Returns V, the last extent of the logits. */
            [[nodiscard]] std::int64_t vocabulary() const noexcept { return vocabulary_; }

        private:
            /** @brief Stores extents make() has checked and the name of the logits. */
            NextTokenObjective(std::int64_t batch, std::int64_t length,
                               std::int64_t vocabulary, std::string output) noexcept;

            /// @brief N, the rows of a batch.
            std::int64_t batch_ = 0;

            /// @brief T, the positions of a row.
            std::int64_t length_ = 0;

            /// @brief V, the last extent of the logits.
            std::int64_t vocabulary_ = 0;

            /// @brief The name of the output that holds the logits; empty for the only one.
            std::string output_;
    };

    /**
     * @brief Builds a next-token objective from the batch and the logits of @p given.
     * @param attributes May state "output", the name of the published output that holds
     *        the logits; a model that publishes one output needs no key.
     * @return The objective, or InvalidDescription for another key, for logits that are
     *         not one [N, T, V] output over [T] samples, for data of more classes than V,
     *         or for what NextTokenObjective::make refuses.
     */
    [[nodiscard]] Result<std::unique_ptr<Train::Objective>> make_next_token_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes,
        std::string_view context);

} // namespace SushiAI::Language::Training
