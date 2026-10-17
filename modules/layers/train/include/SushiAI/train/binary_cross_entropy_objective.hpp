/**************************************************************************/
/* binary_cross_entropy_objective.hpp                                     */
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
 * @file binary_cross_entropy_objective.hpp
 * @brief Declares the binary objective: cross entropy of one logit per sample against 0 or 1.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/train/decision_threshold.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>

namespace SushiAI
{
    namespace Train
    {
        /** @brief Holds the "type" a training file states for the binary objective. */
        inline constexpr std::string_view K_BINARY_CROSS_ENTROPY_TYPE = "binary_cross_entropy";

        /**
         * @brief Computes the mean cross entropy of [N, 1] logits against [N] labels of 0 or 1.
         *
         * The loss is emit_logistic_loss(), computed from the logits. The metrics are
         * accuracy, precision and recall, a logit reading as 1 where its sigmoid is at
         * least the threshold.
         */
        class BinaryCrossEntropyObjective final : public Objective
        {
            public:
                /**
                 * @brief Builds the objective for @p batch samples.
                 * @param threshold The probability at which a prediction reads as 1.
                 * @param dtype The element type of the target leaf, float32 or float64.
                 * @return The objective, or InvalidDescription for a batch below 1, a
                 *         threshold outside (0, 1) or another element type.
                 */
                [[nodiscard]] static Result<std::unique_ptr<BinaryCrossEntropyObjective>> make(
                    std::int64_t batch, double threshold = K_DEFAULT_THRESHOLD,
                    DType dtype = DType::Float32);

                /** @brief Declares the [N, 1] "targets" leaf and requires float32 [N] labels. */
                void declare(const Graph::GraphBuilder& builder,
                             ObjectiveLeaves& leaves) const override;

                /**
                 * @brief Emits the mean logistic loss of the model's only output.
                 * @pre @p outputs holds one [N, 1] value; anything else stops the process.
                 */
                Graph::ValueID loss(const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                                    const NN::ModelOutputs& outputs,
                                    const ObjectiveLeaves& leaves) const override;

                /** @brief Copies the labels role into the leaf with fill_binary_targets(). */
                [[nodiscard]] Status feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                                          Graph::TensorPool& pool) const override;

                /** @brief Returns true: a batch alone fills the leaves. */
                [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

                /**
                 * @brief Appends "accuracy", "precision" and "recall" at the threshold.
                 * @return InvalidDescription for a step of several outputs, values the host
                 *         cannot read or a batch with no labels role; Incompatible for
                 *         another count of logits or labels.
                 */
                [[nodiscard]] Status measure(const StepView& step,
                                             Metrics& metrics) const override;

                /** @brief Returns "binary_cross_entropy". */
                std::string_view name() const noexcept override
                {
                    return K_BINARY_CROSS_ENTROPY_TYPE;
                }

                /** @brief Returns N, the samples of a batch. */
                [[nodiscard]] std::int64_t batch() const noexcept { return batch_; }

                /** @brief Returns the probability at which a prediction reads as 1. */
                [[nodiscard]] double threshold() const noexcept { return threshold_; }

            private:
                /** @brief Stores what make() has checked. */
                BinaryCrossEntropyObjective(std::int64_t batch, double threshold,
                                            DType dtype) noexcept;

                /// @brief N, the samples of a batch.
                std::int64_t batch_ = 0;

                /// @brief The probability at which a prediction reads as 1.
                double threshold_ = K_DEFAULT_THRESHOLD;

                /// @brief The element type of the target leaf.
                DType dtype_ = DType::Float32;
        };

        /**
         * @brief Builds a binary objective over the batch and the one output of @p given.
         * @param attributes The objective's own keys; it reads "threshold" alone.
         * @param context What to call this objective in a diagnostic.
         * @return The objective, or InvalidDescription for what read_threshold() refuses,
         *         for a model that does not publish one [N, 1] output, or for what
         *         BinaryCrossEntropyObjective::make refuses.
         */
        [[nodiscard]] Result<std::unique_ptr<Objective>> make_binary_cross_entropy(
            const ObjectiveContext& given, const Config::Value& attributes,
            std::string_view context);

    } // namespace Train
} // namespace SushiAI
