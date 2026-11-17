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
#include <SushiAI/train/objective_output.hpp>

namespace SushiAI
{
    namespace Train
    {
        /** @brief Holds the "type" a training file states for the binary objective. */
        inline constexpr std::string_view K_BINARY_CROSS_ENTROPY_TYPE = "binary_cross_entropy";

        /**
         * @brief Computes the mean cross entropy of [N, 1] logits against [N] labels of 0 or 1.
         *
         * The loss is emit_logistic_loss(), computed from the logits of the selected output;
         * a type mask keeps the loss of its rows alone. The metrics are
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
                 * @param selection The output it reads and the rows it keeps.
                 * @return The objective, or InvalidDescription for a batch below 1, a
                 *         threshold outside (0, 1) or another element type.
                 */
                [[nodiscard]] static Result<std::unique_ptr<BinaryCrossEntropyObjective>> make(
                    std::int64_t batch, double threshold = K_DEFAULT_THRESHOLD,
                    DType dtype = DType::Float32, ObjectiveSelection selection = {}) noexcept;

                /**
                 * @brief Declares the [N, 1] "targets" leaf and requires float32 [N] labels,
                 *        and the weights leaf and the mask role when the mask is on.
                 */
                void declare(const Graph::GraphBuilder& builder,
                             ObjectiveLeaves& leaves) const override;

                /**
                 * @brief Emits the mean logistic loss of the selected output.
                 * @pre @p outputs holds the selected output, an [N, 1] value; anything else
                 *      stops the process.
                 */
                [[nodiscard]] Result<LossTrace> loss(
                    const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                    const NN::ModelOutputs& outputs, const ObjectiveLeaves& leaves) const override;

                /**
                 * @brief Copies the labels role into the leaf with fill_binary_targets(), then
                 *        writes the mask weights.
                 */
                [[nodiscard]] Status feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                                          Graph::TensorPool& pool) const override;

                /** @brief Returns true: a batch alone fills the leaves. */
                [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

                /**
                 * @brief Appends "accuracy", "precision" and "recall" at the threshold.
                 * @return InvalidDescription for a step that publishes no selected output,
                 *         values the host cannot read or a batch with no labels role;
                 *         Incompatible for another count of logits or labels.
                 */
                [[nodiscard]] Status measure(const StepView& step,
                                             Metrics& metrics) const override;

                /** @brief Returns K_ACCURACY_METRIC, K_PRECISION_METRIC and K_RECALL_METRIC. */
                [[nodiscard]] SushiRuntime::span<const std::string_view> metric_names()
                    const noexcept override;

                /** @brief Returns "binary_cross_entropy". */
                std::string_view name() const noexcept override
                {
                    return K_BINARY_CROSS_ENTROPY_TYPE;
                }

                /** @brief Returns N, the samples of a batch. */
                [[nodiscard]] std::int64_t batch() const noexcept { return batch_; }

                /** @brief Returns the probability at which a prediction reads as 1. */
                [[nodiscard]] double threshold() const noexcept { return threshold_; }

                /** @brief Returns the output the objective reads and the rows it keeps. */
                [[nodiscard]] const ObjectiveSelection& selection() const noexcept
                {
                    return selection_;
                }

            private:
                /** @brief Stores what make() has checked. */
                BinaryCrossEntropyObjective(std::int64_t batch, double threshold, DType dtype,
                                            ObjectiveSelection selection) noexcept;

                /// @brief N, the samples of a batch.
                std::int64_t batch_ = 0;

                /// @brief The probability at which a prediction reads as 1.
                double threshold_ = K_DEFAULT_THRESHOLD;

                /// @brief The element type of the target leaf.
                DType dtype_ = DType::Float32;

                /// @brief The output it reads and the rows it keeps.
                ObjectiveSelection selection_;
        };

        /**
         * @brief Builds a binary objective over the batch and the selected output of @p given.
         * @param attributes The objective's own keys: "threshold", "output" and "type_mask".
         * @param context What to call this objective in a diagnostic.
         * @return The objective, or InvalidDescription for what read_selection() and
         *         read_threshold() refuse, for a model that does not publish the selected
         *         [N, 1] output, or for what BinaryCrossEntropyObjective::make refuses.
         */
        [[nodiscard]] Result<std::unique_ptr<Objective>> make_binary_cross_entropy(
            const ObjectiveContext& given, const Config::Value& attributes,
            std::string_view context) noexcept;

    } // namespace Train
} // namespace SushiAI
