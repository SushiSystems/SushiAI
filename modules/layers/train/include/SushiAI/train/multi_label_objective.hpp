/**************************************************************************/
/* multi_label_objective.hpp                                              */
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
 * @file multi_label_objective.hpp
 * @brief Declares the multi-label objective: one yes-or-no cross entropy per class.
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
        /** @brief Holds the "type" a training file states for the multi-label objective. */
        inline constexpr std::string_view K_MULTI_LABEL_TYPE = "multi_label";

        /**
         * @brief Computes the mean cross entropy of [N, C] logits against [N, C] labels.
         *
         * The loss is emit_logistic_loss() over the N * C elements. The metric is the F1
         * of each class at the threshold, averaged over the C classes; a class with no
         * true and no predicted 1 in the batch counts as 0.
         */
        class MultiLabelObjective final : public Objective
        {
            public:
                /**
                 * @brief Builds the objective for @p batch samples of @p classes labels.
                 * @param threshold The probability at which a prediction reads as 1.
                 * @param dtype The element type of the target leaf, float32 or float64.
                 * @return The objective, or InvalidDescription for an extent below 1, a
                 *         threshold outside (0, 1) or another element type.
                 */
                [[nodiscard]] static Result<std::unique_ptr<MultiLabelObjective>> make(
                    std::int64_t batch, std::int64_t classes,
                    double threshold = K_DEFAULT_THRESHOLD, DType dtype = DType::Float32);

                /** @brief Declares the [N, C] "targets" leaf and requires float32 [N, C] labels. */
                void declare(const Graph::GraphBuilder& builder,
                             ObjectiveLeaves& leaves) const override;

                /**
                 * @brief Emits the mean logistic loss of the model's only output.
                 * @pre @p outputs holds one [N, C] value; anything else stops the process.
                 */
                [[nodiscard]] Result<Graph::ValueID> loss(
                    const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                    const NN::ModelOutputs& outputs, const ObjectiveLeaves& leaves) const override;

                /** @brief Copies the labels role into the leaf with fill_binary_targets(). */
                [[nodiscard]] Status feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                                          Graph::TensorPool& pool) const override;

                /** @brief Returns true: a batch alone fills the leaves. */
                [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

                /**
                 * @brief Appends "f1": the mean, over the classes, of each class's F1.
                 * @return InvalidDescription for a step of several outputs, values the host
                 *         cannot read or a batch with no labels role; Incompatible for
                 *         another count of logits or labels.
                 */
                [[nodiscard]] Status measure(const StepView& step,
                                             Metrics& metrics) const override;

                /** @brief Returns "multi_label". */
                std::string_view name() const noexcept override { return K_MULTI_LABEL_TYPE; }

                /** @brief Returns N, the samples of a batch. */
                [[nodiscard]] std::int64_t batch() const noexcept { return batch_; }

                /** @brief Returns C, the labels of a sample. */
                [[nodiscard]] std::int64_t classes() const noexcept { return classes_; }

                /** @brief Returns the probability at which a prediction reads as 1. */
                [[nodiscard]] double threshold() const noexcept { return threshold_; }

            private:
                /** @brief Stores what make() has checked. */
                MultiLabelObjective(std::int64_t batch, std::int64_t classes, double threshold,
                                    DType dtype) noexcept;

                /// @brief N, the samples of a batch.
                std::int64_t batch_ = 0;

                /// @brief C, the labels of a sample.
                std::int64_t classes_ = 0;

                /// @brief The probability at which a prediction reads as 1.
                double threshold_ = K_DEFAULT_THRESHOLD;

                /// @brief The element type of the target leaf.
                DType dtype_ = DType::Float32;
        };

        /**
         * @brief Builds a multi-label objective over the batch and the one output of @p given.
         * @param attributes The objective's own keys; it reads "threshold" alone.
         * @param context What to call this objective in a diagnostic.
         * @return The objective, or InvalidDescription for what read_threshold() refuses,
         *         for a model that does not publish one [N, C] output, or for what
         *         MultiLabelObjective::make refuses.
         */
        [[nodiscard]] Result<std::unique_ptr<Objective>> make_multi_label(
            const ObjectiveContext& given, const Config::Value& attributes,
            std::string_view context);

    } // namespace Train
} // namespace SushiAI
