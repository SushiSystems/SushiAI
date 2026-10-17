/**************************************************************************/
/* mean_squared_error_objective.hpp                                       */
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
 * @file mean_squared_error_objective.hpp
 * @brief Declares the regression objective: the mean squared error of K values per sample.
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
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>

namespace SushiAI
{
    namespace Train
    {
        /** @brief Holds the "type" a training file states for the regression objective. */
        inline constexpr std::string_view K_MEAN_SQUARED_ERROR_TYPE = "mean_squared_error";

        /**
         * @brief Computes the mean squared error of [N, K] outputs against [N, K] values.
         *
         * The loss is emit_squared_error() over the N * K elements. The metric is the
         * mean absolute error over the same elements.
         */
        class MeanSquaredErrorObjective final : public Objective
        {
            public:
                /**
                 * @brief Builds the objective for @p batch samples of @p width values.
                 * @param dtype The element type of the target leaf, float32 or float64.
                 * @return The objective, or InvalidDescription for an extent below 1 or
                 *         another element type.
                 */
                [[nodiscard]] static Result<std::unique_ptr<MeanSquaredErrorObjective>> make(
                    std::int64_t batch, std::int64_t width, DType dtype = DType::Float32);

                /** @brief Declares the [N, K] "targets" leaf and requires float32 [N, K] values. */
                void declare(const Graph::GraphBuilder& builder,
                             ObjectiveLeaves& leaves) const override;

                /**
                 * @brief Emits the mean squared error of the model's only output.
                 * @pre @p outputs holds one [N, K] value; anything else stops the process.
                 */
                Graph::ValueID loss(const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                                    const NN::ModelOutputs& outputs,
                                    const ObjectiveLeaves& leaves) const override;

                /** @brief Copies the values role into the leaf with fill_dense_targets(). */
                [[nodiscard]] Status feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                                          Graph::TensorPool& pool) const override;

                /** @brief Returns true: a batch alone fills the leaves. */
                [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

                /**
                 * @brief Appends "mean_absolute_error" over the N * K elements.
                 * @return InvalidDescription for a step of several outputs, values the host
                 *         cannot read or a batch with no values role; Incompatible for
                 *         another count of outputs or values.
                 */
                [[nodiscard]] Status measure(const StepView& step,
                                             Metrics& metrics) const override;

                /** @brief Returns "mean_squared_error". */
                std::string_view name() const noexcept override
                {
                    return K_MEAN_SQUARED_ERROR_TYPE;
                }

                /** @brief Returns N, the samples of a batch. */
                [[nodiscard]] std::int64_t batch() const noexcept { return batch_; }

                /** @brief Returns K, the values of a sample. */
                [[nodiscard]] std::int64_t width() const noexcept { return width_; }

            private:
                /** @brief Stores what make() has checked. */
                MeanSquaredErrorObjective(std::int64_t batch, std::int64_t width,
                                          DType dtype) noexcept;

                /// @brief N, the samples of a batch.
                std::int64_t batch_ = 0;

                /// @brief K, the values of a sample.
                std::int64_t width_ = 0;

                /// @brief The element type of the target leaf.
                DType dtype_ = DType::Float32;
        };

        /**
         * @brief Builds a regression objective over the batch and the one output of @p given.
         * @param attributes The objective's own keys; it has none.
         * @param context What to call this objective in a diagnostic.
         * @return The objective, or InvalidDescription for a key, for a model that does not
         *         publish one [N, K] output, or for what MeanSquaredErrorObjective::make
         *         refuses.
         */
        [[nodiscard]] Result<std::unique_ptr<Objective>> make_mean_squared_error(
            const ObjectiveContext& given, const Config::Value& attributes,
            std::string_view context);

    } // namespace Train
} // namespace SushiAI
