/**************************************************************************/
/* replica_test_support.hpp                                               */
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
 * @file replica_test_support.hpp
 * @brief Declares the model and the single-machine step the distribution tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/distribution/replica_step.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/shape/shape.hpp>

/** @brief Holds what the tests of the distribution module share. */
namespace SushiAI::Distribution::Support
{
    /** @brief Holds the input features of the model. */
    inline constexpr std::int64_t K_FEATURES = 5;

    /** @brief Holds the classes the model ends in. */
    inline constexpr std::int64_t K_CLASSES = 3;

    /** @brief Holds the seed every case's task carries. */
    inline constexpr std::uint64_t K_SEED = 0x5EEDC0DE20261024ull;

    /** @brief Holds the run index every case's task carries. */
    inline constexpr std::uint64_t K_RUN = 3;

    /** @brief Holds the hash the test model reports as its fingerprint. */
    inline constexpr std::uint64_t K_MODEL_FINGERPRINT = 0xA2A2A2A2A2A2A2A2ull;


    /** @brief Returns element @p index of the parameters, exact in float32. */
    inline float parameter_value(std::size_t index)
    {
        return static_cast<float>(static_cast<int>((index * 37) % 17) - 8) / 16.0f;
    }

    /** @brief Returns feature @p feature of the sample at position @p position, exact. */
    inline float feature_value(std::uint64_t position, std::int64_t feature)
    {
        const std::uint64_t mixed = position * 13 + static_cast<std::uint64_t>(feature) * 7;
        return static_cast<float>(static_cast<int>(mixed % 11) - 5) / 8.0f;
    }

    /**
     * @brief Holds logits = x W + b under a mean cross-entropy, differentiated for W and b.
     *
     * The sample at position p has the features feature_value(p, d) and the class p mod
     * K_CLASSES; the epoch's order is the identity.
     */
    class LinearModel final : public ReplicaModel
    {
        public:
            /** @brief Traces the model over @p rows rows and binds every value it packs. */
            static Result<std::unique_ptr<ReplicaModel>> make(SushiBLAS::Engine& engine,
                                                              std::int64_t rows)
            {
                std::unique_ptr<LinearModel> model(new LinearModel(engine, rows));
                SA_TRY(model->bind());
                return Result<std::unique_ptr<ReplicaModel>>(std::move(model));
            }

            const Graph::Graph& graph() const noexcept override
            {
                return graph_;
            }

            Graph::TensorPool& pool() noexcept override
            {
                return pool_;
            }

            SushiRuntime::span<const SushiBLAS::Tensor> parameters() const noexcept override
            {
                return SushiRuntime::span<const SushiBLAS::Tensor>(parameters_.data(),
                                                                   parameters_.size());
            }

            SushiRuntime::span<const SushiBLAS::Tensor> results() const noexcept override
            {
                return SushiRuntime::span<const SushiBLAS::Tensor>(results_.data(),
                                                                   results_.size());
            }

            std::uint64_t fingerprint() const noexcept override
            {
                return K_MODEL_FINGERPRINT;
            }

            Status feed(const ReplicaRows& rows) override
            {
                SA_FAIL_IF(static_cast<std::int64_t>(rows.rows.count) != rows_,
                           StatusCode::InvalidDescription, "{} rows fed to a model of {}",
                           rows.rows.count, rows_);

                float* const inputs = pool_.tensor(inputs_).data_as<float>();
                float* const targets = pool_.tensor(targets_).data_as<float>();
                for (std::int64_t row = 0; row < rows_; ++row)
                {
                    const std::uint64_t position =
                        rows.rows.first + static_cast<std::uint64_t>(row);
                    for (std::int64_t feature = 0; feature < K_FEATURES; ++feature)
                        inputs[row * K_FEATURES + feature] = feature_value(position, feature);

                    for (std::int64_t label = 0; label < K_CLASSES; ++label)
                        targets[row * K_CLASSES + label] =
                            static_cast<std::int64_t>(position % K_CLASSES) == label ? 1.0f
                                                                                     : 0.0f;
                }
                return Status();
            }

            /** @brief Writes parameter_value() into every parameter, in pack order. */
            void fill_parameters()
            {
                std::size_t index = 0;
                for (SushiBLAS::Tensor parameter : parameters_)
                {
                    float* const values = parameter.data_as<float>();
                    for (std::int64_t element = 0; element < parameter.num_elements; ++element)
                        values[element] = parameter_value(index++);
                }
            }

        private:
            /** @brief Traces the graph; bind() allocates its values. */
            LinearModel(SushiBLAS::Engine& engine, std::int64_t rows)
                : pool_(engine), rows_(rows)
            {
                const Graph::GraphBuilder builder(graph_);
                inputs_ = builder.input(Shape{rows, K_FEATURES}, "inputs");
                weight_ = builder.parameter(Shape{K_FEATURES, K_CLASSES}, "weight");
                bias_ = builder.parameter(Shape{K_CLASSES}, "bias");
                targets_ = builder.input(Shape{rows, K_CLASSES}, "targets");
                const Graph::ValueID logits = builder.bias_add(builder.matmul(inputs_, weight_),
                                                               bias_);
                loss_ = builder.nll_loss(builder.log_softmax(logits), targets_);

                const std::vector<Graph::ValueID> wrt = {weight_, bias_};
                const SushiRuntime::span<const Graph::ValueID> wrt_view(wrt.data(), wrt.size());
                gradients_ = Autograd::differentiate(graph_, loss_, wrt_view);
            }

            /** @brief Allocates every leaf and every value the packs name, and lists them. */
            Status bind()
            {
                SA_TRY(pool_.materialise_leaves(graph_));
                parameters_ = {pool_.tensor(weight_), pool_.tensor(bias_)};
                for (const Graph::ValueID gradient : gradients_)
                {
                    SA_TRY_ASSIGN(SushiBLAS::Tensor& tensor,
                                  pool_.materialise(gradient, graph_.value(gradient)));
                    results_.push_back(tensor);
                }

                SA_TRY_ASSIGN(SushiBLAS::Tensor& loss,
                              pool_.materialise(loss_, graph_.value(loss_)));
                results_.push_back(loss);
                return Status();
            }

            /// @brief The forward and backward graph.
            Graph::Graph graph_;

            /// @brief The values of graph_.
            Graph::TensorPool pool_;

            /// @brief The rows of one micro-batch.
            std::int64_t rows_;

            /// @brief The [rows, K_FEATURES] input leaf.
            Graph::ValueID inputs_ = Graph::K_NO_VALUE;

            /// @brief The [rows, K_CLASSES] one-hot target leaf.
            Graph::ValueID targets_ = Graph::K_NO_VALUE;

            /// @brief The [K_FEATURES, K_CLASSES] weight.
            Graph::ValueID weight_ = Graph::K_NO_VALUE;

            /// @brief The [K_CLASSES] bias.
            Graph::ValueID bias_ = Graph::K_NO_VALUE;

            /// @brief The mean cross-entropy.
            Graph::ValueID loss_ = Graph::K_NO_VALUE;

            /// @brief The gradients of the weight and of the bias, in that order.
            std::vector<Graph::ValueID> gradients_;

            /// @brief The weight and the bias.
            std::vector<SushiBLAS::Tensor> parameters_;

            /// @brief The two gradients, then the loss.
            std::vector<SushiBLAS::Tensor> results_;
    };

    /** @brief Returns a factory that builds LinearModel over a micro-batch of rows. */
    inline ReplicaModelFactory linear_factory()
    {
        return [](SushiBLAS::Engine& engine, const ReplicaDraws&, std::int64_t rows)
        {
            return LinearModel::make(engine, rows);
        };
    }

    /**
     * @brief Holds the results of one forward and backward replay, with no optimizer, over
     *        @p rows rows from position @p first, on a runtime configured as a replica's.
     */
    struct SingleMachineStep
    {
        /** @brief Compiles and runs the step; a failure is a failed expectation. */
        explicit SingleMachineStep(std::int64_t rows, std::uint64_t first = 0)
        {
            SushiRuntime::Runtime runtime =
                SushiAI::Detail::create_runtime(replica_runtime_config()).value();
            SushiBLAS::Engine engine(runtime);
            std::unique_ptr<ReplicaModel> built = LinearModel::make(engine, rows).value();
            LinearModel& model = static_cast<LinearModel&>(*built);
            model.fill_parameters();

            std::unique_ptr<Graph::CompiledStep> step =
                Graph::CompiledStep::compile(engine, model.graph(), model.pool()).value();
            ReplicaRows fed;
            fed.rows = RowRange{first, static_cast<std::uint64_t>(rows)};
            SA_EXPECT_OK(model.feed(fed));
            SA_EXPECT_OK(step->run());

            for (const SushiBLAS::Tensor& result : model.results())
            {
                const std::size_t bytes = static_cast<std::size_t>(result.num_elements)
                                          * sizeof(float);
                std::vector<unsigned char> copy(bytes);
                std::memcpy(copy.data(), result.data(), bytes);
                results.push_back(std::move(copy));
            }
        }

        /// @brief The bytes of each result, in the order of the result pack.
        std::vector<std::vector<unsigned char>> results;
    };
} // namespace SushiAI::Distribution::Support
