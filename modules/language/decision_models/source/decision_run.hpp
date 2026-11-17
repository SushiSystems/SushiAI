/**************************************************************************/
/* decision_run.hpp                                                       */
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
 * @file decision_run.hpp
 * @brief Declares DecisionRun, one compiled step of a decision model over the state the session
 *        shares between its steps.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/data/dataset.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/decision_models/decision_session.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

namespace SushiAI::Language::DecisionModels
{
    /** @brief Names what a DecisionRun does with the gradients of its step. */
    enum class RunKind : std::uint8_t
    {
        Training, /**< The optimizer of the training file applies the gradients, so a step learns. */
        Probe     /**< SGD at a rate of 0 applies them, so a step only reads. */
    };

    /**
     * @brief Holds the graph, the pool, the optimizer and the trainer of one compiled step.
     *
     * Two runs over one model state share its parameters: the training run learns and the probe
     * run reads the outputs a batch gives, over the same weights.
     */
    class DecisionRun final
    {
        public:
            /**
             * @brief Traces @p model under the objective of the training file, binds it to
             *        @p state and compiles the step, as TrainingSetup does for `sushiai train`.
             * @param state Created from @p model at the first run, when it holds none.
             * @param dataset Fixes the batch, the input shape and the roles the step reads.
             * @return The run, or the failure of the objective, the trace, a role match or the
             *         compilation.
             */
            [[nodiscard]] static Result<std::unique_ptr<DecisionRun>> make(
                RunKind kind, ExecutionBackend& backend, NN::ModelGraph& model,
                std::unique_ptr<NN::ModelState>& state, const DecisionSessionOptions& options,
                const Data::Dataset& dataset) noexcept;

            DecisionRun(const DecisionRun&) = delete;
            DecisionRun& operator=(const DecisionRun&) = delete;
            DecisionRun(DecisionRun&&) = delete;
            DecisionRun& operator=(DecisionRun&&) = delete;
            ~DecisionRun() = default;

            /** @brief Returns the trainer of the step. */
            [[nodiscard]] Train::Trainer& trainer() noexcept;

            /**
             * @brief Returns the tensor a step wrote for the model output @p name.
             * @return The tensor, or InvalidDescription when the model publishes no such output.
             */
            [[nodiscard]] Result<const SushiBLAS::Tensor*> output(std::string_view name) const noexcept;

        private:
            /** @brief Builds the pool over @p engine; make() fills the rest. */
            explicit DecisionRun(SushiBLAS::Engine& engine);

            /// @brief Holds the objective the graph and the feed come from.
            std::unique_ptr<Train::Objective> objective_;

            /// @brief Holds the traced and differentiated graph.
            Train::TrainingGraph built_;

            /// @brief Holds the tensor of every value of the graph.
            Graph::TensorPool pool_;

            /// @brief Holds the optimizer the trainer applies.
            std::unique_ptr<Optim::Optimizer> optimizer_;

            /// @brief Holds the compiled step.
            std::unique_ptr<Train::Trainer> trainer_;
    };
} // namespace SushiAI::Language::DecisionModels
