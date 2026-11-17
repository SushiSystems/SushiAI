/**************************************************************************/
/* decision_session.hpp                                                   */
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
 * @file decision_session.hpp
 * @brief Declares DecisionSession, a decision model built from its model file and its training
 *        file, with the two compiled steps that train it and read what it answers.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>

#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/decision_models/decision_predictions.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <SushiAI/train/trainer.hpp>

namespace SushiAI::Language::DecisionModels
{
    class DecisionRun;

    /** @brief Holds the two files a decision session is built from, and the rows they may name. */
    struct DecisionSessionOptions
    {
        /** @brief Holds the rows the model file may name; it must outlive the session's make(). */
        const NN::NodeRegistry* nodes = nullptr;

        /** @brief Holds the types the objective block may name; it must outlive make() as well. */
        const Train::ObjectiveRegistry* objectives = nullptr;

        /** @brief Holds the model file, as Config::load_model() reads it. */
        Config::ModelSpec model;

        /** @brief Holds the training file, as Config::load_train() reads it. */
        Config::TrainSpec train;
    };

    /**
     * @brief Owns a decision model, its state and the two steps compiled over that state.
     *
     * The model, the objective and the optimizer come from the two files by the calls
     * `sushiai train` makes. fit() trains; predict() runs the second step, whose optimizer is
     * an SGD of rate 0, so it reads the weights fit() left. Both steps compile once.
     * @see modules/language/decision_models/README.md, section "The session".
     */
    class DecisionSession final
    {
        public:
            /**
             * @brief Builds the model of @p options and compiles both steps.
             * @param backend Outlives the session.
             * @param training Its batch size, texts and roles fix the compiled shapes.
             * @return The session; Unsupported for a training file that states a schedule, a
             *         cluster or a file to start from; or the first failure of the build.
             */
            [[nodiscard]] static Result<std::unique_ptr<DecisionSession>> make(
                ExecutionBackend& backend, const DecisionSessionOptions& options,
                const Data::Dataset& training) noexcept;

            DecisionSession(const DecisionSession&) = delete;
            DecisionSession& operator=(const DecisionSession&) = delete;
            DecisionSession(DecisionSession&&) = delete;
            DecisionSession& operator=(DecisionSession&&) = delete;
            ~DecisionSession();

            /**
             * @brief Trains over @p dataset for the epochs of the training file.
             * @param dataset Its batch size, texts and roles are those of the data given to make().
             * @return The report of Trainer::fit().
             */
            [[nodiscard]] Result<Train::RunReport> fit(const Data::Dataset& dataset) noexcept;

            /**
             * @brief Reads the answers of the model to every whole batch of @p dataset.
             * @return The three outputs of every record with its answers, or the failure of a
             *         read or of a step; InvalidDescription for a dataset that publishes no
             *         `choice` role.
             */
            [[nodiscard]] Result<DecisionPredictions> predict(const Data::Dataset& dataset) noexcept;

        private:
            /** @brief Takes the model make() built and the epochs of the training file. */
            DecisionSession(NN::ModelGraph model, std::int64_t epochs) noexcept;

            /// @brief Holds the epochs the training file states.
            std::int64_t epochs_;

            /// @brief Holds the model both steps trace.
            NN::ModelGraph model_;

            /// @brief Holds the parameters both steps share; null until the first step is made.
            std::unique_ptr<NN::ModelState> state_;

            /// @brief Holds the step that learns.
            std::unique_ptr<DecisionRun> training_;

            /// @brief Holds the step that reads.
            std::unique_ptr<DecisionRun> probe_;
    };
} // namespace SushiAI::Language::DecisionModels
