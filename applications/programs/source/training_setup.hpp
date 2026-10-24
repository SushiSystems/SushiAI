/**************************************************************************/
/* training_setup.hpp                                                     */
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
 * @file training_setup.hpp
 * @brief Declares TrainingSetup, the model, data and training graph a training file builds.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/training_graph.hpp>

#include "registries.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @enum SetupScope
         * @brief Names how much of the training graph a setup holds memory for.
         */
        enum class SetupScope : std::uint8_t
        {
            Step,    /**< Every value of the graph, planned, so the setup can run its step. */
            Results, /**< The state, a gradient per parameter, the loss and the update sources. */
        };

        /**
         * @class TrainingSetup
         * @brief Holds what a training file builds on one engine before a run: the data, the
         *        model, the objective, the training graph, the state and the bound pool.
         *
         * `sushiai train` builds one over the data's batch, a replica over its micro-batch,
         * the master of a cluster run one of SetupScope::Results. It holds no optimizer.
         */
        class TrainingSetup final
        {
            public:
                /**
                 * @brief Builds the setup the files describe on @p engine.
                 * @param engine Outlives the setup.
                 * @param seed Seeds the random sites of the training trace.
                 * @param rows The rows of the traced batch; nothing takes the data's batch.
                 * @param scope What the pool holds memory for.
                 * @return The setup, or the first refusal of a factory, a trace or a bind.
                 */
                [[nodiscard]] static Result<std::unique_ptr<TrainingSetup>> make(
                    const Registries& registries, const Config::TrainSpec& spec,
                    const Config::ModelSpec& model_spec, SushiBLAS::Engine& engine,
                    std::uint64_t seed, std::optional<std::int64_t> rows = std::nullopt,
                    SetupScope scope = SetupScope::Step);

                TrainingSetup(const TrainingSetup&) = delete;
                TrainingSetup& operator=(const TrainingSetup&) = delete;
                TrainingSetup(TrainingSetup&&) = delete;
                TrainingSetup& operator=(TrainingSetup&&) = delete;
                ~TrainingSetup() = default;

                /** @brief Returns the batches of one epoch, read on the setup's engine. */
                [[nodiscard]] const Data::Dataset& dataset() const noexcept;

                /** @brief Returns the model the architecture file describes. */
                [[nodiscard]] const NN::ModelGraph& model() const noexcept;

                /** @brief Returns what the objective was built from. */
                [[nodiscard]] const Train::ObjectiveContext& given() const noexcept;

                /** @brief Returns the objective the training graph was built with. */
                [[nodiscard]] const Train::Objective& objective() const noexcept;

                /** @brief Returns the training graph over the traced batch. */
                [[nodiscard]] const Train::TrainingGraph& built() const noexcept;

                /** @brief Returns the parameters and buffers of the model. */
                [[nodiscard]] NN::ModelState& state() noexcept;

                /** @brief Returns the pool that holds the tensors of the training graph. */
                [[nodiscard]] Graph::TensorPool& pool() noexcept;

                /** @brief Returns each parameter with its gradient, in the order of the graph. */
                [[nodiscard]] const std::vector<Optim::ParameterSlot>& parameters()
                    const noexcept;

                /** @brief Returns the running-statistic updates of the training trace. */
                [[nodiscard]] const std::vector<Train::BufferUpdateSlot>& updates()
                    const noexcept;

                /**
                 * @brief Moves the running-statistic updates out, for a step to own.
                 * @return The updates; updates() is empty after the call.
                 */
                [[nodiscard]] std::vector<Train::BufferUpdateSlot> take_updates() noexcept;

            private:
                /** @brief Holds the parts read first; make() fills every other member. */
                TrainingSetup(std::unique_ptr<Data::Dataset> read_dataset,
                              NN::ModelGraph built_model, Train::ObjectiveContext context,
                              SushiBLAS::Engine& engine) noexcept;

                /// @brief The batches of one epoch, read on the setup's engine.
                std::unique_ptr<Data::Dataset> dataset_;

                /// @brief The model the architecture file describes.
                NN::ModelGraph model_;

                /// @brief What the objective was built from.
                Train::ObjectiveContext given_;

                /// @brief The objective the training graph was built with.
                std::unique_ptr<Train::Objective> objective_;

                /// @brief The training graph over the traced batch.
                Train::TrainingGraph built_;

                /// @brief The parameters and buffers of the model, as the initialisers left them.
                std::unique_ptr<NN::ModelState> state_;

                /// @brief The pool that holds the tensors of the training graph.
                std::unique_ptr<Graph::TensorPool> pool_;

                /// @brief Each parameter with its gradient, in the order of the training graph.
                std::vector<Optim::ParameterSlot> parameters_;

                /// @brief The running-statistic updates of the training trace.
                std::vector<Train::BufferUpdateSlot> updates_;
        };

    } // namespace Apps
} // namespace SushiAI
