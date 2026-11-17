/**************************************************************************/
/* inference_preparation.hpp                                              */
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
 * @file inference_preparation.hpp
 * @brief Declares the build `sushiai inference` runs before a report starts, and what owns it.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "inference_preparation.hpp".
 */

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include "inference_run.hpp"
#include "registries.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @struct InferenceHooks
         * @brief Names the two steps of the build a caller may replace; an empty one keeps the
         *        behaviour of `sushiai inference`.
         */
        struct InferenceHooks
        {
            /** @brief Fills the state of the model; empty loads the weight file of the request. */
            std::function<Status(ExecutionBackend&, NN::ModelState&)> fill_state;

            /** @brief Wraps the dataset the data block built; empty serves it as it is. */
            std::function<std::unique_ptr<Data::Dataset>(std::unique_ptr<Data::Dataset>)>
                wrap_dataset;
        };

        /**
         * @struct InferenceRequest
         * @brief Names the files and the settings a PreparedInference is built from.
         *
         * Each reference must outlive the InferenceBuild that prepare_inference() returns,
         * because the PreparedInference refers to them; @c data is read during the call only.
         */
        struct InferenceRequest
        {
            /** @brief The model file, as the command line names it. */
            const std::filesystem::path& model_path;

            /** @brief The weight file, as the command line names it. */
            const std::filesystem::path& weights_path;

            /** @brief The parsed model file. */
            const Config::ModelSpec& model;

            /** @brief The parsed data section. */
            const Config::DataSpec& data;

            /** @brief The thresholds the evaluation file states. */
            Config::DetectionSettings detection = {};

            /** @brief Whether a detection report scores its detections against the data's truth. */
            Comparison comparison = Comparison::WithTruth;

            /** @brief The steps of the build the caller replaces. */
            InferenceHooks hooks = {};
        };

        /**
         * @class InferenceBuild
         * @brief Owns everything `sushiai inference` builds before a report starts and the
         *        PreparedInference over it.
         */
        class InferenceBuild final
        {
            public:
                InferenceBuild(const InferenceBuild&) = delete;
                InferenceBuild& operator=(const InferenceBuild&) = delete;
                InferenceBuild(InferenceBuild&&) = delete;
                InferenceBuild& operator=(InferenceBuild&&) = delete;
                ~InferenceBuild() = default;

                /** @brief Returns the report input over the members of this build. */
                [[nodiscard]] PreparedInference& prepared() noexcept;

            private:
                friend Result<std::unique_ptr<InferenceBuild>> prepare_inference(
                    const Registries& registries, ExecutionBackend& backend,
                    const InferenceRequest& request);

                /** @brief Builds the pool on the engine of @p backend; prepare_inference() fills
                 *         the rest. */
                explicit InferenceBuild(ExecutionBackend& backend);

                /// @brief The batches the report runs over.
                std::unique_ptr<Data::Dataset> dataset_;

                /// @brief The model the evaluation graph is traced from.
                std::optional<NN::ModelGraph> graph_;

                /// @brief The evaluation graph of the model over one batch.
                std::optional<Train::TracedStep> traced_;

                /// @brief The parameters and buffers of the model.
                std::optional<NN::ModelState> state_;

                /// @brief The pool the evaluation step lives in.
                Graph::TensorPool pool_;

                /// @brief The compiled forward pass over the state.
                std::unique_ptr<Train::EvaluationStep> step_;

                /// @brief The report input over the members above.
                std::optional<PreparedInference> prepared_;
        };

        /**
         * @brief Builds the data, the model, the evaluation graph, the state and the compiled
         *        forward pass of @p request, and the PreparedInference over them.
         * @param registries Outlives the call; the build reads the node and dataset rows.
         * @param backend Outlives the build; its engine allocates the tensors.
         * @return The build; InvalidDescription on an input-shape mismatch or an untasked model
         *         with other than one output; else the first failure of a step.
         */
        [[nodiscard]] Result<std::unique_ptr<InferenceBuild>> prepare_inference(
            const Registries& registries, ExecutionBackend& backend,
            const InferenceRequest& request);
    } // namespace Apps
} // namespace SushiAI
