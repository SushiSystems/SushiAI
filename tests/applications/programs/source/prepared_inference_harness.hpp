/**************************************************************************/
/* prepared_inference_harness.hpp                                         */
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
 * @file prepared_inference_harness.hpp
 * @brief Declares the builder of a PreparedInference over the tiny_yolo fixture, through
 *        prepare_inference().
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <SushiAI/config/data_adapt.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/model_state.hpp>

#include "inference_preparation.hpp"
#include "inference_run.hpp"
#include "registries.hpp"
#include "role_hidden_dataset.hpp"
#include "tiny_yolo_files.hpp"

namespace SushiAI::Apps::PreparedTest
{
    /**
     * @struct InferenceHarness
     * @brief Holds the files of the tiny_yolo fixture and the InferenceBuild that
     *        prepare_inference() made over them, with initialised weights.
     */
    struct InferenceHarness
    {
        InferenceHarness() = default;
        InferenceHarness(const InferenceHarness&) = delete;
        InferenceHarness& operator=(const InferenceHarness&) = delete;
        InferenceHarness(InferenceHarness&&) = delete;
        InferenceHarness& operator=(InferenceHarness&&) = delete;
        ~InferenceHarness() = default;

        /** @brief The model file of the fixture. */
        std::filesystem::path model_path;

        /** @brief The weight file the report names; no file stands behind it. */
        std::filesystem::path weights_path = "weights.sushi";

        /** @brief The parsed model file; a test may change what a report reads from it. */
        Config::ModelSpec model;

        /** @brief The registries the model and the data were built from. */
        std::optional<Registries> registries;

        /** @brief What prepare_inference() built over the members above. */
        std::unique_ptr<InferenceBuild> build;

        /** @brief The report input of @c build; a test may change its thresholds. */
        PreparedInference* prepared = nullptr;
    };

    /** @brief Parses the data section of the tiny_yolo fixture. */
    [[nodiscard]] inline Result<Config::DataSpec> fixture_data_spec()
    {
        SA_TRY_ASSIGN(const Config::Value document,
                      Config::parse_json(R"({"sushiai_data": 1, "data": )"
                                             + TinyYoloTest::data_block() + "}",
                                         "<test>"));
        return Config::parse_data(document, "<test>", {});
    }

    /**
     * @brief Builds the PreparedInference of the tiny_yolo fixture on @p backend.
     * @param hidden_role A role the data does not publish; empty hides none.
     * @return The harness, or the first failure of a read, a build or a bind.
     */
    [[nodiscard]] inline Result<std::unique_ptr<InferenceHarness>> make_inference_harness(
        ExecutionBackend& backend, std::string_view hidden_role = {})
    {
        std::unique_ptr<InferenceHarness> made = std::make_unique<InferenceHarness>();
        made->model_path = TinyYoloTest::fixture_directory() / "model.json";
        SA_TRY_ASSIGN(made->model, Config::load_model(made->model_path));
        SA_TRY_ASSIGN(made->registries, compose_registries());

        SA_TRY_ASSIGN(const Config::DataSpec data_spec, fixture_data_spec());

        InferenceRequest request{made->model_path, made->weights_path, made->model, data_spec};
        request.hooks.fill_state = [](ExecutionBackend& filled, NN::ModelState& state)
        {
            return NN::initialize(filled, state);
        };
        if (!hidden_role.empty())
        {
            request.hooks.wrap_dataset = [hidden_role](std::unique_ptr<Data::Dataset> read)
            {
                return std::make_unique<RoleHiddenDataset>(std::move(read), hidden_role);
            };
        }
        SA_TRY_ASSIGN(made->build, prepare_inference(*made->registries, backend, request));
        made->prepared = &made->build->prepared();
        return std::move(made);
    }
} // namespace SushiAI::Apps::PreparedTest
