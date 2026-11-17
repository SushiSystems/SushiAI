/**************************************************************************/
/* test_inference_preparation.cpp                                         */
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
 * @file test_inference_preparation.cpp
 * @brief Tests that prepare_inference() honours its two hooks and the weight file it names.
 * @author Mustafa Garip
 */

#include "inference_preparation.hpp"

#include <filesystem>
#include <memory>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/nn/init.hpp>

#include "prepared_inference_harness.hpp"
#include "registries.hpp"
#include "role_hidden_dataset.hpp"
#include "tiny_yolo_files.hpp"

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/applications/programs/README.md.
namespace SaTest = SushiAI::Test;
namespace Prepared = SushiAI::Apps::PreparedTest;
namespace TinyYolo = SushiAI::Apps::TinyYoloTest;

namespace
{
    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_InferencePreparation : public SaTest::EngineTest {};

    /** @brief Holds the files of the tiny_yolo fixture a request is made from. */
    struct FixtureFiles
    {
        /** @brief The model file of the fixture. */
        std::filesystem::path model_path = TinyYolo::fixture_directory() / "model.json";

        /** @brief A weight file no file stands behind. */
        std::filesystem::path weights_path = "absent_weights.sushi";

        /** @brief The parsed model file. */
        Config::ModelSpec model;

        /** @brief The data section of the fixture. */
        Config::DataSpec data;

        /** @brief The registries the build reads. */
        Apps::Registries registries;
    };

    /** @brief Reads the fixture files, or stops with the refusal. */
    FixtureFiles read_fixture_files()
    {
        FixtureFiles files;
        files.model = SaTest::require_value(Config::load_model(files.model_path));
        files.data = SaTest::require_value(Prepared::fixture_data_spec());
        files.registries = SaTest::require_value(Apps::compose_registries());
        return files;
    }
} // namespace

/** @brief Tests that a state left to prepare_inference() is loaded from the weight file. */
TEST_F(Medium_InferencePreparation, AWeightFileThatDoesNotExistRefusesTheBuild)
{
    const FixtureFiles files = read_fixture_files();
    const Apps::InferenceRequest request{files.model_path, files.weights_path, files.model,
                                         files.data};

    const auto build = Apps::prepare_inference(files.registries, backend(), request);

    EXPECT_FALSE(build.has_value());
}

/** @brief Tests that the fill hook runs once and takes the place of the weight load. */
TEST_F(Medium_InferencePreparation, TheFillHookRunsOnceInPlaceOfTheWeightLoad)
{
    const FixtureFiles files = read_fixture_files();
    int calls = 0;
    Apps::InferenceRequest request{files.model_path, files.weights_path, files.model, files.data};
    request.hooks.fill_state = [&calls](ExecutionBackend& filled, NN::ModelState& state)
    {
        ++calls;
        return NN::initialize(filled, state);
    };

    const auto build = Apps::prepare_inference(files.registries, backend(), request);

    ASSERT_TRUE(build.has_value()) << build.status().message();
    EXPECT_EQ(calls, 1);
}

/** @brief Tests that the dataset of the build is the one the wrap hook returned. */
TEST_F(Medium_InferencePreparation, TheWrapHookServesTheDatasetItReturns)
{
    const FixtureFiles files = read_fixture_files();
    Apps::InferenceRequest request{files.model_path, files.weights_path, files.model, files.data};
    request.hooks.fill_state = [](ExecutionBackend& filled, NN::ModelState& state)
    {
        return NN::initialize(filled, state);
    };
    request.hooks.wrap_dataset = [](std::unique_ptr<Data::Dataset> read)
    {
        return std::make_unique<Prepared::RoleHiddenDataset>(std::move(read), Data::K_ROLE_FIT);
    };

    const auto build = Apps::prepare_inference(files.registries, backend(), request);

    ASSERT_TRUE(build.has_value()) << build.status().message();
    EXPECT_TRUE(Data::find_target(build.value()->prepared().dataset.targets(), Data::K_ROLE_FIT)
                == nullptr);
}
