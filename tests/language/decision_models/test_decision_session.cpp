/**************************************************************************/
/* test_decision_session.cpp                                              */
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
 * @file test_decision_session.cpp
 * @brief Tests that DecisionSession refuses a training file that states what it does not run.
 * @author Mustafa Garip
 */

#include <filesystem>
#include <memory>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/decision_dataset.hpp>
#include <SushiAI/language/decision_models/decision_model.hpp>
#include <SushiAI/language/decision_models/decision_session.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include "decision_fixture.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::DecisionModels;
using DecisionTest::example_file;
using DecisionTest::K_TRAIN_SEED;
using DecisionTest::make_split;
// SaTest stays reachable inside a TEST_F body; see tests/language/decision_models/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine for the cases that read the example files and a dataset. */
    class Medium_DecisionSession : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that a session refuses a training file that states what it does not run. */
TEST_F(Medium_DecisionSession, ASessionRefusesAFileThatStatesAScheduleOrAStart)
{
    const SaTest::ScratchDirectory scratch("sushiai_decision_refusals");
    std::filesystem::create_directories(scratch.path());
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Datasets::DecisionDataset> dataset,
        make_split(scratch.path(), "refusals.sadc", K_TRAIN_SEED, K_TEST_EXTENTS.batch));
    const NN::NodeRegistry nodes{};
    const Train::ObjectiveRegistry objectives{};
    DecisionSessionOptions scheduled;
    scheduled.nodes = &nodes;
    scheduled.objectives = &objectives;
    SA_ASSERT_OK_AND_ASSIGN(scheduled.model, Config::load_model(example_file("model.json")));
    SA_ASSERT_OK_AND_ASSIGN(scheduled.train, Config::load_train(example_file("train.json")));
    DecisionSessionOptions resumed = scheduled;
    scheduled.train.schedule = Config::Value::object({"type"}, {Config::Value::string("cosine")});
    resumed.train.resume = "previous.sushi";

    SA_EXPECT_FAILURE(DecisionSession::make(SaTest::TestEnvironment::backend(), scheduled,
                                            *dataset),
                      StatusCode::Unsupported, "'schedule'");
    SA_EXPECT_FAILURE(DecisionSession::make(SaTest::TestEnvironment::backend(), resumed,
                                            *dataset),
                      StatusCode::Unsupported, "'resume'");
}
