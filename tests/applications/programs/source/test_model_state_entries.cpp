/**************************************************************************/
/* test_model_state_entries.cpp                                           */
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
 * @file test_model_state_entries.cpp
 * @brief Tests that model_state_entries lists the state records of a model file without an engine.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/shape/shape.hpp>
#include "model_state_entries.hpp"
#include "registries.hpp"

using namespace SushiAI;

namespace
{
    /** @brief Returns the model file of the tiny run: linear 6 to 8, gelu, linear 8 to 4. */
    std::filesystem::path tiny_model_file()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "tests" / "applications" / "programs"
               / "fixtures" / "tiny_run" / "tiny_model.json";
    }
} // namespace

/** @brief Tests that the two linear rows list a weight and a bias each, in the order of the rows. */
TEST(Small_ModelStateEntries, EachLinearRowListsItsWeightAndItsBiasInRowOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries registries, Apps::compose_registries());
    SA_ASSERT_OK_AND_ASSIGN(const Config::ModelSpec spec, Config::load_model(tiny_model_file()));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<NN::StateEntry> entries,
                            Apps::model_state_entries(registries.nodes, spec));

    // The gelu row holds no state, so the model lists four records: [8, 6], [8], [4, 8] and [4].
    ASSERT_EQ(entries.size(), 4u);
    EXPECT_EQ(entries[0].shape, (Shape{8, 6}));
    EXPECT_EQ(entries[1].shape, (Shape{8}));
    EXPECT_EQ(entries[2].shape, (Shape{4, 8}));
    EXPECT_EQ(entries[3].shape, (Shape{4}));
}

/** @brief Tests that every record of the model is a float32 parameter and that no two share a name. */
TEST(Small_ModelStateEntries, EveryRecordIsAFloat32ParameterUnderItsOwnName)
{
    SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries registries, Apps::compose_registries());
    SA_ASSERT_OK_AND_ASSIGN(const Config::ModelSpec spec, Config::load_model(tiny_model_file()));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<NN::StateEntry> entries,
                            Apps::model_state_entries(registries.nodes, spec));

    for (std::size_t first = 0; first < entries.size(); ++first)
    {
        EXPECT_EQ(entries[first].kind, NN::StateKind::Parameter) << entries[first].name;
        EXPECT_EQ(entries[first].dtype, DType::Float32) << entries[first].name;
        for (std::size_t second = first + 1; second < entries.size(); ++second)
            EXPECT_NE(entries[first].name, entries[second].name);
    }
}

/** @brief Tests that a registry that holds no row for a type of the file refuses the model. */
TEST(Small_ModelStateEntries, ARegistryWithoutTheRowsOfTheFileRefusesTheModel)
{
    SA_ASSERT_OK_AND_ASSIGN(const Config::ModelSpec spec, Config::load_model(tiny_model_file()));
    const NN::NodeRegistry empty;

    const Result<std::vector<NN::StateEntry>> entries = Apps::model_state_entries(empty, spec);

    EXPECT_FALSE(entries.has_value());
}
