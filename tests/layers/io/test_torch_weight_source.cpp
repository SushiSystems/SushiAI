/**************************************************************************/
/* test_torch_weight_source.cpp                                           */
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
 * @file test_torch_weight_source.cpp
 * @brief Tests the weight source of a PyTorch file: its matcher, its tensors and its state key.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/torch_weight_source.hpp>
#include <SushiAI/io/weight_source.hpp>
#include <SushiAI/shape/shape.hpp>
#include "archive_bytes.hpp"
#include "weight_fixtures.hpp"

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Names the file in every message. */
    constexpr std::string_view K_ORIGIN = "fixture.bin";

    /** @brief Returns the registry that holds the torch row alone. */
    IO::WeightSourceRegistry torch_only()
    {
        IO::WeightSourceRegistry registry;
        EXPECT_TRUE(registry.add(IO::K_TORCH_WEIGHT_SOURCE_ROW).is_ok());
        return registry;
    }
} // namespace

/** @brief Tests that the matcher takes a zip archive and no other bytes. */
TEST(Small_TorchWeightSource, TheMatcherTakesAZipArchiveAndNoOtherBytes)
{
    const std::vector<std::byte> zip = SaTest::torch_fixture();
    const std::vector<std::byte> text = SaTest::bytes_of("hello, weights");

    EXPECT_TRUE(IO::is_torch_file(SaTest::view_of(zip)));
    EXPECT_FALSE(IO::is_torch_file(SaTest::view_of(text)));
}

/** @brief Tests that the source lists one tensor per entry of the dictionary, in pickle order. */
TEST(Small_TorchWeightSource, TheSourceListsOneTensorPerEntryInPickleOrder)
{
    IO::WeightSourceOptions options;
    options.origin = K_ORIGIN;
    options.state_key = "state_dict";

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<IO::WeightSource> source,
                            IO::open_torch_source(SaTest::torch_fixture(), options));

    ASSERT_EQ(source->tensors().size(), 3u);
    EXPECT_EQ(source->tensors()[0].name, "net.fc.weight");
    EXPECT_EQ(source->tensors()[0].stored_type, "float32");
    EXPECT_EQ(source->tensors()[0].shape, (Shape{2, 3}));
    EXPECT_EQ(source->tensors()[1].name, "net.fc.bias");
    EXPECT_EQ(source->tensors()[2].name, "net.bn.num_batches_tracked");
    EXPECT_EQ(source->tensors()[2].stored_type, "int64");
    EXPECT_EQ(source->origin(), K_ORIGIN);
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> weight, source->read_float32(0));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> bias, source->read_float32(1));
    EXPECT_EQ(weight, SaTest::fixture_weight());
    EXPECT_EQ(bias, SaTest::fixture_bias());
}

/** @brief Tests that a torch file whose root is a checkpoint says which key to name. */
TEST(Small_TorchWeightSource, ACheckpointRootNeedsTheStateKey)
{
    const IO::WeightSourceRegistry registry = torch_only();
    IO::WeightSourceOptions options;
    options.origin = K_ORIGIN;

    SA_EXPECT_FAILURE(IO::open_weight_source(registry, SaTest::torch_fixture(), options),
                      StatusCode::InvalidFile, "'state_dict'", "no tensor", "name map");
}

/** @brief Tests that a state key the file does not hold is refused, naming it. */
TEST(Small_TorchWeightSource, RefusesAStateKeyTheFileDoesNotHold)
{
    const IO::WeightSourceRegistry registry = torch_only();
    IO::WeightSourceOptions options;
    options.origin = K_ORIGIN;
    options.state_key = "model";

    SA_EXPECT_FAILURE(IO::open_weight_source(registry, SaTest::torch_fixture(), options),
                      StatusCode::InvalidFile, "fixture.bin", "'model'");
}
