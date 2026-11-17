/**************************************************************************/
/* test_safetensors_weight_source.cpp                                     */
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
 * @file test_safetensors_weight_source.cpp
 * @brief Tests the weight source of a safetensors file: its matcher, its tensors and its state key.
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
#include <SushiAI/io/safetensors_weight_source.hpp>
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

    /** @brief Returns the registry that holds the safetensors row alone. */
    IO::WeightSourceRegistry safetensors_only()
    {
        IO::WeightSourceRegistry registry;
        EXPECT_TRUE(registry.add(IO::K_SAFETENSORS_WEIGHT_SOURCE_ROW).is_ok());
        return registry;
    }
} // namespace

/** @brief Tests that the source lists the tensors of the file in the order of their bytes. */
TEST(Small_SafetensorsWeightSource, TheSourceListsTheTensorsOfTheFileWithTheirValues)
{
    IO::WeightSourceOptions options;
    options.origin = K_ORIGIN;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<IO::WeightSource> source,
                            IO::open_safetensors_source(SaTest::safetensors_fixture(), options));

    ASSERT_EQ(source->tensors().size(), 2u);
    EXPECT_EQ(source->tensors()[0].name, "net.fc.weight");
    EXPECT_EQ(source->tensors()[0].stored_type, "F32");
    EXPECT_EQ(source->tensors()[0].shape, (Shape{2, 3}));
    EXPECT_EQ(source->tensors()[1].name, "net.fc.bias");
    EXPECT_EQ(source->tensors()[1].shape, (Shape{2}));
    EXPECT_EQ(source->origin(), K_ORIGIN);
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> weight, source->read_float32(0));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> bias, source->read_float32(1));
    EXPECT_EQ(weight, SaTest::fixture_weight());
    EXPECT_EQ(bias, SaTest::fixture_bias());
}

/** @brief Tests that a header length beyond the file, or no '{', is no safetensors file. */
TEST(Small_SafetensorsWeightSource, ASafetensorsFileNeedsAFittingLengthAndAnOpeningBrace)
{
    std::string long_header;
    SaTest::put_le(long_header, 1000, 8);
    long_header += "{}";
    std::string no_brace;
    SaTest::put_le(no_brace, 2, 8);
    no_brace += "[]";
    std::string short_file;
    SaTest::put_le(short_file, 0, 8);

    const std::vector<std::byte> long_bytes = SaTest::bytes_of(long_header);
    const std::vector<std::byte> brace_bytes = SaTest::bytes_of(no_brace);
    const std::vector<std::byte> short_bytes = SaTest::bytes_of(short_file);
    EXPECT_FALSE(IO::is_safetensors_file(SaTest::view_of(long_bytes)));
    EXPECT_FALSE(IO::is_safetensors_file(SaTest::view_of(brace_bytes)));
    EXPECT_FALSE(IO::is_safetensors_file(SaTest::view_of(short_bytes)));
}

/** @brief Tests that the safetensors source ignores a state key, which the file cannot nest. */
TEST(Small_SafetensorsWeightSource, SafetensorsIgnoresTheStateKey)
{
    const IO::WeightSourceRegistry registry = safetensors_only();
    IO::WeightSourceOptions options;
    options.origin = K_ORIGIN;
    options.state_key = "model";

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<IO::WeightSource> source,
        IO::open_weight_source(registry, SaTest::safetensors_fixture(), options));
    EXPECT_EQ(source->tensors().size(), 2u);
}
