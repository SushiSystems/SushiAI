/**************************************************************************/
/* test_weight_source.cpp                                                 */
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
 * @file test_weight_source.cpp
 * @brief Tests that a weight file is sniffed into its container and read through the same seam.
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

    /** @brief Returns the registry of the two containers, torch first. */
    IO::WeightSourceRegistry both_containers()
    {
        IO::WeightSourceRegistry registry;
        EXPECT_TRUE(registry.add(IO::K_TORCH_WEIGHT_SOURCE_ROW).is_ok());
        EXPECT_TRUE(registry.add(IO::K_SAFETENSORS_WEIGHT_SOURCE_ROW).is_ok());
        return registry;
    }

    /** @brief Returns the names of the tensors of @p source, in the order it lists them. */
    std::vector<std::string> names_of(const IO::WeightSource& source)
    {
        std::vector<std::string> names;
        for (const IO::SourceTensor& tensor : source.tensors())
            names.push_back(tensor.name);
        return names;
    }
} // namespace

/** @brief Tests that each matcher accepts its own container and not the other's. */
TEST(Small_WeightSource, EachMatcherAcceptsOnlyItsOwnContainer)
{
    const std::vector<std::byte> torch = SaTest::torch_fixture();
    const std::vector<std::byte> safetensors = SaTest::safetensors_fixture();

    EXPECT_TRUE(IO::is_torch_file(SaTest::view_of(torch)));
    EXPECT_FALSE(IO::is_safetensors_file(SaTest::view_of(torch)));
    EXPECT_TRUE(IO::is_safetensors_file(SaTest::view_of(safetensors)));
    EXPECT_FALSE(IO::is_torch_file(SaTest::view_of(safetensors)));
}

/** @brief Tests that sniffing opens a zip as torch and a length-prefixed header as safetensors. */
TEST(Small_WeightSource, SniffingPicksTheSourceOfTheContainer)
{
    const IO::WeightSourceRegistry registry = both_containers();
    IO::WeightSourceOptions options;
    options.origin = K_ORIGIN;
    options.state_key = "state_dict";

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<IO::WeightSource> torch,
                            IO::open_weight_source(registry, SaTest::torch_fixture(), options));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<IO::WeightSource> safetensors,
        IO::open_weight_source(registry, SaTest::safetensors_fixture(), options));

    // Only the torch file holds the counter, and only the torch file spells types as DType.
    EXPECT_EQ(names_of(*torch),
              (std::vector<std::string>{"net.fc.weight", "net.fc.bias",
                                        "net.bn.num_batches_tracked"}));
    EXPECT_EQ(torch->tensors()[0].stored_type, "float32");
    EXPECT_EQ(torch->tensors()[2].stored_type, "int64");
    EXPECT_EQ(names_of(*safetensors),
              (std::vector<std::string>{"net.fc.weight", "net.fc.bias"}));
    EXPECT_EQ(safetensors->tensors()[0].stored_type, "F32");
    EXPECT_EQ(torch->origin(), K_ORIGIN);
}

/** @brief Tests that both sources give the same shapes and the same float32 values. */
TEST(Small_WeightSource, BothSourcesReadTheSameShapesAndValues)
{
    const IO::WeightSourceRegistry registry = both_containers();
    IO::WeightSourceOptions options;
    options.origin = K_ORIGIN;
    options.state_key = "state_dict";
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<IO::WeightSource> torch,
                            IO::open_weight_source(registry, SaTest::torch_fixture(), options));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<IO::WeightSource> safetensors,
        IO::open_weight_source(registry, SaTest::safetensors_fixture(), options));

    for (std::size_t i = 0; i < 2; ++i)
        EXPECT_EQ(torch->tensors()[i].shape, safetensors->tensors()[i].shape);
    EXPECT_EQ(torch->tensors()[0].shape, (Shape{2, 3}));
    EXPECT_EQ(torch->tensors()[1].shape, (Shape{2}));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> torch_weight, torch->read_float32(0));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> torch_bias, torch->read_float32(1));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> file_weight, safetensors->read_float32(0));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> file_bias, safetensors->read_float32(1));
    EXPECT_EQ(torch_weight, SaTest::fixture_weight());
    EXPECT_EQ(file_weight, SaTest::fixture_weight());
    EXPECT_EQ(torch_bias, SaTest::fixture_bias());
    EXPECT_EQ(file_bias, SaTest::fixture_bias());
}

/** @brief Tests that bytes of neither container are refused, naming the origin and both formats. */
TEST(Small_WeightSource, RefusesBytesOfNeitherContainer)
{
    const IO::WeightSourceRegistry registry = both_containers();
    IO::WeightSourceOptions options;
    options.origin = K_ORIGIN;

    SA_EXPECT_FAILURE(IO::open_weight_source(registry, SaTest::bytes_of("hello, weights"), options),
                      StatusCode::InvalidFile, "fixture.bin", "'torch'", "'safetensors'");
}

/** @brief Tests that a second row of one format is refused, and that the rows keep their order. */
TEST(Small_WeightSource, TheRegistryRefusesAFormatTwiceAndKeepsTheOrder)
{
    IO::WeightSourceRegistry registry = both_containers();

    SA_EXPECT_FAILURE(registry.add(IO::K_TORCH_WEIGHT_SOURCE_ROW), StatusCode::InvalidDescription,
                      "'torch'");
    ASSERT_EQ(registry.rows().size(), 2u);
    EXPECT_EQ(registry.rows()[0].format, "torch");
    EXPECT_EQ(registry.rows()[1].format, "safetensors");
}
