/**************************************************************************/
/* test_torch_writer.cpp                                                  */
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
 * @file test_torch_writer.cpp
 * @brief Tests the PyTorch state dictionary writer against the PyTorch weight source, at the
 *        root and under a state key.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/torch_weight_source.hpp>
#include <SushiAI/io/torch_writer.hpp>
#include <SushiAI/io/weight_source.hpp>
#include <SushiAI/shape/shape.hpp>

using namespace SushiAI;
using SushiAI::Test::require_value;

namespace
{
    /** @brief Returns a float32 tensor named @p name of @p shape holding seeded values. */
    IO::ImportedTensor make_tensor(const std::string& name, const Shape& shape,
                                   std::vector<float>& values)
    {
        values.resize(static_cast<std::size_t>(shape.num_elements()));
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>((i * 17u + name.size()) % 23u) * 0.25f - 2.0f;
        IO::ImportedTensor tensor;
        tensor.record.name = name;
        tensor.record.dtype = DType::Float32;
        tensor.record.shape = shape;
        tensor.payload.resize(values.size() * sizeof(float));
        std::memcpy(tensor.payload.data(), values.data(), tensor.payload.size());
        return tensor;
    }

    /** @brief Writes three tensors under @p state_key and reads each back through the source. */
    void expect_round_trip(std::string_view state_key)
    {
        std::vector<std::vector<float>> values(3);
        const std::vector<IO::ImportedTensor> tensors = {
            make_tensor("conv.weight", Shape{4, 3, 3, 2}, values[0]),
            make_tensor("conv.bias", Shape{4}, values[1]),
            make_tensor("fc.weight", Shape{2, 6}, values[2])};

        SA_ASSERT_OK_AND_ASSIGN(std::vector<std::byte> file,
                                IO::write_torch_state_dict(
                                    SushiRuntime::span<const IO::ImportedTensor>(tensors.data(),
                                                                                 tensors.size()),
                                    "model", state_key));
        ASSERT_TRUE(IO::is_torch_file(
            SushiRuntime::span<const std::byte>(file.data(), file.size())));
        IO::WeightSourceOptions options;
        options.origin = "written.pth";
        options.state_key = state_key;
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<IO::WeightSource> source,
                                IO::open_torch_source(std::move(file), options));

        ASSERT_EQ(source->tensors().size(), tensors.size());
        for (std::size_t i = 0; i < tensors.size(); ++i)
        {
            EXPECT_EQ(source->tensors()[i].name, tensors[i].record.name);
            EXPECT_EQ(source->tensors()[i].shape, tensors[i].record.shape);
            EXPECT_EQ(require_value(source->read_float32(i)), values[i]);
        }
    }
} // namespace

/** @brief Tests that a dictionary written as the root reads back tensor by tensor. */
TEST(Small_TorchWriter, TheSourceReadsBackADictionaryAtTheRoot)
{
    expect_round_trip("");
}

/** @brief Tests that a dictionary written under a state key reads back tensor by tensor. */
TEST(Small_TorchWriter, TheSourceReadsBackADictionaryUnderAStateKey)
{
    expect_round_trip("state_dict");
}

/** @brief Tests that an integer tensor and a folder with a slash are refused. */
TEST(Small_TorchWriter, WhatItCannotWriteIsRefused)
{
    IO::ImportedTensor steps;
    steps.record.name = "steps";
    steps.record.dtype = DType::Int64;
    steps.record.shape = Shape{1};
    steps.payload.resize(8);

    SA_EXPECT_FAILURE(IO::write_torch_state_dict(
                          SushiRuntime::span<const IO::ImportedTensor>(&steps, 1), "model", ""),
                      StatusCode::Unsupported, "'steps'", "int64");
    SA_EXPECT_FAILURE(IO::write_torch_state_dict({}, "a/b", ""), StatusCode::InvalidDescription,
                      "'a/b'");
}
