/**************************************************************************/
/* test_safetensors_writer.cpp                                            */
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
 * @file test_safetensors_writer.cpp
 * @brief Tests the safetensors writer against the reader for each element type it writes.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/safetensors_reader.hpp>
#include <SushiAI/io/safetensors_writer.hpp>
#include <SushiAI/shape/shape.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Returns a tensor named @p name of @p dtype and @p shape holding seeded bytes. */
    IO::ImportedTensor make_tensor(const std::string& name, DType dtype, const Shape& shape)
    {
        IO::ImportedTensor tensor;
        tensor.record.name = name;
        tensor.record.dtype = dtype;
        tensor.record.shape = shape;
        tensor.payload.resize(tensor.record.payload_bytes());
        for (std::size_t i = 0; i < tensor.payload.size(); ++i)
            tensor.payload[i] = static_cast<std::byte>((i * 29u + name.size()) & 0x3Fu);
        return tensor;
    }
} // namespace

/** @brief Tests that the reader reads back the name, type, shape and bytes of each tensor. */
TEST(Small_SafetensorsWriter, TheReaderReadsBackEveryTensor)
{
    const std::vector<IO::ImportedTensor> tensors = {
        make_tensor("conv.weight", DType::Float32, Shape{2, 3, 3, 1}),
        make_tensor("bn.running_var", DType::Float64, Shape{5}),
        make_tensor("head.bias", DType::Half, Shape{3})};
    const IO::SafetensorsType types[] = {IO::SafetensorsType::Float32,
                                         IO::SafetensorsType::Float64,
                                         IO::SafetensorsType::Float16};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::byte> file,
                            IO::write_safetensors(SushiRuntime::span<const IO::ImportedTensor>(
                                tensors.data(), tensors.size())));
    SA_ASSERT_OK_AND_ASSIGN(const IO::SafetensorsReader reader,
                            IO::SafetensorsReader::read(
                                SushiRuntime::span<const std::byte>(file.data(), file.size()),
                                "written"));

    ASSERT_EQ(reader.tensors().size(), tensors.size());
    EXPECT_EQ(reader.tensors()[0].offset % 8u, 0u);
    for (std::size_t i = 0; i < tensors.size(); ++i)
    {
        const IO::SafetensorsTensor* read = reader.find(tensors[i].record.name);
        ASSERT_NE(read, nullptr) << tensors[i].record.name;
        EXPECT_EQ(read->type, types[i]);
        EXPECT_EQ(read->shape, tensors[i].record.shape);
        const SushiRuntime::span<const std::byte> bytes = reader.contents(*read);
        EXPECT_EQ(std::vector<std::byte>(bytes.begin(), bytes.end()), tensors[i].payload);
    }
}

/** @brief Tests that an integer tensor is refused by name. */
TEST(Small_SafetensorsWriter, AnIntegerTensorIsRefused)
{
    const std::vector<IO::ImportedTensor> tensors = {
        make_tensor("steps", DType::Int64, Shape{1})};

    SA_EXPECT_FAILURE(IO::write_safetensors(SushiRuntime::span<const IO::ImportedTensor>(
                          tensors.data(), tensors.size())),
                      StatusCode::Unsupported, "'steps'", "int64");
}
