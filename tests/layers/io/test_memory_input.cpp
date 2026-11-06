/**************************************************************************/
/* test_memory_input.cpp                                                  */
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
 * @file test_memory_input.cpp
 * @brief Tests that MemoryInput serves the bytes of a checkpoint to its reader.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/memory_input.hpp>

using namespace SushiAI;

/** @brief Tests that a checkpoint held in memory opens a reader that yields its record. */
TEST(Small_MemoryInput, ServesTheBytesOfACheckpointToItsReader)
{
    IO::TensorRecord header;
    header.name = "0.bias";
    header.shape = Shape{3};

    const float values[3] = {1.5f, -2.0f, 4.0f};
    std::ostringstream out;
    SA_ASSERT_OK_AND_ASSIGN(IO::CheckpointWriter writer, IO::CheckpointWriter::create(out, 1));
    SA_EXPECT_OK(writer.write(header, values));
    writer.finish();

    const std::string bytes = out.str();
    IO::MemoryInput in(SushiRuntime::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));

    SA_ASSERT_OK_AND_ASSIGN(IO::CheckpointReader reader, IO::CheckpointReader::open(in));
    EXPECT_EQ(reader.count(), 1u);

    SA_ASSERT_OK_AND_ASSIGN(const IO::TensorRecord record, reader.next());
    EXPECT_EQ(record.name, "0.bias");

    float read[3] = {0.0f, 0.0f, 0.0f};
    SA_EXPECT_OK(reader.read_payload(record, read));
    EXPECT_EQ(std::memcmp(read, values, sizeof values), 0);
}
