/**************************************************************************/
/* test_safetensors_reader.cpp                                            */
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
 * @file test_safetensors_reader.cpp
 * @brief Tests the safetensors reader on files written by hand: the ranges it lists and the files it refuses.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/safetensors_reader.hpp>
#include "archive_bytes.hpp"

using namespace SushiAI;

namespace
{
    /** @brief Names the file in every message. */
    constexpr std::string_view K_ORIGIN = "fixture.safetensors";

    /** @brief Returns a file of the 8-byte length of @p header, @p header and @p data. */
    std::vector<std::byte> file_of(const std::string& header, const std::string& data)
    {
        std::string out;
        SushiAI::Test::put_le(out, header.size(), 8);
        out += header;
        out += data;
        return SushiAI::Test::bytes_of(out);
    }

    /** @brief Returns 16 bytes of data, which every refusal case states its ranges against. */
    std::string sixteen_bytes()
    {
        return std::string(16, '\x01');
    }
} // namespace

/** @brief Tests that every tensor lists its dtype, shape, offset and length, in the order of its bytes. */
TEST(Small_SafetensorsReader, ListsEveryTensorInTheOrderOfItsBytes)
{
    const std::string header =
        R"({"__metadata__":{"format":"pt"},)"
        R"("a":{"dtype":"F16","shape":[3],"data_offsets":[8,14]},)"
        R"("d":{"dtype":"F64","shape":[],"data_offsets":[16,24]},)"
        R"("b":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},)"
        R"("c":{"dtype":"BF16","shape":[1],"data_offsets":[14,16]}})";
    std::string data;
    SushiAI::Test::put_le(data, 0x3F800000u, 4);
    SushiAI::Test::put_le(data, 0xC0000000u, 4);
    SushiAI::Test::put_le(data, 0x3C00u, 2);
    SushiAI::Test::put_le(data, 0x3800u, 2);
    SushiAI::Test::put_le(data, 0xC000u, 2);
    SushiAI::Test::put_le(data, 0x3F80u, 2);
    SushiAI::Test::put_le(data, 0x3FE0000000000000ull, 8);
    const std::vector<std::byte> bytes = file_of(header, data);
    const std::size_t data_start = 8 + header.size();

    SA_ASSERT_OK_AND_ASSIGN(
        const IO::SafetensorsReader reader,
        IO::SafetensorsReader::read(SushiAI::Test::view_of(bytes), K_ORIGIN));

    ASSERT_EQ(reader.tensors().size(), 4u);
    const IO::SafetensorsTensor& b = reader.tensors()[0];
    EXPECT_EQ(b.name, "b");
    EXPECT_EQ(b.type, IO::SafetensorsType::Float32);
    ASSERT_EQ(b.shape.rank(), 2);
    EXPECT_EQ(b.shape[0], 1);
    EXPECT_EQ(b.shape[1], 2);
    EXPECT_EQ(b.offset, data_start + 0);
    EXPECT_EQ(b.length, 8u);
    EXPECT_EQ(reader.read_float32(b), (std::vector<float>{1.0f, -2.0f}));

    const IO::SafetensorsTensor& a = reader.tensors()[1];
    EXPECT_EQ(a.name, "a");
    EXPECT_EQ(a.type, IO::SafetensorsType::Float16);
    ASSERT_EQ(a.shape.rank(), 1);
    EXPECT_EQ(a.shape[0], 3);
    EXPECT_EQ(a.offset, data_start + 8);
    EXPECT_EQ(a.length, 6u);
    EXPECT_EQ(reader.read_float32(a), (std::vector<float>{1.0f, 0.5f, -2.0f}));

    const IO::SafetensorsTensor& c = reader.tensors()[2];
    EXPECT_EQ(c.name, "c");
    EXPECT_EQ(c.type, IO::SafetensorsType::BFloat16);
    EXPECT_EQ(c.offset, data_start + 14);
    EXPECT_EQ(c.length, 2u);
    EXPECT_EQ(reader.read_float32(c), (std::vector<float>{1.0f}));

    const IO::SafetensorsTensor& d = reader.tensors()[3];
    EXPECT_EQ(d.name, "d");
    EXPECT_EQ(d.type, IO::SafetensorsType::Float64);
    EXPECT_EQ(d.shape.rank(), 0);
    EXPECT_EQ(d.offset, data_start + 16);
    EXPECT_EQ(d.length, 8u);
    EXPECT_EQ(reader.read_float32(d), (std::vector<float>{0.5f}));

    EXPECT_EQ(reader.find("a"), &a);
    EXPECT_EQ(reader.find("__metadata__"), nullptr);
}

/** @brief Tests that a view of a tensor is the bytes of its range, not a copy. */
TEST(Small_SafetensorsReader, HandsOutTheBytesOfARangeWithoutCopyingThem)
{
    const std::string header = R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    const std::vector<std::byte> bytes = file_of(header, std::string("\x01\x02\x03\x04", 4));

    SA_ASSERT_OK_AND_ASSIGN(
        const IO::SafetensorsReader reader,
        IO::SafetensorsReader::read(SushiAI::Test::view_of(bytes), K_ORIGIN));

    const SushiRuntime::span<const std::byte> view = reader.contents(reader.tensors()[0]);
    EXPECT_EQ(view.data(), bytes.data() + 8 + header.size());
    EXPECT_EQ(view.size(), 4u);
}

/** @brief Tests that a file too short to hold the length field is refused as a header. */
TEST(Small_SafetensorsReader, RefusesAFileShorterThanTheLengthField)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of(std::string("\x02\x00\x00", 3));

    SA_EXPECT_FAILURE(IO::SafetensorsReader::read(SushiAI::Test::view_of(bytes), K_ORIGIN),
                      StatusCode::InvalidFile, "3 bytes", "header");
}

/** @brief Tests that a header stated longer than the file is refused, naming the header. */
TEST(Small_SafetensorsReader, RefusesAHeaderLongerThanTheFile)
{
    std::string text;
    SushiAI::Test::put_le(text, 40, 8);
    text += "{}";
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of(text);

    SA_EXPECT_FAILURE(IO::SafetensorsReader::read(SushiAI::Test::view_of(bytes), K_ORIGIN),
                      StatusCode::InvalidFile, "header states 40 bytes", "file holds 2");
}

/** @brief Tests that a header that is no JSON object is refused, naming the header. */
TEST(Small_SafetensorsReader, RefusesAHeaderThatIsNoJsonObject)
{
    const std::vector<std::byte> text = file_of("[1,2]", "");
    const std::vector<std::byte> broken = file_of("{\"w\":", "");

    SA_EXPECT_FAILURE(IO::SafetensorsReader::read(SushiAI::Test::view_of(text), K_ORIGIN),
                      StatusCode::InvalidFile, "header is no JSON object");
    SA_EXPECT_FAILURE(IO::SafetensorsReader::read(SushiAI::Test::view_of(broken), K_ORIGIN),
                      StatusCode::InvalidFile, "fixture.safetensors header");
}

/** @brief Tests that a range ending past the data is refused, naming the tensor. */
TEST(Small_SafetensorsReader, RefusesARangeOutsideTheData)
{
    const std::vector<std::byte> bytes = file_of(
        R"({"w":{"dtype":"F32","shape":[5],"data_offsets":[0,20]}})", sixteen_bytes());

    SA_EXPECT_FAILURE(IO::SafetensorsReader::read(SushiAI::Test::view_of(bytes), K_ORIGIN),
                      StatusCode::InvalidFile, "tensor 'w'", "[0, 20]", "outside the 16 data bytes");
}

/** @brief Tests that a dtype the reader does not read is refused, naming the tensor and the dtype. */
TEST(Small_SafetensorsReader, RefusesAnUnknownDtype)
{
    const std::vector<std::byte> bytes = file_of(
        R"({"w":{"dtype":"I8","shape":[4],"data_offsets":[0,4]}})", sixteen_bytes());

    SA_EXPECT_FAILURE(IO::SafetensorsReader::read(SushiAI::Test::view_of(bytes), K_ORIGIN),
                      StatusCode::Unsupported, "tensor 'w'", "'I8'");
}

/** @brief Tests that two ranges that share a byte are refused, naming both tensors. */
TEST(Small_SafetensorsReader, RefusesOverlappingRanges)
{
    // x holds bytes 0 to 8 and y starts at byte 4.
    const std::vector<std::byte> bytes = file_of(
        R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
        R"("y":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})",
        sixteen_bytes());

    SA_EXPECT_FAILURE(IO::SafetensorsReader::read(SushiAI::Test::view_of(bytes), K_ORIGIN),
                      StatusCode::InvalidFile, "'x' and 'y' overlap", "ends at byte 8",
                      "starts at 4");
}

/** @brief Tests that a range whose end lies before its start is refused, naming the tensor. */
TEST(Small_SafetensorsReader, RefusesARangeThatRunsBackwards)
{
    const std::vector<std::byte> bytes = file_of(
        R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}})", sixteen_bytes());

    SA_EXPECT_FAILURE(IO::SafetensorsReader::read(SushiAI::Test::view_of(bytes), K_ORIGIN),
                      StatusCode::InvalidFile, "tensor 'w'", "[8, 4]", "run backwards");
}

/** @brief Tests that a range that does not hold the bytes of its shape is refused, naming the tensor. */
TEST(Small_SafetensorsReader, RefusesARangeThatDoesNotMatchItsShape)
{
    // Three F32 elements need 12 bytes; the range holds 8.
    const std::vector<std::byte> bytes = file_of(
        R"({"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", sixteen_bytes());

    SA_EXPECT_FAILURE(IO::SafetensorsReader::read(SushiAI::Test::view_of(bytes), K_ORIGIN),
                      StatusCode::InvalidFile, "tensor 'w'", "need 12 bytes", "hold 8");
}

/** @brief Tests that an entry lacking a field, or holding a non-positive extent, is refused by name. */
TEST(Small_SafetensorsReader, RefusesAnEntryThatIsMissingAFieldOrHoldsABadExtent)
{
    const std::vector<std::byte> no_offsets =
        file_of(R"({"w":{"dtype":"F32","shape":[1]}})", sixteen_bytes());
    const std::vector<std::byte> zero_extent = file_of(
        R"({"w":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}})", sixteen_bytes());

    SA_EXPECT_FAILURE(IO::SafetensorsReader::read(SushiAI::Test::view_of(no_offsets), K_ORIGIN),
                      StatusCode::InvalidFile, "tensor 'w'", "data_offsets");
    SA_EXPECT_FAILURE(IO::SafetensorsReader::read(SushiAI::Test::view_of(zero_extent), K_ORIGIN),
                      StatusCode::InvalidFile, "tensor 'w'", "extent 0");
}
