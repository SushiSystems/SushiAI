/**************************************************************************/
/* test_protobuf_wire_writer.cpp                                          */
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
 * @file test_protobuf_wire_writer.cpp
 * @brief Tests the protobuf wire writer against the reader: every field kind, a nested message
 *        and a packed float field.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/protobuf_wire.hpp>
#include <SushiAI/io/protobuf_wire_writer.hpp>

using namespace SushiAI;
using SushiAI::Test::require_value;

namespace
{
    /** @brief Reads the next tag of @p wire and expects @p number and @p type. */
    void expect_field(IO::ProtobufWire& wire, std::uint32_t number, IO::WireType type)
    {
        ASSERT_FALSE(wire.at_end());
        const IO::WireField field = require_value(wire.next_field());
        EXPECT_EQ(field.number, number);
        EXPECT_EQ(field.type, type);
    }
} // namespace

/** @brief Tests that each scalar kind reads back with its number, wire type and value. */
TEST(Small_ProtobufWireWriter, EveryScalarKindReadsBack)
{
    IO::ProtobufWireWriter writer;
    writer.write_varint(1, 300);
    writer.write_varint(2, ~std::uint64_t(0));
    writer.write_zigzag(3, -5);
    writer.write_fixed32(4, 0xDEADBEEFu);
    writer.write_fixed64(5, 0x0123456789ABCDEFull);
    writer.write_float(6, 1.5f);
    writer.write_double(7, -2.25);
    writer.write_string(536870911u, "text");

    const std::vector<std::byte>& bytes = writer.bytes();
    IO::ProtobufWire wire(SushiRuntime::span<const std::byte>(bytes.data(), bytes.size()),
                          "written");
    expect_field(wire, 1, IO::WireType::Varint);
    EXPECT_EQ(require_value(wire.read_varint()), 300u);
    expect_field(wire, 2, IO::WireType::Varint);
    EXPECT_EQ(require_value(wire.read_varint()), ~std::uint64_t(0));
    expect_field(wire, 3, IO::WireType::Varint);
    EXPECT_EQ(require_value(wire.read_zigzag()), -5);
    expect_field(wire, 4, IO::WireType::Fixed32);
    EXPECT_EQ(require_value(wire.read_fixed32()), 0xDEADBEEFu);
    expect_field(wire, 5, IO::WireType::Fixed64);
    EXPECT_EQ(require_value(wire.read_fixed64()), 0x0123456789ABCDEFull);
    expect_field(wire, 6, IO::WireType::Fixed32);
    EXPECT_EQ(IO::float_from_bits(require_value(wire.read_fixed32())), 1.5f);
    expect_field(wire, 7, IO::WireType::Fixed64);
    EXPECT_EQ(IO::double_from_bits(require_value(wire.read_fixed64())), -2.25);
    expect_field(wire, 536870911u, IO::WireType::LengthDelimited);
    const SushiRuntime::span<const std::byte> text = require_value(wire.read_length_delimited());
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(text.data()), text.size()), "text");
    EXPECT_TRUE(wire.at_end());
}

/** @brief Tests that a nested message and a packed float field read back. */
TEST(Small_ProtobufWireWriter, ANestedMessageAndAPackedFieldReadBack)
{
    IO::ProtobufWireWriter inner;
    inner.write_varint(1, 7);
    const std::vector<float> values = {0.5f, -1.0f, 3.25f};
    IO::ProtobufWireWriter writer;
    writer.write_message(2, inner);
    writer.write_packed_floats(3, SushiRuntime::span<const float>(values.data(), values.size()));

    const std::vector<std::byte>& bytes = writer.bytes();
    IO::ProtobufWire wire(SushiRuntime::span<const std::byte>(bytes.data(), bytes.size()),
                          "written");
    expect_field(wire, 2, IO::WireType::LengthDelimited);
    IO::ProtobufWire nested = require_value(wire.message());
    expect_field(nested, 1, IO::WireType::Varint);
    EXPECT_EQ(require_value(nested.read_varint()), 7u);
    EXPECT_TRUE(nested.at_end());

    expect_field(wire, 3, IO::WireType::LengthDelimited);
    IO::ProtobufWire packed = require_value(wire.message());
    std::vector<float> read;
    while (!packed.at_end())
        read.push_back(IO::float_from_bits(require_value(packed.read_fixed32())));
    EXPECT_EQ(read, values);
    EXPECT_TRUE(wire.at_end());
}
