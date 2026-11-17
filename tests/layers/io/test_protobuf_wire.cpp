/**************************************************************************/
/* test_protobuf_wire.cpp                                                 */
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
 * @file test_protobuf_wire.cpp
 * @brief Tests the protobuf wire reader: every wire type, skipped unknown fields and the refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/protobuf_wire.hpp>
#include "archive_bytes.hpp"

using namespace SushiAI;
using namespace std::string_view_literals;

namespace
{
    /** @brief Names the stream in every message. */
    constexpr std::string_view K_ORIGIN = "fixture.pb";

    /** @brief Names one read a row of a refusal table runs. */
    enum class Read : std::uint8_t
    {
        Tag,             /**< next_field. */
        Varint,          /**< read_varint. */
        Zigzag,          /**< read_zigzag. */
        Fixed32,         /**< read_fixed32. */
        Fixed64,         /**< read_fixed64. */
        LengthDelimited, /**< read_length_delimited. */
    };

    /** @brief Holds the fields the test message of the protobuf guide carries, read by walk. */
    struct Walked
    {
        std::uint64_t a = 0;                 /**< Field 1, a varint. */
        std::string b;                       /**< Field 2, a string. */
        std::uint64_t c_a = 0;               /**< Field 1 of the sub-message in field 3. */
        std::vector<std::uint64_t> d;        /**< Field 4, a packed repeated varint. */
        std::size_t skipped = 0;             /**< The fields the walk did not know. */
    };

    /** @brief Reads @p reader to its end, keeping fields 1 to 4 and skipping every other. */
    Result<Walked> walk(IO::ProtobufWire& reader)
    {
        Walked out;
        while (!reader.at_end())
        {
            SA_TRY_ASSIGN(const IO::WireField field, reader.next_field());
            if (field.number == 1 && field.type == IO::WireType::Varint)
            {
                SA_TRY_ASSIGN(out.a, reader.read_varint());
            }
            else if (field.number == 2 && field.type == IO::WireType::LengthDelimited)
            {
                SA_TRY_ASSIGN(const SushiRuntime::span<const std::byte> text,
                              reader.read_length_delimited());
                out.b.assign(reinterpret_cast<const char*>(text.data()), text.size());
            }
            else if (field.number == 3 && field.type == IO::WireType::LengthDelimited)
            {
                SA_TRY_ASSIGN(IO::ProtobufWire inner, reader.message());
                SA_TRY(inner.next_field());
                SA_TRY_ASSIGN(out.c_a, inner.read_varint());
            }
            else if (field.number == 4 && field.type == IO::WireType::LengthDelimited)
            {
                SA_TRY_ASSIGN(IO::ProtobufWire packed, reader.message());
                while (!packed.at_end())
                {
                    SA_TRY_ASSIGN(const std::uint64_t element, packed.read_varint());
                    out.d.push_back(element);
                }
            }
            else
            {
                SA_TRY(reader.skip(field.type));
                ++out.skipped;
            }
        }
        return out;
    }

    /** @brief Returns a reader over @p bytes, which must outlive it. */
    IO::ProtobufWire reader_over(const std::vector<std::byte>& bytes)
    {
        return IO::ProtobufWire(SushiAI::Test::view_of(bytes), K_ORIGIN);
    }

    /** @brief Reads @p count varints from @p reader and drops them. */
    void drop_varints(IO::ProtobufWire& reader, std::size_t count)
    {
        for (std::size_t i = 0; i < count; ++i)
            (void)reader.read_varint();
    }

    /** @brief Runs the read @p read on @p reader and returns its status. */
    Status run_read(IO::ProtobufWire& reader, Read read)
    {
        switch (read)
        {
            case Read::Tag:             return reader.next_field().status();
            case Read::Varint:          return reader.read_varint().status();
            case Read::Zigzag:          return reader.read_zigzag().status();
            case Read::Fixed32:         return reader.read_fixed32().status();
            case Read::Fixed64:         return reader.read_fixed64().status();
            case Read::LengthDelimited: return reader.read_length_delimited().status();
        }
        return Status();
    }

    /** @brief Names one varint stream and the number it encodes. */
    struct VarintRow
    {
        std::string_view name;   /**< The row's name. */
        std::string_view stream; /**< The encoded bytes. */
        std::uint64_t expected;  /**< The value, decoded from base-128 groups of seven bits. */
    };

    /** @brief Lists the varints of the protobuf guide and the edges of 64 bits. */
    const VarintRow K_VARINT_ROWS[] = {
        {"Zero", "\x00"sv, 0},
        {"One", "\x01"sv, 1},
        {"LargestOneByte", "\x7f"sv, 127},
        {"SmallestTwoBytes", "\x80\x01"sv, 128},
        {"ThreeHundred", "\xac\x02"sv, 300},
        {"OneHundredFifty", "\x96\x01"sv, 150},
        {"LargestSixtyFourBit",
         "\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"sv, 0xFFFFFFFFFFFFFFFFull},
        {"RedundantHighGroups", "\x81\x80\x00"sv, 1},
    };

    /** @brief Runs one row of K_VARINT_ROWS. */
    class Small_ProtobufWireVarint : public ::testing::TestWithParam<VarintRow>
    {

    };

    /** @brief Returns the name of the varint row @p info runs. */
    std::string varint_name(const ::testing::TestParamInfo<VarintRow>& info)
    {
        return std::string(info.param.name);
    }

    /** @brief Names one zigzag stream and the signed number it encodes. */
    struct ZigzagRow
    {
        std::string_view name;   /**< The row's name. */
        std::string_view stream; /**< The encoded bytes. */
        std::int64_t expected;   /**< The value: n even is n / 2, n odd is -(n + 1) / 2. */
    };

    /** @brief Lists the zigzag pairs of the protobuf guide and both 32-bit extremes. */
    const ZigzagRow K_ZIGZAG_ROWS[] = {
        {"Zero", "\x00"sv, 0},
        {"MinusOne", "\x01"sv, -1},
        {"One", "\x02"sv, 1},
        {"MinusTwo", "\x03"sv, -2},
        {"LargestInt32", "\xfe\xff\xff\xff\x0f"sv, 2147483647},
        {"SmallestInt32", "\xff\xff\xff\xff\x0f"sv, -2147483648ll},
        {"SmallestInt64", "\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"sv,
         static_cast<std::int64_t>(0x8000000000000000ull)},
    };

    /** @brief Runs one row of K_ZIGZAG_ROWS. */
    class Small_ProtobufWireZigzag : public ::testing::TestWithParam<ZigzagRow>
    {

    };

    /** @brief Returns the name of the zigzag row @p info runs. */
    std::string zigzag_name(const ::testing::TestParamInfo<ZigzagRow>& info)
    {
        return std::string(info.param.name);
    }

    /** @brief Names one stream a reader must refuse and what the refusal must say. */
    struct RefusalRow
    {
        std::string_view name;     /**< The row's name. */
        std::string_view stream;   /**< The bytes. */
        std::size_t lead;          /**< The varints read before the read under test. */
        Read read;                 /**< The read under test. */
        std::string_view location; /**< The byte the refusal must name. */
        std::string_view what;     /**< What the refusal must say. */
    };

    /** @brief Lists a refusal for every way a value or a tag can be cut short or be invalid. */
    const RefusalRow K_REFUSAL_ROWS[] = {
        {"VarintCutAtItsFirstByte", "\x80"sv, 0, Read::Varint, "byte 0", "is truncated"},
        {"VarintCutAfterTwoBytes", "\x08\x01\x80"sv, 2, Read::Varint, "byte 2", "is truncated"},
        {"VarintOfElevenBytes",
         "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x01"sv, 0, Read::Varint, "byte 0",
         "does not fit 64 bits"},
        {"VarintWhoseTenthByteIsTwo",
         "\xff\xff\xff\xff\xff\xff\xff\xff\xff\x02"sv, 0, Read::Varint, "byte 0",
         "does not fit 64 bits"},
        {"ZigzagCut", "\xff"sv, 0, Read::Zigzag, "byte 0", "is truncated"},
        {"Fixed32OfThreeBytes", "\x01\x02\x03"sv, 0, Read::Fixed32, "fixed32 at byte 0",
         "is truncated"},
        {"Fixed32AfterTwoVarints", "\x08\x01\x01\x02\x03"sv, 2, Read::Fixed32,
         "fixed32 at byte 2", "is truncated"},
        {"Fixed64OfSevenBytes", "\x01\x02\x03\x04\x05\x06\x07"sv, 0, Read::Fixed64,
         "fixed64 at byte 0", "is truncated"},
        {"LengthPastTheEnd", "\x05" "abc"sv, 0, Read::LengthDelimited, "byte 0",
         "claims 5 bytes and 3 remain"},
        {"LengthAfterTwoVarints", "\x08\x01\x03" "ab"sv, 2, Read::LengthDelimited, "byte 2",
         "claims 3 bytes and 2 remain"},
        {"LengthOfTwoToTheSixtyFourMinusOne",
         "\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"sv, 0, Read::LengthDelimited, "byte 0",
         "claims 18446744073709551615 bytes"},
        {"LengthCut", "\x80"sv, 0, Read::LengthDelimited, "byte 0", "is truncated"},
        {"FieldNumberZero", "\x00"sv, 0, Read::Tag, "byte 0", "field number 0"},
        {"FieldNumberAboveTheLimit", "\x80\x80\x80\x80\x10"sv, 0, Read::Tag, "byte 0",
         "field number 536870912"},
        {"WireTypeSix", "\x0e"sv, 0, Read::Tag, "byte 0", "wire type 6"},
        {"WireTypeSevenAfterTwoVarints", "\x08\x01\x0f"sv, 2, Read::Tag, "byte 2",
         "wire type 7"},
    };

    /** @brief Runs one row of K_REFUSAL_ROWS. */
    class Small_ProtobufWireRefusal : public ::testing::TestWithParam<RefusalRow>
    {

    };

    /** @brief Returns the name of the refusal row @p info runs. */
    std::string refusal_name(const ::testing::TestParamInfo<RefusalRow>& info)
    {
        return std::string(info.param.name);
    }

    /**
     * @brief Holds the message of the protobuf guide, then one unknown field of each wire type.
     *
     * Field 1 = 150, field 2 = "testing", field 3 = { field 1 = 150 }, field 4 = packed
     * [3, 270, 86942], then fields 15 (varint), 16 (fixed64), 17 (length-delimited),
     * 18 (fixed32) and 19 (a group holding field 1 and a group 20), whose tags are
     * (number << 3) | wire type, as varints.
     */
    constexpr std::string_view K_GUIDE_MESSAGE =
        "\x08\x96\x01"
        "\x12\x07testing"
        "\x1a\x03\x08\x96\x01"
        "\x22\x06\x03\x8e\x02\x9e\xa7\x05"sv;

    /** @brief Holds one unknown field of each wire type, the group nested in a group. */
    constexpr std::string_view K_UNKNOWN_FIELDS =
        "\x78\x05"
        "\x81\x01\x01\x02\x03\x04\x05\x06\x07\x08"
        "\x8a\x01\x02\xab\xcd"
        "\x95\x01\x01\x02\x03\x04"
        "\x9b\x01\x08\x01\xa3\x01\xa4\x01\x9c\x01"sv;
} // namespace

/** @brief Tests that a varint decodes to the number its base-128 groups state. */
TEST_P(Small_ProtobufWireVarint, ReadsTheNumberItEncodes)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of(GetParam().stream);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t value, reader.read_varint());

    EXPECT_EQ(value, GetParam().expected);
    EXPECT_TRUE(reader.at_end());
}

INSTANTIATE_TEST_SUITE_P(Encodings, Small_ProtobufWireVarint, ::testing::ValuesIn(K_VARINT_ROWS),
                         varint_name);

/** @brief Tests that a zigzag varint decodes to the signed number the encoding defines. */
TEST_P(Small_ProtobufWireZigzag, ReadsTheSignedNumberItEncodes)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of(GetParam().stream);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t value, reader.read_zigzag());

    EXPECT_EQ(value, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(Encodings, Small_ProtobufWireZigzag, ::testing::ValuesIn(K_ZIGZAG_ROWS),
                         zigzag_name);

/** @brief Tests that a fixed32 is four bytes read low byte first and moves four bytes on. */
TEST(Small_ProtobufWire, ReadsAFixed32LowByteFirst)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of("\x78\x56\x34\x12\xff"sv);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const std::uint32_t value, reader.read_fixed32());

    EXPECT_EQ(value, 0x12345678u);
    EXPECT_EQ(reader.offset(), 4u);
}

/** @brief Tests that a fixed64 is eight bytes read low byte first and moves eight bytes on. */
TEST(Small_ProtobufWire, ReadsAFixed64LowByteFirst)
{
    const std::vector<std::byte> bytes =
        SushiAI::Test::bytes_of("\xef\xcd\xab\x89\x67\x45\x23\x01\xff"sv);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t value, reader.read_fixed64());

    EXPECT_EQ(value, 0x0123456789ABCDEFull);
    EXPECT_EQ(reader.offset(), 8u);
}

/** @brief Tests that the fixed bits of 1.0f and 1.0 read back as those numbers. */
TEST(Small_ProtobufWire, TurnsFixedBitsIntoTheFloatsTheyEncode)
{
    // 1.0f is sign 0, exponent 127, fraction 0: 0x3F800000. 1.0 is exponent 1023: 0x3FF0...0.
    EXPECT_EQ(IO::float_from_bits(0x3F800000u), 1.0f);
    EXPECT_EQ(IO::double_from_bits(0x3FF0000000000000ull), 1.0);
    EXPECT_EQ(IO::float_from_bits(0xC0000000u), -2.0f);
}

/** @brief Tests that a length-delimited value is a view into the caller's bytes, not a copy. */
TEST(Small_ProtobufWire, HandsOutLengthDelimitedBytesWithoutCopyingThem)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of("\x03" "abcd"sv);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const SushiRuntime::span<const std::byte> view,
                            reader.read_length_delimited());

    EXPECT_EQ(view.data(), bytes.data() + 1);
    EXPECT_EQ(view.size(), 3u);
    EXPECT_EQ(reader.offset(), 4u);
}

/** @brief Tests that an empty length-delimited value reads as an empty view. */
TEST(Small_ProtobufWire, ReadsAnEmptyLengthDelimitedValue)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of("\x00"sv);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const SushiRuntime::span<const std::byte> view,
                            reader.read_length_delimited());

    EXPECT_EQ(view.size(), 0u);
    EXPECT_TRUE(reader.at_end());
}

/** @brief Tests that a tag splits into its field number and its wire type. */
TEST(Small_ProtobufWire, SplitsATagIntoItsFieldNumberAndWireType)
{
    // 0x12 = 0b00010010: number 2 (0b10 after dropping the low three bits), wire type 2.
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of("\x12\x95\x01"sv);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const IO::WireField first, reader.next_field());
    SA_ASSERT_OK_AND_ASSIGN(const IO::WireField second, reader.next_field());

    EXPECT_EQ(first.number, 2u);
    EXPECT_EQ(first.type, IO::WireType::LengthDelimited);
    // 0x95 0x01 is 149 = (18 << 3) | 5: number 18, fixed32.
    EXPECT_EQ(second.number, 18u);
    EXPECT_EQ(second.type, IO::WireType::Fixed32);
}

/** @brief Tests that the message of the protobuf guide reads field by field, packed included. */
TEST(Small_ProtobufWire, ReadsEveryKnownFieldOfTheGuideMessage)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of(K_GUIDE_MESSAGE);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const Walked walked, walk(reader));

    EXPECT_EQ(walked.a, 150u);
    EXPECT_EQ(walked.b, "testing");
    EXPECT_EQ(walked.c_a, 150u);
    EXPECT_EQ(walked.d, (std::vector<std::uint64_t>{3, 270, 86942}));
    EXPECT_EQ(walked.skipped, 0u);
}

/** @brief Tests that unknown fields of every wire type are skipped and known ones still read. */
TEST(Small_ProtobufWire, SkipsUnknownFieldsOfEveryWireType)
{
    const std::string stream = std::string(K_UNKNOWN_FIELDS.substr(0, 2)) +
                               std::string(K_GUIDE_MESSAGE) +
                               std::string(K_UNKNOWN_FIELDS.substr(2));
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of(stream);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const Walked walked, walk(reader));

    EXPECT_EQ(walked.a, 150u);
    EXPECT_EQ(walked.b, "testing");
    EXPECT_EQ(walked.c_a, 150u);
    EXPECT_EQ(walked.d, (std::vector<std::uint64_t>{3, 270, 86942}));
    EXPECT_EQ(walked.skipped, 5u);
    EXPECT_TRUE(reader.at_end());
}

/** @brief Tests that a packed fixed32 field reads as the values it holds, in order. */
TEST(Small_ProtobufWire, ReadsAPackedFixed32Field)
{
    const std::vector<std::byte> bytes =
        SushiAI::Test::bytes_of("\x22\x08\x01\x00\x00\x00\x02\x01\x00\x00"sv);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const IO::WireField field, reader.next_field());
    SA_ASSERT_OK_AND_ASSIGN(IO::ProtobufWire packed, reader.message());
    SA_ASSERT_OK_AND_ASSIGN(const std::uint32_t first, packed.read_fixed32());
    SA_ASSERT_OK_AND_ASSIGN(const std::uint32_t second, packed.read_fixed32());

    EXPECT_EQ(field.number, 4u);
    EXPECT_EQ(first, 1u);
    EXPECT_EQ(second, 0x102u);
    EXPECT_TRUE(packed.at_end());
}

/** @brief Tests that a sub-message reader counts its offsets from the start of the file. */
TEST(Small_ProtobufWire, ASubMessageContinuesTheOffsetsOfItsParent)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of("\x08\x01\x12\x02\x08\x05"sv);
    IO::ProtobufWire reader = reader_over(bytes);
    drop_varints(reader, 2);
    SA_ASSERT_OK_AND_ASSIGN(const IO::WireField field, reader.next_field());

    SA_ASSERT_OK_AND_ASSIGN(IO::ProtobufWire inner, reader.message());

    EXPECT_EQ(field.number, 2u);
    EXPECT_EQ(inner.offset(), 4u);
    EXPECT_EQ(reader.offset(), 6u);
}

/** @brief Tests that a failure inside a sub-message names its byte in the whole file. */
TEST(Small_ProtobufWire, AFailureInASubMessageNamesTheByteInTheFile)
{
    // The sub-message holds 08 80 at bytes 4 and 5; the varint at byte 5 is cut.
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of("\x08\x01\x12\x02\x08\x80"sv);
    IO::ProtobufWire reader = reader_over(bytes);
    drop_varints(reader, 2);
    SA_ASSERT_OK_AND_ASSIGN(const IO::WireField field, reader.next_field());
    SA_ASSERT_OK_AND_ASSIGN(IO::ProtobufWire inner, reader.message());
    SA_ASSERT_OK_AND_ASSIGN(const IO::WireField inner_field, inner.next_field());

    SA_EXPECT_FAILURE(inner.read_varint(), StatusCode::InvalidFile, "'fixture.pb'", "byte 5",
                      "is truncated");
    EXPECT_EQ(field.number, 2u);
    EXPECT_EQ(inner_field.number, 1u);
}

/** @brief Tests that each truncated or invalid value is refused with its byte offset. */
TEST_P(Small_ProtobufWireRefusal, NamesTheByteItFailsAt)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of(GetParam().stream);
    IO::ProtobufWire reader = reader_over(bytes);
    drop_varints(reader, GetParam().lead);

    SA_EXPECT_FAILURE(run_read(reader, GetParam().read), StatusCode::InvalidFile, "'fixture.pb'",
                      GetParam().location, GetParam().what);
}

INSTANTIATE_TEST_SUITE_P(Streams, Small_ProtobufWireRefusal, ::testing::ValuesIn(K_REFUSAL_ROWS),
                         refusal_name);

/** @brief Tests that skipping a fixed64 that is cut short is refused at its byte. */
TEST(Small_ProtobufWire, RefusesToSkipAValueThatIsCutShort)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of("\x01\x02\x03"sv);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_EXPECT_FAILURE(reader.skip(IO::WireType::Fixed64), StatusCode::InvalidFile,
                      "fixed64 at byte 0", "is truncated");
}

/** @brief Tests that a group that never reaches its end tag is refused at the end of the bytes. */
TEST(Small_ProtobufWire, RefusesAGroupThatNeverEnds)
{
    // The group holds field 1 = 1 and then the bytes stop at offset 2.
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of("\x08\x01"sv);
    IO::ProtobufWire reader = reader_over(bytes);

    SA_EXPECT_FAILURE(reader.skip(IO::WireType::StartGroup), StatusCode::InvalidFile,
                      "byte 2", "without its end-group tag");
}

/** @brief Tests that an end-group tag with no group open is refused. */
TEST(Small_ProtobufWire, RefusesAnEndGroupThatClosesNothing)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of("\x0c"sv);
    IO::ProtobufWire reader = reader_over(bytes);
    SA_ASSERT_OK_AND_ASSIGN(const IO::WireField field, reader.next_field());

    SA_EXPECT_FAILURE(reader.skip(field.type), StatusCode::InvalidFile, "byte 1",
                      "closes no group");
    EXPECT_EQ(field.type, IO::WireType::EndGroup);
}

/** @brief Tests that an empty stream is at its end and a reader reports no bytes read. */
TEST(Small_ProtobufWire, AnEmptyStreamIsAtItsEnd)
{
    const std::vector<std::byte> bytes;
    IO::ProtobufWire reader = reader_over(bytes);

    EXPECT_TRUE(reader.at_end());
    EXPECT_EQ(reader.offset(), 0u);
}
