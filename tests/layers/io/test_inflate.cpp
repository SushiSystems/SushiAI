/**************************************************************************/
/* test_inflate.cpp                                                       */
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
 * @file test_inflate.cpp
 * @brief Tests the DEFLATE decoder: the three block kinds, the output bounds and each refusal.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/inflate.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Holds a stored block, made by `c = zlib.compressobj(0, zlib.DEFLATED, -15, 9,
     *         zlib.Z_DEFAULT_STRATEGY); c.compress(b"stored block bytes") + c.flush()`. */
    const std::vector<unsigned> K_STORED = {
        0x01, 0x12, 0x00, 0xED, 0xFF, 0x73, 0x74, 0x6F, 0x72, 0x65, 0x64, 0x20,
        0x62, 0x6C, 0x6F, 0x63, 0x6B, 0x20, 0x62, 0x79, 0x74, 0x65, 0x73};

    /** @brief Holds an empty stored block, made by the line of K_STORED over `b""`. */
    const std::vector<unsigned> K_STORED_EMPTY = {0x01, 0x00, 0x00, 0xFF, 0xFF};

    /** @brief Holds a fixed-Huffman block, made by `c = zlib.compressobj(9, zlib.DEFLATED, -15,
     *         9, zlib.Z_FIXED); c.compress(b"hello hello hello hello") + c.flush()`. */
    const std::vector<unsigned> K_FIXED = {0xCB, 0x48, 0xCD, 0xC9, 0xC9,
                                           0x57, 0xC8, 0x40, 0x27, 0x01};

    /**
     * @brief Holds a dynamic-Huffman block, made by `c = zlib.compressobj(9, zlib.DEFLATED,
     *        -15, 9, zlib.Z_DEFAULT_STRATEGY); c.compress(b"".join(b"%d," % i for i in
     *        range(60))) + c.flush()`.
     */
    const std::vector<unsigned> K_DYNAMIC = {
        0x0D, 0x8E, 0x47, 0x01, 0xC0, 0x30, 0x10, 0xC3, 0x08, 0xE9, 0x11, 0xDB, 0xE7, 0x0C,
        0xFE, 0xC4, 0x5A, 0x00, 0x1A, 0x0B, 0x61, 0xC2, 0x50, 0x36, 0x87, 0xCB, 0x43, 0x0B,
        0x09, 0x19, 0x05, 0x0D, 0x2A, 0xDA, 0xE8, 0xA0, 0x8B, 0x1E, 0x5E, 0xF8, 0x67, 0x8C,
        0x83, 0x07, 0x17, 0x6F, 0x7C, 0xF0, 0xC5, 0x8F, 0x2C, 0x22, 0xF2, 0x2B, 0x43, 0x86,
        0x94, 0x6C, 0x72, 0xC8, 0x25, 0x8F, 0x59, 0x8C, 0x18, 0x33, 0x7F, 0x71, 0x98, 0x32,
        0x9B, 0x39, 0xCC, 0x65, 0x1E, 0x5D, 0x54, 0xD4, 0x34, 0xF4, 0x1F, 0x2A, 0xDD, 0xF4,
        0xD0, 0x4B, 0x1F, 0x1F};

    /**
     * @brief Holds a fixed block, an empty stored block and a last fixed block, made by
     *        `c = zlib.compressobj(9, zlib.DEFLATED, -15); c.compress(b"alpha alpha ") +
     *        c.flush(zlib.Z_FULL_FLUSH) + c.compress(b"beta beta") + c.flush()`.
     */
    const std::vector<unsigned> K_THREE_BLOCKS = {
        0x4A, 0xCC, 0x29, 0xC8, 0x48, 0x54, 0x48, 0x04, 0x93, 0x00, 0x00, 0x00,
        0x00, 0xFF, 0xFF, 0x4B, 0x4A, 0x2D, 0x49, 0x54, 0x48, 0x02, 0x12, 0x00};

    /**
     * @brief Holds a last fixed block whose first symbol is a match of length 3 at distance 1,
     *        written by hand: the bits 1, 01, 0000001, 00000 and seven padding zeros.
     */
    const std::vector<unsigned> K_MATCH_AT_START = {0x03, 0x02, 0x00};

    /**
     * @brief Holds a last dynamic block of 257 length codes, 1 distance code and 4 code-length
     *        symbols, written by hand, each of the 4 given the length 1, which no code allows.
     */
    const std::vector<unsigned> K_OVERSUBSCRIBED = {0x05, 0x00, 0x92, 0x04};

    /**
     * @brief Holds the same block with only the symbol 16 given the length 1, which leaves
     *        the other half of the one-bit codes unused.
     */
    const std::vector<unsigned> K_INCOMPLETE = {0x05, 0x00, 0x02, 0x00};

    /** @brief Holds a last block of the reserved type 3: the bits 1 and 11. */
    const std::vector<unsigned> K_RESERVED_TYPE = {0x07};

    /** @brief Holds a last stored block whose length 5 and check 0 are no complements. */
    const std::vector<unsigned> K_BAD_CHECK = {0x01, 0x05, 0x00, 0x00, 0x00};

    /** @brief Returns the bytes whose values are @p values. */
    std::vector<std::byte> bytes_from(const std::vector<unsigned>& values)
    {
        std::vector<std::byte> out;
        for (const unsigned value : values)
            out.push_back(static_cast<std::byte>(value));
        return out;
    }

    /** @brief Returns the text "0,1,2,...,count-1,", the input of K_DYNAMIC for 60. */
    std::string counting_text(int count)
    {
        std::string out;
        for (int i = 0; i < count; ++i)
            out += std::to_string(i) + ",";
        return out;
    }

    /** @brief Returns the first @p count bytes of @p bytes as text. */
    std::string text_of(const std::vector<std::byte>& bytes, std::size_t count)
    {
        return std::string(reinterpret_cast<const char*>(bytes.data()), count);
    }

    /** @brief Decodes @p stream into @p output, which the caller sizes. */
    Result<std::size_t> inflate_into(const std::vector<unsigned>& stream,
                                     std::vector<std::byte>& output)
    {
        const std::vector<std::byte> bytes = bytes_from(stream);
        return IO::inflate(SushiRuntime::span<const std::byte>(bytes.data(), bytes.size()),
                           SushiRuntime::span<std::byte>(output.data(), output.size()),
                           "fixture.zip");
    }

    /** @brief Holds a stream and the text it decodes to. */
    struct BlockRow
    {
        std::string name;               /**< The row's name. */
        std::vector<unsigned> stream;   /**< The compressed bytes. */
        std::string expected;           /**< The bytes they decode to. */
    };

    /** @brief Returns one row for each kind of block and for a stream of three blocks. */
    std::vector<BlockRow> block_rows()
    {
        return {{"Stored", K_STORED, "stored block bytes"},
                {"StoredEmpty", K_STORED_EMPTY, ""},
                {"Fixed", K_FIXED, "hello hello hello hello"},
                {"Dynamic", K_DYNAMIC, counting_text(60)},
                {"ThreeBlocks", K_THREE_BLOCKS, "alpha alpha beta beta"}};
    }

    /** @brief Returns the rows of block_rows() that decode to at least one byte. */
    std::vector<BlockRow> non_empty_rows()
    {
        std::vector<BlockRow> out;
        for (const BlockRow& row : block_rows())
        {
            if (!row.expected.empty())
                out.push_back(row);
        }
        return out;
    }

    /** @brief Holds a corrupt stream and what the refusal of it names. */
    struct CorruptRow
    {
        std::string name;               /**< The row's name. */
        std::vector<unsigned> stream;   /**< The corrupt bytes. */
        std::string bit;                /**< The bit offset the message states, with its colon. */
        std::string reason;             /**< A part of the reason the message gives. */
    };

    /** @brief Runs the decoder over each block kind. */
    class Small_InflateBlocks : public ::testing::TestWithParam<BlockRow>
    {
    };

    /** @brief Runs the decoder over each block kind with an output one byte short. */
    class Small_InflateOutputBounds : public ::testing::TestWithParam<BlockRow>
    {
    };

    /** @brief Runs the decoder over each corrupt stream. */
    class Small_InflateCorruption : public ::testing::TestWithParam<CorruptRow>
    {
    };
} // namespace

/** @brief Tests that each block kind decodes to the bytes the compressor was given. */
TEST_P(Small_InflateBlocks, InflateReproducesTheBytesTheCompressorWasGiven)
{
    std::vector<std::byte> output(GetParam().expected.size());

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t written, inflate_into(GetParam().stream, output));

    EXPECT_EQ(written, GetParam().expected.size());
    EXPECT_EQ(text_of(output, written), GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_InflateBlocks, ::testing::ValuesIn(block_rows()),
    [](const ::testing::TestParamInfo<BlockRow>& info) { return info.param.name; });

/** @brief Tests that an output one byte short of the stream's bytes is refused as an overrun. */
TEST_P(Small_InflateOutputBounds, InflateRefusesAnOutputTooSmallForTheStream)
{
    std::vector<std::byte> output(GetParam().expected.size() - 1);

    SA_EXPECT_FAILURE(inflate_into(GetParam().stream, output), StatusCode::InvalidFile,
                      "'fixture.zip'", "the deflate stream is corrupt at bit",
                      "writes past the end of the output");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_InflateOutputBounds, ::testing::ValuesIn(non_empty_rows()),
    [](const ::testing::TestParamInfo<BlockRow>& info) { return info.param.name; });

/** @brief Tests that a corrupt stream is refused with the bit offset where it is found. */
TEST_P(Small_InflateCorruption, InflateRefusesACorruptStreamAtItsBitOffset)
{
    std::vector<std::byte> output(64);

    SA_EXPECT_FAILURE(inflate_into(GetParam().stream, output), StatusCode::InvalidFile,
                      "'fixture.zip'", GetParam().bit, GetParam().reason);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_InflateCorruption,
    ::testing::Values(
        CorruptRow{"ReservedBlockType", K_RESERVED_TYPE, "at bit 3:", "reserved type 3"},
        CorruptRow{"StoredLengthAndCheckDisagree", K_BAD_CHECK, "at bit 40:", "disagree"},
        CorruptRow{"CodeLengthCodeOverSubscribed", K_OVERSUBSCRIBED, "at bit 29:",
                   "more codes than its lengths allow"},
        CorruptRow{"CodeLengthCodeIncomplete", K_INCOMPLETE, "at bit 29:",
                   "leaves bit patterns unused"},
        CorruptRow{"MatchBeforeTheStartOfTheOutput", K_MATCH_AT_START, "at bit 15:",
                   "reaches back before the start"},
        CorruptRow{"StreamCutInsideADynamicBlock",
                   std::vector<unsigned>(K_DYNAMIC.begin(), K_DYNAMIC.begin() + 10), "at bit 80:",
                   "ends inside a block"},
        CorruptRow{"EmptyStream", {}, "at bit 0:", "ends inside a block"}),
    [](const ::testing::TestParamInfo<CorruptRow>& info) { return info.param.name; });

/** @brief Tests that inflate returns the bytes written when the output is larger. */
TEST(Small_Inflate, InflateReturnsTheBytesWrittenForALargerOutput)
{
    std::vector<std::byte> output(100);

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t written, inflate_into(K_FIXED, output));

    EXPECT_EQ(written, 23u);
    EXPECT_EQ(text_of(output, written), "hello hello hello hello");
}

/** @brief Tests that bytes after the last block are left unread. */
TEST(Small_Inflate, InflateStopsAtTheLastBlock)
{
    std::vector<unsigned> stream = K_STORED;
    stream.push_back(0xFF);
    stream.push_back(0xFF);
    std::vector<std::byte> output(18);

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t written, inflate_into(stream, output));

    EXPECT_EQ(text_of(output, written), "stored block bytes");
}
