/**************************************************************************/
/* test_checkpoint_format.cpp                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// The checkpoint format's whole malformed-file surface, in the device-free
// tier - which is the point of splitting checkpoint_format.hpp out of
// checkpoint.hpp. Every case here is a byte pattern, so none of it needs an
// Engine, a TensorPool or a device to run.
//
// Each corruption is expressed as a patch at a known offset into a good file
// rather than as a hand-written byte array, so a case says what is wrong with
// the file instead of restating the whole format.

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/io/checkpoint_format.hpp>
#include "../../common/test_throws.hpp"

namespace
{
    using SushiAI::IO::CheckpointReader;
    using SushiAI::IO::CheckpointWriter;
    using SushiAI::IO::TensorRecord;

    // Byte offsets into sample_stream(), whose first record is "0.weight",
    // FLOAT32, [2, 3]. Stated once here so a case names the field it breaks.
    constexpr std::size_t OFFSET_MAGIC        = 0;
    constexpr std::size_t OFFSET_VERSION      = 4;
    constexpr std::size_t OFFSET_COUNT        = 8;
    constexpr std::size_t OFFSET_NAME_LENGTH  = 12;
    constexpr std::size_t OFFSET_DTYPE        = 24;   // after the 8-byte name
    constexpr std::size_t OFFSET_RANK         = 28;
    constexpr std::size_t OFFSET_FIRST_DIM    = 32;

    const std::vector<float> WEIGHT{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> BIAS{7.0f, 8.0f};

    /** @brief A well-formed two-tensor checkpoint, as bytes. */
    std::string sample_stream()
    {
        std::ostringstream out;
        CheckpointWriter writer(out, 2);
        writer.write(TensorRecord{"0.weight", SushiAI::DType::FLOAT32, SushiAI::Shape{2, 3}},
                     WEIGHT.data());
        writer.write(TensorRecord{"0.bias", SushiAI::DType::FLOAT32, SushiAI::Shape{2}},
                     BIAS.data());
        writer.finish();
        return out.str();
    }

    /** @brief Overwrites the four little-endian bytes at @p offset. */
    void patch_u32(std::string& bytes, std::size_t offset, std::uint32_t value)
    {
        for (int i = 0; i < 4; ++i)
            bytes[offset + static_cast<std::size_t>(i)] =
                static_cast<char>((value >> (8 * i)) & 0xFFu);
    }

    /** @brief Overwrites the eight little-endian bytes at @p offset. */
    void patch_i64(std::string& bytes, std::size_t offset, std::int64_t value)
    {
        const std::uint64_t bits = static_cast<std::uint64_t>(value);
        for (int i = 0; i < 8; ++i)
            bytes[offset + static_cast<std::size_t>(i)] =
                static_cast<char>((bits >> (8 * i)) & 0xFFu);
    }

    /** @brief Reads every record of @p bytes, discarding the payloads. */
    void read_everything(const std::string& bytes)
    {
        std::istringstream in(bytes);
        CheckpointReader reader(in);
        std::vector<char> sink;
        while (reader.read_count() < reader.count())
        {
            const TensorRecord record = reader.next();
            sink.assign(record.payload_bytes(), '\0');
            reader.read_payload(record, sink.data());
        }
    }
}

TEST(Unit_CheckpointFormat, RoundTripsNamesShapesAndPayloads)
{
    std::istringstream in(sample_stream());
    CheckpointReader reader(in);

    ASSERT_EQ(reader.count(), 2u);

    const TensorRecord weight = reader.next();
    EXPECT_EQ(weight.name, "0.weight");
    EXPECT_EQ(weight.dtype, SushiAI::DType::FLOAT32);
    EXPECT_EQ(weight.shape, (SushiAI::Shape{2, 3}));
    EXPECT_EQ(weight.payload_bytes(), 24u);

    std::vector<float> weight_payload(6, 0.0f);
    reader.read_payload(weight, weight_payload.data());
    EXPECT_EQ(weight_payload, WEIGHT);

    const TensorRecord bias = reader.next();
    EXPECT_EQ(bias.name, "0.bias");
    EXPECT_EQ(bias.shape, (SushiAI::Shape{2}));

    std::vector<float> bias_payload(2, 0.0f);
    reader.read_payload(bias, bias_payload.data());
    EXPECT_EQ(bias_payload, BIAS);

    EXPECT_EQ(reader.read_count(), 2u);
}

TEST(Unit_CheckpointFormat, DistinguishesTwoShapesWithTheSameElementCount)
{
    // The defect this format exists to close: an element count alone lets a
    // [2, 3] weight load into a [3, 2] one. The shape is recorded, so the two
    // records are not interchangeable.
    std::ostringstream out;
    CheckpointWriter writer(out, 1);
    writer.write(TensorRecord{"w", SushiAI::DType::FLOAT32, SushiAI::Shape{2, 3}}, WEIGHT.data());
    writer.finish();

    std::istringstream in(out.str());
    CheckpointReader reader(in);
    EXPECT_NE(reader.next().shape, (SushiAI::Shape{3, 2}));
}

TEST(Unit_CheckpointFormat, RejectsAForeignMagic)
{
    std::string bytes = sample_stream();
    std::memcpy(&bytes[OFFSET_MAGIC], "JUNK", 4);
    SA_EXPECT_ERROR(read_everything(bytes), "SACP", "not a SushiAI checkpoint");
}

TEST(Unit_CheckpointFormat, RejectsAFutureVersion)
{
    std::string bytes = sample_stream();
    patch_u32(bytes, OFFSET_VERSION, 2u);
    SA_EXPECT_ERROR(read_everything(bytes), "version 2", "version 1");
}

TEST(Unit_CheckpointFormat, RejectsACountOverTheBound)
{
    std::string bytes = sample_stream();
    patch_u32(bytes, OFFSET_COUNT, 4097u);
    SA_EXPECT_ERROR(read_everything(bytes), "4097", "4096");
}

TEST(Unit_CheckpointFormat, RejectsATruncatedHeader)
{
    const std::string bytes = sample_stream().substr(0, 6);
    SA_EXPECT_ERROR(read_everything(bytes), "ended while reading", "version");
}

TEST(Unit_CheckpointFormat, RejectsATruncatedPayload)
{
    const std::string whole = sample_stream();
    const std::string bytes = whole.substr(0, whole.size() - 4);
    SA_EXPECT_ERROR(read_everything(bytes), "ended while reading", "0.bias");
}

TEST(Unit_CheckpointFormat, RejectsANameLengthOverTheBound)
{
    std::string bytes = sample_stream();
    patch_u32(bytes, OFFSET_NAME_LENGTH, 257u);
    SA_EXPECT_ERROR(read_everything(bytes), "257", "256");
}

TEST(Unit_CheckpointFormat, RejectsARankOverTheBound)
{
    std::string bytes = sample_stream();
    patch_u32(bytes, OFFSET_RANK, 7u);
    SA_EXPECT_ERROR(read_everything(bytes), "0.weight", "rank 7", "6");
}

TEST(Unit_CheckpointFormat, RejectsANonPositiveDim)
{
    std::string bytes = sample_stream();
    patch_i64(bytes, OFFSET_FIRST_DIM, 0);
    SA_EXPECT_ERROR(read_everything(bytes), "0.weight", "must be positive");
}

TEST(Unit_CheckpointFormat, RejectsAnUnknownDtype)
{
    std::string bytes = sample_stream();
    patch_u32(bytes, OFFSET_DTYPE, 99u);
    SA_EXPECT_ERROR(read_everything(bytes), "0.weight", "99", "does not know");
}

TEST(Unit_CheckpointFormat, RejectsMoreRecordsThanDeclared)
{
    std::ostringstream out;
    CheckpointWriter writer(out, 2);
    writer.write(TensorRecord{"a", SushiAI::DType::FLOAT32, SushiAI::Shape{2}}, BIAS.data());
    writer.write(TensorRecord{"b", SushiAI::DType::FLOAT32, SushiAI::Shape{2}}, BIAS.data());

    SA_EXPECT_ERROR(
        writer.write(TensorRecord{"c", SushiAI::DType::FLOAT32, SushiAI::Shape{2}}, BIAS.data()),
        "declared 2", "'c'");
}

TEST(Unit_CheckpointFormat, RejectsFewerRecordsThanDeclared)
{
    std::ostringstream out;
    CheckpointWriter writer(out, 2);
    writer.write(TensorRecord{"a", SushiAI::DType::FLOAT32, SushiAI::Shape{2}}, BIAS.data());

    SA_EXPECT_ERROR(writer.finish(), "declared 2", "1 were written");
}

TEST(Unit_CheckpointFormat, RejectsATensorCountOverTheBoundAtConstruction)
{
    std::ostringstream out;
    SA_EXPECT_ERROR((CheckpointWriter(out, 4097u)), "4097", "4096");
}

TEST(Unit_CheckpointFormat, ReadingPastTheDeclaredCountIsAnError)
{
    std::istringstream in(sample_stream());
    CheckpointReader reader(in);

    std::vector<char> sink;
    for (int i = 0; i < 2; ++i)
    {
        const TensorRecord record = reader.next();
        sink.assign(record.payload_bytes(), '\0');
        reader.read_payload(record, sink.data());
    }

    SA_EXPECT_ERROR(reader.next(), "declared 2");
}
