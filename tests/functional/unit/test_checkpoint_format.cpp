/**************************************************************************/
/* test_checkpoint_format.cpp                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_checkpoint_format.cpp
 * @brief Tests the checkpoint format's round trip and its refusal of each malformed byte pattern.
 * @author Mustafa Garip
 */

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

    // Offsets index the first record of sample_stream(); see tests/functional/unit/README.md.
    constexpr std::size_t OFFSET_MAGIC        = 0;
    constexpr std::size_t OFFSET_VERSION      = 4;
    constexpr std::size_t OFFSET_COUNT        = 8;
    constexpr std::size_t OFFSET_NAME_LENGTH  = 12;
    constexpr std::size_t OFFSET_DTYPE        = 24;   // after the 8-byte name
    constexpr std::size_t OFFSET_RANK         = 28;
    constexpr std::size_t OFFSET_FIRST_DIM    = 32;

    const std::vector<float> K_WEIGHT{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> K_BIAS{7.0f, 8.0f};

    /** @brief A well-formed two-tensor checkpoint, as bytes. */
    std::string sample_stream()
    {
        std::ostringstream out;
        CheckpointWriter writer(out, 2);
        writer.write(TensorRecord{"0.weight", SushiAI::DType::FLOAT32, SushiAI::Shape{2, 3}},
                     K_WEIGHT.data());
        writer.write(TensorRecord{"0.bias", SushiAI::DType::FLOAT32, SushiAI::Shape{2}},
                     K_BIAS.data());
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
    EXPECT_EQ(weight_payload, K_WEIGHT);

    const TensorRecord bias = reader.next();
    EXPECT_EQ(bias.name, "0.bias");
    EXPECT_EQ(bias.shape, (SushiAI::Shape{2}));

    std::vector<float> bias_payload(2, 0.0f);
    reader.read_payload(bias, bias_payload.data());
    EXPECT_EQ(bias_payload, K_BIAS);

    EXPECT_EQ(reader.read_count(), 2u);
}

TEST(Unit_CheckpointFormat, DistinguishesTwoShapesWithTheSameElementCount)
{
    // The recorded shape separates [2, 3] from [3, 2]; see tests/functional/unit/README.md.
    std::ostringstream out;
    CheckpointWriter writer(out, 1);
    writer.write(TensorRecord{"w", SushiAI::DType::FLOAT32, SushiAI::Shape{2, 3}}, K_WEIGHT.data());
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
    writer.write(TensorRecord{"a", SushiAI::DType::FLOAT32, SushiAI::Shape{2}}, K_BIAS.data());
    writer.write(TensorRecord{"b", SushiAI::DType::FLOAT32, SushiAI::Shape{2}}, K_BIAS.data());

    SA_EXPECT_ERROR(
        writer.write(TensorRecord{"c", SushiAI::DType::FLOAT32, SushiAI::Shape{2}}, K_BIAS.data()),
        "declared 2", "'c'");
}

TEST(Unit_CheckpointFormat, RejectsFewerRecordsThanDeclared)
{
    std::ostringstream out;
    CheckpointWriter writer(out, 2);
    writer.write(TensorRecord{"a", SushiAI::DType::FLOAT32, SushiAI::Shape{2}}, K_BIAS.data());

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
