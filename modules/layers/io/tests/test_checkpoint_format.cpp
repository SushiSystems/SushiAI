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
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <common/test_status.hpp>

namespace
{
    using SushiAI::Status;
    using SushiAI::StatusCode;
    using SushiAI::IO::CheckpointReader;
    using SushiAI::IO::CheckpointWriter;
    using SushiAI::IO::TensorRecord;

    // Offsets index the first record of sample_stream(); see tests/functional/unit/README.md.
    constexpr std::size_t K_OFFSET_MAGIC        = 0;
    constexpr std::size_t K_OFFSET_VERSION      = 4;
    constexpr std::size_t K_OFFSET_COUNT        = 8;
    constexpr std::size_t K_OFFSET_NAME_LENGTH  = 12;
    constexpr std::size_t K_OFFSET_DTYPE        = 24;   // after the 8-byte name
    constexpr std::size_t K_OFFSET_RANK         = 28;
    constexpr std::size_t K_OFFSET_FIRST_DIM    = 32;

    const std::vector<float> K_WEIGHT{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> K_BIAS{7.0f, 8.0f};

    /** @brief Writes a well-formed two-tensor checkpoint to @p out. */
    Status write_sample_stream(std::ostream& out)
    {
        SA_TRY_ASSIGN(CheckpointWriter writer, CheckpointWriter::create(out, 2));
        SA_TRY(writer.write(TensorRecord{"0.weight", SushiAI::DType::Float32, SushiAI::Shape{2, 3}},
                            K_WEIGHT.data()));
        SA_TRY(writer.write(TensorRecord{"0.bias", SushiAI::DType::Float32, SushiAI::Shape{2}},
                            K_BIAS.data()));
        writer.finish();
        return Status();
    }

    /** @brief A well-formed two-tensor checkpoint, as bytes. */
    std::string sample_stream()
    {
        std::ostringstream out;
        SA_EXPECT_OK(write_sample_stream(out));
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

    /** @brief Reads every record of @p bytes, discarding payloads; returns the first failure. */
    Status read_everything(const std::string& bytes)
    {
        std::istringstream in(bytes);
        SA_TRY_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));
        std::vector<char> sink;
        while (reader.read_count() < reader.count())
        {
            SA_TRY_ASSIGN(const TensorRecord record, reader.next());
            sink.assign(record.payload_bytes(), '\0');
            SA_TRY(reader.read_payload(record, sink.data()));
        }
        return Status();
    }
}

TEST(Unit_CheckpointFormat, RoundTripsNamesShapesAndPayloads)
{
    std::istringstream in(sample_stream());
    SA_ASSERT_OK_AND_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));

    ASSERT_EQ(reader.count(), 2u);

    SA_ASSERT_OK_AND_ASSIGN(const TensorRecord weight, reader.next());
    EXPECT_EQ(weight.name, "0.weight");
    EXPECT_EQ(weight.dtype, SushiAI::DType::Float32);
    EXPECT_EQ(weight.shape, (SushiAI::Shape{2, 3}));
    EXPECT_EQ(weight.payload_bytes(), 24u);

    std::vector<float> weight_payload(6, 0.0f);
    SA_EXPECT_OK(reader.read_payload(weight, weight_payload.data()));
    EXPECT_EQ(weight_payload, K_WEIGHT);

    SA_ASSERT_OK_AND_ASSIGN(const TensorRecord bias, reader.next());
    EXPECT_EQ(bias.name, "0.bias");
    EXPECT_EQ(bias.shape, (SushiAI::Shape{2}));

    std::vector<float> bias_payload(2, 0.0f);
    SA_EXPECT_OK(reader.read_payload(bias, bias_payload.data()));
    EXPECT_EQ(bias_payload, K_BIAS);

    EXPECT_EQ(reader.read_count(), 2u);
}

TEST(Unit_CheckpointFormat, DistinguishesTwoShapesWithTheSameElementCount)
{
    // The recorded shape separates [2, 3] from [3, 2]; see tests/functional/unit/README.md.
    std::ostringstream out;
    SA_ASSERT_OK_AND_ASSIGN(CheckpointWriter writer, CheckpointWriter::create(out, 1));
    SA_EXPECT_OK(writer.write(TensorRecord{"w", SushiAI::DType::Float32, SushiAI::Shape{2, 3}},
                              K_WEIGHT.data()));
    writer.finish();

    std::istringstream in(out.str());
    SA_ASSERT_OK_AND_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));
    SA_ASSERT_OK_AND_ASSIGN(const TensorRecord record, reader.next());
    EXPECT_NE(record.shape, (SushiAI::Shape{3, 2}));
}

TEST(Unit_CheckpointFormat, RejectsAForeignMagic)
{
    std::string bytes = sample_stream();
    std::memcpy(&bytes[K_OFFSET_MAGIC], "JUNK", 4);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "SACP",
                      "not a SushiAI checkpoint");
}

TEST(Unit_CheckpointFormat, RejectsAFutureVersion)
{
    std::string bytes = sample_stream();
    patch_u32(bytes, K_OFFSET_VERSION, 2u);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "version 2", "version 1");
}

TEST(Unit_CheckpointFormat, RejectsACountOverTheBound)
{
    std::string bytes = sample_stream();
    patch_u32(bytes, K_OFFSET_COUNT, 4097u);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "4097", "4096");
}

TEST(Unit_CheckpointFormat, RejectsATruncatedHeader)
{
    const std::string bytes = sample_stream().substr(0, 6);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "ended while reading",
                      "version");
}

TEST(Unit_CheckpointFormat, RejectsATruncatedPayload)
{
    const std::string whole = sample_stream();
    const std::string bytes = whole.substr(0, whole.size() - 4);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "ended while reading",
                      "0.bias");
}

TEST(Unit_CheckpointFormat, RejectsANameLengthOverTheBound)
{
    std::string bytes = sample_stream();
    patch_u32(bytes, K_OFFSET_NAME_LENGTH, 257u);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "257", "256");
}

TEST(Unit_CheckpointFormat, RejectsARankOverTheBound)
{
    std::string bytes = sample_stream();
    patch_u32(bytes, K_OFFSET_RANK, 7u);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "0.weight", "rank 7", "6");
}

TEST(Unit_CheckpointFormat, RejectsANonPositiveDim)
{
    std::string bytes = sample_stream();
    patch_i64(bytes, K_OFFSET_FIRST_DIM, 0);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "0.weight",
                      "must be positive");
}

TEST(Unit_CheckpointFormat, RejectsAnUnknownDtype)
{
    std::string bytes = sample_stream();
    patch_u32(bytes, K_OFFSET_DTYPE, 99u);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "0.weight", "99",
                      "does not know");
}

TEST(Unit_CheckpointFormatDeathTest, RejectsMoreRecordsThanDeclared)
{
    std::ostringstream out;
    SA_ASSERT_OK_AND_ASSIGN(CheckpointWriter writer, CheckpointWriter::create(out, 2));
    SA_EXPECT_OK(writer.write(TensorRecord{"a", SushiAI::DType::Float32, SushiAI::Shape{2}},
                              K_BIAS.data()));
    SA_EXPECT_OK(writer.write(TensorRecord{"b", SushiAI::DType::Float32, SushiAI::Shape{2}},
                              K_BIAS.data()));

    SA_EXPECT_FATAL(
        (void)writer.write(TensorRecord{"c", SushiAI::DType::Float32, SushiAI::Shape{2}},
                           K_BIAS.data()),
        "declared 2", "'c'");
}

TEST(Unit_CheckpointFormatDeathTest, RejectsFewerRecordsThanDeclared)
{
    std::ostringstream out;
    SA_ASSERT_OK_AND_ASSIGN(CheckpointWriter writer, CheckpointWriter::create(out, 2));
    SA_EXPECT_OK(writer.write(TensorRecord{"a", SushiAI::DType::Float32, SushiAI::Shape{2}},
                              K_BIAS.data()));

    SA_EXPECT_FATAL(writer.finish(), "declared 2", "1 were written");
}

TEST(Unit_CheckpointFormat, RejectsATensorCountOverTheBoundAtConstruction)
{
    std::ostringstream out;
    SA_EXPECT_FAILURE(CheckpointWriter::create(out, 4097u), StatusCode::InvalidDescription,
                      "4097", "4096");
}

TEST(Unit_CheckpointFormatDeathTest, ReadingPastTheDeclaredCountIsAnError)
{
    std::istringstream in(sample_stream());
    SA_ASSERT_OK_AND_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));

    std::vector<char> sink;
    for (int i = 0; i < 2; ++i)
    {
        SA_ASSERT_OK_AND_ASSIGN(const TensorRecord record, reader.next());
        sink.assign(record.payload_bytes(), '\0');
        SA_EXPECT_OK(reader.read_payload(record, sink.data()));
    }

    SA_EXPECT_FATAL((void)reader.next(), "declared 2");
}
