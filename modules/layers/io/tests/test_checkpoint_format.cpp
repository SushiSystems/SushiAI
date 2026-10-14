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
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace
{
    using SushiAI::Status;
    using SushiAI::StatusCode;
    using SushiAI::IO::CheckpointReader;
    using SushiAI::IO::CheckpointWriter;
    using SushiAI::IO::TensorRecord;

    // Offsets index the first record of sample_stream(); see tests/unit/README.md.
    constexpr std::size_t K_OFFSET_MAGIC        = 0;
    constexpr std::size_t K_OFFSET_VERSION      = 4;
    constexpr std::size_t K_OFFSET_COUNT        = 8;
    constexpr std::size_t K_OFFSET_RUN_STATE_COUNT = 12;
    constexpr std::size_t K_OFFSET_NAME_LENGTH  = 16;
    constexpr std::size_t K_OFFSET_DTYPE        = 28;   // after the 8-byte name
    constexpr std::size_t K_OFFSET_RANK         = 32;
    constexpr std::size_t K_OFFSET_FIRST_DIM    = 36;

    const std::vector<float> K_WEIGHT{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> K_BIAS{7.0f, 8.0f};

    /** @brief Holds the count 7 as the eight little-endian bytes of an Int64 payload. */
    const std::string K_SEVEN_STEPS{7, 0, 0, 0, 0, 0, 0, 0};

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

    /** @brief Writes one model record and the run state of an "sgd": a count and a tensor. */
    Status write_run_state_stream(std::ostream& out)
    {
        SA_TRY_ASSIGN(CheckpointWriter writer, CheckpointWriter::create(out, 1, 2));
        SA_TRY(writer.write(TensorRecord{"0.bias", SushiAI::DType::Float32, SushiAI::Shape{2}},
                            K_BIAS.data()));
        SA_TRY(writer.begin_run_state("sgd"));
        SA_TRY(writer.write(TensorRecord{"step_count", SushiAI::DType::Int64, SushiAI::Shape{1}},
                            K_SEVEN_STEPS.data()));
        SA_TRY(writer.write(
            TensorRecord{"0.bias.momentum", SushiAI::DType::Float32, SushiAI::Shape{2}},
            K_BIAS.data()));
        writer.finish();
        return Status();
    }

    /** @brief A checkpoint of one model record and two run-state records, as bytes. */
    std::string run_state_stream()
    {
        std::ostringstream out;
        SA_EXPECT_OK(write_run_state_stream(out));
        return out.str();
    }

    /** @brief The sample checkpoint as version 1 wrote it: no run-state count in the header. */
    std::string version_one_stream()
    {
        std::string bytes = sample_stream();
        bytes.erase(K_OFFSET_RUN_STATE_COUNT, 4);
        bytes[K_OFFSET_VERSION] = 1;
        return bytes;
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

    /** @brief Reads both sections of @p bytes, discarding payloads; returns the first failure. */
    Status read_run_state(const std::string& bytes)
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

        SA_TRY_ASSIGN(const std::string owner, reader.begin_run_state());
        while (reader.run_state_read_count() < reader.run_state_count())
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
    // The recorded shape separates [2, 3] from [3, 2]; see tests/unit/README.md.
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
    patch_u32(bytes, K_OFFSET_VERSION, 3u);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "version 3",
                      "versions 1 to 2");
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

TEST(Unit_CheckpointFormat, RejectsAnUnknownDType)
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

/** @brief Tests the sixteen bytes of a version 2 header and the size of the file. */
TEST(Unit_CheckpointFormat, WritesTheVersionTwoHeaderByteByByte)
{
    const std::string bytes = sample_stream();

    const std::string expected_header{'S', 'A', 'C', 'P', 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(bytes.substr(0, K_OFFSET_NAME_LENGTH), expected_header);

    // A record is its name length, name, dtype, rank, extents and payload.
    const std::size_t weight_bytes = 4 + 8 + 4 + 4 + 2 * 8 + 6 * 4;
    const std::size_t bias_bytes = 4 + 6 + 4 + 4 + 1 * 8 + 2 * 4;
    EXPECT_EQ(bytes.size(), K_OFFSET_NAME_LENGTH + weight_bytes + bias_bytes);
}

/** @brief Tests that a version 1 file reads as model state with no run state. */
TEST(Unit_CheckpointFormat, AVersionOneFileReadsAsModelStateAlone)
{
    std::istringstream in(version_one_stream());
    SA_ASSERT_OK_AND_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));

    EXPECT_EQ(reader.version(), 1u);
    ASSERT_EQ(reader.count(), 2u);
    EXPECT_EQ(reader.run_state_count(), 0u);

    SA_ASSERT_OK_AND_ASSIGN(const TensorRecord weight, reader.next());
    std::vector<float> weight_payload(6, 0.0f);
    SA_EXPECT_OK(reader.read_payload(weight, weight_payload.data()));
    EXPECT_EQ(weight.name, "0.weight");
    EXPECT_EQ(weight_payload, K_WEIGHT);

    SA_ASSERT_OK_AND_ASSIGN(const TensorRecord bias, reader.next());
    std::vector<float> bias_payload(2, 0.0f);
    SA_EXPECT_OK(reader.read_payload(bias, bias_payload.data()));
    EXPECT_EQ(bias_payload, K_BIAS);
}

/** @brief Tests that a version below the first one is refused with the versions read. */
TEST(Unit_CheckpointFormat, RejectsAVersionBelowTheFirst)
{
    std::string bytes = sample_stream();
    patch_u32(bytes, K_OFFSET_VERSION, 0u);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "version 0",
                      "versions 1 to 2");
}

/** @brief Tests the bytes of the run-state section and its round trip. */
TEST(Unit_CheckpointFormat, RoundTripsARunStateSection)
{
    const std::string bytes = run_state_stream();

    // The section opens where the one model record ends: 16 + (4 + 6 + 4 + 4 + 8 + 8).
    const std::size_t section = 50;
    const std::string expected_owner{3, 0, 0, 0, 's', 'g', 'd'};
    EXPECT_EQ(bytes.substr(section, expected_owner.size()), expected_owner);
    const std::string expected_count{2, 0, 0, 0};
    EXPECT_EQ(bytes.substr(K_OFFSET_RUN_STATE_COUNT, 4), expected_count);

    std::istringstream in(bytes);
    SA_ASSERT_OK_AND_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));
    EXPECT_EQ(reader.version(), 2u);
    ASSERT_EQ(reader.count(), 1u);
    ASSERT_EQ(reader.run_state_count(), 2u);

    SA_ASSERT_OK_AND_ASSIGN(const TensorRecord bias, reader.next());
    std::vector<float> bias_payload(2, 0.0f);
    SA_EXPECT_OK(reader.read_payload(bias, bias_payload.data()));

    SA_ASSERT_OK_AND_ASSIGN(const std::string owner, reader.begin_run_state());
    EXPECT_EQ(owner, "sgd");

    SA_ASSERT_OK_AND_ASSIGN(const TensorRecord steps, reader.next());
    EXPECT_EQ(steps.name, "step_count");
    EXPECT_EQ(steps.dtype, SushiAI::DType::Int64);
    EXPECT_EQ(steps.shape, (SushiAI::Shape{1}));
    std::string steps_payload(8, '\0');
    SA_EXPECT_OK(reader.read_payload(steps, steps_payload.data()));
    EXPECT_EQ(steps_payload, K_SEVEN_STEPS);

    SA_ASSERT_OK_AND_ASSIGN(const TensorRecord momentum, reader.next());
    EXPECT_EQ(momentum.name, "0.bias.momentum");
    std::vector<float> momentum_payload(2, 0.0f);
    SA_EXPECT_OK(reader.read_payload(momentum, momentum_payload.data()));
    EXPECT_EQ(momentum_payload, K_BIAS);

    EXPECT_EQ(reader.read_count(), 1u);
    EXPECT_EQ(reader.run_state_read_count(), 2u);
}

/** @brief Tests that a reader of the model records alone leaves the run state unread. */
TEST(Unit_CheckpointFormat, AReaderOfModelStateLeavesTheRunStateUnread)
{
    SA_EXPECT_OK(read_everything(run_state_stream()));
}

/** @brief Tests that a run-state count above the bound is refused by reader and writer. */
TEST(Unit_CheckpointFormat, RejectsARunStateCountOverTheBound)
{
    std::string bytes = run_state_stream();
    patch_u32(bytes, K_OFFSET_RUN_STATE_COUNT, 16385u);
    SA_EXPECT_FAILURE(read_everything(bytes), StatusCode::InvalidFile, "16385", "16384",
                      "run-state");

    std::ostringstream out;
    SA_EXPECT_FAILURE(CheckpointWriter::create(out, 0u, 16385u),
                      StatusCode::InvalidDescription, "16385", "16384", "run-state");
}

/** @brief Tests that an owner that is empty, too long or cut short is refused. */
TEST(Unit_CheckpointFormat, RejectsAMalformedRunStateOwner)
{
    const std::size_t owner_length = 50;

    std::string empty = run_state_stream();
    patch_u32(empty, owner_length, 0u);
    SA_EXPECT_FAILURE(read_run_state(empty), StatusCode::InvalidFile, "names no owner");

    std::string long_owner = run_state_stream();
    patch_u32(long_owner, owner_length, 257u);
    SA_EXPECT_FAILURE(read_run_state(long_owner), StatusCode::InvalidFile, "run-state section",
                      "257", "256");

    const std::string cut = run_state_stream().substr(0, owner_length + 5);
    SA_EXPECT_FAILURE(read_run_state(cut), StatusCode::InvalidFile, "ended while reading",
                      "run-state owner");

    std::ostringstream out;
    SA_ASSERT_OK_AND_ASSIGN(CheckpointWriter writer, CheckpointWriter::create(out, 0u, 1u));
    SA_EXPECT_FAILURE(writer.begin_run_state(""), StatusCode::InvalidDescription, "owner",
                      "0 bytes");
}

/** @brief Tests that a run-state section cut short in a payload names the record. */
TEST(Unit_CheckpointFormat, RejectsATruncatedRunStatePayload)
{
    const std::string whole = run_state_stream();
    SA_EXPECT_FAILURE(read_run_state(whole.substr(0, whole.size() - 3)),
                      StatusCode::InvalidFile, "ended while reading", "0.bias.momentum");
}

/** @brief Tests that the run-state section opens only after every model record. */
TEST(Unit_CheckpointFormatDeathTest, TheRunStateOpensAfterTheModelRecords)
{
    std::ostringstream out;
    SA_ASSERT_OK_AND_ASSIGN(CheckpointWriter writer, CheckpointWriter::create(out, 1u, 1u));
    SA_EXPECT_FATAL((void)writer.begin_run_state("sgd"), "declared 1 records", "0 were written");

    std::istringstream in(run_state_stream());
    SA_ASSERT_OK_AND_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));
    SA_EXPECT_FATAL((void)reader.begin_run_state(), "declared 1 records", "0 were read");
}

/** @brief Tests that a file that declared no run state cannot open the section. */
TEST(Unit_CheckpointFormatDeathTest, ARunStateThatWasNotDeclaredCannotOpen)
{
    std::ostringstream out;
    SA_ASSERT_OK_AND_ASSIGN(CheckpointWriter writer, CheckpointWriter::create(out, 0u));
    SA_EXPECT_FATAL((void)writer.begin_run_state("sgd"), "declared no run-state record");

    std::istringstream in(version_one_stream());
    SA_ASSERT_OK_AND_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));
    for (int i = 0; i < 2; ++i)
    {
        SA_ASSERT_OK_AND_ASSIGN(const TensorRecord record, reader.next());
        std::vector<char> sink(record.payload_bytes(), '\0');
        SA_EXPECT_OK(reader.read_payload(record, sink.data()));
    }
    SA_EXPECT_FATAL((void)reader.begin_run_state(), "declares no run-state record");
}

/** @brief Tests that the run-state section holds exactly the records declared. */
TEST(Unit_CheckpointFormatDeathTest, TheRunStateHoldsExactlyTheRecordsDeclared)
{
    std::ostringstream out;
    SA_ASSERT_OK_AND_ASSIGN(CheckpointWriter writer, CheckpointWriter::create(out, 0u, 1u));
    SA_EXPECT_OK(writer.begin_run_state("sgd"));
    SA_EXPECT_FATAL(writer.finish(), "declared 1 run-state records", "0 were written");

    SA_EXPECT_OK(writer.write(TensorRecord{"a", SushiAI::DType::Float32, SushiAI::Shape{2}},
                              K_BIAS.data()));
    SA_EXPECT_FATAL(
        (void)writer.write(TensorRecord{"b", SushiAI::DType::Float32, SushiAI::Shape{2}},
                           K_BIAS.data()),
        "declared 1 run-state records", "'b'");
}
