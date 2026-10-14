/**************************************************************************/
/* test_state_checkpoint.cpp                                              */
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
 * @file test_state_checkpoint.cpp
 * @brief Tests checkpoint save and load over a ModelState, and their shared file format.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_StateCheckpoint : public SaTest::EngineTest {};

    /**
     * @struct NormModel
     * @brief A model holding one BatchNorm2d named "bn", with the graph and updates of its trace.
     */
    struct NormModel
    {
        /** @brief The model, whose samples are [4, 4, 3]. */
        NN::ModelGraph model{Shape{4, 4, 3}};

        /** @brief The graph the training trace fills. */
        Graph::Graph graph;

        /** @brief The updates the training trace requests. */
        NN::BufferUpdateList updates;
    };

    /** @brief Adds the BatchNorm2d to @p norm and traces a training pass through it. */
    void trace_norm(NormModel& norm)
    {
        SA_ASSERT_OK_AND_ASSIGN(const auto out,
                                norm.model.add("bn", NN::BatchNorm2d(3, NN::NormOptions{}),
                                               norm.model.input()));
        SA_EXPECT_OK(norm.model.output("normalised", out));

        const Graph::GraphBuilder builder(norm.graph);
        const Graph::ValueID x = builder.input(Shape{2, 4, 4, 3}, "x");
        SA_EXPECT_OK(norm.model.trace(NN::Trace::training(builder, norm.updates), x));
    }

    /** @brief Writes a distinct value into every element of every entry, buffers included. */
    void fill_state(NN::ModelState& state, float base)
    {
        float offset = base;
        for (const NN::StateEntry& entry : state.entries())
        {
            SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* tensor, state.tensor(entry.state));
            float* values = tensor->data_as<float>();
            for (int64_t i = 0; i < tensor->num_elements; ++i)
                values[i] = offset + 0.25f * static_cast<float>(i);
            offset += 1.0f;
        }
    }

    /** @brief Writes every element of every entry of @p state into @p out as bit patterns. */
    void collect_bits(NN::ModelState& state, std::vector<std::uint32_t>& out)
    {
        out.clear();
        for (const NN::StateEntry& entry : state.entries())
        {
            SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* tensor, state.tensor(entry.state));
            const float* values = tensor->data_as<float>();
            for (int64_t i = 0; i < tensor->num_elements; ++i)
            {
                std::uint32_t pattern = 0;
                std::memcpy(&pattern, &values[i], sizeof(pattern));
                out.push_back(pattern);
            }
        }
    }

    /** @brief Writes a checkpoint that holds the parameters of @p state and no buffer. */
    void write_parameters_only(const std::filesystem::path& path, NN::ModelState& state)
    {
        std::uint32_t count = 0;
        for (const NN::StateEntry& entry : state.entries())
        {
            if (entry.kind == NN::StateKind::Parameter)
                ++count;
        }

        std::ofstream out(path, std::ios::binary);
        SA_ASSERT_OK_AND_ASSIGN(IO::CheckpointWriter writer,
                                IO::CheckpointWriter::create(out, count));
        for (const NN::StateEntry& entry : state.entries())
        {
            if (entry.kind != NN::StateKind::Parameter)
                continue;

            IO::TensorRecord record;
            record.name = entry.name;
            record.dtype = entry.dtype;
            record.shape = entry.shape;
            const std::vector<std::uint8_t> payload(record.payload_bytes(), 0);
            SA_EXPECT_OK(writer.write(record, payload.data()));
        }
        writer.finish();
    }

    using SaTest::ScratchFile;
} // namespace

/**
 * @brief Tests that every parameter and buffer of a state, statistics included, round-trips
 *        bit for bit through a checkpoint.
 */
TEST_F(Integration_StateCheckpoint, SavesAndLoadsParametersAndBuffersBitForBit)
{
    const ScratchFile file("sushiai_state_roundtrip.sushi");

    NormModel saved_model;
    trace_norm(saved_model);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState saved,
                            NN::ModelState::create(*engine, saved_model.model));
    fill_state(saved, 1.0f);
    std::vector<std::uint32_t> expected;
    collect_bits(saved, expected);

    SA_EXPECT_OK(IO::save_checkpoint(file.path(), saved, *engine));

    NormModel loaded_model;
    trace_norm(loaded_model);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState loaded,
                            NN::ModelState::create(*engine, loaded_model.model));
    fill_state(loaded, 2.0f);
    std::vector<std::uint32_t> before;
    collect_bits(loaded, before);
    ASSERT_NE(before, expected);

    SA_EXPECT_OK(IO::load_checkpoint(file.path(), loaded));
    std::vector<std::uint32_t> after;
    collect_bits(loaded, after);
    EXPECT_EQ(after, expected);
}

/**
 * @brief Tests that the file holds one record per entry, in the walk order of the state, and
 *        that each record's payload is the length its header declares.
 */
TEST_F(Integration_StateCheckpoint, TheFileListsEveryEntryInWalkOrder)
{
    const ScratchFile file("sushiai_state_walk_order.sushi");

    NormModel norm;
    trace_norm(norm);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, norm.model));
    fill_state(state, 1.0f);
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), state, *engine));

    std::vector<std::string> expected_names;
    for (const NN::StateEntry& entry : state.entries())
        expected_names.push_back(entry.name);

    std::ifstream in(file.path(), std::ios::binary);
    ASSERT_TRUE(in.is_open());
    SA_ASSERT_OK_AND_ASSIGN(IO::CheckpointReader reader, IO::CheckpointReader::open(in));
    ASSERT_EQ(reader.count(), expected_names.size());

    std::vector<std::string> names;
    while (reader.read_count() < reader.count())
    {
        SA_ASSERT_OK_AND_ASSIGN(IO::TensorRecord record, reader.next());
        names.push_back(record.name);
        std::vector<std::uint8_t> payload(record.payload_bytes());
        SA_EXPECT_OK(reader.read_payload(record, payload.data()));
    }
    EXPECT_EQ(names, expected_names);
}

/**
 * @brief Tests that a checkpoint without the buffers is refused, and the refusal names the
 *        running mean the file does not supply.
 */
TEST_F(Integration_StateCheckpoint, ACheckpointWithoutBuffersIsRefusedAndNamesTheRunningMean)
{
    const ScratchFile file("sushiai_state_no_buffers.sushi");

    NormModel norm;
    trace_norm(norm);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, norm.model));
    write_parameters_only(file.path(), state);

    SA_EXPECT_FAILURE(IO::load_checkpoint(file.path(), state), StatusCode::Incompatible,
                      "running_mean");
}

/**
 * @brief Tests that a path whose directory does not exist is refused as an invalid file.
 */
TEST_F(Integration_StateCheckpoint, AnUnwritablePathIsInvalidFile)
{
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "sushiai_state_no_such_directory";
    std::error_code ignored;
    ASSERT_FALSE(std::filesystem::exists(directory, ignored));

    NormModel norm;
    trace_norm(norm);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, norm.model));

    SA_EXPECT_FAILURE(IO::save_checkpoint(directory / "state.sushi", state, *engine),
                      StatusCode::InvalidFile);
}
