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
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include "../../common/test_status.hpp"
#include "../../common/test_utils.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_StateCheckpoint : public SaTest::EngineTest {};

    /** @brief The architecture the module-form case traces. */
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    Mlp make_mlp()
    {
        return Mlp(NN::Linear{6, 8}, NN::GELU{}, NN::Linear{8, 4});
    }

    /** @brief Traces @p model into @p graph from a [4, 6] input. */
    void trace_mlp(Mlp& model, Graph::Graph& graph)
    {
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueId x = builder.input(Shape{4, 6}, "inputs");
        static_cast<void>(model.forward(builder, x));
    }

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
        const Graph::ValueId x = builder.input(Shape{2, 4, 4, 3}, "x");
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

    /** @brief Writes a distinct value into every parameter of @p model, through @p pool. */
    void fill_module(Mlp& model, Graph::TensorPool& pool)
    {
        float offset = 1.0f;
        NN::for_each_parameter(model, [&](const std::string&, NN::Parameter& parameter)
        {
            SushiBLAS::Tensor& tensor = pool.tensor(parameter.id());
            float* values = tensor.data_as<float>();
            for (int64_t i = 0; i < tensor.num_elements; ++i)
                values[i] = offset + 0.25f * static_cast<float>(i);
            offset += 1.0f;
        });
    }

    /** @brief Writes every parameter of @p model, through @p pool, into @p out as bit patterns. */
    void collect_module_bits(Mlp& model, Graph::TensorPool& pool,
                             std::vector<std::uint32_t>& out)
    {
        out.clear();
        NN::for_each_parameter(model, [&](const std::string&, NN::Parameter& parameter)
        {
            const SushiBLAS::Tensor& tensor = pool.tensor(parameter.id());
            const float* values = tensor.data_as<float>();
            for (int64_t i = 0; i < tensor.num_elements; ++i)
            {
                std::uint32_t pattern = 0;
                std::memcpy(&pattern, &values[i], sizeof(pattern));
                out.push_back(pattern);
            }
        });
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

    /**
     * @class ScratchFile
     * @brief A checkpoint path that removes itself.
     */
    class ScratchFile
    {
        public:
            explicit ScratchFile(const char* name)
                : path_(std::filesystem::temp_directory_path() / name)
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }

            ~ScratchFile()
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }

            ScratchFile(const ScratchFile&) = delete;
            ScratchFile& operator=(const ScratchFile&) = delete;

            /** @brief Returns the path this file removes on destruction. */
            const std::filesystem::path& path() const noexcept { return path_; }

        private:
            /** @brief Holds the path of the scratch file. */
            std::filesystem::path path_;
    };
} // namespace

/**
 * @brief Tests that every parameter and buffer of a state, statistics included, round-trips
 *        bit for bit through a checkpoint.
 */
TEST_F(Integration_StateCheckpoint, SavesAndLoadsParametersAndBuffersBitForBit)
{
    const ScratchFile file("sushiai_state_roundtrip.ckpt");

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
    const ScratchFile file("sushiai_state_walk_order.ckpt");

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
    const ScratchFile file("sushiai_state_no_buffers.ckpt");

    NormModel norm;
    trace_norm(norm);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, norm.model));
    write_parameters_only(file.path(), state);

    SA_EXPECT_FAILURE(IO::load_checkpoint(file.path(), state), StatusCode::Incompatible,
                      "running_mean");
}

/**
 * @brief Tests that a file the module form wrote loads into a state of the same model, bit for
 *        bit, because both forms share one format.
 */
TEST_F(Integration_StateCheckpoint, AModuleFormCheckpointLoadsIntoAState)
{
    const ScratchFile file("sushiai_state_module_form.ckpt");

    Mlp source = make_mlp();
    Graph::Graph source_graph;
    trace_mlp(source, source_graph);
    Graph::TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(source_graph));
    fill_module(source, pool);
    std::vector<std::uint32_t> expected;
    collect_module_bits(source, pool, expected);

    SA_EXPECT_OK(IO::save_checkpoint(file.path(), source, pool, *engine));

    Mlp target = make_mlp();
    Graph::Graph target_graph;
    trace_mlp(target, target_graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, target));
    SA_EXPECT_OK(IO::load_checkpoint(file.path(), state));

    std::vector<std::uint32_t> loaded;
    collect_bits(state, loaded);
    EXPECT_EQ(loaded, expected);
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

    SA_EXPECT_FAILURE(IO::save_checkpoint(directory / "state.ckpt", state, *engine),
                      StatusCode::InvalidFile);
}
