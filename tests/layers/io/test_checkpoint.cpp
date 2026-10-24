/**************************************************************************/
/* test_checkpoint.cpp                                                    */
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
 * @file test_checkpoint.cpp
 * @brief Tests checkpoints: the parameters and buffers a save holds and a load restores.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/layer_rows/layer_registry.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    using Test::make_layer_registry;

    class Medium_Checkpoint : public SaTest::EngineTest {};

    constexpr std::string_view K_MODEL_JSON = R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "units": 8},
                   {"type": "gelu"},
                   {"type": "linear", "units": 4}]
    })";

    /** @brief The same architecture, spelled at compile time. */
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    MLP make_sequential()
    {
        return MLP(NN::Linear{6, 8}, NN::GELU{}, NN::Linear{8, 4});
    }

    NN::LayerStack make_stack()
    {
        return NN::build_layer_stack(
            make_layer_registry(),
            Config::parse_model(Config::parse_json(K_MODEL_JSON, "<test>"), "<test>"));
    }

    /**
     * @brief Traces @p model into @p graph, then creates the state of the traced model.
     * @return The state; a failed create is reported and stops the process on value().
     */
    template<typename M>
    NN::ModelState traced_state(SushiBLAS::Engine& engine, M& model, Graph::Graph& graph)
    {
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{4, 6}, "inputs");
        (void)model.forward(builder, x);

        Result<NN::ModelState> created = NN::ModelState::create(engine, model);
        SA_EXPECT_OK(created);
        return std::move(created).value();
    }

    /**
     * @class Materialised
     * @brief A traced and initialised model, and the state that holds its tensors.
     *
     * Owns the state, because every tensor a test holds has to outlive the comparison.
     */
    template<typename M>
    class Materialised
    {
        public:
            Materialised(SushiBLAS::Engine& engine, M model, std::uint64_t seed)
                : model_(std::move(model)), state_(traced_state(engine, model_, graph_))
            {
                NN::InitPolicy policy;
                policy.seed = seed;
                policy.zero_final_layer = false;   // so every parameter carries real numbers
                SA_EXPECT_OK(NN::initialize(engine, state_, policy));
            }

            M& model() noexcept { return model_; }
            NN::ModelState& state() noexcept { return state_; }

            /** @brief Every parameter's raw bit patterns, in walk order. */
            std::vector<std::uint32_t> bits()
            {
                std::vector<std::uint32_t> out;
                NN::for_each_parameter(model_, [&](const std::string& name, NN::Parameter&)
                {
                    const SushiBLAS::Tensor& tensor = tensor_named(name);
                    const float* values = tensor.data_as<float>();
                    for (int64_t i = 0; i < tensor.num_elements; ++i)
                    {
                        std::uint32_t pattern = 0;
                        std::memcpy(&pattern, &values[i], sizeof(pattern));
                        out.push_back(pattern);
                    }
                });
                return out;
            }

            /** @brief Every parameter's dotted name, in walk order. */
            std::vector<std::string> names()
            {
                std::vector<std::string> out;
                NN::for_each_parameter(model_, [&out](const std::string& name, NN::Parameter&)
                {
                    out.push_back(name);
                });
                return out;
            }

        private:
            /**
             * @brief Returns the state tensor of the parameter called @p name.
             * @return The tensor; stops the process when no entry has that name, which the walk
             *         of the same model rules out.
             */
            SushiBLAS::Tensor& tensor_named(const std::string& name)
            {
                Result<SushiBLAS::Tensor*> found = state_.tensor(name);
                SA_EXPECT_OK(found);
                return *found.value();
            }

            M model_;
            Graph::Graph graph_{};
            NN::ModelState state_;
    };

    using SaTest::ScratchFile;
}

TEST_F(Medium_Checkpoint, SavesAndLoadsEveryParameterBitForBit)
{
    const ScratchFile file("sushiai_roundtrip.sushi");

    Materialised<MLP> trained(*engine, make_sequential(), 11);
    const std::vector<std::uint32_t> expected = trained.bits();

    SA_EXPECT_OK(IO::save_checkpoint(file.path(), trained.state()));

    // A different seed, so a load that did nothing at all would be visible.
    Materialised<MLP> fresh(*engine, make_sequential(), 22);
    ASSERT_NE(fresh.bits(), expected);

    SA_EXPECT_OK(IO::load_checkpoint(file.path(), fresh.state()));
    EXPECT_EQ(fresh.bits(), expected);
}

TEST_F(Medium_Checkpoint, ASequentialCheckpointLoadsIntoAnEquivalentLayerStackAndBack)
{
    // Both spellings of the model swap a checkpoint; see tests/layers/io/README.md.
    const ScratchFile from_sequential("sushiai_equivalence_a.sushi");
    const ScratchFile from_stack("sushiai_equivalence_b.sushi");

    Materialised<MLP> compile_time(*engine, make_sequential(), 7);
    Materialised<NN::LayerStack> run_time(*engine, make_stack(), 99);

    // Names must match before any load is compared; see tests/layers/io/README.md.
    ASSERT_EQ(compile_time.names(), run_time.names());
    ASSERT_EQ(compile_time.names(),
              (std::vector<std::string>{"0.weight", "0.bias", "2.weight", "2.bias"}));

    const std::vector<std::uint32_t> sequential_bits = compile_time.bits();
    ASSERT_NE(run_time.bits(), sequential_bits);

    // Sequential -> LayerStack
    SA_EXPECT_OK(IO::save_checkpoint(from_sequential.path(), compile_time.state()));
    SA_EXPECT_OK(IO::load_checkpoint(from_sequential.path(), run_time.state()));
    EXPECT_EQ(run_time.bits(), sequential_bits);

    // The return trip loads into a third model; see tests/layers/io/README.md.
    Materialised<MLP> third(*engine, make_sequential(), 123);
    ASSERT_NE(third.bits(), sequential_bits);

    SA_EXPECT_OK(IO::save_checkpoint(from_stack.path(), run_time.state()));
    SA_EXPECT_OK(IO::load_checkpoint(from_stack.path(), third.state()));
    EXPECT_EQ(third.bits(), sequential_bits);
}

TEST_F(Medium_Checkpoint, AParameterTheFileDoesNotSupplyIsRefusedAndNamesIt)
{
    const ScratchFile file("sushiai_missing.sushi");

    // A different architecture, so the file lacks 2.weight and 2.bias.
    using Shallow = NN::Sequential<NN::Linear>;
    Materialised<Shallow> shallow(*engine, Shallow(NN::Linear{6, 8}), 5);
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), shallow.state()));

    Materialised<MLP> deep(*engine, make_sequential(), 6);
    SA_EXPECT_FAILURE(IO::load_checkpoint(file.path(), deep.state()),
                      StatusCode::Incompatible, "does not supply", "2.weight");
}

TEST_F(Medium_Checkpoint, AParameterTheModelDoesNotHaveIsRefusedAndNamesIt)
{
    const ScratchFile file("sushiai_extra.sushi");

    Materialised<MLP> deep(*engine, make_sequential(), 5);
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), deep.state()));

    using Shallow = NN::Sequential<NN::Linear>;
    Materialised<Shallow> shallow(*engine, Shallow(NN::Linear{6, 8}), 6);
    SA_EXPECT_FAILURE(IO::load_checkpoint(file.path(), shallow.state()),
                      StatusCode::Incompatible, "2.weight", "this model has no such entry");
}

TEST_F(Medium_Checkpoint, AShapeMismatchIsRefusedWithBothShapes)
{
    const ScratchFile file("sushiai_shape.sushi");

    // Shapes are compared, not element counts; see tests/layers/io/README.md.
    using OneLayer = NN::Sequential<NN::Linear>;
    Materialised<OneLayer> wide(*engine, OneLayer(NN::Linear{6, 8}), 5);
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), wide.state()));

    Materialised<OneLayer> narrow(*engine, OneLayer(NN::Linear{6, 4}), 6);
    SA_EXPECT_FAILURE(IO::load_checkpoint(file.path(), narrow.state()),
                      StatusCode::Incompatible, "0.weight", "[8, 6]", "[4, 6]");
}

TEST_F(Medium_Checkpoint, ADuplicatedNameIsRefusedAndNamesTheDuplicate)
{
    const ScratchFile file("sushiai_duplicate.sushi");

    // save_checkpoint cannot write a duplicated name; see tests/layers/io/README.md.
    Materialised<MLP> model(*engine, make_sequential(), 5);

    {
        const std::vector<float> weight(8 * 6, 1.0f);
        const std::vector<float> bias(8, 2.0f);
        std::ofstream out(file.path(), std::ios::binary);
        SA_ASSERT_OK_AND_ASSIGN(IO::CheckpointWriter writer,
                                IO::CheckpointWriter::create(out, 4));
        SA_EXPECT_OK(writer.write(IO::TensorRecord{"0.weight", DType::Float32, Shape{8, 6}},
                                  weight.data()));
        SA_EXPECT_OK(writer.write(IO::TensorRecord{"0.bias", DType::Float32, Shape{8}},
                                  bias.data()));
        SA_EXPECT_OK(writer.write(IO::TensorRecord{"0.bias", DType::Float32, Shape{8}},
                                  bias.data()));
        SA_EXPECT_OK(writer.write(IO::TensorRecord{"2.bias", DType::Float32, Shape{4}},
                                  bias.data()));
        writer.finish();
    }

    SA_EXPECT_FAILURE(IO::load_checkpoint(file.path(), model.state()),
                      StatusCode::Incompatible, "supplies", "0.bias", "twice");
}

namespace
{
    class Medium_StateCheckpoint : public SaTest::EngineTest {};

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

} // namespace

/**
 * @brief Tests that every parameter and buffer of a state, statistics included, round-trips
 *        bit for bit through a checkpoint.
 */
TEST_F(Medium_StateCheckpoint, SavesAndLoadsParametersAndBuffersBitForBit)
{
    const ScratchFile file("sushiai_state_roundtrip.sushi");

    NormModel saved_model;
    trace_norm(saved_model);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState saved,
                            NN::ModelState::create(*engine, saved_model.model));
    fill_state(saved, 1.0f);
    std::vector<std::uint32_t> expected;
    collect_bits(saved, expected);

    SA_EXPECT_OK(IO::save_checkpoint(file.path(), saved));

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
TEST_F(Medium_StateCheckpoint, TheFileListsEveryEntryInWalkOrder)
{
    const ScratchFile file("sushiai_state_walk_order.sushi");

    NormModel norm;
    trace_norm(norm);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, norm.model));
    fill_state(state, 1.0f);
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), state));

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
TEST_F(Medium_StateCheckpoint, ACheckpointWithoutBuffersIsRefusedAndNamesTheRunningMean)
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
TEST_F(Medium_StateCheckpoint, AnUnwritablePathIsInvalidFile)
{
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "sushiai_state_no_such_directory";
    std::error_code ignored;
    ASSERT_FALSE(std::filesystem::exists(directory, ignored));

    NormModel norm;
    trace_norm(norm);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, norm.model));

    SA_EXPECT_FAILURE(IO::save_checkpoint(directory / "state.sushi", state),
                      StatusCode::InvalidFile);
}
