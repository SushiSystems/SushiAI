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
 * @brief Tests checkpoints over a model state and the Sequential and LayerStack exchange.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/nn/node_registry.hpp>
#include "../../common/test_utils.hpp"
#include "../../common/test_status.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_Checkpoint : public SaTest::EngineTest {};

    constexpr std::string_view K_MODEL_JSON = R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "units": 8},
                   {"type": "gelu"},
                   {"type": "linear", "units": 4}]
    })";

    /** @brief The same architecture, spelled at compile time. */
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    Mlp make_sequential()
    {
        return Mlp(NN::Linear{6, 8}, NN::GELU{}, NN::Linear{8, 4});
    }

    /** @brief Builds a registry holding the rows register_layers adds, one fresh copy per call. */
    NN::NodeRegistry make_layer_registry()
    {
        NN::NodeRegistry registry;
        SA_EXPECT_OK(NN::register_layers(registry));
        return registry;
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
        const Graph::ValueId x = builder.input(Shape{4, 6}, "inputs");
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

TEST_F(Integration_Checkpoint, SavesAndLoadsEveryParameterBitForBit)
{
    const ScratchFile file("sushiai_roundtrip.ckpt");

    Materialised<Mlp> trained(*engine, make_sequential(), 11);
    const std::vector<std::uint32_t> expected = trained.bits();

    SA_EXPECT_OK(IO::save_checkpoint(file.path(), trained.state(), *engine));

    // A different seed, so a load that did nothing at all would be visible.
    Materialised<Mlp> fresh(*engine, make_sequential(), 22);
    ASSERT_NE(fresh.bits(), expected);

    SA_EXPECT_OK(IO::load_checkpoint(file.path(), fresh.state()));
    EXPECT_EQ(fresh.bits(), expected);
}

TEST_F(Integration_Checkpoint, ASequentialCheckpointLoadsIntoAnEquivalentLayerStackAndBack)
{
    // Both spellings of the model swap a checkpoint; see tests/functional/integration/README.md.
    const ScratchFile from_sequential("sushiai_equivalence_a.ckpt");
    const ScratchFile from_stack("sushiai_equivalence_b.ckpt");

    Materialised<Mlp> compile_time(*engine, make_sequential(), 7);
    Materialised<NN::LayerStack> run_time(*engine, make_stack(), 99);

    // Names must match before any load is compared; see tests/functional/integration/README.md.
    ASSERT_EQ(compile_time.names(), run_time.names());
    ASSERT_EQ(compile_time.names(),
              (std::vector<std::string>{"0.weight", "0.bias", "2.weight", "2.bias"}));

    const std::vector<std::uint32_t> sequential_bits = compile_time.bits();
    ASSERT_NE(run_time.bits(), sequential_bits);

    // Sequential -> LayerStack
    SA_EXPECT_OK(IO::save_checkpoint(from_sequential.path(), compile_time.state(), *engine));
    SA_EXPECT_OK(IO::load_checkpoint(from_sequential.path(), run_time.state()));
    EXPECT_EQ(run_time.bits(), sequential_bits);

    // The return trip loads into a third model; see tests/functional/integration/README.md.
    Materialised<Mlp> third(*engine, make_sequential(), 123);
    ASSERT_NE(third.bits(), sequential_bits);

    SA_EXPECT_OK(IO::save_checkpoint(from_stack.path(), run_time.state(), *engine));
    SA_EXPECT_OK(IO::load_checkpoint(from_stack.path(), third.state()));
    EXPECT_EQ(third.bits(), sequential_bits);
}

TEST_F(Integration_Checkpoint, AParameterTheFileDoesNotSupplyIsRefusedAndNamesIt)
{
    const ScratchFile file("sushiai_missing.ckpt");

    // A different architecture, so the file lacks 2.weight and 2.bias.
    using Shallow = NN::Sequential<NN::Linear>;
    Materialised<Shallow> shallow(*engine, Shallow(NN::Linear{6, 8}), 5);
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), shallow.state(), *engine));

    Materialised<Mlp> deep(*engine, make_sequential(), 6);
    SA_EXPECT_FAILURE(IO::load_checkpoint(file.path(), deep.state()),
                      StatusCode::Incompatible, "does not supply", "2.weight");
}

TEST_F(Integration_Checkpoint, AParameterTheModelDoesNotHaveIsRefusedAndNamesIt)
{
    const ScratchFile file("sushiai_extra.ckpt");

    Materialised<Mlp> deep(*engine, make_sequential(), 5);
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), deep.state(), *engine));

    using Shallow = NN::Sequential<NN::Linear>;
    Materialised<Shallow> shallow(*engine, Shallow(NN::Linear{6, 8}), 6);
    SA_EXPECT_FAILURE(IO::load_checkpoint(file.path(), shallow.state()),
                      StatusCode::Incompatible, "2.weight", "this model has no such entry");
}

TEST_F(Integration_Checkpoint, AShapeMismatchIsRefusedWithBothShapes)
{
    const ScratchFile file("sushiai_shape.ckpt");

    // Shapes are compared, not element counts; see tests/functional/integration/README.md.
    using OneLayer = NN::Sequential<NN::Linear>;
    Materialised<OneLayer> wide(*engine, OneLayer(NN::Linear{6, 8}), 5);
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), wide.state(), *engine));

    Materialised<OneLayer> narrow(*engine, OneLayer(NN::Linear{6, 4}), 6);
    SA_EXPECT_FAILURE(IO::load_checkpoint(file.path(), narrow.state()),
                      StatusCode::Incompatible, "0.weight", "[8, 6]", "[4, 6]");
}

TEST_F(Integration_Checkpoint, ADuplicatedNameIsRefusedAndNamesTheDuplicate)
{
    const ScratchFile file("sushiai_duplicate.ckpt");

    // save_checkpoint cannot write a duplicated name; see tests/functional/integration/README.md.
    Materialised<Mlp> model(*engine, make_sequential(), 5);

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
