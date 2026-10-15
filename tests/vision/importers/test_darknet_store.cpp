/**************************************************************************/
/* test_darknet_store.cpp                                                 */
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
 * @file test_darknet_store.cpp
 * @brief Tests the kernel-last permutation, the empty-name rule and the refusals of the Darknet
 *        store against a ModelState.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/vision/importers/darknet_fold.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    using Vision::Importers::DarknetConvolution;
    using Vision::Importers::DarknetConvolutionShape;
    using Vision::Importers::DarknetWeights;
    using Vision::Importers::DarknetWeightTarget;
    using Vision::Importers::fold_darknet_normalisation;
    using Vision::Importers::store_darknet_weights;

    class Integration_DarknetStore : public SaTest::EngineTest {};

    /** @brief Holds the value the test writes into every entry before a store runs. */
    constexpr float K_SENTINEL = -7.0f;

    /** @brief Describes an entry of a given name, identity and extents, for from_entries. */
    NN::StateEntry make_entry(std::string name, std::uint64_t state, Shape shape)
    {
        NN::StateEntry entry;
        entry.name = std::move(name);
        entry.state = Graph::StateID{state};
        entry.shape = shape;
        return entry;
    }

    /**
     * @brief Returns the entries of a model with one plain and one normalised convolution.
     *
     * Layer 0 has filters 2, channels 3 and kernel side 2; layer 1 has filters 2, channels 1
     * and kernel side 1, and carries batch norm in its own entries.
     */
    std::vector<NN::StateEntry> model_entries()
    {
        return {
            make_entry("0.conv.weight", 1, Shape{2, 2, 2, 3}),
            make_entry("0.conv.bias", 2, Shape{2}),
            make_entry("1.conv.weight", 3, Shape{2, 1, 1, 1}),
            make_entry("1.conv.bias", 4, Shape{2}),
            make_entry("1.bn.weight", 5, Shape{2}),
            make_entry("1.bn.bias", 6, Shape{2}),
            make_entry("1.bn.running_mean", 7, Shape{2}),
            make_entry("1.bn.running_var", 8, Shape{2}),
        };
    }

    /** @brief Returns the two layers of the model as the reader would describe them. */
    DarknetWeights make_weights()
    {
        DarknetConvolution plain;
        plain.biases = {0.5f, -0.5f};
        plain.weights.resize(2 * 3 * 2 * 2);
        for (std::size_t k = 0; k < plain.weights.size(); ++k)
            plain.weights[k] = static_cast<float>(k);

        DarknetConvolution normalised;
        normalised.biases = {1.5f, -1.5f};
        normalised.scales = {2.0f, 3.0f};
        normalised.rolling_mean = {0.25f, 0.75f};
        normalised.rolling_variance = {4.0f, 9.0f};
        normalised.weights = {10.0f, 20.0f};

        DarknetWeights weights;
        weights.convolutions = {plain, normalised};
        return weights;
    }

    /** @brief Holds the shapes of the two layers make_weights() returns. */
    const DarknetConvolutionShape K_SHAPES[] = {
        {0, 2, 3, 2, false},
        {1, 2, 1, 1, true},
    };

    /** @brief Names the entries of the unfolded model, one target per layer. */
    const DarknetWeightTarget K_TARGETS[] = {
        {"0.conv.weight", "0.conv.bias", "", "", ""},
        {"1.conv.weight", "1.bn.bias", "1.bn.weight", "1.bn.running_mean", "1.bn.running_var"},
    };

    /** @brief Returns the values of the entry @p name in @p state, in memory order. */
    std::vector<float> values_of(NN::ModelState& state, std::string_view name)
    {
        SushiBLAS::Tensor& tensor = *state.tensor(name).value();
        const float* data = tensor.data_as<float>();
        return std::vector<float>(data, data + tensor.num_elements);
    }

    /** @brief Writes @p value into every element of the entry @p name in @p state. */
    void fill(NN::ModelState& state, std::string_view name, float value)
    {
        SushiBLAS::Tensor& tensor = *state.tensor(name).value();
        float* data = tensor.data_as<float>();
        for (std::int64_t i = 0; i < tensor.num_elements; ++i)
            data[i] = value;
    }

} // namespace

/**
 * @brief Tests that weights reach their entry kernel last and every other vector lands in order.
 *
 * The expected weights are built by walking the destination order and reading the source as
 * [out, in, kh, kw], which the test states on its own, not by the store's index formula.
 */
TEST_F(Integration_DarknetStore, WeightsMoveToKernelLastAndEveryVectorLands)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, model_entries()));
    SA_EXPECT_OK(store_darknet_weights(make_weights(), K_SHAPES, K_TARGETS, state));

    std::vector<float> expected_weights;
    for (std::size_t o = 0; o < 2; ++o)
    {
        for (std::size_t y = 0; y < 2; ++y)
        {
            for (std::size_t x = 0; x < 2; ++x)
            {
                for (std::size_t i = 0; i < 3; ++i)
                    expected_weights.push_back(static_cast<float>(((o * 3 + i) * 2 + y) * 2 + x));
            }
        }
    }
    EXPECT_EQ(values_of(state, "0.conv.weight"), expected_weights);
    EXPECT_EQ(values_of(state, "0.conv.bias"), (std::vector<float>{0.5f, -0.5f}));
    EXPECT_EQ(values_of(state, "1.conv.weight"), (std::vector<float>{10.0f, 20.0f}));
    EXPECT_EQ(values_of(state, "1.bn.weight"), (std::vector<float>{2.0f, 3.0f}));
    EXPECT_EQ(values_of(state, "1.bn.bias"), (std::vector<float>{1.5f, -1.5f}));
    EXPECT_EQ(values_of(state, "1.bn.running_mean"), (std::vector<float>{0.25f, 0.75f}));
    EXPECT_EQ(values_of(state, "1.bn.running_var"), (std::vector<float>{4.0f, 9.0f}));
}

/**
 * @brief Tests that a folded convolution stores its weight and bias and leaves the batch-norm
 *        entries it does not name.
 */
TEST_F(Integration_DarknetStore, AFoldedConvolutionLeavesItsNormalisationEntriesAlone)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, model_entries()));
    fill(state, "1.bn.weight", K_SENTINEL);
    fill(state, "1.bn.bias", K_SENTINEL);
    fill(state, "1.bn.running_mean", K_SENTINEL);
    fill(state, "1.bn.running_var", K_SENTINEL);

    DarknetWeights weights = make_weights();
    fold_darknet_normalisation(weights.convolutions[1]);
    const DarknetWeightTarget folded_targets[] = {
        {"0.conv.weight", "0.conv.bias", "", "", ""},
        {"1.conv.weight", "1.conv.bias", "", "", ""},
    };
    SA_EXPECT_OK(store_darknet_weights(weights, K_SHAPES, folded_targets, state));

    EXPECT_EQ(values_of(state, "1.conv.weight"), weights.convolutions[1].weights);
    EXPECT_EQ(values_of(state, "1.conv.bias"), weights.convolutions[1].biases);
    EXPECT_EQ(values_of(state, "1.bn.weight"), (std::vector<float>{K_SENTINEL, K_SENTINEL}));
    EXPECT_EQ(values_of(state, "1.bn.bias"), (std::vector<float>{K_SENTINEL, K_SENTINEL}));
    EXPECT_EQ(values_of(state, "1.bn.running_mean"),
              (std::vector<float>{K_SENTINEL, K_SENTINEL}));
    EXPECT_EQ(values_of(state, "1.bn.running_var"),
              (std::vector<float>{K_SENTINEL, K_SENTINEL}));
}

/**
 * @brief Tests that a target naming an entry the state lacks is refused, and the layer is named.
 */
TEST_F(Integration_DarknetStore, AMissingEntryIsRefusedAndNamesTheLayer)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, model_entries()));
    const DarknetWeightTarget targets[] = {
        {"0.conv.weight", "0.conv.bias", "", "", ""},
        {"1.conv.weight", "", "1.bn.missing", "", ""},
    };
    SA_EXPECT_FAILURE(store_darknet_weights(make_weights(), K_SHAPES, targets, state),
                      StatusCode::Incompatible, "layer 1", "'1.bn.missing'");
}

/**
 * @brief Tests that a shape that differs is refused, and that no earlier write reaches the state.
 */
TEST_F(Integration_DarknetStore, AShapeMismatchIsRefusedAndNothingIsWritten)
{
    std::vector<NN::StateEntry> entries = model_entries();
    for (NN::StateEntry& entry : entries)
    {
        if (entry.name == "1.bn.running_var")
            entry.shape = Shape{3};
    }
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, std::move(entries)));
    fill(state, "0.conv.weight", K_SENTINEL);

    SA_EXPECT_FAILURE(store_darknet_weights(make_weights(), K_SHAPES, K_TARGETS, state),
                      StatusCode::Incompatible, "'1.bn.running_var'", "shape");
    EXPECT_EQ(values_of(state, "0.conv.weight"),
              std::vector<float>(static_cast<std::size_t>(2 * 2 * 2 * 3), K_SENTINEL));
}

/**
 * @brief Tests that a vector of the wrong length is refused, as a folded scale would be.
 */
TEST_F(Integration_DarknetStore, AVectorOfTheWrongLengthIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, model_entries()));
    DarknetWeights weights = make_weights();
    fold_darknet_normalisation(weights.convolutions[1]);

    SA_EXPECT_FAILURE(store_darknet_weights(weights, K_SHAPES, K_TARGETS, state),
                      StatusCode::Incompatible, "layer 1", "0 values", "'1.bn.weight'");
}

/**
 * @brief Tests that two targets naming one entry are refused, so no entry is filled twice.
 */
TEST_F(Integration_DarknetStore, AnEntryNamedTwiceIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, model_entries()));
    const DarknetWeightTarget targets[] = {
        {"0.conv.weight", "0.conv.bias", "", "", ""},
        {"1.conv.weight", "0.conv.bias", "", "", ""},
    };
    SA_EXPECT_FAILURE(store_darknet_weights(make_weights(), K_SHAPES, targets, state),
                      StatusCode::Incompatible, "layer 1", "'0.conv.bias'", "already fills");
}

/**
 * @brief Tests that a weights file, a shape list and a target list of different lengths are
 *        refused before any entry is touched.
 */
TEST_F(Integration_DarknetStore, MismatchedCountsAreRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, model_entries()));
    const DarknetConvolutionShape one_shape[] = {{0, 2, 3, 2, false}};
    SA_EXPECT_FAILURE(store_darknet_weights(make_weights(), one_shape, K_TARGETS, state),
                      StatusCode::Incompatible, "the weights hold 2 convolutions");
}
