/**************************************************************************/
/* test_darknet_tensors.cpp                                               */
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
 * @file test_darknet_tensors.cpp
 * @brief Tests the listing of Darknet convolution vectors as the tensors of a weight file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_tensors.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using IO::ImportedTensor;
using Vision::Importers::DarknetConvolution;
using Vision::Importers::DarknetConvolutionShape;
using Vision::Importers::DarknetWeights;
using Vision::Importers::DarknetWeightTarget;
using Vision::Importers::list_darknet_tensors;

namespace
{
    /** @brief Returns a plain 2 by 3 by 2 by 2 convolution, then a normalised 2 by 1 by 1 by 1. */
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
    const std::vector<DarknetConvolutionShape> K_SHAPES = {
        {0, 2, 3, 2, false},
        {1, 2, 1, 1, true},
    };

    /** @brief Names the entries of the unfolded model, one target per layer. */
    const std::vector<DarknetWeightTarget> K_TARGETS = {
        {"0.weight", "0.bias", "", "", ""},
        {"1.weight", "2.bias", "2.weight", "2.running_mean", "2.running_var"},
    };

    /** @brief Returns the floats of @p tensor, in memory order. */
    std::vector<float> values_of(const ImportedTensor& tensor)
    {
        std::vector<float> values(tensor.payload.size() / sizeof(float));
        if (!values.empty())
            std::memcpy(values.data(), tensor.payload.data(), tensor.payload.size());
        return values;
    }

    /** @brief Returns the tensor named @p name, or null when @p tensors holds none. */
    const ImportedTensor* find_tensor(const std::vector<ImportedTensor>& tensors,
                                      std::string_view name)
    {
        for (const ImportedTensor& tensor : tensors)
        {
            if (tensor.record.name == name)
                return &tensor;
        }

        return nullptr;
    }
} // namespace

/** @brief Tests that one float32 tensor is listed per target name, with the bytes of its shape. */
TEST(Unit_DarknetTensors, ListsOneTensorPerNamedTarget)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<ImportedTensor> tensors,
                            list_darknet_tensors(make_weights(), K_SHAPES, K_TARGETS));

    ASSERT_EQ(tensors.size(), 7u);
    for (const ImportedTensor& tensor : tensors)
    {
        EXPECT_EQ(tensor.record.dtype, DType::Float32) << tensor.record.name;
        EXPECT_EQ(tensor.payload.size(), tensor.record.payload_bytes()) << tensor.record.name;
    }
}

/** @brief Tests that the weights move from [out, in, kh, kw] to [out, kh, kw, in]. */
TEST(Unit_DarknetTensors, WeightsMoveToKernelLast)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<ImportedTensor> tensors,
                            list_darknet_tensors(make_weights(), K_SHAPES, K_TARGETS));

    const ImportedTensor* const weight = find_tensor(tensors, "0.weight");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->record.shape, (Shape{2, 2, 2, 3}));

    // Source element (o, i, y, x) holds ((o * 3 + i) * 2 + y) * 2 + x.
    const std::vector<float> values = values_of(*weight);
    ASSERT_EQ(values.size(), 24u);
    EXPECT_EQ(values[0], 0.0f);
    EXPECT_EQ(values[1], 4.0f);
    EXPECT_EQ(values[2], 8.0f);
    EXPECT_EQ(values[3], 1.0f);
    EXPECT_EQ(values[23], 23.0f);
}

/** @brief Tests that a normalised convolution's vectors land under the batch-norm names. */
TEST(Unit_DarknetTensors, EveryVectorLandsUnderItsTarget)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<ImportedTensor> tensors,
                            list_darknet_tensors(make_weights(), K_SHAPES, K_TARGETS));

    const ImportedTensor* const scale = find_tensor(tensors, "2.weight");
    const ImportedTensor* const bias = find_tensor(tensors, "2.bias");
    const ImportedTensor* const mean = find_tensor(tensors, "2.running_mean");
    const ImportedTensor* const variance = find_tensor(tensors, "2.running_var");
    ASSERT_NE(scale, nullptr);
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(mean, nullptr);
    ASSERT_NE(variance, nullptr);

    EXPECT_EQ(scale->record.shape, (Shape{2}));
    EXPECT_EQ(values_of(*scale), (std::vector<float>{2.0f, 3.0f}));
    EXPECT_EQ(values_of(*bias), (std::vector<float>{1.5f, -1.5f}));
    EXPECT_EQ(values_of(*mean), (std::vector<float>{0.25f, 0.75f}));
    EXPECT_EQ(values_of(*variance), (std::vector<float>{4.0f, 9.0f}));
    EXPECT_EQ(find_tensor(tensors, "0.running_mean"), nullptr);
}

/** @brief Tests that a vector of another length is refused and names the layer and the entry. */
TEST(Unit_DarknetTensors, AVectorOfTheWrongLengthIsRefused)
{
    DarknetWeights weights = make_weights();
    weights.convolutions[1].scales.push_back(1.0f);

    SA_EXPECT_FAILURE(list_darknet_tensors(weights, K_SHAPES, K_TARGETS),
                      StatusCode::Incompatible, "layer 1", "'2.weight'", "3 values");
}

/** @brief Tests that two targets naming one entry are refused. */
TEST(Unit_DarknetTensors, AnEntryNamedTwiceIsRefused)
{
    std::vector<DarknetWeightTarget> targets = K_TARGETS;
    targets[1].bias = "0.bias";

    SA_EXPECT_FAILURE(list_darknet_tensors(make_weights(), K_SHAPES, targets),
                      StatusCode::Incompatible, "layer 1", "'0.bias'", "already fills");
}

/** @brief Tests that a shape that is not positive is refused before any vector is read. */
TEST(Unit_DarknetTensors, AShapeThatIsNotPositiveIsRefused)
{
    std::vector<DarknetConvolutionShape> shapes = K_SHAPES;
    shapes[0].channels = 0;

    SA_EXPECT_FAILURE(list_darknet_tensors(make_weights(), shapes, K_TARGETS),
                      StatusCode::InvalidDescription, "layer 0", "channels 0");
}

/** @brief Tests that weights, shapes and targets of different counts are refused. */
TEST(Unit_DarknetTensors, MismatchedCountsAreRefused)
{
    std::vector<DarknetWeightTarget> targets = K_TARGETS;
    targets.pop_back();

    SA_EXPECT_FAILURE(list_darknet_tensors(make_weights(), K_SHAPES, targets),
                      StatusCode::Incompatible, "2 convolutions", "the targets 1");
}
