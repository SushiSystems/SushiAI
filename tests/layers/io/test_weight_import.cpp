/**************************************************************************/
/* test_weight_import.cpp                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_weight_import.cpp
 * @brief Tests the strict match of a weight source against the state records of a small model.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/filter_layout.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/name_map.hpp>
#include <SushiAI/io/safetensors_weight_source.hpp>
#include <SushiAI/io/torch_weight_source.hpp>
#include <SushiAI/io/weight_import.hpp>
#include <SushiAI/io/weight_source.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/shape/shape.hpp>
#include "archive_bytes.hpp"
#include "weight_fixtures.hpp"

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Names the file in every message. */
    constexpr std::string_view K_ORIGIN = "fixture.bin";

    /** @brief Holds the map both fixtures share: the net. prefix goes, the counter is dropped. */
    constexpr std::string_view K_MAP =
        R"({"state_dict":"state_dict","rename":[{"from":"net.","to":""}],)"
        R"("drop":["num_batches_tracked"]})";

    /** @brief Returns a float32 record named @p name of shape @p shape. */
    NN::StateEntry record(std::string name, const Shape& shape, DType dtype = DType::Float32)
    {
        NN::StateEntry entry;
        entry.name = std::move(name);
        entry.shape = shape;
        entry.dtype = dtype;
        return entry;
    }

    /** @brief Returns the records of the small model: fc.weight [2, 3] and fc.bias [2]. */
    std::vector<NN::StateEntry> model_records()
    {
        return {record("fc.weight", Shape{2, 3}), record("fc.bias", Shape{2})};
    }

    /** @brief Returns the registry of the two containers. */
    IO::WeightSourceRegistry both_containers()
    {
        IO::WeightSourceRegistry registry;
        EXPECT_TRUE(registry.add(IO::K_TORCH_WEIGHT_SOURCE_ROW).is_ok());
        EXPECT_TRUE(registry.add(IO::K_SAFETENSORS_WEIGHT_SOURCE_ROW).is_ok());
        return registry;
    }

    /** @brief Opens @p bytes through the registry of both containers. */
    std::unique_ptr<IO::WeightSource> open(std::vector<std::byte> bytes)
    {
        IO::WeightSourceOptions options;
        options.origin = K_ORIGIN;
        options.state_key = "state_dict";
        Result<std::unique_ptr<IO::WeightSource>> opened =
            IO::open_weight_source(both_containers(), std::move(bytes), options);
        EXPECT_TRUE(opened.has_value()) << opened.status().message();
        return std::move(opened).value();
    }

    /** @brief Runs the strict import of @p source against @p records under the shared map. */
    Result<std::vector<IO::ImportedTensor>> import_with(const std::vector<NN::StateEntry>& records,
                                                        const IO::WeightSource& source,
                                                        std::string_view map_text = K_MAP)
    {
        SA_TRY_ASSIGN(const IO::NameMap map, IO::NameMap::parse(map_text, "map.json"));
        return IO::import_state_weights(
            SushiRuntime::span<const NN::StateEntry>(records.data(), records.size()), source, map);
    }

    /** @brief Returns the bytes of @p values as the float32 payload a record states. */
    std::vector<std::byte> payload_of(const std::vector<float>& values)
    {
        return SaTest::bytes_of(SaTest::float_bytes(values));
    }

    /** @brief Returns a safetensors file of one float32 tensor "conv.weight" of @p dims. */
    std::vector<std::byte> conv_file(const std::string& dims, const std::vector<float>& values)
    {
        const std::string header = R"({"conv.weight":{"dtype":"F32","shape":)" + dims +
                                   R"(,"data_offsets":[0,)" + std::to_string(values.size() * 4) +
                                   "]}}";
        std::string file;
        SaTest::put_le(file, header.size(), 8);
        file += header;
        file += SaTest::float_bytes(values);
        return SaTest::bytes_of(file);
    }

    /** @brief Returns 1, 2, ... @p count as floats. */
    std::vector<float> counting(int count)
    {
        std::vector<float> values;
        for (int i = 1; i <= count; ++i)
            values.push_back(static_cast<float>(i));
        return values;
    }

    /**
     * @brief Returns the [out, kh, kw, in] layout of the [out, in, kh, kw] weight @p values by
     *        the index formula alone: element (o, h, w, i) is values[((o * in + i) * kh + h) * kw + w].
     */
    std::vector<float> kernel_last_by_formula(const std::vector<float>& values, int out, int in,
                                              int kh, int kw)
    {
        std::vector<float> result;
        for (int o = 0; o < out; ++o)
            for (int h = 0; h < kh; ++h)
                for (int w = 0; w < kw; ++w)
                    for (int i = 0; i < in; ++i)
                        result.push_back(values[static_cast<std::size_t>(
                            ((o * in + i) * kh + h) * kw + w)]);
        return result;
    }
} // namespace

/** @brief Tests that a torch file and a safetensors file of one model give the same records. */
TEST(Small_WeightImport, BothContainersGiveTheSameRecordsAndBytes)
{
    const std::unique_ptr<IO::WeightSource> torch = open(SaTest::torch_fixture());
    const std::unique_ptr<IO::WeightSource> safetensors = open(SaTest::safetensors_fixture());
    const std::vector<NN::StateEntry> records = model_records();

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> from_torch,
                            import_with(records, *torch));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> from_file,
                            import_with(records, *safetensors));

    ASSERT_EQ(from_torch.size(), 2u);
    ASSERT_EQ(from_file.size(), 2u);
    // The records come in the model's order, whatever order the file lists its tensors in.
    const std::vector<std::vector<float>> expected = {SaTest::fixture_weight(),
                                                      SaTest::fixture_bias()};
    for (std::size_t i = 0; i < 2; ++i)
    {
        EXPECT_EQ(from_torch[i].record.name, records[i].name);
        EXPECT_EQ(from_torch[i].record.shape, records[i].shape);
        EXPECT_EQ(from_torch[i].record.dtype, DType::Float32);
        EXPECT_EQ(from_file[i].record.name, from_torch[i].record.name);
        EXPECT_EQ(from_file[i].record.shape, from_torch[i].record.shape);
        EXPECT_EQ(from_file[i].record.dtype, from_torch[i].record.dtype);
        // Every value is exact in float32, so the payloads equal the bytes worked out by hand.
        EXPECT_EQ(from_torch[i].payload, payload_of(expected[i]));
        EXPECT_EQ(from_file[i].payload, from_torch[i].payload);
    }
}

/** @brief Tests that a missing, an extra and a misshapen tensor come out in one message. */
TEST(Small_WeightImport, OneMessageNamesTheMissingTheExtraAndTheMisshapenTensor)
{
    // The file holds net.fc.weight [2, 3] and net.fc.bias [2]; the model fc.weight [3, 2].
    const std::unique_ptr<IO::WeightSource> source = open(SaTest::safetensors_fixture());
    const std::vector<NN::StateEntry> records = {record("fc.weight", Shape{3, 2}),
                                                 record("fc.missing", Shape{4})};

    const Result<std::vector<IO::ImportedTensor>> imported = import_with(records, *source);

    // fc.missing has no tensor, fc.bias has no record, and fc.weight differs in shape.
    SA_EXPECT_FAILURE(imported, StatusCode::Incompatible, "fixture.bin", "'fc.missing'",
                      "'net.fc.bias' as 'fc.bias'",
                      "'fc.weight' is [3, 2] in the model and [2, 3] in the file ('net.fc.weight')");
}

/** @brief Tests that the one message groups the three kinds under their own headings and counts. */
TEST(Small_WeightImport, TheMessageGroupsEachKindWithItsCount)
{
    const std::unique_ptr<IO::WeightSource> source = open(SaTest::safetensors_fixture());
    const std::vector<NN::StateEntry> records = {record("fc.weight", Shape{3, 2}),
                                                 record("fc.missing", Shape{4}),
                                                 record("fc.other", Shape{4})};

    const Result<std::vector<IO::ImportedTensor>> imported = import_with(records, *source);

    SA_EXPECT_FAILURE(imported, StatusCode::Incompatible,
                      "records the file does not supply (2): 'fc.missing', 'fc.other'",
                      "tensors of the file that no record takes (1): 'net.fc.bias' as 'fc.bias'",
                      "tensors of another shape than their record (1)");
}

/** @brief Tests that two tensors that rename to one record are refused, naming both. */
TEST(Small_WeightImport, TwoTensorsRenamingToOneRecordAreRefused)
{
    const std::unique_ptr<IO::WeightSource> source = open(SaTest::safetensors_fixture());
    const std::vector<NN::StateEntry> records = {record("fc.weight", Shape{2, 3}),
                                                 record("fc.bias", Shape{2})};

    // Two whole-name rules send both tensors of the file to fc.weight.
    const Result<std::vector<IO::ImportedTensor>> imported = import_with(
        records, *source,
        R"({"rename":[{"from":"net.fc.weight","to":"fc.weight"},)"
        R"({"from":"net.fc.bias","to":"fc.weight"}]})");

    SA_EXPECT_FAILURE(imported, StatusCode::Incompatible,
                      "'net.fc.bias' as 'fc.weight', which 'net.fc.weight' already supplies",
                      "records the file does not supply (1): 'fc.bias'");
}

/** @brief Tests that a record of an element type the import cannot fill is refused by name. */
TEST(Small_WeightImport, RefusesARecordOfAnotherElementType)
{
    const std::unique_ptr<IO::WeightSource> source = open(SaTest::safetensors_fixture());
    const std::vector<NN::StateEntry> records = {
        record("fc.weight", Shape{2, 3}), record("fc.bias", Shape{2}, DType::Int32)};

    SA_EXPECT_FAILURE(import_with(records, *source), StatusCode::Unsupported, "'fc.bias'",
                      "int32");
}

/** @brief Tests that a float64 record gets the float32 values widened exactly. */
TEST(Small_WeightImport, AFloat64RecordIsFilledWithTheWidenedValues)
{
    const std::unique_ptr<IO::WeightSource> source = open(SaTest::safetensors_fixture());
    const std::vector<NN::StateEntry> records = {
        record("fc.weight", Shape{2, 3}), record("fc.bias", Shape{2}, DType::Float64)};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> imported,
                            import_with(records, *source));

    ASSERT_EQ(imported[1].payload.size(), 2 * sizeof(double));
    double widened[2] = {0.0, 0.0};
    std::memcpy(widened, imported[1].payload.data(), sizeof(widened));
    EXPECT_EQ(widened[0], 0.5);
    EXPECT_EQ(widened[1], -0.5);
}

/** @brief Tests that an import with no rules and no drops fills a model whose names match. */
TEST(Small_WeightImport, AnEmptyMapFillsAModelThatStatesTheFilesNames)
{
    const std::unique_ptr<IO::WeightSource> source = open(SaTest::safetensors_fixture());
    const std::vector<NN::StateEntry> records = {record("net.fc.weight", Shape{2, 3}),
                                                 record("net.fc.bias", Shape{2})};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> imported,
                            import_with(records, *source, "{}"));

    ASSERT_EQ(imported.size(), 2u);
    EXPECT_EQ(imported[0].record.name, "net.fc.weight");
    EXPECT_EQ(imported[0].payload, payload_of(SaTest::fixture_weight()));
}

/** @brief Tests that an OIHW [2, 3, 3, 3] weight imports as the bytes copy_kernel_last gives. */
TEST(Small_WeightImport, AnOihwConvolutionWeightImportsInTheKernelLastLayout)
{
    const std::vector<float> weight = counting(54);
    const std::unique_ptr<IO::WeightSource> source = open(conv_file("[2,3,3,3]", weight));
    const std::vector<NN::StateEntry> records = {record("conv.weight", Shape{2, 3, 3, 3})};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> imported,
                            import_with(records, *source, "{}"));

    std::vector<float> by_function(weight.size());
    IO::copy_kernel_last(weight.data(), IO::FilterExtents{2, 3, 3, 3}, by_function.data());
    ASSERT_EQ(imported.size(), 1u);
    EXPECT_EQ(imported[0].payload, payload_of(by_function));
    EXPECT_EQ(imported[0].payload, payload_of(kernel_last_by_formula(weight, 2, 3, 3, 3)));
}

/** @brief Tests that a weight whose [out, in, kh, kw] shape differs from its record's moves. */
TEST(Small_WeightImport, AnOihwWeightOfUnequalExtentsGivesTheRecordsShapeAndOrder)
{
    const std::vector<float> weight = counting(12);
    const std::unique_ptr<IO::WeightSource> source = open(conv_file("[2,3,1,2]", weight));
    const std::vector<NN::StateEntry> records = {record("conv.weight", Shape{2, 1, 2, 3})};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> imported,
                            import_with(records, *source, "{}"));

    ASSERT_EQ(imported.size(), 1u);
    EXPECT_EQ(imported[0].record.shape, (Shape{2, 1, 2, 3}));
    EXPECT_EQ(imported[0].payload, payload_of(kernel_last_by_formula(weight, 2, 3, 1, 2)));
}

/** @brief Tests that a rank-4 tensor already in the record's shape is copied as stored. */
TEST(Small_WeightImport, ARankFourTensorInTheRecordsShapeIsCopiedAsStored)
{
    const std::vector<float> weight = counting(6);
    const std::unique_ptr<IO::WeightSource> source = open(conv_file("[2,1,1,3]", weight));
    const std::vector<NN::StateEntry> records = {record("conv.weight", Shape{2, 1, 1, 3})};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> imported,
                            import_with(records, *source, "{}"));

    ASSERT_EQ(imported.size(), 1u);
    EXPECT_EQ(imported[0].payload, payload_of(weight));
}

/** @brief Tests that a rank-4 tensor matching neither layout is reported as misshapen. */
TEST(Small_WeightImport, ARankFourTensorMatchingNeitherLayoutIsMisshapen)
{
    const std::unique_ptr<IO::WeightSource> source =
        open(conv_file("[2,3,1,1]", counting(6)));
    const std::vector<NN::StateEntry> records = {record("conv.weight", Shape{3, 1, 1, 3})};

    const auto result = import_with(records, *source, "{}");

    SA_EXPECT_FAILURE(result, StatusCode::Incompatible, "tensors of another shape than their record",
                      "'conv.weight' is [3, 1, 1, 3] in the model");
}

/** @brief Tests that a [C, 1, kh, kw] file tensor fills a rank-3 [kh, kw, C] record depthwise. */
TEST(Small_WeightImport, ADepthwiseWeightImportsAsKernelHeightWidthChannels)
{
    const std::vector<float> weight = counting(24);
    const std::unique_ptr<IO::WeightSource> source = open(conv_file("[4,1,2,3]", weight));
    const std::vector<NN::StateEntry> records = {record("conv.weight", Shape{2, 3, 4})};

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> imported,
                            import_with(records, *source, "{}"));

    // Element (h, w, c) is source element (c * 2 + h) * 3 + w, by the index formula alone.
    std::vector<float> expected;
    for (int h = 0; h < 2; ++h)
        for (int w = 0; w < 3; ++w)
            for (int c = 0; c < 4; ++c)
                expected.push_back(weight[static_cast<std::size_t>((c * 2 + h) * 3 + w)]);

    std::vector<float> by_function(weight.size());
    IO::copy_depthwise_kernel_last(weight.data(), IO::FilterExtents{4, 1, 2, 3},
                                   by_function.data());
    ASSERT_EQ(imported.size(), 1u);
    EXPECT_EQ(imported[0].record.shape, (Shape{2, 3, 4}));
    EXPECT_EQ(imported[0].payload, payload_of(expected));
    EXPECT_EQ(imported[0].payload, payload_of(by_function));
}

/** @brief Tests that a [C, 2, kh, kw] tensor is not taken for a depthwise weight. */
TEST(Small_WeightImport, AGroupedWeightIsNotTakenForADepthwiseOne)
{
    const std::unique_ptr<IO::WeightSource> source = open(conv_file("[4,2,2,3]", counting(48)));
    const std::vector<NN::StateEntry> records = {record("conv.weight", Shape{2, 3, 4})};

    const auto result = import_with(records, *source, "{}");

    SA_EXPECT_FAILURE(result, StatusCode::Incompatible, "tensors of another shape than their record",
                      "'conv.weight' is [2, 3, 4] in the model");
}
