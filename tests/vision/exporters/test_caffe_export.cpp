/**************************************************************************/
/* test_caffe_export.cpp                                                  */
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
 * @file test_caffe_export.cpp
 * @brief Tests the Caffe exporter: the LeNet fixed point, one model per row and the refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/exporters/caffe_export.hpp>
#include <SushiAI/vision/importers/caffe_import.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Names a weight of a test model and its extents. */
    struct WeightSpec
    {
        std::string name;                /**< The record name, `<row>.<field>`. */
        std::vector<std::int64_t> shape; /**< The extents of the record. */
    };

    /** @brief Returns a float32 weight of @p spec whose values follow from @p seed. */
    Result<IO::ImportedTensor> make_weight(const WeightSpec& spec, std::size_t seed)
    {
        SA_TRY_ASSIGN(const Shape shape,
                      Shape::make(SushiRuntime::span<const std::int64_t>(spec.shape.data(),
                                                                         spec.shape.size())));
        std::vector<float> values(static_cast<std::size_t>(shape.num_elements()));
        for (std::size_t at = 0; at < values.size(); ++at)
            values[at] = static_cast<float>((at * 7919 + seed * 104729) % 2001) / 1000.0f;
        IO::ImportedTensor tensor;
        tensor.record.name = spec.name;
        tensor.record.dtype = DType::Float32;
        tensor.record.shape = shape;
        tensor.payload.resize(values.size() * sizeof(float));
        std::memcpy(tensor.payload.data(), values.data(), tensor.payload.size());
        return tensor;
    }

    /** @brief Returns the model the JSON text @p text states, with seeded @p weights. */
    Result<IO::ImportedModel> make_model(std::string_view text,
                                         const std::vector<WeightSpec>& weights)
    {
        SA_TRY_ASSIGN(const Config::Value document, Config::parse_json(text, "test model"));
        IO::ImportedModel model;
        SA_TRY_ASSIGN(model.model, Config::parse_model(document, "test model"));
        for (std::size_t at = 0; at < weights.size(); ++at)
        {
            SA_TRY_ASSIGN(IO::ImportedTensor tensor, make_weight(weights[at], at + 1));
            model.weights.push_back(std::move(tensor));
        }
        return model;
    }

    /** @brief Holds a model read back from an export and the files it was read from. */
    struct RoundTrip
    {
        std::vector<IO::ExportedFile> files; /**< The prototxt, then the caffemodel if any. */
        IO::ImportedModel model;             /**< The model the Caffe importer read back. */
    };

    /** @brief Exports @p model to Caffe and imports the files back. */
    Result<RoundTrip> round_trip(const IO::ImportedModel& model)
    {
        SA_TRY_ASSIGN(IO::ExportedModel exported,
                      Vision::Exporters::export_caffe_files(model, "model"));
        const std::size_t expected = model.weights.empty() ? 1 : 2;
        SA_FAIL_IF(exported.files.size() != expected ||
                       exported.files[0].name != "model.prototxt" ||
                       (expected == 2 && exported.files[1].name != "model.caffemodel"),
                   StatusCode::InvalidDescription,
                   "the export wrote other than model.prototxt and model.caffemodel");
        RoundTrip trip;
        trip.files = std::move(exported.files);
        const std::vector<std::byte>& prototxt = trip.files[0].bytes;
        std::string text(prototxt.size(), '\0');
        if (!prototxt.empty())
            std::memcpy(text.data(), prototxt.data(), prototxt.size());
        SA_TRY_ASSIGN(Interchange::CaffeNet net,
                      Interchange::read_caffe_prototxt(text, "model.prototxt"));
        if (expected == 2)
        {
            const std::vector<std::byte>& bytes = trip.files[1].bytes;
            SA_TRY_ASSIGN(const Interchange::CaffeNet weights,
                          Interchange::read_caffe_model(
                              SushiRuntime::span<const std::byte>(bytes.data(), bytes.size()),
                              "model.caffemodel"));
            SA_TRY(Interchange::attach_caffe_weights(net, weights));
        }
        SA_TRY_ASSIGN(trip.model, Vision::Importers::import_caffe_net(net, "model.prototxt"));
        return trip;
    }

    /** @brief Expects the same model text and every tensor equal bit for bit. */
    void expect_same_model(const IO::ImportedModel& expected, const IO::ImportedModel& actual)
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::string expected_text,
                                Config::write_model(expected.model));
        SA_ASSERT_OK_AND_ASSIGN(const std::string actual_text, Config::write_model(actual.model));
        EXPECT_EQ(expected_text, actual_text);
        ASSERT_EQ(expected.weights.size(), actual.weights.size());
        for (const IO::ImportedTensor& tensor : expected.weights)
        {
            const IO::ImportedTensor* found = nullptr;
            for (const IO::ImportedTensor& candidate : actual.weights)
            {
                if (candidate.record.name == tensor.record.name)
                    found = &candidate;
            }
            ASSERT_NE(found, nullptr) << tensor.record.name;
            EXPECT_EQ(found->record.dtype, tensor.record.dtype) << tensor.record.name;
            EXPECT_EQ(found->record.shape.to_string(), tensor.record.shape.to_string())
                << tensor.record.name;
            EXPECT_EQ(found->payload, tensor.payload) << tensor.record.name;
        }
    }

    /** @brief Expects @p text with @p weights to come back from an export unchanged. */
    void expect_round_trip(std::string_view text, const std::vector<WeightSpec>& weights)
    {
        SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel model, make_model(text, weights));
        SA_ASSERT_OK_AND_ASSIGN(const RoundTrip trip, round_trip(model));
        expect_same_model(model, trip.model);
    }

    /** @brief Returns the folder of the Caffe importer's fixture. */
    std::filesystem::path fixture_folder()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "tests" / "vision" / "importers" /
               "fixtures" / "caffe";
    }

    /** @brief Holds row 0 of a model whose tested rows hold no weight: a 1x1 conv2d. */
    constexpr std::string_view K_WEIGHTED_ROW =
        R"({"type": "conv2d", "filters": 3, "kernel": 1, "stride": 1, "padding": 0,
            "use_bias": false})";

    /**
     * @brief Expects a model over [H, W, 3] of K_WEIGHTED_ROW and @p rows, publishing row
     *        @p published, to come back from an export unchanged.
     */
    void expect_rows_round_trip(std::string_view shape, std::string_view rows, int published)
    {
        const std::string text = R"({"format_version": 2, "input": {"shape": )" +
                                 std::string(shape) + R"(}, "layers": [)" +
                                 std::string(K_WEIGHTED_ROW) + ", " + std::string(rows) +
                                 R"(], "outputs": [{"node": )" + std::to_string(published) +
                                 R"(, "port": "output", "name": "y"}]})";
        expect_round_trip(text, {{"0.weight", {3, 1, 1, 3}}});
    }
} // namespace

/** @brief Tests that LeNet imports, exports and imports back to the same model and bytes. */
TEST(Small_CaffeExport, TheLeNetFixtureComesBackAsItWasImported)
{
    const std::filesystem::path files[] = {fixture_folder() / "lenet.prototxt",
                                           fixture_folder() / "lenet.caffemodel"};
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_caffe_files(
                                SushiRuntime::span<const std::filesystem::path>(files, 2)));
    SA_ASSERT_OK_AND_ASSIGN(const RoundTrip first, round_trip(imported));
    expect_same_model(imported, first.model);

    SA_ASSERT_OK_AND_ASSIGN(const RoundTrip second, round_trip(first.model));
    ASSERT_EQ(first.files.size(), second.files.size());
    for (std::size_t at = 0; at < first.files.size(); ++at)
        EXPECT_EQ(first.files[at].bytes, second.files[at].bytes) << first.files[at].name;
}

/** @brief Tests that a conv2d with a stride, a padding and a bias comes back. */
TEST(Small_CaffeExport, AConvolutionComesBack)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [5, 5, 3]},
        "layers": [{"type": "conv2d", "filters": 4, "kernel": 3, "stride": 2, "padding": 1,
                    "use_bias": true}],
        "outputs": [{"node": 0, "port": "output", "name": "y"}]})",
                      {{"0.weight", {4, 3, 3, 3}}, {"0.bias", {4}}});
}

/** @brief Tests that a conv2d of two groups, a dilation and no bias comes back. */
TEST(Small_CaffeExport, AGroupedDilatedConvolutionComesBack)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [6, 6, 4]},
        "layers": [{"type": "conv2d", "filters": 6, "kernel": [1, 2], "stride": 1,
                    "padding": 0, "dilation": 2, "groups": 2, "use_bias": false}],
        "outputs": [{"node": 0, "port": "output", "name": "y"}]})",
                      {{"0.weight", {6, 1, 2, 2}}});
}

/** @brief Tests that a depthwise conv2d, its kernel [kh, kw, C], comes back. */
TEST(Small_CaffeExport, ADepthwiseConvolutionComesBack)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [4, 4, 3]},
        "layers": [{"type": "conv2d", "filters": 3, "kernel": 3, "stride": 1, "padding": 1,
                    "groups": 3, "use_bias": true}],
        "outputs": [{"node": 0, "port": "output", "name": "y"}]})",
                      {{"0.weight", {3, 3, 3}}, {"0.bias", {3}}});
}

/** @brief Tests that a maxpool2d whose end padding is Caffe's rounding up comes back. */
TEST(Small_CaffeExport, AMaxPoolComesBack)
{
    expect_rows_round_trip("[5, 5, 3]",
                           R"({"type": "maxpool2d", "kernel": 2, "stride": 2, "padding": 0,
                               "trailing_padding": [1, 1]})",
                           1);
}

/** @brief Tests that an avgpool2d whose windows fit the input comes back. */
TEST(Small_CaffeExport, AnAveragePoolComesBack)
{
    expect_rows_round_trip("[4, 4, 3]",
                           R"({"type": "avgpool2d", "kernel": 2, "stride": 2, "padding": 0})", 1);
}

/** @brief Tests that a global_avgpool comes back. */
TEST(Small_CaffeExport, AGlobalAveragePoolComesBack)
{
    expect_rows_round_trip("[4, 4, 3]", R"({"type": "global_avgpool"})", 1);
}

/** @brief Tests that relu, leaky_relu and dropout come back. */
TEST(Small_CaffeExport, EachPointwiseRowComesBack)
{
    expect_rows_round_trip("[4, 4, 3]",
                           R"({"type": "relu"}, {"type": "leaky_relu", "slope": 0.1},
                              {"type": "dropout", "rate": 0.25})",
                           3);
}

/** @brief Tests that a softmax over the channels and one over the rows come back. */
TEST(Small_CaffeExport, ASoftmaxComesBack)
{
    expect_rows_round_trip("[4, 4, 3]", R"({"type": "softmax"}, {"type": "softmax", "axis": 0})",
                           2);
}

/** @brief Tests that a concat of two images along the channels comes back. */
TEST(Small_CaffeExport, AConcatComesBack)
{
    expect_rows_round_trip("[4, 4, 3]",
                           R"({"type": "relu"}, {"type": "concat", "from": [0, 1]})", 2);
}

/** @brief Tests that add, mul and maximum of two images of one shape come back. */
TEST(Small_CaffeExport, EachEltwiseRowComesBack)
{
    for (const std::string type : {"add", "mul", "maximum"})
    {
        SCOPED_TRACE(type);
        expect_rows_round_trip("[4, 4, 3]",
                               R"({"type": "relu"}, {"type": "dropout", "rate": 0.5, "from": 0},
                                  {"type": ")" + type + R"(", "from": [1, 2]})",
                               3);
    }
}

/** @brief Tests that a batchnorm2d and its four vectors come back. */
TEST(Small_CaffeExport, ABatchNormalisationComesBack)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [4, 4, 3]},
        "layers": [{"type": "batchnorm2d", "bn_epsilon": 0.001}],
        "outputs": [{"node": 0, "port": "output", "name": "y"}]})",
                      {{"0.weight", {3}}, {"0.bias", {3}}, {"0.running_mean", {3}},
                       {"0.running_var", {3}}});
}

/** @brief Tests that a flatten and the linear that reads it come back, columns in place. */
TEST(Small_CaffeExport, AFlattenAndItsLinearComeBack)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [2, 3, 4]},
        "layers": [{"type": "flatten"}, {"type": "linear", "units": 5, "use_bias": true}],
        "outputs": [{"node": 1, "port": "output", "name": "y"}]})",
                      {{"1.weight", {5, 24}}, {"1.bias", {5}}});
}

/** @brief Tests that a linear over a vector input comes back. */
TEST(Small_CaffeExport, ALinearOverAVectorComesBack)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [6]},
        "layers": [{"type": "linear", "units": 2, "use_bias": false}],
        "outputs": [{"node": 0, "port": "output", "name": "y"}]})",
                      {{"0.weight", {2, 6}}});
}

/** @brief Tests that a model without weights writes the prototxt alone and comes back. */
TEST(Small_CaffeExport, AModelWithoutWeightsWritesThePrototxtAlone)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [4, 4, 3]},
        "layers": [{"type": "relu"}],
        "outputs": [{"node": 0, "port": "output", "name": "y"}]})",
                      {});
}

/** @brief Tests that every row type the Caffe importer has no row for is named in one message. */
TEST(Small_CaffeExport, EveryRefusedRowTypeIsNamed)
{
    const char* const refused[] = {
        "sigmoid",   "silu",       "mish",      "gelu",       "tanh",           "upsample",
        "slice_channels", "sub",   "div",       "transpose",  "unflatten",      "embedding",
        "layernorm", "conv",       "bottleneck", "c3",        "c3k",            "c3k2",
        "c2f",       "sppf",       "c2psa",     "focus",      "detect",         "decoupled_head",
        "segment_head"};
    IO::ImportedModel model;
    model.model.input_shape = Shape{4, 4, 3};
    for (const char* type : refused)
    {
        Config::LayerSpec row;
        row.type = type;
        row.from.push_back(Config::SourceSpec{});
        model.model.layers.push_back(std::move(row));
    }
    model.model.outputs.push_back(Config::OutputSpec{0, "output", "y"});

    const Result<Interchange::CaffeNet> exported = Vision::Exporters::export_caffe_net(model);
    ASSERT_FALSE(exported.has_value());
    EXPECT_EQ(exported.status().code(), StatusCode::Unsupported);
    const std::string message(exported.status().message());
    for (std::size_t at = 0; at < std::size(refused); ++at)
    {
        const std::string named = "row " + std::to_string(at) + " '" + refused[at] + "'";
        EXPECT_NE(message.find(named), std::string::npos) << named;
    }
}

/** @brief Tests that the rows whose keys the Caffe importer cannot read back are all named. */
TEST(Small_CaffeExport, EveryRefusedKeyIsNamedWithItsRow)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::ImportedModel model,
        make_model(R"({"format_version": 2, "input": {"shape": [5, 5, 3]},
            "layers": [{"type": "conv2d", "filters": 2, "kernel": 3, "trailing_padding": 1,
                        "use_bias": false, "from": "input"},
                       {"type": "maxpool2d", "kernel": 2, "stride": 2, "padding": 0,
                        "from": "input"},
                       {"type": "avgpool2d", "kernel": 2, "stride": 2, "padding": 0,
                        "from": "input"},
                       {"type": "leaky_relu", "slope": 0.0, "from": "input"},
                       {"type": "flatten", "from": "input"},
                       {"type": "relu", "from": 4},
                       {"type": "batchnorm2d", "bn_momentum": 0.5, "from": "input"},
                       {"type": "concat", "from": [0, 1, 2, 3, 5, 6]}],
            "outputs": [{"node": 7, "port": "output", "name": "y"}]})",
                   {}));
    SA_EXPECT_FAILURE(Vision::Exporters::export_caffe_net(model), StatusCode::Unsupported,
                      "row 0 'conv2d'", "trailing padding", "row 1 'maxpool2d'",
                      "row 2 'avgpool2d'", "rounding up", "row 3 'leaky_relu'",
                      "row 4 'flatten'", "row 6 'batchnorm2d'", "bn_momentum");
}

/** @brief Tests that an output the Caffe importer would not publish back is refused. */
TEST(Small_CaffeExport, AnOutputOfARowAnotherReadsIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel model,
                            make_model(R"({"format_version": 2, "input": {"shape": [4, 4, 3]},
            "layers": [{"type": "relu"}, {"type": "dropout", "rate": 0.5}],
            "outputs": [{"node": 0, "port": "output", "name": "a"},
                        {"node": 1, "port": "output", "name": "b"}]})",
                                       {}));
    SA_EXPECT_FAILURE(Vision::Exporters::export_caffe_net(model), StatusCode::Unsupported,
                      "output 'a'", "a later row reads it");
}
