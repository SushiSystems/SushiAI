/**************************************************************************/
/* test_onnx_export.cpp                                                   */
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
 * @file test_onnx_export.cpp
 * @brief Tests the ONNX exporter: the fixed point, one model per row and the refusals, and
 *        that the Caffe fixture and its exported copy give the same output.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
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
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/exporters/onnx_export.hpp>
#include <SushiAI/vision/importers/onnx_import.hpp>
#include <memory>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/io/exported_model_writer.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/vision/darknet_import_cases/darknet_import.hpp>
#include <SushiAI/vision/exporters/caffe_export.hpp>
#include <SushiAI/vision/importers/caffe_import.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Names a weight of a test model and its extents. */
    struct WeightSpec
    {
        std::string name;                 /**< The record name, `<row>.<field>`. */
        std::vector<std::int64_t> shape;  /**< The extents of the record. */
    };

    /** @brief Returns a float32 weight of @p spec whose values follow from @p seed. */
    Result<IO::ImportedTensor> make_weight(const WeightSpec& spec, std::size_t seed)
    {
        SA_TRY_ASSIGN(const Shape shape,
                      Shape::make(SushiRuntime::span<const std::int64_t>(spec.shape.data(),
                                                                         spec.shape.size())));
        std::vector<float> values(static_cast<std::size_t>(shape.num_elements()));
        for (std::size_t at = 0; at < values.size(); ++at)
            values[at] = static_cast<float>((at * 7919 + seed * 104729) % 2001) / 1000.0f - 1.0f;
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

    /** @brief Holds a model read back from an export and the file it was read from. */
    struct RoundTrip
    {
        std::vector<std::byte> bytes; /**< The exported `.onnx` file. */
        IO::ImportedModel model;      /**< The model the ONNX importer read from it. */
    };

    /** @brief Exports @p model to ONNX and imports the file back. */
    Result<RoundTrip> round_trip(const IO::ImportedModel& model)
    {
        SA_TRY_ASSIGN(IO::ExportedModel exported,
                      Vision::Exporters::export_onnx_files(model, "model"));
        SA_FAIL_IF(exported.files.size() != 1 || exported.files[0].name != "model.onnx",
                   StatusCode::InvalidDescription, "the export wrote other than model.onnx");
        RoundTrip trip;
        trip.bytes = std::move(exported.files[0].bytes);
        SA_TRY_ASSIGN(const Interchange::OnnxGraph graph,
                      Interchange::read_onnx_graph(
                          SushiRuntime::span<const std::byte>(trip.bytes.data(),
                                                              trip.bytes.size()),
                          "model.onnx"));
        SA_TRY_ASSIGN(trip.model, Vision::Importers::import_onnx_model(graph, "model.onnx"));
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

    /** @brief Returns the folder of the ONNX importer's fixture. */
    std::filesystem::path fixture_folder()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "tests" / "vision" / "importers" /
               "fixtures" / "onnx";
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

/** @brief Tests that the fixture is refused, since it publishes a flatten ONNX orders (C, H, W). */
TEST(Small_OnnxExport, TheFixtureIsRefusedForItsPublishedFlatten)
{
    const std::filesystem::path files[] = {fixture_folder() / "silu_resize.onnx"};
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_onnx_files(
                                SushiRuntime::span<const std::filesystem::path>(files, 1)));
    SA_EXPECT_FAILURE(Vision::Exporters::export_onnx_graph(imported), StatusCode::Unsupported,
                      "row 6 'flatten'", "the model publishes it");
}

/** @brief Tests that a conv2d with a stride, a padding and a trailing padding comes back. */
TEST(Small_OnnxExport, AConvolutionComesBack)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [5, 5, 3]},
        "layers": [{"type": "conv2d", "filters": 4, "kernel": 3, "stride": 2, "padding": 1,
                    "trailing_padding": 1, "use_bias": true}],
        "outputs": [{"node": 0, "port": "output", "name": "y"}]})",
                      {{"0.weight", {4, 3, 3, 3}}, {"0.bias", {4}}});
}

/** @brief Tests that a conv2d of two groups and no bias comes back. */
TEST(Small_OnnxExport, AGroupedConvolutionComesBack)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [4, 4, 4]},
        "layers": [{"type": "conv2d", "filters": 6, "kernel": [1, 2], "stride": 1,
                    "padding": 0, "groups": 2, "use_bias": false}],
        "outputs": [{"node": 0, "port": "output", "name": "y"}]})",
                      {{"0.weight", {6, 1, 2, 2}}});
}

/** @brief Tests that a depthwise conv2d, its kernel [kh, kw, C], comes back. */
TEST(Small_OnnxExport, ADepthwiseConvolutionComesBack)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [4, 4, 3]},
        "layers": [{"type": "conv2d", "filters": 3, "kernel": 3, "stride": 1, "padding": 1,
                    "groups": 3, "use_bias": true}],
        "outputs": [{"node": 0, "port": "output", "name": "y"}]})",
                      {{"0.weight", {3, 3, 3}}, {"0.bias", {3}}});
}

/** @brief Tests that a batchnorm2d and its four vectors come back. */
TEST(Small_OnnxExport, ABatchNormalisationComesBack)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [4, 4, 3]},
        "layers": [{"type": "batchnorm2d", "bn_epsilon": 0.0009765625}],
        "outputs": [{"node": 0, "port": "output", "name": "y"}]})",
                      {{"0.weight", {3}}, {"0.bias", {3}}, {"0.running_mean", {3}},
                       {"0.running_var", {3}}});
}

/** @brief Tests that each row of one operand and no key comes back. */
TEST(Small_OnnxExport, EachPointwiseRowComesBack)
{
    for (const std::string type : {"relu", "sigmoid", "silu"})
    {
        SCOPED_TRACE(type);
        expect_rows_round_trip("[4, 4, 3]", R"({"type": ")" + type + R"("})", 1);
    }
}

/** @brief Tests that add, sub, mul and div of two values of one shape come back. */
TEST(Small_OnnxExport, EachBinaryRowComesBack)
{
    for (const std::string type : {"add", "sub", "mul", "div"})
    {
        SCOPED_TRACE(type);
        expect_rows_round_trip("[4, 4, 3]",
                               R"({"type": "relu"}, {"type": "sigmoid", "from": 0},
                                  {"type": ")" + type + R"(", "from": [1, 2]})",
                               3);
    }
}

/** @brief Tests that a softmax over the channels and one over the rows come back. */
TEST(Small_OnnxExport, ASoftmaxComesBack)
{
    expect_rows_round_trip("[4, 4, 3]", R"({"type": "softmax"}, {"type": "softmax", "axis": 0})",
                           2);
}

/** @brief Tests that a transpose of the three axes of an image comes back. */
TEST(Small_OnnxExport, ATransposeComesBack)
{
    expect_rows_round_trip("[4, 5, 3]", R"({"type": "transpose", "perm": [2, 0, 1]})", 1);
}

/** @brief Tests that a concat of two images along the channels comes back. */
TEST(Small_OnnxExport, AConcatComesBack)
{
    expect_rows_round_trip("[4, 4, 3]",
                           R"({"type": "relu"}, {"type": "concat", "from": [0, 1]})", 2);
}

/** @brief Tests that a maxpool2d, padded alike and padded unevenly, comes back. */
TEST(Small_OnnxExport, AMaxPoolComesBack)
{
    expect_rows_round_trip("[6, 6, 3]",
                           R"({"type": "maxpool2d", "kernel": 2, "stride": 2, "padding": 0},
                              {"type": "maxpool2d", "kernel": 2, "stride": 1,
                               "leading_padding": 0, "end_padding": 1})",
                           2);
}

/** @brief Tests that a nearest upsample comes back. */
TEST(Small_OnnxExport, AnUpsampleComesBack)
{
    expect_rows_round_trip("[3, 3, 3]", R"({"type": "upsample", "scale": 3})", 1);
}

/** @brief Tests that a slice of the channels comes back. */
TEST(Small_OnnxExport, AChannelSliceComesBack)
{
    expect_rows_round_trip("[4, 4, 3]", R"({"type": "slice_channels", "offset": 1, "count": 2})",
                           1);
}

/** @brief Tests that a flatten and the linear that reads it come back, columns in place. */
TEST(Small_OnnxExport, AFlattenAndItsLinearComeBack)
{
    expect_round_trip(R"({"format_version": 2, "input": {"shape": [2, 3, 4]},
        "layers": [{"type": "flatten"}, {"type": "linear", "units": 5, "use_bias": true}],
        "outputs": [{"node": 1, "port": "output", "name": "y"}]})",
                      {{"1.weight", {5, 24}}, {"1.bias", {5}}});
}

/** @brief Tests that every row type the ONNX importer has no row for is named in one message. */
TEST(Small_OnnxExport, EveryRefusedRowTypeIsNamed)
{
    const char* const refused[] = {
        "leaky_relu", "mish",  "gelu",   "tanh",    "avgpool2d",      "global_avgpool",
        "maximum",    "dropout", "unflatten", "embedding", "layernorm", "conv",
        "bottleneck", "c3",    "c3k",    "c3k2",    "c2f",            "sppf",
        "c2psa",      "focus", "detect", "decoupled_head", "segment_head"};
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
    SA_ASSERT_OK_AND_ASSIGN(IO::ImportedTensor weight, make_weight({"9.unused", {1}}, 1));
    model.weights.push_back(std::move(weight));

    const Result<Interchange::OnnxGraph> exported = Vision::Exporters::export_onnx_graph(model);
    ASSERT_FALSE(exported.has_value());
    EXPECT_EQ(exported.status().code(), StatusCode::Unsupported);
    const std::string message(exported.status().message());
    for (std::size_t at = 0; at < std::size(refused); ++at)
    {
        const std::string named = "row " + std::to_string(at) + " '" + refused[at] + "'";
        EXPECT_NE(message.find(named), std::string::npos) << named;
    }
}

/** @brief Tests that the rows whose keys the ONNX importer cannot read back are all named. */
TEST(Small_OnnxExport, EveryRefusedKeyIsNamedWithItsRow)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::ImportedModel model,
        make_model(R"({"format_version": 2, "input": {"shape": [4, 4, 3]},
            "layers": [{"type": "conv2d", "filters": 2, "kernel": 3, "dilation": 2,
                        "from": "input"},
                       {"type": "upsample", "scale": 2, "mode": "bilinear", "from": "input"},
                       {"type": "flatten", "from": "input"},
                       {"type": "relu", "from": 2},
                       {"type": "transpose", "perm": [0, 1, 2], "from": "input"},
                       {"type": "batchnorm2d", "bn_momentum": 0.5, "from": "input"}],
            "outputs": [{"node": 5, "port": "output", "name": "y"}]})",
                   {{"0.weight", {2, 3, 3, 3}}, {"0.bias", {2}}}));
    SA_EXPECT_FAILURE(Vision::Exporters::export_onnx_graph(model), StatusCode::Unsupported,
                      "row 0 'conv2d'", "dilation", "row 1 'upsample'", "bilinear",
                      "row 2 'flatten'", "row 3 'relu'", "row 4 'transpose'",
                      "row 5 'batchnorm2d'", "bn_momentum");
}

/** @brief Tests that a model without weights is refused before any row. */
TEST(Small_OnnxExport, AModelWithoutWeightsIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel model,
                            make_model(R"({"format_version": 2, "input": {"shape": [4, 4, 3]},
            "layers": [{"type": "relu"}],
            "outputs": [{"node": 0, "port": "output", "name": "y"}]})",
                                       {}));
    SA_EXPECT_FAILURE(Vision::Exporters::export_onnx_graph(model), StatusCode::Unsupported,
                      "no weights");
}

namespace
{
    /** @brief Reads files of one format into a model. */
    using ImportFiles = Result<IO::ImportedModel> (*)(
        SushiRuntime::span<const std::filesystem::path> files);

    /** @brief Writes a model as the files of one format. */
    using ExportFiles = Result<IO::ExportedModel> (*)(const IO::ImportedModel& model,
                                                      std::string_view stem);

    /** @brief Returns the folder of the importers' fixtures for @p format. */
    std::filesystem::path fixture_folder(std::string_view format)
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "tests" / "vision" / "importers" /
               "fixtures" / std::string(format);
    }

    /** @brief Runs fixtures and their round-tripped copies on the test engine. */
    class Medium_ExportForward : public SushiAI::Test::EngineTest
    {
        protected:
            /**
             * @brief Exports @p model through @p write, saves the files into @p directory and
             *        reads them back through @p read.
             */
            static Result<IO::ImportedModel> round_trip(const IO::ImportedModel& model,
                                                        ExportFiles write, ImportFiles read,
                                                        const std::filesystem::path& directory)
            {
                SA_TRY_ASSIGN(const IO::ExportedModel exported, write(model, "model"));
                SA_TRY_ASSIGN(const std::vector<std::filesystem::path> written,
                              IO::write_exported_model(exported, directory,
                                                       IO::ExistingFiles::Refuse));
                return read(SushiRuntime::span<const std::filesystem::path>(written.data(),
                                                                            written.size()));
            }

            /** @brief Runs @p imported over a seeded input and returns its first output. */
            Result<std::vector<float>> run_model(const IO::ImportedModel& imported)
            {
                SushiBLAS::Engine& engine = backend().engine();
                const Shape& sample = imported.model.input_shape;
                std::vector<std::int64_t> extents = {1};
                for (std::int32_t axis = 0; axis < sample.rank(); ++axis)
                    extents.push_back(sample[axis]);
                SA_TRY_ASSIGN(const Shape batch_shape,
                              Shape::make(SushiRuntime::span<const std::int64_t>(
                                  extents.data(), extents.size())));
                SA_FAIL_IF(imported.model.outputs.empty(), StatusCode::Incompatible,
                           "the model publishes no output");

                NN::NodeRegistry registry;
                SA_TRY(NN::register_layers(registry));
                SA_TRY_ASSIGN(NN::ModelGraph graph, NN::build_model(registry, imported.model));
                SA_TRY_ASSIGN(const Train::TracedStep traced,
                              Train::build_evaluation_graph(graph, batch_shape));
                SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, graph));
                SA_TRY(SushiAI::Test::fill_imported_state(
                    SushiRuntime::span<const IO::ImportedTensor>(imported.weights.data(),
                                                                 imported.weights.size()),
                    state));

                Graph::TensorPool pool(engine);
                SA_TRY_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                              Train::EvaluationStep::make(backend(), traced, state, pool));
                SA_TRY_ASSIGN(SushiBLAS::Tensor image, allocate_tensor(engine, batch_shape));
                std::vector<float> pixels(static_cast<std::size_t>(image.num_elements));
                for (std::int64_t at = 0; at < image.num_elements; ++at)
                    pixels[static_cast<std::size_t>(at)] =
                        static_cast<float>((at * 7919) % 2001) / 1000.0f - 1.0f;
                SA_TRY(::SushiAI::write_elements(image, pixels));
                SA_TRY(step->run(image));

                SA_TRY_ASSIGN(const SushiBLAS::Tensor* const output,
                              step->output(imported.model.outputs[0].name));
                return ::SushiAI::read_elements<float>(*output);
            }

            /** @brief Expects @p files and their exported copy to give the same bits. */
            void expect_same_output(const std::vector<std::filesystem::path>& files,
                                    ImportFiles read, ExportFiles write, const char* scratch)
            {
                const SushiAI::Test::ScratchDirectory directory(scratch);
                SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                                        read(SushiRuntime::span<const std::filesystem::path>(
                                            files.data(), files.size())));
                SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel copy,
                                        round_trip(imported, write, read, directory.path()));

                SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> expected, run_model(imported));
                SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> actual, run_model(copy));
                ASSERT_EQ(expected.size(), actual.size());
                EXPECT_EQ(std::memcmp(expected.data(), actual.data(),
                                      expected.size() * sizeof(float)),
                          0);
            }
    };
} // namespace

/** @brief Tests that the Caffe LeNet and its exported copy give the same output bits. */
TEST_F(Medium_ExportForward, TheCaffeFixtureAndItsCopyGiveTheSameOutput)
{
    expect_same_output({fixture_folder("caffe") / "lenet.prototxt",
                        fixture_folder("caffe") / "lenet.caffemodel"},
                       &Vision::Importers::import_caffe_files,
                       &Vision::Exporters::export_caffe_files, "sushiai_export_forward_caffe");
}
