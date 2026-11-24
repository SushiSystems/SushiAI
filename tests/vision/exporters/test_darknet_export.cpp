/**************************************************************************/
/* test_darknet_export.cpp                                                */
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
 * @file test_darknet_export.cpp
 * @brief Tests the Darknet exporter: a written .cfg's fixed point, one model per row, the
 *        anchors and the refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/exporters/darknet_export.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_import.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

using namespace SushiAI;

namespace
{
    using Vision::Importers::DarknetConfig;
    using Vision::Importers::DarknetModel;
    using Vision::Importers::DarknetWeights;

    /** @brief Names a weight of a test model and its extents. */
    struct WeightSpec
    {
        std::string name;                /**< The record name, `<row>.<field>`. */
        std::vector<std::int64_t> shape; /**< The extents of the record. */
    };

    /** @brief Returns @p count floats that follow from @p seed. */
    std::vector<float> make_values(std::size_t count, std::size_t seed)
    {
        std::vector<float> values(count);
        for (std::size_t at = 0; at < count; ++at)
            values[at] = static_cast<float>((at * 7919 + seed * 104729) % 2001) / 1000.0f - 1.0f;
        return values;
    }

    /** @brief Returns a float32 weight of @p spec whose values follow from @p seed. */
    Result<IO::ImportedTensor> make_weight(const WeightSpec& spec, std::size_t seed)
    {
        SA_TRY_ASSIGN(const Shape shape,
                      Shape::make(SushiRuntime::span<const std::int64_t>(spec.shape.data(),
                                                                         spec.shape.size())));
        const std::vector<float> values =
            make_values(static_cast<std::size_t>(shape.num_elements()), seed);
        IO::ImportedTensor tensor;
        tensor.record.name = spec.name;
        tensor.record.dtype = DType::Float32;
        tensor.record.shape = shape;
        tensor.payload.resize(values.size() * sizeof(float));
        std::memcpy(tensor.payload.data(), values.data(), tensor.payload.size());
        return tensor;
    }

    /** @brief Returns a task of one head over one anchor, @p classes classes. */
    Result<Config::Value> make_task(std::int64_t classes)
    {
        Vision::Detect::Anchor::AnchorTask task;
        task.num_classes = classes;
        task.layout.anchors = {Vision::Detect::Anchor::AnchorSize{0.25f, 0.5f}};
        task.layout.head_anchors = {{0}};
        Vision::Detect::Anchor::AnchorTaskHead head;
        head.output = "head.0";
        head.options.num_classes = classes;
        task.heads = {head};
        return Vision::Detect::Anchor::write_anchor_task(task);
    }

    /**
     * @brief Returns the model of the JSON rows @p layers over [H, W, C] @p shape, with the
     *        output `head.0` of row @p head, a task of one class and seeded @p weights.
     */
    Result<IO::ImportedModel> make_model(std::string_view shape, std::string_view layers,
                                         int head, const std::vector<WeightSpec>& weights)
    {
        const std::string text = R"({"format_version": 2, "input": {"shape": )" +
                                 std::string(shape) + R"(}, "layers": [)" + std::string(layers) +
                                 R"(], "outputs": [{"node": )" + std::to_string(head) +
                                 R"(, "port": "output", "name": "head.0"}]})";
        SA_TRY_ASSIGN(const Config::Value document, Config::parse_json(text, "test model"));
        IO::ImportedModel model;
        SA_TRY_ASSIGN(model.model, Config::parse_model(document, "test model"));
        model.model.input_fit = std::string(Config::K_FIT_RESIZE);
        SA_TRY_ASSIGN(model.model.task, make_task(1));
        for (std::size_t at = 0; at < weights.size(); ++at)
        {
            SA_TRY_ASSIGN(IO::ImportedTensor tensor, make_weight(weights[at], at + 1));
            model.weights.push_back(std::move(tensor));
        }
        return model;
    }

    /** @brief Holds the files of an export and the model the Darknet importer read back. */
    struct RoundTrip
    {
        std::vector<IO::ExportedFile> files; /**< The .cfg, then the .weights if any. */
        IO::ImportedModel model;             /**< The model read from the files. */
    };

    /** @brief Exports @p model to Darknet and imports the files back. */
    Result<RoundTrip> round_trip(const IO::ImportedModel& model)
    {
        SA_TRY_ASSIGN(IO::ExportedModel exported,
                      Vision::Exporters::export_darknet_files(model, "model"));
        const std::size_t expected = model.weights.empty() ? 1 : 2;
        SA_FAIL_IF(exported.files.size() != expected || exported.files[0].name != "model.cfg",
                   StatusCode::InvalidDescription, "the export wrote other than model.cfg");
        RoundTrip trip;
        trip.files = std::move(exported.files);
        const std::vector<std::byte>& cfg = trip.files[0].bytes;
        const std::string text(reinterpret_cast<const char*>(cfg.data()), cfg.size());
        SA_TRY_ASSIGN(const DarknetConfig config,
                      Vision::Importers::parse_darknet_config(text, "model.cfg"));
        SA_TRY_ASSIGN(const DarknetModel translated,
                      Vision::Importers::translate_darknet_model(
                          config, Vision::Importers::DarknetTranslation{}));
        if (expected == 1)
        {
            SA_TRY_ASSIGN(trip.model,
                          Vision::Importers::import_darknet_model(config, translated, nullptr));
            return trip;
        }
        SA_FAIL_IF(trip.files[1].name != "model.weights", StatusCode::InvalidDescription,
                   "the export wrote '{}' for the weights", trip.files[1].name);
        const std::vector<std::byte>& bytes = trip.files[1].bytes;
        SA_TRY_ASSIGN(const DarknetWeights weights,
                      Vision::Importers::read_darknet_weights(
                          SushiRuntime::span<const std::uint8_t>(
                              reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()),
                          "model.weights",
                          SushiRuntime::span<const Vision::Importers::DarknetConvolutionShape>(
                              translated.convolutions.data(), translated.convolutions.size())));
        SA_TRY_ASSIGN(trip.model,
                      Vision::Importers::import_darknet_model(config, translated, &weights));
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

    /** @brief Expects the rows @p layers, head row @p head, to come back from an export. */
    void expect_rows_round_trip(std::string_view shape, std::string_view layers, int head,
                                const std::vector<WeightSpec>& weights)
    {
        SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel model,
                                make_model(shape, layers, head, weights));
        SA_ASSERT_OK_AND_ASSIGN(const RoundTrip trip, round_trip(model));
        expect_same_model(model, trip.model);
    }

    /** @brief Returns the head row, a 1x1 conv2d of 6 filters, one anchor and one class. */
    std::string head_row(std::string_view from)
    {
        return R"({"type": "conv2d", "filters": 6, "kernel": 1, "stride": 1, "padding": 0,
                   "use_bias": true, "from": )" +
               std::string(from) + "}";
    }

    /** @brief Holds a .cfg of every section the exporter writes, two heads and training keys. */
    constexpr std::string_view K_FIXTURE_CFG = R"([net]
width=16
height=8
channels=3
letter_box=1
momentum=0.949
decay=0.0005
learning_rate=0.0013
burn_in=1000
policy=steps
steps=400000,450000
scales=.1,.1
hue=.1
saturation=1.5
exposure=1.5

[convolutional]
batch_normalize=1
filters=4
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
filters=8
size=1
stride=1
pad=1
activation=mish

[route]
layers=-1
groups=2
group_id=1

[shortcut]
from=1
activation=linear

[convolutional]
filters=14
size=1
stride=1
pad=1
activation=linear

[yolo]
mask=0,1
anchors=10,13,16,30,33,23
classes=2
num=3
scale_x_y=1.05
nms_kind=diounms
beta_nms=0.5
iou_loss=ciou
iou_normalizer=0.07
ignore_thresh=.7
iou_thresh=0.213
max_delta=5
jitter=.3

[route]
layers=2

[upsample]
stride=2

[route]
layers=-1,0

[convolutional]
filters=7
size=1
stride=1
pad=1
activation=logistic

[yolo]
mask=2
anchors=10,13,16,30,33,23
classes=2
num=3
scale_x_y=1.05
nms_kind=diounms
beta_nms=0.5
iou_loss=ciou
iou_normalizer=0.07
ignore_thresh=.7
iou_thresh=0.213
max_delta=5
jitter=.3
)";

    /** @brief Returns seeded floats of the lengths @p model's convolutions state. */
    DarknetWeights make_fixture_weights(const DarknetModel& model)
    {
        DarknetWeights weights;
        weights.header.minor = 2;
        std::size_t seed = 1;
        for (const Vision::Importers::DarknetConvolutionShape& shape : model.convolutions)
        {
            const std::size_t filters = static_cast<std::size_t>(shape.filters);
            Vision::Importers::DarknetConvolution convolution;
            convolution.biases = make_values(filters, seed++);
            if (shape.normalised)
            {
                convolution.scales = make_values(filters, seed++);
                convolution.rolling_mean = make_values(filters, seed++);
                convolution.rolling_variance = make_values(filters, seed++);
            }
            convolution.weights = make_values(
                filters * static_cast<std::size_t>(shape.channels * shape.size * shape.size),
                seed++);
            weights.convolutions.push_back(std::move(convolution));
        }
        return weights;
    }

    /** @brief Returns the JSON text of @p value, for a comparison. */
    std::string spell(const Config::Value& value)
    {
        const Result<std::string> text = Config::write_json(value);
        return text.has_value() ? text.value() : std::string("<unwritable>");
    }
} // namespace

/** @brief Tests that a .cfg and its weights import, export and import back unchanged. */
TEST(Small_DarknetExport, TheFixtureComesBackAsItWasImported)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetConfig config,
                            Vision::Importers::parse_darknet_config(K_FIXTURE_CFG, "fixture.cfg"));
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel translated,
                            Vision::Importers::translate_darknet_model(
                                config, Vision::Importers::DarknetTranslation{}));
    const DarknetWeights weights = make_fixture_weights(translated);
    SA_ASSERT_OK_AND_ASSIGN(const IO::ImportedModel imported,
                            Vision::Importers::import_darknet_model(config, translated, &weights));

    SA_ASSERT_OK_AND_ASSIGN(const RoundTrip first, round_trip(imported));
    expect_same_model(imported, first.model);
    EXPECT_EQ(spell(imported.training), spell(first.model.training));
    EXPECT_EQ(spell(imported.data), spell(first.model.data));

    SA_ASSERT_OK_AND_ASSIGN(const RoundTrip second, round_trip(first.model));
    ASSERT_EQ(first.files.size(), second.files.size());
    for (std::size_t at = 0; at < first.files.size(); ++at)
        EXPECT_EQ(first.files[at].bytes, second.files[at].bytes) << first.files[at].name;
}

/** @brief Tests that a conv2d of a stride and a padding of half its kernel comes back. */
TEST(Small_DarknetExport, AConvolutionComesBack)
{
    expect_rows_round_trip(
        "[8, 8, 3]",
        R"({"type": "conv2d", "filters": 4, "kernel": 3, "stride": 2, "padding": 1,
            "use_bias": true, "from": "input"}, )" + head_row("0"),
        1, {{"0.weight", {4, 3, 3, 3}}, {"0.bias", {4}}, {"1.weight", {6, 1, 1, 4}},
            {"1.bias", {6}}});
}

/** @brief Tests that a conv2d of a padding other than half its kernel comes back. */
TEST(Small_DarknetExport, AnUnpaddedConvolutionComesBack)
{
    expect_rows_round_trip(
        "[8, 8, 3]",
        R"({"type": "conv2d", "filters": 4, "kernel": 3, "stride": 1, "padding": 0,
            "use_bias": true, "from": "input"}, )" + head_row("0"),
        1, {{"0.weight", {4, 3, 3, 3}}, {"0.bias", {4}}, {"1.weight", {6, 1, 1, 4}},
            {"1.bias", {6}}});
}

/** @brief Tests that a conv2d without a bias and its batchnorm2d come back as one section. */
TEST(Small_DarknetExport, ANormalisedConvolutionComesBack)
{
    expect_rows_round_trip(
        "[8, 8, 3]",
        R"({"type": "conv2d", "filters": 4, "kernel": 3, "stride": 1, "padding": 1,
            "use_bias": false, "from": "input"},
           {"type": "batchnorm2d", "bn_epsilon": 1e-5, "from": 0}, )" + head_row("1"),
        2, {{"0.weight", {4, 3, 3, 3}}, {"1.weight", {4}}, {"1.bias", {4}},
            {"1.running_mean", {4}}, {"1.running_var", {4}}, {"2.weight", {6, 1, 1, 4}},
            {"2.bias", {6}}});
}

/** @brief Tests that each activation a [convolutional] states comes back after its conv2d. */
TEST(Small_DarknetExport, EachActivationComesBack)
{
    for (const std::string activation : {R"({"type": "leaky_relu", "slope": 0.1)",
                                         R"({"type": "mish")", R"({"type": "silu")",
                                         R"({"type": "sigmoid")"})
    {
        SCOPED_TRACE(activation);
        expect_rows_round_trip(
            "[8, 8, 3]",
            R"({"type": "conv2d", "filters": 4, "kernel": 1, "stride": 1, "padding": 0,
                "use_bias": true, "from": "input"}, )" +
                activation + R"(, "from": 0}, )" + head_row("1"),
            2, {{"0.weight", {4, 1, 1, 3}}, {"0.bias", {4}}, {"2.weight", {6, 1, 1, 4}},
                {"2.bias", {6}}});
    }
}

/** @brief Tests that a maxpool2d of no trailing padding and one of a trailing row come back. */
TEST(Small_DarknetExport, AMaxPoolComesBack)
{
    expect_rows_round_trip(
        "[8, 8, 3]",
        R"({"type": "maxpool2d", "kernel": 2, "stride": 2, "padding": 0,
            "trailing_padding": [0, 0], "from": "input"},
           {"type": "maxpool2d", "kernel": 2, "stride": 1, "padding": 0,
            "trailing_padding": [1, 1], "from": 0}, )" + head_row("1"),
        2, {{"2.weight", {6, 1, 1, 3}}, {"2.bias", {6}}});
}

/** @brief Tests that an upsample comes back. */
TEST(Small_DarknetExport, AnUpsampleComesBack)
{
    expect_rows_round_trip("[4, 4, 3]",
                           R"({"type": "upsample", "scale": 2, "from": "input"}, )" +
                               head_row("0"),
                           1, {{"1.weight", {6, 1, 1, 3}}, {"1.bias", {6}}});
}

/** @brief Tests that a concat of two layers, the later first, comes back as one [route]. */
TEST(Small_DarknetExport, AConcatComesBack)
{
    expect_rows_round_trip(
        "[4, 4, 3]",
        R"({"type": "conv2d", "filters": 3, "kernel": 1, "stride": 1, "padding": 0,
            "use_bias": true, "from": "input"},
           {"type": "conv2d", "filters": 3, "kernel": 1, "stride": 1, "padding": 0,
            "use_bias": true, "from": 0},
           {"type": "concat", "from": [1, 0]}, )" + head_row("2"),
        3, {{"0.weight", {3, 1, 1, 3}}, {"0.bias", {3}}, {"1.weight", {3, 1, 1, 3}},
            {"1.bias", {3}}, {"3.weight", {6, 1, 1, 6}}, {"3.bias", {6}}});
}

/** @brief Tests that a slice of one group of the channels comes back as a [route] group. */
TEST(Small_DarknetExport, AChannelSliceComesBack)
{
    expect_rows_round_trip(
        "[4, 4, 3]",
        R"({"type": "conv2d", "filters": 4, "kernel": 1, "stride": 1, "padding": 0,
            "use_bias": true, "from": "input"},
           {"type": "slice_channels", "offset": 2, "count": 2, "from": 0}, )" + head_row("1"),
        2, {{"0.weight", {4, 1, 1, 3}}, {"0.bias", {4}}, {"2.weight", {6, 1, 1, 2}},
            {"2.bias", {6}}});
}

/** @brief Tests that an add of the last layer and an earlier one comes back as a [shortcut]. */
TEST(Small_DarknetExport, AnAddComesBack)
{
    expect_rows_round_trip(
        "[4, 4, 3]",
        R"({"type": "conv2d", "filters": 3, "kernel": 1, "stride": 1, "padding": 0,
            "use_bias": true, "from": "input"},
           {"type": "conv2d", "filters": 3, "kernel": 1, "stride": 1, "padding": 0,
            "use_bias": true, "from": 0},
           {"type": "add", "from": [1, 0]}, )" + head_row("2"),
        3, {{"0.weight", {3, 1, 1, 3}}, {"0.bias", {3}}, {"1.weight", {3, 1, 1, 3}},
            {"1.bias", {3}}, {"3.weight", {6, 1, 1, 3}}, {"3.bias", {6}}});
}

/** @brief Tests that written anchors import to the same float32 fractions, bit for bit. */
TEST(Small_DarknetExport, TheAnchorsComeBackBitForBit)
{
    SA_ASSERT_OK_AND_ASSIGN(IO::ImportedModel model,
                            make_model("[416, 608, 3]", head_row(R"("input")"), 0,
                                       {{"0.weight", {6, 1, 1, 3}}, {"0.bias", {6}}}));
    Vision::Detect::Anchor::AnchorTask task;
    task.num_classes = 1;
    task.layout.anchors = {{0.1f, 1.0f / 3.0f}, {0.7f, 0.123456789f}, {1.0f / 608.0f, 0.9999f}};
    task.layout.head_anchors = {{0}};
    Vision::Detect::Anchor::AnchorTaskHead head;
    head.output = "head.0";
    head.options.num_classes = 1;
    task.heads = {head};
    SA_ASSERT_OK_AND_ASSIGN(model.model.task, Vision::Detect::Anchor::write_anchor_task(task));

    SA_ASSERT_OK_AND_ASSIGN(const RoundTrip trip, round_trip(model));
    SA_ASSERT_OK_AND_ASSIGN(const Vision::Detect::Anchor::AnchorTask read,
                            Vision::Detect::Anchor::read_anchor_task(trip.model.model.task));
    ASSERT_EQ(read.layout.anchors.size(), task.layout.anchors.size());
    for (std::size_t at = 0; at < task.layout.anchors.size(); ++at)
    {
        EXPECT_EQ(std::memcmp(&read.layout.anchors[at], &task.layout.anchors[at],
                              sizeof(task.layout.anchors[at])),
                  0)
            << "anchor " << at;
    }
}

/** @brief Tests that a model without weights is written as a .cfg alone. */
TEST(Small_DarknetExport, AModelWithoutWeightsWritesTheCfgAlone)
{
    expect_rows_round_trip("[4, 4, 3]", head_row(R"("input")"), 0, {});
}

/** @brief Tests that every row type no Darknet section gives back is named in one message. */
TEST(Small_DarknetExport, EveryRefusedRowTypeIsNamed)
{
    const char* const refused[] = {
        "relu",       "leaky_relu", "mish",      "silu",       "sigmoid",   "batchnorm2d",
        "gelu",       "tanh",       "avgpool2d", "global_avgpool", "maximum", "dropout",
        "sub",        "mul",        "div",       "softmax",    "transpose", "flatten",
        "unflatten",  "linear",     "embedding", "layernorm",  "conv",      "bottleneck",
        "c3",         "c3k",        "c3k2",      "c2f",        "sppf",      "c2psa",
        "focus",      "detect",     "decoupled_head", "segment_head"};
    IO::ImportedModel model;
    model.model.input_shape = Shape{4, 4, 3};
    for (const char* type : refused)
    {
        Config::LayerSpec row;
        row.type = type;
        row.from.push_back(Config::SourceSpec{});
        model.model.layers.push_back(std::move(row));
    }
    model.model.outputs.push_back(Config::OutputSpec{0, "output", "head.0"});
    SA_ASSERT_OK_AND_ASSIGN(model.model.task, make_task(1));

    const Result<IO::ExportedModel> exported =
        Vision::Exporters::export_darknet_files(model, "model");
    ASSERT_FALSE(exported.has_value());
    EXPECT_EQ(exported.status().code(), StatusCode::Unsupported);
    const std::string message(exported.status().message());
    for (std::size_t at = 0; at < std::size(refused); ++at)
    {
        const std::string named = "row " + std::to_string(at) + " '" + refused[at] + "'";
        EXPECT_NE(message.find(named), std::string::npos) << named;
    }
}

/** @brief Tests that the rows whose keys or operands Darknet cannot state are all named. */
TEST(Small_DarknetExport, EveryRefusedKeyIsNamedWithItsRow)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const IO::ImportedModel model,
        make_model("[8, 8, 3]",
                   R"({"type": "conv2d", "filters": 2, "kernel": 3, "dilation": 2,
                       "from": "input"},
                      {"type": "conv2d", "filters": 3, "kernel": 1, "groups": 3,
                       "from": "input"},
                      {"type": "upsample", "scale": 2, "mode": "bilinear", "from": "input"},
                      {"type": "conv2d", "filters": 3, "kernel": 1, "stride": 1, "padding": 0,
                       "use_bias": true, "from": "input"},
                      {"type": "relu", "from": 3},
                      {"type": "leaky_relu", "slope": 0.2, "from": 3},
                      {"type": "slice_channels", "offset": 1, "count": 2, "from": 3},
                      {"type": "maxpool2d", "kernel": 2, "stride": 2, "padding": 0,
                       "trailing_padding": [1, 1], "from": 3},
                      {"type": "conv2d", "filters": 3, "kernel": 1, "use_bias": false,
                       "from": 3},
                      {"type": "conv2d", "filters": 3, "kernel": 1, "use_bias": false,
                       "from": 3},
                      {"type": "batchnorm2d", "bn_epsilon": 0.001, "from": 9}, )" +
                       head_row("3"),
                   11,
                   {{"3.weight", {3, 1, 1, 3}}, {"3.bias", {3}}, {"11.weight", {6, 1, 1, 3}},
                    {"11.bias", {6}}}));
    SA_EXPECT_FAILURE(Vision::Exporters::export_darknet_files(model, "model"),
                      StatusCode::Unsupported, "row 0 'conv2d'", "dilation", "row 1 'conv2d'",
                      "groups", "row 2 'upsample'", "bilinear", "row 4 'relu'",
                      "row 5 'leaky_relu'", "row 6 'slice_channels'", "row 7 'maxpool2d'",
                      "row 8 'conv2d'", "row 9 'conv2d'", "bn_epsilon", "row 10 'batchnorm2d'");
}

/** @brief Tests that a model without a task is refused before any row. */
TEST(Small_DarknetExport, AModelWithoutATaskIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(IO::ImportedModel model,
                            make_model("[4, 4, 3]", head_row(R"("input")"), 0, {}));
    model.model.task = Config::Value();
    SA_EXPECT_FAILURE(Vision::Exporters::export_darknet_files(model, "model"),
                      StatusCode::Unsupported, "no task");
}

/** @brief Tests that the fit `letterbox`, which letter_box cannot state, is refused. */
TEST(Small_DarknetExport, ALetterboxFitIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(IO::ImportedModel model,
                            make_model("[4, 4, 3]", head_row(R"("input")"), 0, {}));
    model.model.input_fit = std::string(Config::K_FIT_LETTERBOX);
    SA_EXPECT_FAILURE(Vision::Exporters::export_darknet_files(model, "model"),
                      StatusCode::Unsupported, "letterbox");
}
