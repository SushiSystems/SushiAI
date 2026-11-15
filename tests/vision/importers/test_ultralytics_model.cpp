/**************************************************************************/
/* test_ultralytics_model.cpp                                             */
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
 * @file test_ultralytics_model.cpp
 * @brief Tests the translation of Ultralytics model descriptions into SushiAI model files.
 * @author Mustafa Garip
 */

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/importers/ultralytics_checkpoint.hpp>
#include <SushiAI/vision/importers/ultralytics_model.hpp>

using namespace SushiAI;

namespace
{
    /**
     * @brief Holds the "yaml" of yolo26n.pt as JSON: ultralytics/cfg/models/26/yolo26.yaml at the
     *        pinned commit, with the scale "n" the file name selects.
     */
    constexpr std::string_view K_YOLO26N = R"({
        "nc": 80, "end2end": true, "reg_max": 1,
        "scales": {"n": [0.5, 0.25, 1024], "s": [0.5, 0.5, 1024], "m": [0.5, 1.0, 512],
                   "l": [1.0, 1.0, 512], "x": [1.0, 1.5, 512]},
        "backbone": [[-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]],
                     [-1, 2, "C3k2", [256, false, 0.25]], [-1, 1, "Conv", [256, 3, 2]],
                     [-1, 2, "C3k2", [512, false, 0.25]], [-1, 1, "Conv", [512, 3, 2]],
                     [-1, 2, "C3k2", [512, true]], [-1, 1, "Conv", [1024, 3, 2]],
                     [-1, 2, "C3k2", [1024, true]], [-1, 1, "SPPF", [1024, 5, 3, true]],
                     [-1, 2, "C2PSA", [1024]]],
        "head": [[-1, 1, "nn.Upsample", ["None", 2, "nearest"]], [[-1, 6], 1, "Concat", [1]],
                 [-1, 2, "C3k2", [512, true]], [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
                 [[-1, 4], 1, "Concat", [1]], [-1, 2, "C3k2", [256, true]],
                 [-1, 1, "Conv", [256, 3, 2]], [[-1, 13], 1, "Concat", [1]],
                 [-1, 2, "C3k2", [512, true]], [-1, 1, "Conv", [512, 3, 2]],
                 [[-1, 10], 1, "Concat", [1]], [-1, 1, "C3k2", [1024, true, 0.5, true]],
                 [[16, 19, 22], 1, "Detect", ["nc"]]],
        "scale": "n", "channels": 3
    })";

    /** @brief Holds the "yaml" of yolov8n.pt: ultralytics/cfg/models/v8/yolov8.yaml at n. */
    constexpr std::string_view K_YOLOV8N = R"({
        "nc": 80, "depth_multiple": 0.33, "width_multiple": 0.25,
        "backbone": [[-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]],
                     [-1, 3, "C2f", [128, true]], [-1, 1, "Conv", [256, 3, 2]],
                     [-1, 6, "C2f", [256, true]], [-1, 1, "Conv", [512, 3, 2]],
                     [-1, 6, "C2f", [512, true]], [-1, 1, "Conv", [1024, 3, 2]],
                     [-1, 3, "C2f", [1024, true]], [-1, 1, "SPPF", [1024, 5]]],
        "head": [[-1, 1, "nn.Upsample", ["None", 2, "nearest"]], [[-1, 6], 1, "Concat", [1]],
                 [-1, 3, "C2f", [512]], [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
                 [[-1, 4], 1, "Concat", [1]], [-1, 3, "C2f", [256]],
                 [-1, 1, "Conv", [256, 3, 2]], [[-1, 12], 1, "Concat", [1]],
                 [-1, 3, "C2f", [512]], [-1, 1, "Conv", [512, 3, 2]],
                 [[-1, 9], 1, "Concat", [1]], [-1, 3, "C2f", [1024]],
                 [[15, 18, 21], 1, "Detect", ["nc"]]],
        "ch": 3
    })";

    /** @brief Holds the "yaml" of yolo11n.pt: ultralytics/cfg/models/11/yolo11.yaml at n. */
    constexpr std::string_view K_YOLO11N = R"({
        "nc": 80,
        "scales": {"n": [0.5, 0.25, 1024], "s": [0.5, 0.5, 1024], "m": [0.5, 1.0, 512],
                   "l": [1.0, 1.0, 512], "x": [1.0, 1.5, 512]},
        "backbone": [[-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]],
                     [-1, 2, "C3k2", [256, false, 0.25]], [-1, 1, "Conv", [256, 3, 2]],
                     [-1, 2, "C3k2", [512, false, 0.25]], [-1, 1, "Conv", [512, 3, 2]],
                     [-1, 2, "C3k2", [512, true]], [-1, 1, "Conv", [1024, 3, 2]],
                     [-1, 2, "C3k2", [1024, true]], [-1, 1, "SPPF", [1024, 5]],
                     [-1, 2, "C2PSA", [1024]]],
        "head": [[-1, 1, "nn.Upsample", ["None", 2, "nearest"]], [[-1, 6], 1, "Concat", [1]],
                 [-1, 2, "C3k2", [512, false]], [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
                 [[-1, 4], 1, "Concat", [1]], [-1, 2, "C3k2", [256, false]],
                 [-1, 1, "Conv", [256, 3, 2]], [[-1, 13], 1, "Concat", [1]],
                 [-1, 2, "C3k2", [512, false]], [-1, 1, "Conv", [512, 3, 2]],
                 [[-1, 10], 1, "Concat", [1]], [-1, 2, "C3k2", [1024, true]],
                 [[16, 19, 22], 1, "Detect", ["nc"]]],
        "scale": "n", "ch": 3
    })";

    /** @brief Holds a small YOLOv5 description: a padded stem, a C3 and an anchored Detect. */
    constexpr std::string_view K_SMALL_YOLOV5 = R"({
        "nc": 2, "depth_multiple": 0.33, "width_multiple": 0.25,
        "anchors": [[10, 13, 16, 30], [30, 61, 62, 45]],
        "backbone": [[-1, 1, "Conv", [64, 6, 2, 2]], [-1, 3, "C3", [128]]],
        "head": [[[0, 1], 1, "Detect", ["nc", "anchors"]]]
    })";

    /** @brief Holds no pickled module classes, so each module is built as Ultralytics builds it. */
    const SushiRuntime::span<const Vision::Importers::UltralyticsModuleClass> K_NO_MODULES;

    /** @brief Returns @p text parsed, failing the test when it is not JSON. */
    Config::Value parsed(std::string_view text)
    {
        Result<Config::Value> value = Config::parse_json(text, "description");
        EXPECT_TRUE(value.has_value());
        return value.has_value() ? value.value() : Config::Value();
    }

    /** @brief Names one released checkpoint's description and its hand-written model file. */
    struct HandFileRow
    {
        const char* name;             /**< The row's name. */
        std::string_view description; /**< The "yaml" of the checkpoint. */
        const char* folder;           /**< Its folder under applications/examples. */
        const char* sppf_activation;  /**< The class its checkpoint pickles at 9.cv1.act. */
    };

    /** @brief Lists the released descriptions whose model file was written by hand. */
    const HandFileRow K_HAND_FILES[] = {
        {"Yolo26n", K_YOLO26N, "yolo26", "Identity"},
        {"Yolov8n", K_YOLOV8N, "yolov8n", "SiLU"},
        {"Yolo11n", K_YOLO11N, "yolo11n", "SiLU"},
    };

    /** @brief Returns the one module class @p row's checkpoint pickles that the translation reads. */
    std::vector<Vision::Importers::UltralyticsModuleClass> modules_of(const HandFileRow& row)
    {
        return {Vision::Importers::UltralyticsModuleClass{"9.cv1.act", row.sppf_activation}};
    }

    /** @brief Runs one hand-written model file against the translation of its description. */
    class Medium_UltralyticsHandFile : public ::testing::TestWithParam<HandFileRow>
    {

    };
} // namespace

/** @brief Tests that a released description becomes the model file written by hand for it. */
TEST_P(Medium_UltralyticsHandFile, TheDescriptionIsTheHandWrittenModelFile)
{
    const HandFileRow& row = GetParam();
    const std::filesystem::path path = std::filesystem::path(SA_REPOSITORY_ROOT) /
                                       "applications" / "examples" / row.folder / "model.json";
    SA_ASSERT_OK_AND_ASSIGN(const Config::ModelSpec wanted, Config::load_model(path));

    SA_ASSERT_OK_AND_ASSIGN(
        const Config::ModelSpec found,
        Vision::Importers::translate_ultralytics_model(parsed(row.description), 640,
                                                       modules_of(row), row.name));

    SA_ASSERT_OK_AND_ASSIGN(const std::string found_text, Config::write_model(found));
    SA_ASSERT_OK_AND_ASSIGN(const std::string wanted_text, Config::write_model(wanted));
    EXPECT_EQ(found_text, wanted_text);
}

INSTANTIATE_TEST_SUITE_P(Released, Medium_UltralyticsHandFile, ::testing::ValuesIn(K_HAND_FILES),
                         [](const ::testing::TestParamInfo<HandFileRow>& info)
                         { return std::string(info.param.name); });

/** @brief Tests that a YOLOv5 Detect becomes one 1x1 convolution per level and the anchor task. */
TEST(Small_UltralyticsModel, AnAnchoredDetectBecomesOneConvolutionPerLevel)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const Vision::Importers::UltralyticsTranslation translation,
        Vision::Importers::describe_ultralytics_model(parsed(K_SMALL_YOLOV5), 64, K_NO_MODULES, "v5.pt"));

    // Rows 0 (Conv) and 1 (C3), then one convolution per level: 2 anchors x (2 classes + 5).
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& layers, translation.document.at("layers", "v5"));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Config::Value>& rows, layers.as_array("v5"));
    ASSERT_EQ(rows.size(), 4u);
    const Config::Value& head = rows[3];
    EXPECT_EQ(head.at("filters", "v5").value().as_int("filters").value(), 14);
    EXPECT_EQ(head.at("from", "v5").value().as_int("from").value(), 1);
}

/** @brief Tests that an anchor in pixels is stated as a fraction of the input side. */
TEST(Small_UltralyticsModel, AnAnchorIsAFractionOfTheInputSide)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const Vision::Importers::UltralyticsTranslation translation,
        Vision::Importers::describe_ultralytics_model(parsed(K_SMALL_YOLOV5), 64, K_NO_MODULES, "v5.pt"));

    // The first anchor is 10 x 13 pixels over a side of 64; the quotient is computed the same way.
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& task, translation.document.at("task", "v5"));
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& anchors, task.at("anchors", "v5"));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Config::Value>& pairs, anchors.as_array("v5"));
    ASSERT_EQ(pairs.size(), 4u);
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Config::Value>& first, pairs[0].as_array("v5"));
    ASSERT_EQ(first.size(), 2u);
    EXPECT_EQ(first[0].as_real("anchor").value(), 10.0 / 64.0);
    EXPECT_EQ(first[1].as_real("anchor").value(), 13.0 / 64.0);
}

/** @brief Tests that the tensors of an anchored level are renamed to the row of its convolution. */
TEST(Small_UltralyticsModel, TheTensorsOfALevelTakeTheRowOfItsConvolution)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const Vision::Importers::UltralyticsTranslation translation,
        Vision::Importers::describe_ultralytics_model(parsed(K_SMALL_YOLOV5), 64, K_NO_MODULES, "v5.pt"));

    ASSERT_EQ(translation.tensor_rules.renames.size(), 2u);
    EXPECT_EQ(translation.tensor_rules.renames[1].first, "2.m.1.");
    EXPECT_EQ(translation.tensor_rules.renames[1].second, "3.");
}

/** @brief Tests that a padding the file states is kept on the conv row. */
TEST(Small_UltralyticsModel, AStatedPaddingIsKeptOnTheConvRow)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const Vision::Importers::UltralyticsTranslation translation,
        Vision::Importers::describe_ultralytics_model(parsed(K_SMALL_YOLOV5), 64, K_NO_MODULES, "v5.pt"));

    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& layers, translation.document.at("layers", "v5"));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Config::Value>& rows, layers.as_array("v5"));
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows[0].at("padding", "v5").value().as_int("padding").value(), 2);
}

/** @brief Tests that the m scale turns the inner block of a C3k2 into C3k, as Ultralytics does. */
TEST(Small_UltralyticsModel, TheMScaleTurnsTheInnerBlockOfC3k2IntoC3k)
{
    const Config::Value description = parsed(R"({
        "nc": 2, "scales": {"m": [0.5, 1.0, 512]}, "scale": "m",
        "backbone": [[-1, 1, "Conv", [64, 3, 2]], [-1, 2, "C3k2", [128, false, 0.25]]],
        "head": [[[1], 1, "Detect", ["nc"]]]
    })");

    SA_ASSERT_OK_AND_ASSIGN(
        const Vision::Importers::UltralyticsTranslation translation,
        Vision::Importers::describe_ultralytics_model(description, 64, K_NO_MODULES, "small.pt"));

    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& layers,
                            translation.document.at("layers", "small.pt"));
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& c3k2, layers.as_array("layers").value()[1].at(
                                                           "inner", "layers"));
    EXPECT_EQ(c3k2.as_string("inner").value(), "c3k");
}

/** @brief Tests that the widths are scaled, capped and rounded up to a multiple of eight. */
TEST(Small_UltralyticsModel, AWidthIsCappedScaledAndRoundedUpToEight)
{
    // min(1000, 512) * 0.3 = 153.6, which rounds up to 160.
    const Config::Value description = parsed(R"({
        "nc": 2, "scales": {"q": [1.0, 0.3, 512]},
        "backbone": [[-1, 1, "Conv", [1000, 1, 1]]],
        "head": [[[0], 1, "Detect", ["nc"]]]
    })");

    SA_ASSERT_OK_AND_ASSIGN(
        const Vision::Importers::UltralyticsTranslation translation,
        Vision::Importers::describe_ultralytics_model(description, 64, K_NO_MODULES, "small.pt"));

    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& layers,
                            translation.document.at("layers", "small.pt"));
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& filters, layers.as_array("layers").value()[0].at(
                                                              "filters", "layers"));
    EXPECT_EQ(filters.as_int("filters").value(), 160);
}

/** @brief Tests that every module no translator reads is listed with the entries that use it. */
TEST(Small_UltralyticsModel, EveryUnreadModuleIsListedWithItsEntry)
{
    const Config::Value description = parsed(R"({
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Focus", [32]],
                     [-1, 1, "RepNCSPELAN4", [32]]],
        "head": [[[2], 1, "Detect", ["nc"]]]
    })");

    SA_EXPECT_FAILURE(
        Vision::Importers::describe_ultralytics_model(description, 64, K_NO_MODULES, "small.pt"),
        StatusCode::Unsupported, "Focus (entry 1)", "RepNCSPELAN4 (entry 2)");
}

/** @brief Tests that a description without a Detect entry is refused. */
TEST(Small_UltralyticsModel, ADescriptionWithoutADetectEntryIsRefused)
{
    const Config::Value description = parsed(R"({
        "nc": 2, "backbone": [[-1, 1, "Conv", [16, 3, 2]]], "head": []
    })");

    SA_EXPECT_FAILURE(
        Vision::Importers::describe_ultralytics_model(description, 64, K_NO_MODULES, "small.pt"),
        StatusCode::Unsupported, "no Detect entry");
}

/** @brief Tests that an SPPF whose checkpoint pickles a SiLU after cv1 activates its narrowing. */
TEST(Small_UltralyticsModel, APickledSiluAfterTheSppfNarrowingIsKept)
{
    const Config::Value description = parsed(R"({
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "SPPF", [16, 5]]],
        "head": [[[1], 1, "Detect", ["nc"]]]
    })");
    const Vision::Importers::UltralyticsModuleClass silu{"1.cv1.act", "SiLU"};

    SA_ASSERT_OK_AND_ASSIGN(
        const Vision::Importers::UltralyticsTranslation translation,
        Vision::Importers::describe_ultralytics_model(
            description, 64, SushiRuntime::span<const Vision::Importers::UltralyticsModuleClass>(
                                 &silu, 1),
            "small.pt"));

    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& layers,
                            translation.document.at("layers", "small.pt"));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Config::Value>& rows, layers.as_array("small"));
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_TRUE(rows[1].at("activate_narrowing", "small").value().as_bool("sppf").value());
}

/** @brief Tests that an SPPF without a pickled class keeps cv1 linear, as Ultralytics builds it. */
TEST(Small_UltralyticsModel, AnSppfWithoutAPickledClassKeepsItsNarrowingLinear)
{
    const Config::Value description = parsed(R"({
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "SPPF", [16, 5]]],
        "head": [[[1], 1, "Detect", ["nc"]]]
    })");

    SA_ASSERT_OK_AND_ASSIGN(
        const Vision::Importers::UltralyticsTranslation translation,
        Vision::Importers::describe_ultralytics_model(description, 64, K_NO_MODULES, "small.pt"));

    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& layers,
                            translation.document.at("layers", "small.pt"));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Config::Value>& rows, layers.as_array("small"));
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_FALSE(rows[1].has("activate_narrowing"));
}
