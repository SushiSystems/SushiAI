/**************************************************************************/
/* test_darknet_decode.cpp                                                */
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
 * @file test_darknet_decode.cpp
 * @brief Tests that decode_head reproduces the boxes and scores Darknet dumped before its
 *        threshold, for the heads of six models.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/head_decode.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/darknet_reference/darknet_reference.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    using SaTest::DarknetFile;
    using SaTest::DarknetReference;
    using SaTest::DarknetTensor;
    using Vision::Detect::Anchor::HeadOptions;
    using Vision::Detect::Anchor::HeadShape;
    using Vision::Importers::DarknetConfig;
    using Vision::Importers::DarknetModel;
    using Vision::Importers::DarknetTranslation;

    /** @brief Holds the absolute bound on a decoded box and on a decoded score. */
    constexpr double K_DECODE_ABSOLUTE = 1e-6;

    /** @brief Holds the class count of the six cfgs this test reads (COCO). */
    constexpr std::int64_t K_CLASSES = 80;

    /** @brief Names one [yolo] head and the reference tensors of its decode. */
    struct DecodeHead
    {
        std::int64_t layer;      /**< The index of the [yolo] section among the layers. */
        std::int64_t grid;       /**< The rows and the columns of the head's grid. */
        std::string_view input;  /**< The raw head, `layer_<k-1>_yolo_input`, as CHW. */
        std::string_view boxes;  /**< The decoded boxes before the threshold, `[rows, 4]`. */
        std::string_view scores; /**< The decoded scores before the threshold, `[rows, C]`. */
    };

    /** @brief Names one model: its cfg, its run folder and its heads in file order. */
    struct DecodeModel
    {
        std::string_view cfg;          /**< The cfg's file name in the data directory. */
        std::string_view directory;    /**< The run folder under `references`. */
        std::vector<DecodeHead> heads; /**< The [yolo] heads of the cfg, in file order. */
    };

    /** @brief Returns why the first file this model reads is missing, or an empty string. */
    std::string first_missing_file(const DecodeModel& model)
    {
        const DarknetFile cfg = SaTest::find_darknet_file(model.cfg);
        if (!cfg.found)
            return cfg.reason;

        for (const DecodeHead& head : model.heads)
        {
            for (const std::string_view name : {head.input, head.boxes, head.scores})
            {
                const DarknetReference reference =
                    SaTest::find_darknet_reference(model.directory, name);
                if (!reference.found)
                    return reference.reason;
            }
        }

        return std::string();
    }

    /**
     * @brief Translates the cfg named @p cfg with the default translation.
     * @return The translated model, or the failure of the lookup, the parse or the translation.
     *
     * The decode reads the heads' options and the layout only, which the normalisation does not
     * change, so the weights are not read here.
     */
    Result<DarknetModel> translate_model(std::string_view cfg)
    {
        const DarknetFile file = SaTest::find_darknet_file(cfg);
        SA_FAIL_IF(!file.found, StatusCode::FileNotFound, "{}", file.reason);
        SA_TRY_ASSIGN(const DarknetConfig config,
                      Vision::Importers::load_darknet_config(file.path));
        return Vision::Importers::translate_darknet_model(config, DarknetTranslation{});
    }

    /** @brief Loads the reference tensor @p name of the run in @p directory. */
    Result<DarknetTensor> load_reference(std::string_view directory, std::string_view name)
    {
        return SaTest::load_darknet_tensor(SaTest::find_darknet_reference(directory, name));
    }

    /**
     * @brief Checks @p actual against @p reference within K_DECODE_ABSOLUTE and prints the
     *        largest difference measured.
     * @return Success, or a failure naming the count outside the bound and the worst element.
     */
    ::testing::AssertionResult within_absolute(std::string_view what,
                                               SushiRuntime::span<const float> actual,
                                               SushiRuntime::span<const float> reference)
    {
        if (actual.size() != reference.size())
            return ::testing::AssertionFailure() << what << ": " << actual.size()
                                                 << " values against " << reference.size();

        double largest = 0.0;
        std::size_t worst = 0;
        std::size_t outside = 0;
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            const double difference = std::fabs(static_cast<double>(actual[i]) -
                                                static_cast<double>(reference[i]));
            if (!(difference <= K_DECODE_ABSOLUTE))
                ++outside;
            if (!(difference <= largest))
            {
                largest = difference;
                worst = i;
            }
        }

        std::cout << "[ MEASURED ] " << what << ": largest difference " << largest
                  << " at element " << worst << "; bound " << K_DECODE_ABSOLUTE << "\n";
        if (outside == 0)
            return ::testing::AssertionSuccess();

        return ::testing::AssertionFailure()
               << what << ": " << outside << " of " << actual.size()
               << " values are outside " << K_DECODE_ABSOLUTE << "; worst at " << worst << " ("
               << actual[worst] << " against " << reference[worst] << ")";
    }

    /**
     * @brief Decodes every head of @p model through the library and compares each box and each
     *        score with the rows Darknet dumped for that head.
     *
     * Both sides are in row order `(row * W + column) * A + anchor`; no threshold is applied to
     * either side.
     */
    void expect_decode_matches(const DecodeModel& model)
    {
        SA_ASSERT_OK_AND_ASSIGN(const DarknetModel translated, translate_model(model.cfg));
        ASSERT_EQ(translated.heads.size(), model.heads.size());
        ASSERT_EQ(translated.layout.head_anchors.size(), model.heads.size());

        for (std::size_t k = 0; k < model.heads.size(); ++k)
        {
            const DecodeHead& head = model.heads[k];
            const std::int64_t index = static_cast<std::int64_t>(k);
            SCOPED_TRACE(std::string(model.directory) + ", head " + std::to_string(k) +
                         ", layer " + std::to_string(head.layer));

            EXPECT_EQ(translated.heads[k].layer, head.layer);
            const HeadOptions& options = translated.heads[k].options;
            EXPECT_EQ(options.num_classes, K_CLASSES);
            const Status valid =
                Vision::Detect::Anchor::validate_head(options, translated.layout, index);
            ASSERT_TRUE(valid.is_ok()) << valid.message();

            const HeadShape shape{1, head.grid, head.grid};
            const std::int64_t rows =
                Vision::Detect::Anchor::count_head_rows(shape, translated.layout, index);
            const std::int64_t channels =
                Vision::Detect::Anchor::count_head_channels(options, translated.layout, index);

            SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor input,
                                    load_reference(model.directory, head.input));
            ASSERT_EQ(input.extents, (std::vector<std::int64_t>{head.grid, head.grid, channels}));
            SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor boxes,
                                    load_reference(model.directory, head.boxes));
            ASSERT_EQ(boxes.extents, (std::vector<std::int64_t>{rows, 4}));
            SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor scores,
                                    load_reference(model.directory, head.scores));
            ASSERT_EQ(scores.extents, (std::vector<std::int64_t>{rows, options.num_classes}));

            std::vector<float> our_boxes(static_cast<std::size_t>(rows * 4));
            std::vector<float> our_scores(static_cast<std::size_t>(rows * options.num_classes));
            Vision::Detect::Anchor::decode_head(input.values, shape, options, translated.layout,
                                                index, our_boxes, our_scores);

            EXPECT_TRUE(within_absolute("boxes", our_boxes, boxes.values));
            EXPECT_TRUE(within_absolute("scores", our_scores, scores.values));
        }
    }

    /** @brief Lists the six models with the heads REPORT sections 6 and 14 name, by layer. */
    std::vector<DecodeModel> list_models()
    {
        return {
            {"yolov3-tiny.cfg",
             "yolov3_tiny",
             {{16, 13, "layer_015_yolo_input", "layer_016_yolo_boxes", "layer_016_yolo_scores"},
              {23, 26, "layer_022_yolo_input", "layer_023_yolo_boxes", "layer_023_yolo_scores"}}},
            {"yolov7-tiny.cfg",
             "yolov7_tiny",
             {{90, 52, "layer_089_yolo_input", "layer_090_yolo_boxes", "layer_090_yolo_scores"},
              {94, 26, "layer_093_yolo_input", "layer_094_yolo_boxes", "layer_094_yolo_scores"},
              {98, 13, "layer_097_yolo_input", "layer_098_yolo_boxes", "layer_098_yolo_scores"}}},
            {"yolov7.cfg",
             "yolov7_thresh_0p01",
             {{134, 80, "layer_133_yolo_input", "layer_134_yolo_boxes", "layer_134_yolo_scores"},
              {138, 40, "layer_137_yolo_input", "layer_138_yolo_boxes", "layer_138_yolo_scores"},
              {142, 20, "layer_141_yolo_input", "layer_142_yolo_boxes", "layer_142_yolo_scores"}}},
            {"yolov3.cfg",
             "yolov3",
             {{82, 13, "layer_081_yolo_input", "layer_082_yolo_boxes", "layer_082_yolo_scores"},
              {94, 26, "layer_093_yolo_input", "layer_094_yolo_boxes", "layer_094_yolo_scores"},
              {106, 52, "layer_105_yolo_input", "layer_106_yolo_boxes", "layer_106_yolo_scores"}}},
            {"yolov4-tiny.cfg",
             "yolov4_tiny",
             {{30, 13, "layer_029_yolo_input", "layer_030_yolo_boxes", "layer_030_yolo_scores"},
              {37, 26, "layer_036_yolo_input", "layer_037_yolo_boxes", "layer_037_yolo_scores"}}},
            {"yolov4.cfg",
             "yolov4",
             {{139, 76, "layer_138_yolo_input", "layer_139_yolo_boxes", "layer_139_yolo_scores"},
              {150, 38, "layer_149_yolo_input", "layer_150_yolo_boxes", "layer_150_yolo_scores"},
              {161, 19, "layer_160_yolo_input", "layer_161_yolo_boxes", "layer_161_yolo_scores"}}},
        };
    }

    /** @brief Prints the run folder of @p model where the test runner names a parameter. */
    void PrintTo(const DecodeModel& model, std::ostream* stream)
    {
        *stream << model.directory;
    }

    /** @brief Runs once per model; every case reads Darknet's dumps and skips without them. */
    class Integration_DarknetDecode : public ::testing::TestWithParam<DecodeModel>
    {
    };

} // namespace

/**
 * @brief Tests that every head of the model decodes to the boxes and scores Darknet dumped
 *        before its threshold, within 1e-6 absolute, and skips naming a missing file.
 */
TEST_P(Integration_DarknetDecode, TheHeadsDecodeToTheDumpedRows)
{
    const DecodeModel& model = GetParam();
    const std::string missing = first_missing_file(model);
    if (!missing.empty())
        GTEST_SKIP() << missing;

    expect_decode_matches(model);
}

INSTANTIATE_TEST_SUITE_P(Models, Integration_DarknetDecode, ::testing::ValuesIn(list_models()),
                         [](const ::testing::TestParamInfo<DecodeModel>& info)
                         {
                             return std::string(info.param.directory);
                         });
