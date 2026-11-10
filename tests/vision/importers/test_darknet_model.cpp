/**************************************************************************/
/* test_darknet_model.cpp                                                 */
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
 * @file test_darknet_model.cpp
 * @brief Tests how Darknet sections translate to model rows: grouped routes, shortcuts and the
 *        rest.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/layer_traces/trace_summary.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/vision/darknet_forward_cases/darknet_forward.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Returns the array under @p key of @p value; a refusal fails the test run. */
    const std::vector<Config::Value>& array_of(const Config::Value& value, std::string_view key)
    {
        return SaTest::require_value(SaTest::require_value(value.at(key, key)).as_array(key));
    }

    /** @brief Returns the whole number under @p key of @p value; a refusal fails the test run. */
    std::int64_t int_of(const Config::Value& value, std::string_view key)
    {
        return SaTest::require_value(SaTest::require_value(value.at(key, key)).as_int(key));
    }

    /** @brief Returns the real under @p key of @p value; a refusal fails the test run. */
    double real_of(const Config::Value& value, std::string_view key)
    {
        return SaTest::require_value(SaTest::require_value(value.at(key, key)).as_real(key));
    }

    /** @brief Returns the flag under @p key of @p value; a refusal fails the test run. */
    bool bool_of(const Config::Value& value, std::string_view key)
    {
        return SaTest::require_value(SaTest::require_value(value.at(key, key)).as_bool(key));
    }

    using SaTest::sources_of_row;
    using SaTest::translate_darknet_text;
    using Vision::Detect::Anchor::HeadCoding;
    using Vision::Detect::SuppressionCriterion;
    using Vision::Importers::DarknetConvolutionShape;
    using Vision::Importers::DarknetModel;
    using Vision::Importers::DarknetNormalisation;
    using Vision::Importers::DarknetWeightTarget;

    /** @brief Builds the model state of a translated text on the host engine. */
    class Small_DarknetModelState : public ::testing::Test
    {
        protected:
            /** @brief Holds the host engine the state is built on. */
            HostExecutionBackend backend;
    };

    /** @brief Holds the first section and layers 0 to 11 of yolov3-tiny.cfg, as input data. */
    constexpr std::string_view K_TINY_BACKBONE =
        "[net]\nbatch=1\nsubdivisions=1\nwidth=416\nheight=416\nchannels=3\nmomentum=0.9\n"
        "decay=0.0005\nangle=0\nsaturation = 1.5\nexposure = 1.5\nhue=.1\n"
        "learning_rate=0.001\nburn_in=1000\nmax_batches = 500200\npolicy=steps\n"
        "steps=400000,450000\nscales=.1,.1\n"
        "[convolutional]\nbatch_normalize=1\nfilters=16\nsize=3\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[maxpool]\nsize=2\nstride=2\n"
        "[convolutional]\nbatch_normalize=1\nfilters=32\nsize=3\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[maxpool]\nsize=2\nstride=2\n"
        "[convolutional]\nbatch_normalize=1\nfilters=64\nsize=3\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[maxpool]\nsize=2\nstride=2\n"
        "[convolutional]\nbatch_normalize=1\nfilters=128\nsize=3\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[maxpool]\nsize=2\nstride=2\n"
        "[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[maxpool]\nsize=2\nstride=2\n"
        "[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[maxpool]\nsize=2\nstride=1\n";

    /** @brief Holds layers 12 to 23 of yolov3-tiny.cfg: the two heads and what feeds them. */
    constexpr std::string_view K_TINY_HEADS =
        "[convolutional]\nbatch_normalize=1\nfilters=1024\nsize=3\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[convolutional]\nbatch_normalize=1\nfilters=256\nsize=1\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[convolutional]\nbatch_normalize=1\nfilters=512\nsize=3\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[convolutional]\nsize=1\nstride=1\npad=1\nfilters=255\nactivation=linear\n"
        "[yolo]\nmask = 3,4,5\nanchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319\n"
        "classes=80\nnum=6\njitter=.3\nignore_thresh = .7\ntruth_thresh = 1\nrandom=1\n"
        "[route]\nlayers = -4\n"
        "[convolutional]\nbatch_normalize=1\nfilters=128\nsize=1\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[upsample]\nstride=2\n"
        "[route]\nlayers = -1, 8\n"
        "[convolutional]\nbatch_normalize=1\nfilters=256\nsize=3\nstride=1\npad=1\n"
        "activation=leaky\n"
        "[convolutional]\nsize=1\nstride=1\npad=1\nfilters=255\nactivation=linear\n"
        "[yolo]\nmask = 0,1,2\nanchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319\n"
        "classes=80\nnum=6\njitter=.3\nignore_thresh = .7\ntruth_thresh = 1\nrandom=1\n";

    /** @brief Holds the opening of a 32 by 32 network whose layer 0 has 18 plain channels. */
    constexpr std::string_view K_SMALL_NETWORK =
        "[net]\nwidth=32\nheight=32\nchannels=3\n"
        "[convolutional]\nfilters=18\nsize=1\nstride=1\npad=1\nactivation=logistic\n";

    /** @brief Holds the opening of a [yolo] section with three anchors and one class. */
    constexpr std::string_view K_SMALL_HEAD =
        "[yolo]\nmask=0,1,2\nanchors=4,8, 8,16, 16,24\nclasses=1\nnum=3\n";

    /** @brief Returns the translation of the whole yolov3-tiny text in @p normalisation. */
    Result<DarknetModel> translate_tiny(DarknetNormalisation normalisation)
    {
        return translate_darknet_text(std::string(K_TINY_BACKBONE).append(K_TINY_HEADS),
                                      normalisation);
    }

    /** @brief Returns the type of row @p row of @p model. */
    const std::string& type_of(const DarknetModel& model, std::int64_t row)
    {
        return model.spec.layers.at(static_cast<std::size_t>(row)).type;
    }

    /** @brief Returns every name a target of @p model holds, empty names left out. */
    std::vector<std::string> target_names(const DarknetModel& model)
    {
        std::vector<std::string> names;
        for (const DarknetWeightTarget& target : model.targets)
        {
            for (const std::string* name :
                 {&target.weight, &target.bias, &target.scale, &target.mean, &target.variance})
            {
                if (!name->empty())
                    names.push_back(*name);
            }
        }
        return names;
    }

    /** @brief Returns a registry holding the rows of register_layers. */
    Result<NN::NodeRegistry> registered()
    {
        NN::NodeRegistry registry;
        SA_TRY(NN::register_layers(registry));

        return Result<NN::NodeRegistry>(std::move(registry));
    }

    /**
     * @brief Expects that @p model builds, publishes the two heads of yolov3-tiny, and that
     *        every target name is an entry of the state of the traced graph.
     */
    void expect_builds_and_targets_exist(SushiBLAS::Engine& engine, const DarknetModel& model)
    {
        SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, registered());
        SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph graph, NN::build_model(registry, model.spec));
        SA_ASSERT_OK_AND_ASSIGN(const SaTest::TraceSummary trace,
                                SaTest::trace_summary(graph, model.spec.input_shape, ""));
        EXPECT_EQ(trace.outputs, (std::vector<std::string>{"head.0 [2, 13, 13, 255]",
                                                           "head.1 [2, 26, 26, 255]"}));

        SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, graph));
        std::set<std::string> entries;
        for (const NN::StateEntry& entry : state.entries())
            entries.insert(entry.name);

        const std::vector<std::string> names = target_names(model);
        EXPECT_EQ(std::set<std::string>(names.begin(), names.end()).size(), names.size());
        for (const std::string& name : names)
            EXPECT_EQ(entries.count(name), 1u) << "no state entry is named " << name;
    }

    /** @brief Returns the whole numbers of the array under @p key of @p value, in order. */
    std::vector<std::int64_t> ints_of_array(const Config::Value& value, std::string_view key)
    {
        std::vector<std::int64_t> ints;
        for (const Config::Value& item : array_of(value, key))
            ints.push_back(SaTest::require_value(item.as_int(key)));

        return ints;
    }

    /** @brief Returns the trailing padding of the pool on row @p row of @p model. */
    std::vector<std::int64_t> trailing_padding_of(const DarknetModel& model, std::int64_t row)
    {
        return ints_of_array(model.spec.layers.at(static_cast<std::size_t>(row)).attributes,
                             "trailing_padding");
    }

    /** @brief Returns how many rows of @p model have the type @p type. */
    std::size_t rows_of_type(const DarknetModel& model, std::string_view type)
    {
        return static_cast<std::size_t>(
            std::count_if(model.spec.layers.begin(), model.spec.layers.end(),
                          [type](const Config::LayerSpec& row) { return row.type == type; }));
    }

    /** @brief Returns the floats the weights file holds for the convolutions of @p model. */
    std::int64_t floats_of(const DarknetModel& model)
    {
        std::int64_t floats = 0;
        for (const DarknetConvolutionShape& shape : model.convolutions)
        {
            floats += shape.filters * (shape.normalised ? 4 : 1);
            floats += shape.filters * shape.channels * shape.size * shape.size;
        }

        return floats;
    }

    /** @brief Returns the parts joined into one text. */
    std::string joined(std::initializer_list<std::string_view> parts)
    {
        std::string text;
        for (std::string_view part : parts)
            text.append(part);

        return text;
    }

    /** @brief Holds the keys of a [yolo] section with the coding of YOLOv7. */
    constexpr std::string_view K_ACTIVATED_KEYS =
        "scale_x_y = 2.0\nobjectness_smooth=1\nresize=1.5\n"
        "iou_thresh=0.2\niou_normalizer=0.05\n"
        "cls_normalizer=0.5\nobj_normalizer=1.0\niou_loss=ciou\n"
        "nms_kind=diounms\nbeta_nms=0.6\nnew_coords=1\n"
        "max_delta=2\n";

    /** @brief Returns the translation of the small network with a head of the YOLOv7 keys. */
    Result<DarknetModel> translate_activated_head()
    {
        return translate_darknet_text(joined({K_SMALL_NETWORK, K_SMALL_HEAD, K_ACTIVATED_KEYS}),
                                      DarknetNormalisation::Separate);
    }

    /** @brief Holds a text the importer refuses, its code and three substrings of its message. */
    struct Refusal
    {
        const char* name;   /**< Names the row. */
        std::string text;   /**< Holds the config text to translate. */
        StatusCode code;    /**< Holds the code of the refusal. */
        const char* first;  /**< Holds a substring the message names; may be empty. */
        const char* second; /**< Holds a substring the message names; may be empty. */
        const char* third;  /**< Holds a substring the message names; may be empty. */
    };

    /** @brief Expects the translation of the text of @p row to be refused as the row says. */
    void expect_refusal(const Refusal& row)
    {
        SA_EXPECT_FAILURE(translate_darknet_text(row.text, DarknetNormalisation::Separate),
                          row.code, row.first, row.second, row.third);
    }

    /** @brief Holds one head of yolov3-tiny: its output, layer and the row it publishes. */
    struct HeadRow
    {
        const char* name;         /**< Names the row. */
        std::size_t index;        /**< Holds the position of the head. */
        const char* output;       /**< Holds the output name of the head. */
        std::int64_t layer;       /**< Holds the layer of the head. */
        std::int64_t row;         /**< Holds the row the output publishes. */
    };

    /** @brief Holds one anchor of yolov3-tiny as a pixel size over the 416 pixel network. */
    struct AnchorRow
    {
        const char* name;  /**< Names the row. */
        std::size_t index; /**< Holds the position of the anchor. */
        float width;       /**< Holds the width in pixels. */
        float height;      /**< Holds the height in pixels. */
    };

    /** @brief Holds one nms_kind and beta_nms of a head and the exponent they give. */
    struct NmsRow
    {
        const char* name;  /**< Names the row. */
        const char* kind;  /**< Holds the nms_kind value. */
        const char* beta;  /**< Holds the beta_nms value. */
        float exponent;    /**< Holds the distance exponent of the model. */
    };

    /** @brief Holds a shortcut section and nothing else of the model. */
    struct ShortcutRow
    {
        const char* name;    /**< Names the row. */
        const char* section; /**< Holds the shortcut section to add. */
    };

    /** @brief Reads the heads of yolov3-tiny one per row. */
    class Small_DarknetModelHeads : public ::testing::TestWithParam<HeadRow>
    {
    };

    /** @brief Reads the anchors of yolov3-tiny one per row. */
    class Small_DarknetModelAnchors : public ::testing::TestWithParam<AnchorRow>
    {
    };

    /** @brief Reads the suppression of a head one nms_kind per row. */
    class Small_DarknetModelSuppression : public ::testing::TestWithParam<NmsRow>
    {
    };

    /** @brief Refuses the keys that no section translates, one per row. */
    class Small_DarknetModelUnknownKeys : public ::testing::TestWithParam<Refusal>
    {
    };

    /** @brief Refuses the values the importer does not translate, one per row. */
    class Small_DarknetModelUntranslated : public ::testing::TestWithParam<Refusal>
    {
    };

    /** @brief Refuses the values that do not fit their layer, one per row. */
    class Small_DarknetModelMisfits : public ::testing::TestWithParam<Refusal>
    {
    };

    /** @brief Refuses the splits of a grouped route, one per row. */
    class Small_DarknetRouteSplits : public ::testing::TestWithParam<Refusal>
    {
    };

    /** @brief Reads the row a shortcut section adds, one spelling of from per row. */
    class Small_DarknetShortcutFrom : public ::testing::TestWithParam<ShortcutRow>
    {
    };

    /** @brief Names a row of a table by its field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    }
} // namespace

/**
 * @brief Tests the row count and the row each layer ends on in the Separate form.
 *
 * The expected rows are counted by hand from the text: a normalised convolution is three rows,
 * a plain one with a linear activation one, a pool, an upsample and a two-layer route one each,
 * and a one-layer route and a [yolo] none.
 */
TEST(Small_DarknetModel, SeparateFormHasTheCountedRows)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_EQ(model.spec.layers.size(), 43u);
    EXPECT_EQ(model.row_of_layer,
              (std::vector<std::int64_t>{2,  3,  6,  7,  10, 11, 14, 15, 18, 19, 22, 23,
                                         26, 29, 32, 33, 33, 29, 36, 37, 38, 41, 42, 42}));
}

/** @brief Tests that the Separate form reads the input shape, the fit and the network size. */
TEST(Small_DarknetModel, SeparateFormReadsTheInputOfTheNetSection)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_EQ(model.spec.input_shape, (Shape{416, 416, 3}));
    EXPECT_EQ(model.spec.input_fit, Config::K_FIT_RESIZE);
    EXPECT_EQ(model.network.height, 416);
    EXPECT_EQ(model.network.width, 416);
}

/** @brief Tests that a normalised convolution is a conv2d, a batchnorm2d and a leaky_relu. */
TEST(Small_DarknetModel, ANormalisedConvolutionIsThreeRowsReadingTheInput)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_EQ(type_of(model, 0), "conv2d");
    EXPECT_EQ(type_of(model, 1), "batchnorm2d");
    EXPECT_EQ(type_of(model, 2), "leaky_relu");
    EXPECT_EQ(sources_of_row(model, 0), (std::vector<std::int64_t>{Config::K_MODEL_INPUT}));
}

/** @brief Tests the leaky slope and the batch-norm epsilon the Separate form writes. */
TEST(Small_DarknetModel, SeparateFormWritesTheSlopeAndTheEpsilon)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_DOUBLE_EQ(real_of(model.spec.layers[2].attributes, "slope"), 0.1);
    EXPECT_DOUBLE_EQ(real_of(model.spec.layers[1].attributes, "bn_epsilon"), 1e-5);
}

/** @brief Tests that only a convolution without batch normalisation carries a bias. */
TEST(Small_DarknetModel, OnlyAPlainConvolutionHasABiasInTheSeparateForm)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_FALSE(bool_of(model.spec.layers[0].attributes, "use_bias"));
    EXPECT_TRUE(bool_of(model.spec.layers[33].attributes, "use_bias"));
}

/** @brief Tests that the convolution after a route of layer 13 alone reads layer 13's row. */
TEST(Small_DarknetModel, ARouteOfOneLayerAddsNoRow)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    // Layer 17 is a route of layer 13 alone, so layer 18's convolution reads layer 13's row.
    EXPECT_EQ(type_of(model, 34), "conv2d");
    EXPECT_EQ(sources_of_row(model, 34), (std::vector<std::int64_t>{29}));
}

/** @brief Tests that a route of two layers is a concat in the order the file lists them. */
TEST(Small_DarknetModel, ARouteOfTwoLayersIsAConcatInFileOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    // Layer 20 joins layer 19, the upsample, with layer 8.
    EXPECT_EQ(type_of(model, 38), "concat");
    EXPECT_EQ(sources_of_row(model, 38), (std::vector<std::int64_t>{37, 18}));
}

/** @brief Tests that the pool of layer 11 keeps the extent and pads one cell at the end. */
TEST(Small_DarknetModel, AStrideOnePoolPadsTheTrailingEdge)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_EQ(type_of(model, 23), "maxpool2d");
    const Config::Value& pool = model.spec.layers[23].attributes;
    EXPECT_EQ(int_of(pool, "kernel"), 2);
    EXPECT_EQ(int_of(pool, "stride"), 1);
    EXPECT_EQ(int_of(pool, "padding"), 0);
    EXPECT_EQ(trailing_padding_of(model, 23), (std::vector<std::int64_t>{1, 1}));
}

/** @brief Tests that a pool that halves an even extent needs no trailing pad. */
TEST(Small_DarknetModel, AStrideTwoPoolOfAnEvenExtentNeedsNoTrailingPad)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_EQ(trailing_padding_of(model, 3), (std::vector<std::int64_t>{0, 0}));
}

/** @brief Tests that the Folded form has the counted rows and the row each layer ends on. */
TEST(Small_DarknetModel, FoldedFormHasTheCountedRows)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    EXPECT_EQ(model.spec.layers.size(), 32u);
    EXPECT_EQ(model.row_of_layer,
              (std::vector<std::int64_t>{1,  2,  4,  5,  7,  8,  10, 11, 13, 14, 16, 17,
                                         19, 21, 23, 24, 24, 21, 26, 27, 28, 30, 31, 31}));
}

/** @brief Tests that the Folded form drops every batch-norm row. */
TEST(Small_DarknetModel, FoldedFormHasNoNormalisationRow)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    EXPECT_EQ(rows_of_type(model, "batchnorm2d"), 0u);
}

/** @brief Tests that every convolution of the Folded form carries the bias the fold wrote. */
TEST(Small_DarknetModel, FoldedFormConvolutionsHaveABias)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    EXPECT_TRUE(bool_of(model.spec.layers[0].attributes, "use_bias"));
}

/** @brief Tests that the Folded form keeps the sources of the routes. */
TEST(Small_DarknetModel, FoldedFormKeepsTheSourcesOfTheRoutes)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    EXPECT_EQ(sources_of_row(model, 25), (std::vector<std::int64_t>{21}));
    EXPECT_EQ(sources_of_row(model, 28), (std::vector<std::int64_t>{27, 13}));
}

/** @brief Tests that the first target of the Folded form names a weight and a bias only. */
TEST(Small_DarknetModel, FoldedFormTargetsNameAWeightAndABiasOnly)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    ASSERT_EQ(model.targets.size(), 13u);
    EXPECT_EQ(model.targets[0].weight, "0.weight");
    EXPECT_EQ(model.targets[0].bias, "0.bias");
    EXPECT_TRUE(model.targets[0].scale.empty());
    EXPECT_TRUE(model.targets[0].mean.empty());
    EXPECT_TRUE(model.targets[0].variance.empty());
}

/** @brief Tests that yolov3-tiny has two heads and two outputs. */
TEST(Small_DarknetModel, TwoHeadsPublishTwoOutputs)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_EQ(model.heads.size(), 2u);
    EXPECT_EQ(model.spec.outputs.size(), 2u);
}

/** @brief Tests that a head names its output and the layer it ends. */
TEST_P(Small_DarknetModelHeads, NamesItsOutputAndLayer)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    ASSERT_EQ(model.heads.size(), 2u);
    EXPECT_EQ(model.heads[GetParam().index].output, GetParam().output);
    EXPECT_EQ(model.heads[GetParam().index].layer, GetParam().layer);
}

/** @brief Tests that the output of a head publishes the row of its layer. */
TEST_P(Small_DarknetModelHeads, OutputPublishesTheRowOfItsLayer)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    ASSERT_EQ(model.spec.outputs.size(), 2u);
    EXPECT_EQ(model.spec.outputs[GetParam().index].node, GetParam().row);
    EXPECT_EQ(model.spec.outputs[GetParam().index].port, "output");
    EXPECT_EQ(model.spec.outputs[GetParam().index].name, GetParam().output);
}

/** @brief Tests that a head of yolov3-tiny has 80 classes, the Logit coding and no stretch. */
TEST_P(Small_DarknetModelHeads, ReadsTheOptionsOfTheFile)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    ASSERT_EQ(model.heads.size(), 2u);
    const auto& options = model.heads[GetParam().index].options;
    EXPECT_EQ(options.num_classes, 80);
    EXPECT_EQ(options.coding, HeadCoding::Logit);
    EXPECT_EQ(options.scale_xy, 1.0f);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DarknetModelHeads,
                         ::testing::Values(HeadRow{"First", 0, "head.0", 16, 33},
                                           HeadRow{"Second", 1, "head.1", 23, 42}),
                         row_name<HeadRow>);

/** @brief Tests that the criterion is IoU and the exponent is its default of 0.6. */
TEST(Small_DarknetModel, SuppressionDefaultsToIoUWithTheDefaultExponent)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_EQ(model.criterion, SuppressionCriterion::IoU);
    EXPECT_EQ(model.distance_exponent, 0.6f);
}

/**
 * @brief Tests that an anchor is its pixel size over the network size.
 *
 * A fraction is compared to four units in the last place: the translation divides in double
 * and rounds once, the test divides in float.
 */
TEST_P(Small_DarknetModelAnchors, IsItsPixelSizeOverTheNetworkSize)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    ASSERT_EQ(model.layout.anchors.size(), 6u);
    EXPECT_FLOAT_EQ(model.layout.anchors[GetParam().index].width, GetParam().width / 416.0f);
    EXPECT_FLOAT_EQ(model.layout.anchors[GetParam().index].height, GetParam().height / 416.0f);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DarknetModelAnchors,
                         ::testing::Values(AnchorRow{"First", 0, 10.0f, 14.0f},
                                           AnchorRow{"Second", 1, 23.0f, 27.0f},
                                           AnchorRow{"Third", 2, 37.0f, 58.0f},
                                           AnchorRow{"Fourth", 3, 81.0f, 82.0f},
                                           AnchorRow{"Fifth", 4, 135.0f, 169.0f},
                                           AnchorRow{"Sixth", 5, 344.0f, 319.0f}),
                         row_name<AnchorRow>);

/** @brief Tests that the masks of the two heads list the anchors each head reads. */
TEST(Small_DarknetModel, MasksListTheAnchorsOfEachHead)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_EQ(model.layout.head_anchors,
              (std::vector<std::vector<std::int64_t>>{{3, 4, 5}, {0, 1, 2}}));
}

/**
 * @brief Tests that the mask a cfg states wins over the split by area and stride.
 *
 * The six anchors of yolov3-tiny give {3, 4, 5} and {0, 1, 2} by area; the second head of
 * yolov4-tiny states mask 1,2,3, which pretrained weights are trained against.
 */
TEST(Small_DarknetModel, MasksOfACfgOverrideTheAlgorithmicSplit)
{
    std::string text = std::string(K_TINY_BACKBONE).append(K_TINY_HEADS);
    const std::string_view algorithmic = "mask = 0,1,2";
    const std::size_t at = text.find(algorithmic);
    ASSERT_NE(at, std::string::npos);
    text.replace(at, algorithmic.size(), "mask = 1,2,3");

    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_darknet_text(text, DarknetNormalisation::Separate));

    EXPECT_EQ(model.layout.head_anchors,
              (std::vector<std::vector<std::int64_t>>{{3, 4, 5}, {1, 2, 3}}));
}

/** @brief Tests that the shapes name the layer of each convolution of the file. */
TEST(Small_DarknetModel, ShapesNameTheLayerOfEachConvolution)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    std::vector<std::int64_t> layers;
    for (const DarknetConvolutionShape& shape : model.convolutions)
        layers.push_back(shape.layer);

    EXPECT_EQ(layers, (std::vector<std::int64_t>{0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 18, 21, 22}));
}

/** @brief Tests that a shape is normalised unless its convolution is a head of 255 filters. */
TEST(Small_DarknetModel, EveryShapeIsNormalisedExceptTheHeadConvolutions)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    ASSERT_EQ(model.convolutions.size(), 13u);
    const auto agrees = [](const DarknetConvolutionShape& shape)
    { return shape.normalised == (shape.filters != 255); };
    EXPECT_TRUE(std::all_of(model.convolutions.begin(), model.convolutions.end(), agrees));
}

/**
 * @brief Tests that the shapes count the floats of the weights file.
 *
 * 8 858 734 is the float count of yolov3-tiny.weights after its header (SPEC 4.2), measured
 * from the file's size.
 */
TEST(Small_DarknetModel, ShapesCountTheFloatsOfTheWeightsFile)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    EXPECT_EQ(floats_of(model), 8858734);
}

/** @brief Tests that the convolution of layer 21 reads the join of both routes' channels. */
TEST(Small_DarknetModel, TheConvolutionAfterTheJoinReadsBothChannelCounts)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    // Layer 21 reads the join of 128 upsampled channels with the 256 of layer 8.
    ASSERT_EQ(model.convolutions.size(), 13u);
    EXPECT_EQ(model.convolutions[11].channels, 384);
}

/** @brief Tests that the Separate model builds and every target names an entry of its state. */
TEST_F(Small_DarknetModelState, SeparateFormBuildsAndItsTargetsAreStateEntries)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_EQ(model.targets[0].weight, "0.weight");
    EXPECT_EQ(model.targets[0].bias, "1.bias");
    EXPECT_EQ(model.targets[0].scale, "1.weight");
    EXPECT_EQ(model.targets[0].mean, "1.running_mean");
    EXPECT_EQ(model.targets[0].variance, "1.running_var");
    EXPECT_EQ(target_names(model).size(), 11u * 5u + 2u * 2u);
    expect_builds_and_targets_exist(backend.engine(), model);
}

/** @brief Tests that the Folded model builds and every target names an entry of its state. */
TEST_F(Small_DarknetModelState, FoldedFormBuildsAndItsTargetsAreStateEntries)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    EXPECT_EQ(target_names(model).size(), 13u * 2u);
    expect_builds_and_targets_exist(backend.engine(), model);
}

/** @brief Tests that a head with new_coords=1 is Activated. */
TEST(Small_DarknetModel, AnActivatedHeadHasTheActivatedCoding)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_activated_head());

    ASSERT_EQ(model.heads.size(), 1u);
    EXPECT_EQ(model.heads[0].options.coding, HeadCoding::Activated);
}

/** @brief Tests that a head reads its stretch from scale_x_y. */
TEST(Small_DarknetModel, AnActivatedHeadReadsItsStretch)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_activated_head());

    ASSERT_EQ(model.heads.size(), 1u);
    EXPECT_EQ(model.heads[0].options.scale_xy, 2.0f);
}

/** @brief Tests that a head reads its class count from classes. */
TEST(Small_DarknetModel, AnActivatedHeadReadsItsClassCount)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_activated_head());

    ASSERT_EQ(model.heads.size(), 1u);
    EXPECT_EQ(model.heads[0].options.num_classes, 1);
}

/** @brief Tests that nms_kind=diounms selects the DistanceIoU criterion. */
TEST(Small_DarknetModel, DiounmsSelectsTheDistanceCriterion)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_activated_head());

    EXPECT_EQ(model.criterion, SuppressionCriterion::DistanceIoU);
}

/** @brief Tests that beta_nms=0.6 gives the distance exponent 0.6. */
TEST(Small_DarknetModel, DiounmsReadsItsExponentFromBetaNMS)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_activated_head());

    EXPECT_EQ(model.distance_exponent, 0.6f);
}

/** @brief Tests that the activated head is a plain convolution and a published sigmoid row. */
TEST(Small_DarknetModel, AnActivatedHeadPublishesASigmoidRow)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_activated_head());

    // Layer 0 is a plain convolution and a sigmoid row, and the head publishes the sigmoid.
    ASSERT_EQ(model.spec.layers.size(), 2u);
    EXPECT_EQ(type_of(model, 1), "sigmoid");
    EXPECT_EQ(model.spec.outputs[0].node, 1);
}

/** @brief Tests that only diounms takes its exponent from beta_nms. */
TEST_P(Small_DarknetModelSuppression, OnlyDiounmsReadsBetaNMS)
{
    const std::string text = joined({K_SMALL_NETWORK, K_SMALL_HEAD, "nms_kind=", GetParam().kind,
                                     "\nbeta_nms=", GetParam().beta, "\n"});
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_darknet_text(text, DarknetNormalisation::Separate));

    EXPECT_EQ(model.criterion, SuppressionCriterion::DistanceIoU);
    EXPECT_EQ(model.distance_exponent, GetParam().exponent);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DarknetModelSuppression,
                         ::testing::Values(NmsRow{"GreedynmsKeepsTheDefault", "greedynms", "0.3",
                                                  0.6f},
                                           NmsRow{"DiounmsTakesBetaNMS", "diounms", "0.3", 0.3f}),
                         row_name<NmsRow>);

/** @brief Tests that the criterion is the one of the last [yolo] section of the file. */
TEST(Small_DarknetModel, TheLastHeadDecidesTheCriterion)
{
    const std::string text =
        std::string(K_SMALL_NETWORK)
            .append(K_SMALL_HEAD)
            .append("nms_kind=diounms\nbeta_nms=0.3\n[route]\nlayers=0\n")
            .append(K_SMALL_HEAD);
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_darknet_text(text, DarknetNormalisation::Separate));

    ASSERT_EQ(model.heads.size(), 2u);
    EXPECT_EQ(model.criterion, SuppressionCriterion::IoU);
    EXPECT_EQ(model.distance_exponent, 0.6f);
}

/** @brief Tests that a section type outside the table is refused and its line is named. */
TEST(Small_DarknetModel, AnUnknownSectionIsRefusedNamingItsLine)
{
    const std::string text = std::string(K_SMALL_NETWORK).append("[reorg]\nstride=2\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text, DarknetNormalisation::Separate),
                      StatusCode::Unsupported, "line 11", "layer 1", "[reorg]");
}

/** @brief Tests that an unknown key of a layer, a head and the first section is refused. */
TEST_P(Small_DarknetModelUnknownKeys, IsRefusedNamingItsLine)
{
    expect_refusal(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DarknetModelUnknownKeys,
    ::testing::Values(
        Refusal{"AGroupedKeyOfAConvolution",
                "[net]\nwidth=32\nheight=32\nchannels=3\n[convolutional]\nfilters=4\ngroups=2\n",
                StatusCode::Unsupported, "line 7", "[convolutional]", "'groups'"},
        Refusal{"AKeyOfAHead", joined({K_SMALL_NETWORK, K_SMALL_HEAD, "label_smooth_eps=0.1\n"}),
                StatusCode::Unsupported, "line 16", "[yolo]", "'label_smooth_eps'"},
        Refusal{"AKeyOfTheFirstSection", "[net]\nwidth=32\nheight=32\nchannels=3\ncrop=1\n",
                StatusCode::Unsupported, "line 5", "'crop'", ""}),
    row_name<Refusal>);

namespace
{
    /** @brief Returns the small network with letter_box set to @p value on its line 5. */
    std::string letter_box_text(std::string_view value)
    {
        const std::string_view small = K_SMALL_NETWORK;

        return joined({"[net]\nwidth=32\nheight=32\nchannels=3\nletter_box=", value, "\n",
                       small.substr(small.find("[convolutional]")), K_SMALL_HEAD});
    }

} // namespace

/** @brief Tests that letter_box=1 gives the letterbox fit. */
TEST(Small_DarknetModel, LetterBoxOneSelectsTheLetterboxFit)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetModel boxed,
        translate_darknet_text(letter_box_text("1"), DarknetNormalisation::Separate));

    EXPECT_EQ(boxed.spec.input_fit, Config::K_FIT_LETTERBOX);
}

/** @brief Tests that a letter_box value past 1 is refused naming its line. */
TEST(Small_DarknetModel, LetterBoxTwoIsRefusedNamingItsLine)
{
    SA_EXPECT_FAILURE(translate_darknet_text(letter_box_text("2"), DarknetNormalisation::Separate),
                      StatusCode::InvalidDescription, "line 5", "letter_box=2");
}

/** @brief Tests the values this importer refuses although the key is known. */
TEST_P(Small_DarknetModelUntranslated, IsRefusedNamingItsLine)
{
    expect_refusal(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DarknetModelUntranslated,
    ::testing::Values(
        Refusal{"AnUpsampleScale", joined({K_SMALL_NETWORK, "[upsample]\nscale=2\n"}),
                StatusCode::Unsupported, "line 11", "scale 2", ""},
        Refusal{"AnActivationOfRelu",
                "[net]\nwidth=32\nheight=32\nchannels=3\n[convolutional]\nactivation=relu\n",
                StatusCode::Unsupported, "line 6", "'relu'", ""}),
    row_name<Refusal>);

/** @brief Tests the refusals of a value that is malformed or does not fit its layer. */
TEST_P(Small_DarknetModelMisfits, IsRefusedNamingItsLine)
{
    expect_refusal(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DarknetModelMisfits,
    ::testing::Values(
        Refusal{"AWordForANumber",
                "[net]\nwidth=32\nheight=32\nchannels=3\n[convolutional]\nfilters=many\n",
                StatusCode::InvalidDescription, "line 6", "'filters'", "'many'"},
        Refusal{"ARouteToALaterLayer", joined({K_SMALL_NETWORK, "[route]\nlayers=1\n"}),
                StatusCode::InvalidDescription, "line 12", "names layer 1", ""},
        Refusal{"AHeadOverTooFewChannels",
                "[net]\nwidth=32\nheight=32\nchannels=3\n"
                "[convolutional]\nfilters=12\nactivation=linear\n"
                "[yolo]\nmask=0,1,2\nanchors=4,8, 8,16, 16,24\nclasses=1\n",
                StatusCode::InvalidDescription, "line 8", "reads 18 channels", "has 12"},
        Refusal{"AMaskPastTheAnchors",
                joined({K_SMALL_NETWORK,
                        "[yolo]\nmask=1,2,3\nanchors=4,8, 8,16, 16,24\nclasses=1\n"}),
                StatusCode::InvalidDescription, "line 12", "names anchor 3", ""},
        Refusal{"ATextWithoutAHead", std::string(K_SMALL_NETWORK), StatusCode::InvalidDescription,
                "no [yolo] section", "", ""}),
    row_name<Refusal>);

namespace
{
    using Test::sources_of_row;
    using Test::translate_darknet_text;

    /** @brief Holds an 8 by 8 network of 3 channels and one 64-filter 1 by 1 convolution. */
    constexpr std::string_view K_WIDE_NETWORK =
        "[net]\nwidth=8\nheight=8\nchannels=3\n"
        "[convolutional]\nfilters=64\nsize=1\nstride=1\npad=0\nactivation=linear\n";

    /** @brief Holds one 18-filter 1 by 1 convolution and the [yolo] head that reads it. */
    constexpr std::string_view K_HEAD_TAIL =
        "[convolutional]\nfilters=18\nsize=1\nstride=1\npad=0\nactivation=linear\n"
        "[yolo]\nmask=0,1,2\nanchors=4,8, 8,16, 16,24\nclasses=1\nnum=3\n";

    /** @brief Returns the integer attribute @p key of row @p row of @p model. */
    std::int64_t int_attribute(const DarknetModel& model, std::int64_t row, std::string_view key)
    {
        const Config::Value& attributes =
            model.spec.layers.at(static_cast<std::size_t>(row)).attributes;
        return int_of(attributes, key);
    }

} // namespace

/**
 * @brief Tests that a grouped route of one source is one slice row of the second half.
 *
 * The 64 channels of layer 1 split into halves of 32; group 1 is channels 32 to 63. Layer 1 is
 * the row 0 output, so the slice is row 1, and the head convolution reads it.
 */
TEST(Small_DarknetRouteGroups, OneSourceGroupIsOneSliceRow)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetModel model,
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\ngroups=2\ngroup_id=1\n")
                                   .append(K_HEAD_TAIL)));

    ASSERT_EQ(model.spec.layers.size(), 3u);
    EXPECT_EQ(type_of(model, 0), "conv2d");
    EXPECT_EQ(type_of(model, 1), "slice_channels");
    EXPECT_EQ(sources_of_row(model, 1), (std::vector<std::int64_t>{0}));
    EXPECT_EQ(int_attribute(model, 1, "offset"), 32);
    EXPECT_EQ(int_attribute(model, 1, "count"), 32);
    EXPECT_EQ(sources_of_row(model, 2), (std::vector<std::int64_t>{1}));

    ASSERT_EQ(model.convolutions.size(), 2u);
    EXPECT_EQ(model.convolutions[1].channels, 32);
}

/**
 * @brief Tests that each source of a grouped route is sliced before one concat joins them.
 *
 * Layer 2 lists -1 and -2, so its first part is layer 1 (row 1) and its second is layer 0
 * (row 0). Each contributes its group 1 of 32 channels, in list order: rows 2 and 3, then the
 * concat of row 4 over both. The head convolution reads 64 channels.
 */
TEST(Small_DarknetRouteGroups, EverySourceIsSlicedBeforeTheJoin)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetModel model,
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[convolutional]\nfilters=64\nsize=1\nstride=1\npad=0\n"
                              "activation=linear\n")
                                   .append("[route]\nlayers=-1,-2\ngroups=2\ngroup_id=1\n")
                                   .append(K_HEAD_TAIL)));

    ASSERT_EQ(model.spec.layers.size(), 6u);
    EXPECT_EQ(type_of(model, 2), "slice_channels");
    EXPECT_EQ(sources_of_row(model, 2), (std::vector<std::int64_t>{1}));
    EXPECT_EQ(int_attribute(model, 2, "offset"), 32);
    EXPECT_EQ(int_attribute(model, 2, "count"), 32);

    EXPECT_EQ(type_of(model, 3), "slice_channels");
    EXPECT_EQ(sources_of_row(model, 3), (std::vector<std::int64_t>{0}));
    EXPECT_EQ(int_attribute(model, 3, "offset"), 32);
    EXPECT_EQ(int_attribute(model, 3, "count"), 32);

    EXPECT_EQ(type_of(model, 4), "concat");
    EXPECT_EQ(sources_of_row(model, 4), (std::vector<std::int64_t>{2, 3}));
    EXPECT_EQ(sources_of_row(model, 5), (std::vector<std::int64_t>{4}));

    ASSERT_EQ(model.convolutions.size(), 3u);
    EXPECT_EQ(model.convolutions[2].channels, 64);
}

/**
 * @brief Tests that an ungrouped route of one source adds no row and keeps every channel.
 */
TEST(Small_DarknetRouteGroups, UngroupedSourceAddsNoRow)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetModel model,
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\n")
                                   .append(K_HEAD_TAIL)));

    ASSERT_EQ(model.spec.layers.size(), 2u);
    EXPECT_EQ(type_of(model, 1), "conv2d");
    EXPECT_EQ(sources_of_row(model, 1), (std::vector<std::int64_t>{0}));

    ASSERT_EQ(model.convolutions.size(), 2u);
    EXPECT_EQ(model.convolutions[1].channels, 64);
}

/**
 * @brief Tests that a group count that does not divide the channels, or a group index out of
 *        range, is refused naming its line.
 *
 * The [route] section starts on line 11 and its layers line is line 12, which names the channel
 * count. A group index is refused on the section line.
 */
TEST_P(Small_DarknetRouteSplits, IsRefusedNamingItsLine)
{
    expect_refusal(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DarknetRouteSplits,
    ::testing::Values(
        Refusal{"GroupsThatDoNotDivideTheChannels",
                joined({K_WIDE_NETWORK, "[route]\nlayers=-1\ngroups=3\ngroup_id=0\n", K_HEAD_TAIL}),
                StatusCode::InvalidDescription, "line 12", "groups 3", "64 channels"},
        Refusal{"AGroupIdPastTheGroups",
                joined({K_WIDE_NETWORK, "[route]\nlayers=-1\ngroups=2\ngroup_id=2\n", K_HEAD_TAIL}),
                StatusCode::InvalidDescription, "line 11", "group_id 2", ""},
        Refusal{"ZeroGroups",
                joined({K_WIDE_NETWORK, "[route]\nlayers=-1\ngroups=0\ngroup_id=0\n", K_HEAD_TAIL}),
                StatusCode::InvalidDescription, "line 11", "groups 0", ""},
        Refusal{"AGroupIdWithoutGroups",
                joined({K_WIDE_NETWORK, "[route]\nlayers=-1\ngroup_id=1\n", K_HEAD_TAIL}),
                StatusCode::InvalidDescription, "line 11", "group_id 1", ""}),
    row_name<Refusal>);

namespace
{
    using Test::sources_of_row;
    using Test::translate_darknet_text;

    /** @brief Holds the first section of an 8 by 8 network with 3 input channels. */
    constexpr std::string_view K_NETWORK = "[net]\nwidth=8\nheight=8\nchannels=3\n";

    /** @brief Holds layers 0 and 1: two 1 by 1 linear convolutions of 4 filters, lines 5 to 16. */
    constexpr std::string_view K_TWO_CONVOLUTIONS =
        "[convolutional]\nfilters=4\nsize=1\nstride=1\npad=0\nactivation=linear\n"
        "[convolutional]\nfilters=4\nsize=1\nstride=1\npad=0\nactivation=linear\n";

} // namespace

/**
 * @brief Tests that a shortcut adds the layer before it and the named layer, whichever way
 *        from spells the named layer.
 */
TEST_P(Small_DarknetShortcutFrom, AddsTheLayerBeforeAndTheNamedLayer)
{
    const std::string text = joined({K_NETWORK, K_TWO_CONVOLUTIONS, GetParam().section,
                                     K_HEAD_TAIL});

    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_darknet_text(text));

    ASSERT_EQ(model.spec.layers.size(), 4u);
    EXPECT_EQ(model.spec.layers[2].type, "add");
    EXPECT_TRUE(model.spec.layers[2].attributes.keys().empty());
    EXPECT_EQ(sources_of_row(model, 2), (std::vector<std::int64_t>{1, 0}));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DarknetShortcutFrom,
    ::testing::Values(ShortcutRow{"ANegativeFrom", "[shortcut]\nfrom=-2\n"},
                      ShortcutRow{"AnAbsoluteFrom", "[shortcut]\nfrom=0\nactivation=linear\n"}),
    row_name<ShortcutRow>);

TEST(Small_DarknetShortcut, FromAboveTheShortcutIsRefusedNamingItsLine)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=2\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::InvalidDescription, "line 18",
                      "only layers 0 to 1");
}

TEST(Small_DarknetShortcut, DifferentChannelCountsAreRefusedNamingItsLine)
{
    const std::string text =
        std::string(K_NETWORK)
            .append("[convolutional]\nfilters=4\nsize=1\nstride=1\npad=0\nactivation=linear\n")
            .append("[convolutional]\nfilters=5\nsize=1\nstride=1\npad=0\nactivation=linear\n")
            .append("[shortcut]\nfrom=0\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::InvalidDescription, "line 18",
                      "share their shape");
}

TEST(Small_DarknetShortcut, NonLinearActivationIsRefusedNamingItsLine)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-2\nactivation=leaky\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::Unsupported, "line 19", "'leaky'");
}

TEST(Small_DarknetShortcut, WeightsTypeIsRefusedNamingItsLine)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-2\nweights_type=per_feature\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::Unsupported, "line 19",
                      "weights_type");
}

TEST(Small_DarknetShortcut, MissingFromIsRefusedNamingTheSection)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nactivation=linear\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::InvalidDescription, "no 'from'");
}

TEST(Small_DarknetShortcut, MoreThanOneFromIsRefused)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-1,-2\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::InvalidDescription,
                      "names one layer");
}
