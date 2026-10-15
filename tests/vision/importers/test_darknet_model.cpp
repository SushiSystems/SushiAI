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
 * @file test_darknet_model.cpp
 * @brief Tests how Darknet sections translate to model rows: grouped routes, shortcuts and the
 *        rest.
 * @author Mustafa Garip
 */

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
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
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

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    using SaTest::sources_of_row;
    using SaTest::translate_darknet_text;
    using Vision::Detect::Anchor::HeadCoding;
    using Vision::Detect::SuppressionCriterion;
    using Vision::Importers::DarknetConvolutionShape;
    using Vision::Importers::DarknetModel;
    using Vision::Importers::DarknetNormalisation;
    using Vision::Importers::DarknetWeightTarget;

    class Medium_DarknetModelState : public SaTest::EngineTest {};

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

} // namespace

/**
 * @brief Tests the row count, the sources of both routes and the pool of layer 11 in the
 *        Separate form.
 *
 * The expected rows are counted by hand from the text: a normalised convolution is three rows,
 * a plain one with a linear activation one, a pool, an upsample and a two-layer route one each,
 * and a one-layer route and a [yolo] none.
 */
TEST(Small_DarknetModel, SeparateFormHasTheCountedRowsAndSources)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_EQ(model.spec.layers.size(), 43u);
    EXPECT_EQ(model.spec.input_shape, (Shape{416, 416, 3}));
    EXPECT_EQ(model.network.height, 416);
    EXPECT_EQ(model.network.width, 416);
    EXPECT_EQ(model.row_of_layer,
              (std::vector<std::int64_t>{2,  3,  6,  7,  10, 11, 14, 15, 18, 19, 22, 23,
                                         26, 29, 32, 33, 33, 29, 36, 37, 38, 41, 42, 42}));

    EXPECT_EQ(type_of(model, 0), "conv2d");
    EXPECT_EQ(type_of(model, 1), "batchnorm2d");
    EXPECT_EQ(type_of(model, 2), "leaky_relu");
    EXPECT_EQ(sources_of_row(model, 0), (std::vector<std::int64_t>{Config::K_MODEL_INPUT}));
    EXPECT_DOUBLE_EQ(model.spec.layers[2].attributes.at("slope", "s").as_real("s"), 0.1);
    EXPECT_DOUBLE_EQ(model.spec.layers[1].attributes.at("bn_epsilon", "e").as_real("e"), 1e-5);
    EXPECT_FALSE(model.spec.layers[0].attributes.at("use_bias", "b").as_bool("b"));
    EXPECT_TRUE(model.spec.layers[33].attributes.at("use_bias", "b").as_bool("b"));

    // Layer 17 is a route of layer 13 alone, so layer 18's convolution reads layer 13's row.
    EXPECT_EQ(type_of(model, 34), "conv2d");
    EXPECT_EQ(sources_of_row(model, 34), (std::vector<std::int64_t>{29}));

    // Layer 20 joins layer 19, the upsample, with layer 8, in the order the file lists them.
    EXPECT_EQ(type_of(model, 38), "concat");
    EXPECT_EQ(sources_of_row(model, 38), (std::vector<std::int64_t>{37, 18}));

    EXPECT_EQ(type_of(model, 23), "maxpool2d");
    const Config::Value& pool = model.spec.layers[23].attributes;
    EXPECT_EQ(pool.at("kernel", "k").as_int("k"), 2);
    EXPECT_EQ(pool.at("stride", "s").as_int("s"), 1);
    EXPECT_EQ(pool.at("padding", "p").as_int("p"), 0);
    const std::vector<Config::Value>& trailing = pool.at("trailing_padding", "t").as_array("t");
    ASSERT_EQ(trailing.size(), 2u);
    EXPECT_EQ(trailing[0].as_int("t"), 1);
    EXPECT_EQ(trailing[1].as_int("t"), 1);

    // Layer 1 halves an even extent, which needs no trailing pad.
    const std::vector<Config::Value>& even =
        model.spec.layers[3].attributes.at("trailing_padding", "t").as_array("t");
    ASSERT_EQ(even.size(), 2u);
    EXPECT_EQ(even[0].as_int("t"), 0);
    EXPECT_EQ(even[1].as_int("t"), 0);
}

/** @brief Tests that the Folded form drops every batch-norm row and keeps the sources. */
TEST(Small_DarknetModel, FoldedFormHasNoNormalisationRow)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    EXPECT_EQ(model.spec.layers.size(), 32u);
    for (const Config::LayerSpec& row : model.spec.layers)
        EXPECT_NE(row.type, "batchnorm2d");

    EXPECT_EQ(model.row_of_layer,
              (std::vector<std::int64_t>{1,  2,  4,  5,  7,  8,  10, 11, 13, 14, 16, 17,
                                         19, 21, 23, 24, 24, 21, 26, 27, 28, 30, 31, 31}));
    EXPECT_TRUE(model.spec.layers[0].attributes.at("use_bias", "b").as_bool("b"));
    EXPECT_EQ(sources_of_row(model, 25), (std::vector<std::int64_t>{21}));
    EXPECT_EQ(sources_of_row(model, 28), (std::vector<std::int64_t>{27, 13}));

    ASSERT_EQ(model.targets.size(), 13u);
    EXPECT_EQ(model.targets[0].weight, "0.weight");
    EXPECT_EQ(model.targets[0].bias, "0.bias");
    EXPECT_TRUE(model.targets[0].scale.empty());
    EXPECT_TRUE(model.targets[0].mean.empty());
    EXPECT_TRUE(model.targets[0].variance.empty());
}

/**
 * @brief Tests the two outputs, their options, the anchors as fractions and the masks.
 *
 * A fraction is compared to four units in the last place: the translation divides in double
 * and rounds once, the test divides in float.
 */
TEST(Small_DarknetModel, HeadsCarryTheirOutputsOptionsAndAnchors)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    ASSERT_EQ(model.heads.size(), 2u);
    ASSERT_EQ(model.spec.outputs.size(), 2u);
    const std::int64_t layers[] = {16, 23};
    const std::int64_t rows[] = {33, 42};
    const std::string names[] = {"head.0", "head.1"};
    for (std::size_t k = 0; k < 2; ++k)
    {
        EXPECT_EQ(model.heads[k].output, names[k]);
        EXPECT_EQ(model.heads[k].layer, layers[k]);
        EXPECT_EQ(model.heads[k].options.num_classes, 80);
        EXPECT_EQ(model.heads[k].options.coding, HeadCoding::Logit);
        EXPECT_EQ(model.heads[k].options.scale_xy, 1.0f);
        EXPECT_EQ(model.spec.outputs[k].node, rows[k]);
        EXPECT_EQ(model.spec.outputs[k].port, "output");
        EXPECT_EQ(model.spec.outputs[k].name, names[k]);
    }
    EXPECT_EQ(model.criterion, SuppressionCriterion::IoU);
    EXPECT_EQ(model.distance_exponent, 0.6f);

    const float pixels[6][2] = {{10, 14}, {23, 27}, {37, 58}, {81, 82}, {135, 169}, {344, 319}};
    ASSERT_EQ(model.layout.anchors.size(), 6u);
    for (std::size_t i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(model.layout.anchors[i].width, pixels[i][0] / 416.0f);
        EXPECT_FLOAT_EQ(model.layout.anchors[i].height, pixels[i][1] / 416.0f);
    }
    EXPECT_EQ(model.layout.head_anchors,
              (std::vector<std::vector<std::int64_t>>{{3, 4, 5}, {0, 1, 2}}));
}

/**
 * @brief Tests that the shapes name each convolution's layer and count the file's floats.
 *
 * 8 858 734 is the float count of yolov3-tiny.weights after its header (SPEC 4.2), measured
 * from the file's size.
 */
TEST(Small_DarknetModel, ShapesCountTheFloatsOfTheWeightsFile)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    const std::int64_t layers[] = {0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 18, 21, 22};
    ASSERT_EQ(model.convolutions.size(), 13u);
    std::int64_t floats = 0;
    for (std::size_t i = 0; i < 13; ++i)
    {
        const DarknetConvolutionShape& shape = model.convolutions[i];
        EXPECT_EQ(shape.layer, layers[i]);
        EXPECT_EQ(shape.normalised, shape.filters != 255);
        floats += shape.filters * (shape.normalised ? 4 : 1);
        floats += shape.filters * shape.channels * shape.size * shape.size;
    }
    EXPECT_EQ(floats, 8858734);

    // Layer 21 reads the join of 128 upsampled channels with the 256 of layer 8.
    EXPECT_EQ(model.convolutions[11].channels, 384);
}

/** @brief Tests that the Separate model builds and every target names an entry of its state. */
TEST_F(Medium_DarknetModelState, SeparateFormBuildsAndItsTargetsAreStateEntries)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Separate));

    EXPECT_EQ(model.targets[0].weight, "0.weight");
    EXPECT_EQ(model.targets[0].bias, "1.bias");
    EXPECT_EQ(model.targets[0].scale, "1.weight");
    EXPECT_EQ(model.targets[0].mean, "1.running_mean");
    EXPECT_EQ(model.targets[0].variance, "1.running_var");
    EXPECT_EQ(target_names(model).size(), 11u * 5u + 2u * 2u);
    expect_builds_and_targets_exist(*engine, model);
}

/** @brief Tests that the Folded model builds and every target names an entry of its state. */
TEST_F(Medium_DarknetModelState, FoldedFormBuildsAndItsTargetsAreStateEntries)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_tiny(DarknetNormalisation::Folded));

    EXPECT_EQ(target_names(model).size(), 13u * 2u);
    expect_builds_and_targets_exist(*engine, model);
}

/** @brief Tests that a head with the keys of YOLOv7 is Activated, stretched and DistanceIoU. */
TEST(Small_DarknetModel, AnActivatedHeadReadsItsStretchAndItsSuppression)
{
    const std::string text = std::string(K_SMALL_NETWORK)
                                 .append(K_SMALL_HEAD)
                                 .append("scale_x_y = 2.0\nobjectness_smooth=1\nresize=1.5\n"
                                         "iou_thresh=0.2\niou_normalizer=0.05\n"
                                         "cls_normalizer=0.5\nobj_normalizer=1.0\niou_loss=ciou\n"
                                         "nms_kind=diounms\nbeta_nms=0.6\nnew_coords=1\n"
                                         "max_delta=2\n");
    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model,
                            translate_darknet_text(text, DarknetNormalisation::Separate));

    ASSERT_EQ(model.heads.size(), 1u);
    EXPECT_EQ(model.heads[0].options.coding, HeadCoding::Activated);
    EXPECT_EQ(model.heads[0].options.scale_xy, 2.0f);
    EXPECT_EQ(model.heads[0].options.num_classes, 1);
    EXPECT_EQ(model.criterion, SuppressionCriterion::DistanceIoU);
    EXPECT_EQ(model.distance_exponent, 0.6f);

    // Layer 0 is a plain convolution and a sigmoid row, and the head publishes the sigmoid.
    ASSERT_EQ(model.spec.layers.size(), 2u);
    EXPECT_EQ(type_of(model, 1), "sigmoid");
    EXPECT_EQ(model.spec.outputs[0].node, 1);
}

/** @brief Tests that diounms takes its exponent from beta_nms and greedynms never does. */
TEST(Small_DarknetModel, OnlyDiounmsReadsBetaNMS)
{
    const std::string head = std::string(K_SMALL_NETWORK).append(K_SMALL_HEAD);

    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetModel greedy,
        translate_darknet_text(head + "nms_kind=greedynms\nbeta_nms=0.3\n",
                               DarknetNormalisation::Separate));
    EXPECT_EQ(greedy.criterion, SuppressionCriterion::DistanceIoU);
    EXPECT_EQ(greedy.distance_exponent, 0.6f);

    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetModel distance,
        translate_darknet_text(head + "nms_kind=diounms\nbeta_nms=0.3\n",
                               DarknetNormalisation::Separate));
    EXPECT_EQ(distance.criterion, SuppressionCriterion::DistanceIoU);
    EXPECT_EQ(distance.distance_exponent, 0.3f);
}

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

/** @brief Tests that an unknown key of a layer and of the first section is refused by line. */
TEST(Small_DarknetModel, AnUnknownKeyIsRefusedNamingItsLine)
{
    const std::string grouped =
        "[net]\nwidth=32\nheight=32\nchannels=3\n[convolutional]\nfilters=4\ngroups=2\n";
    SA_EXPECT_FAILURE(translate_darknet_text(grouped, DarknetNormalisation::Separate),
                      StatusCode::Unsupported, "line 7", "[convolutional]", "'groups'");

    const std::string smoothed =
        std::string(K_SMALL_NETWORK).append(K_SMALL_HEAD).append("label_smooth_eps=0.1\n");
    SA_EXPECT_FAILURE(translate_darknet_text(smoothed, DarknetNormalisation::Separate),
                      StatusCode::Unsupported, "line 16", "[yolo]", "'label_smooth_eps'");

    const std::string cropped = "[net]\nwidth=32\nheight=32\nchannels=3\ncrop=1\n";
    SA_EXPECT_FAILURE(translate_darknet_text(cropped, DarknetNormalisation::Separate),
                      StatusCode::Unsupported, "line 5", "'crop'");
}

/** @brief Tests the values this importer refuses although the key is known. */
TEST(Small_DarknetModel, UntranslatedValuesAreRefusedNamingTheirLine)
{
    const std::string boxed = "[net]\nwidth=32\nheight=32\nchannels=3\nletter_box=1\n";
    SA_EXPECT_FAILURE(translate_darknet_text(boxed, DarknetNormalisation::Separate),
                      StatusCode::Unsupported, "line 5", "letter_box=1");

    const std::string scaled = std::string(K_SMALL_NETWORK).append("[upsample]\nscale=2\n");
    SA_EXPECT_FAILURE(translate_darknet_text(scaled, DarknetNormalisation::Separate),
                      StatusCode::Unsupported, "line 11", "scale 2");

    const std::string activated =
        "[net]\nwidth=32\nheight=32\nchannels=3\n[convolutional]\nactivation=relu\n";
    SA_EXPECT_FAILURE(translate_darknet_text(activated, DarknetNormalisation::Separate),
                      StatusCode::Unsupported, "line 6", "'relu'");
}

/** @brief Tests the refusals of a value that is malformed or does not fit its layer. */
TEST(Small_DarknetModel, ValuesThatDoNotFitAreRefusedNamingTheirLine)
{
    const std::string worded =
        "[net]\nwidth=32\nheight=32\nchannels=3\n[convolutional]\nfilters=many\n";
    SA_EXPECT_FAILURE(translate_darknet_text(worded, DarknetNormalisation::Separate),
                      StatusCode::InvalidDescription, "line 6", "'filters'", "'many'");

    const std::string forward = std::string(K_SMALL_NETWORK).append("[route]\nlayers=1\n");
    SA_EXPECT_FAILURE(translate_darknet_text(forward, DarknetNormalisation::Separate),
                      StatusCode::InvalidDescription, "line 12", "names layer 1");

    const std::string narrow =
        "[net]\nwidth=32\nheight=32\nchannels=3\n"
        "[convolutional]\nfilters=12\nactivation=linear\n"
        "[yolo]\nmask=0,1,2\nanchors=4,8, 8,16, 16,24\nclasses=1\n";
    SA_EXPECT_FAILURE(translate_darknet_text(narrow, DarknetNormalisation::Separate),
                      StatusCode::InvalidDescription, "line 8", "reads 18 channels", "has 12");

    const std::string masked = std::string(K_SMALL_NETWORK).append(
        "[yolo]\nmask=1,2,3\nanchors=4,8, 8,16, 16,24\nclasses=1\n");
    SA_EXPECT_FAILURE(translate_darknet_text(masked, DarknetNormalisation::Separate),
                      StatusCode::InvalidDescription, "line 12", "names anchor 3");

    SA_EXPECT_FAILURE(translate_darknet_text(K_SMALL_NETWORK, DarknetNormalisation::Separate),
                      StatusCode::InvalidDescription, "no [yolo] section");
}

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
        return attributes.at(key, "test").as_int("test");
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
TEST(Small_DarknetRouteGroups, UnequalSplitsAndGroupsOutOfRangeAreRefused)
{
    SA_EXPECT_FAILURE(
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\ngroups=3\ngroup_id=0\n")
                                   .append(K_HEAD_TAIL)),
        StatusCode::InvalidDescription, "line 12", "groups 3", "64 channels");

    SA_EXPECT_FAILURE(
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\ngroups=2\ngroup_id=2\n")
                                   .append(K_HEAD_TAIL)),
        StatusCode::InvalidDescription, "line 11", "group_id 2");

    SA_EXPECT_FAILURE(
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\ngroups=0\ngroup_id=0\n")
                                   .append(K_HEAD_TAIL)),
        StatusCode::InvalidDescription, "line 11", "groups 0");

    SA_EXPECT_FAILURE(
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\ngroup_id=1\n")
                                   .append(K_HEAD_TAIL)),
        StatusCode::InvalidDescription, "line 11", "group_id 1");
}

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

TEST(Small_DarknetShortcut, NegativeFromAddsTheLayerBeforeAndTheNamedLayer)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-2\n").append(K_HEAD_TAIL);

    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_darknet_text(text));

    ASSERT_EQ(model.spec.layers.size(), 4u);
    EXPECT_EQ(model.spec.layers[2].type, "add");
    EXPECT_TRUE(model.spec.layers[2].attributes.keys().empty());
    EXPECT_EQ(sources_of_row(model, 2), (std::vector<std::int64_t>{1, 0}));
}

TEST(Small_DarknetShortcut, AbsoluteFromAddsTheSameTwoRows)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=0\nactivation=linear\n")
                                 .append(K_HEAD_TAIL);

    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_darknet_text(text));

    ASSERT_EQ(model.spec.layers.size(), 4u);
    EXPECT_EQ(model.spec.layers[2].type, "add");
    EXPECT_EQ(sources_of_row(model, 2), (std::vector<std::int64_t>{1, 0}));
}

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
