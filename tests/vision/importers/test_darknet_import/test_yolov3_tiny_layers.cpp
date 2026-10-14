/**************************************************************************/
/* test_yolov3_tiny_layers.cpp                                            */
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
 * @file test_yolov3_tiny_layers.cpp
 * @brief Tests every layer of an imported Darknet model against the Darknet dump of its output.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/darknet_reference/darknet_reference.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/darknet_forward_cases/darknet_forward.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    using SaTest::DarknetForwardRequest;
    using SaTest::DarknetForwardRun;
    using SaTest::DarknetTensor;
    using SaTest::find_output;
    using SaTest::within_tolerance;
    using Vision::Importers::DarknetConfig;
    using Vision::Importers::DarknetHead;
    using Vision::Importers::DarknetModel;
    using Vision::Importers::DarknetNormalisation;

    /** @brief Names one model that has a dump of every layer and the folder that holds it. */
    struct FullBodyCase
    {
        std::string_view stem;      /**< Names `<stem>.cfg` and `<stem>.weights`. */
        std::string_view directory; /**< The folder of the dumps under `references`. */
    };

    /** @brief Names one layer of a file and the dump REPORT section 14 holds for it. */
    struct BodyLayer
    {
        std::int64_t index; /**< The index of the layer among the sections of the file. */
        std::string type;   /**< The section type the file gives the layer. */
        std::string dump;   /**< The reference name of the layer's output. */
        bool is_head;       /**< Whether the layer is a [yolo] section, not compared here. */
    };

    /** @brief Lists every model whose layers are compared one by one. */
    constexpr FullBodyCase K_FULL_BODY_CASES[] = {
        {"yolov3-tiny", "yolov3_tiny"},
        {"yolov7-tiny", "yolov7_tiny_layers_plain_pool"},
    };

    /** @brief Holds the row of K_FULL_BODY_CASES whose sections K_YOLOV3_TINY_SECTIONS lists. */
    constexpr const FullBodyCase& K_YOLOV3_TINY = K_FULL_BODY_CASES[0];

    /** @brief Lists the type of each of the 24 sections of yolov3-tiny.cfg, in file order. */
    constexpr std::string_view K_YOLOV3_TINY_SECTIONS[] = {
        "convolutional", "maxpool",       "convolutional", "maxpool",       "convolutional",
        "maxpool",       "convolutional", "maxpool",       "convolutional", "maxpool",
        "convolutional", "maxpool",       "convolutional", "convolutional", "convolutional",
        "convolutional", "yolo",          "route",         "convolutional", "upsample",
        "route",         "convolutional", "convolutional", "yolo",
    };

    /** @brief Holds the absolute bound on a layer value (SPEC 6.3, the head tolerance). */
    constexpr double K_LAYER_ABSOLUTE = 1e-3;

    /** @brief Holds the relative bound on a layer value (SPEC 6.3, the head tolerance). */
    constexpr double K_LAYER_RELATIVE = 1e-4;

    /** @brief Holds the type of the section whose output is the host computation of SPEC 4.4. */
    constexpr std::string_view K_HEAD_TYPE = "yolo";

    /** @brief Holds the name of the reference of the network input. */
    constexpr std::string_view K_INPUT = SaTest::K_DARKNET_INPUT;

    /** @brief Prints the folder of @p body where the test runner names a parameter. */
    void PrintTo(const FullBodyCase& body, std::ostream* stream)
    {
        *stream << body.directory;
    }

    /**
     * @brief Lists the layers of @p config in file order, each with the name of its dump.
     *
     * A [yolo] layer adds no row; the forward test of its model compares its output with the
     * `layer_<k>_yolo` dump.
     */
    std::vector<BodyLayer> list_layers(const DarknetConfig& config)
    {
        std::vector<BodyLayer> layers;
        layers.reserve(config.layers.size());
        for (std::size_t i = 0; i < config.layers.size(); ++i)
        {
            const std::int64_t index = static_cast<std::int64_t>(i);
            const std::string& type = config.layers[i].type;
            layers.push_back(
                {index, type, SaTest::name_darknet_layer(index, type), type == K_HEAD_TYPE});
        }

        return layers;
    }

    /** @brief Lists the index of every layer of @p layers that is not a [yolo] section. */
    std::vector<std::int64_t> list_compared_layers(const std::vector<BodyLayer>& layers)
    {
        std::vector<std::int64_t> indices;
        for (const BodyLayer& layer : layers)
        {
            if (!layer.is_head)
                indices.push_back(layer.index);
        }

        return indices;
    }

    /** @brief Lists the index of every [yolo] section of @p layers, in file order. */
    std::vector<std::int64_t> list_head_layers(const std::vector<BodyLayer>& layers)
    {
        std::vector<std::int64_t> indices;
        for (const BodyLayer& layer : layers)
        {
            if (layer.is_head)
                indices.push_back(layer.index);
        }

        return indices;
    }

    /** @brief Lists the layer of every head of @p model, in file order. */
    std::vector<std::int64_t> list_translated_heads(const DarknetModel& model)
    {
        std::vector<std::int64_t> indices;
        for (const DarknetHead& head : model.heads)
            indices.push_back(head.layer);

        return indices;
    }

    /** @brief Appends @p layer to @p reads unless @p reads already holds it. */
    void append_once(std::vector<std::int64_t>& reads, std::int64_t layer)
    {
        for (const std::int64_t held : reads)
        {
            if (held == layer)
                return;
        }

        reads.push_back(layer);
    }

    /**
     * @brief Appends to @p reads the layers whose values row @p row of @p model reads.
     *
     * A source row that @p layer_of_row lacks was added by the reading layer itself, so its own
     * sources are followed.
     * @param layer_of_row The first layer that holds each row, over the layers before the reader.
     */
    void collect_read_layers(const DarknetModel& model,
                             const std::map<std::int64_t, std::int64_t>& layer_of_row,
                             std::int64_t row, std::vector<std::int64_t>& reads)
    {
        for (const std::int64_t source : SaTest::sources_of_row(model, row))
        {
            const auto held = layer_of_row.find(source);
            if (source == Config::K_MODEL_INPUT)
                append_once(reads, Config::K_MODEL_INPUT);
            else if (held != layer_of_row.end())
                append_once(reads, held->second);
            else
                collect_read_layers(model, layer_of_row, source, reads);
        }
    }

    /**
     * @brief Lists the layers whose values layer @p layer of @p model reads, in reading order.
     *
     * A layer that added no row, such as a one-layer route, reads the layer that holds its row.
     * @return The layer indices; Config::K_MODEL_INPUT stands for the network input.
     * @pre @p layer is a layer of the model.
     */
    std::vector<std::int64_t> list_read_layers(const DarknetModel& model, std::int64_t layer)
    {
        std::map<std::int64_t, std::int64_t> layer_of_row;
        for (std::int64_t earlier = 0; earlier < layer; ++earlier)
            layer_of_row.emplace(model.row_of_layer[static_cast<std::size_t>(earlier)], earlier);

        const std::int64_t row = model.row_of_layer[static_cast<std::size_t>(layer)];
        const auto held = layer_of_row.find(row);
        if (held != layer_of_row.end())
            return {held->second};

        std::vector<std::int64_t> reads;
        collect_read_layers(model, layer_of_row, row, reads);
        return reads;
    }

    /** @brief Returns the index and the type of layer @p index, or names the network input. */
    std::string name_layer(const std::vector<BodyLayer>& layers, std::int64_t index)
    {
        if (index == Config::K_MODEL_INPUT)
            return "the network input";

        return "layer " + std::to_string(index) + " [" +
               layers[static_cast<std::size_t>(index)].type + "]";
    }

    /** @brief Returns one sentence naming @p layer and every layer of @p model it reads. */
    std::string name_reads(const DarknetModel& model, const std::vector<BodyLayer>& layers,
                           const BodyLayer& layer)
    {
        std::string names;
        for (const std::int64_t read : list_read_layers(model, layer.index))
            names += (names.empty() ? "" : ", ") + name_layer(layers, read);

        return name_layer(layers, layer.index) + " reads " + names;
    }

    /**
     * @brief Returns what a measurement of @p layer is printed under: the model, the layer's
     *        index and type, and the extents of its dump as `H x W x C`.
     */
    std::string name_measurement(const FullBodyCase& body, const std::vector<BodyLayer>& layers,
                                 const BodyLayer& layer, const std::vector<std::int64_t>& extents)
    {
        std::string shape;
        for (const std::int64_t extent : extents)
            shape += (shape.empty() ? "" : " x ") + std::to_string(extent);

        return std::string(body.stem) + " " + name_layer(layers, layer.index) + ", " + shape;
    }

    /** @brief Runs on an engine over one model, and skips when a file of that model is missing. */
    class FullBodyTest : public SaTest::EngineTest
    {
        protected:
            /**
             * @brief Loads the cfg of @p body into config, or skips naming the model and the
             *        first of its files that is missing.
             */
            void load_case(const FullBodyCase& body)
            {
                std::vector<std::string> names{std::string(K_INPUT)};
                const std::string unread =
                    SaTest::find_missing_darknet_file(body.stem, body.directory, names);
                if (!unread.empty())
                    GTEST_SKIP() << body.stem << ": " << unread;

                SA_ASSERT_OK_AND_ASSIGN(config, SaTest::load_darknet_config_file(body.stem));
                for (const BodyLayer& layer : list_layers(config))
                {
                    if (!layer.is_head)
                        names.push_back(layer.dump);
                }

                const std::string missing =
                    SaTest::find_missing_darknet_file(body.stem, body.directory, names);
                if (!missing.empty())
                    GTEST_SKIP() << body.stem << ": " << missing;
            }

            /** @brief Holds the sections of the model's cfg. */
            DarknetConfig config;
    };

    /** @brief Runs over yolov3-tiny, the model whose sections this file lists. */
    class Integration_DarknetYolov3TinyLayers : public FullBodyTest
    {
        protected:
            /** @brief Builds the engine, then loads yolov3-tiny or skips. */
            void SetUp() override
            {
                FullBodyTest::SetUp();
                load_case(K_YOLOV3_TINY);
            }
    };

    /** @brief Runs once per row of K_FULL_BODY_CASES. */
    class Integration_DarknetLayers : public FullBodyTest,
                                      public ::testing::WithParamInterface<FullBodyCase>
    {
        protected:
            /** @brief Builds the engine, then loads the model of the parameter or skips. */
            void SetUp() override
            {
                FullBodyTest::SetUp();
                load_case(GetParam());
            }
    };

} // namespace

/**
 * @brief Tests that the cfg has the 24 sections REPORT section 14 names, in that order, that
 *        its [yolo] sections are those of the parity table, that a one-layer route takes no
 *        row of its own, and that the layers a layer reads are those the cfg gives it.
 */
TEST_F(Integration_DarknetYolov3TinyLayers, TheSectionsAndRowsAreThoseOfTheDumps)
{
    const std::size_t count = std::size(K_YOLOV3_TINY_SECTIONS);
    const std::vector<BodyLayer> layers = list_layers(config);
    ASSERT_EQ(layers.size(), count);
    for (std::size_t i = 0; i < count; ++i)
    {
        SCOPED_TRACE(layers[i].dump);
        EXPECT_EQ(layers[i].type, std::string(K_YOLOV3_TINY_SECTIONS[i]));
    }
    EXPECT_EQ(list_head_layers(layers),
              SaTest::find_darknet_parity_model(K_YOLOV3_TINY.stem).yolo_layers);

    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetModel model,
        SaTest::translate_darknet_file(K_YOLOV3_TINY.stem, DarknetNormalisation::Folded));
    ASSERT_EQ(model.row_of_layer.size(), count);
    EXPECT_EQ(list_translated_heads(model), list_head_layers(layers));
    // SPEC 2.1: a one-layer route has no row; layer 17 reads the row of layer 13.
    EXPECT_EQ(model.row_of_layer[17], model.row_of_layer[13]);
    EXPECT_EQ(list_read_layers(model, 0), (std::vector<std::int64_t>{Config::K_MODEL_INPUT}));
    EXPECT_EQ(list_read_layers(model, 17), (std::vector<std::int64_t>{13}));
    EXPECT_EQ(list_read_layers(model, 20), (std::vector<std::int64_t>{19, 8}));
}

/**
 * @brief Tests that the output of every non-[yolo] layer of the folded model equals its Darknet
 *        dump within 1e-3 absolute plus 1e-4 relative.
 *
 * Every layer prints its index, type, extents, largest difference and bound, in file order.
 * The failure names the first layer outside the bound and the layers that layer reads.
 */
TEST_P(Integration_DarknetLayers, EveryLayerMatchesItsDumpWithinTolerance)
{
    const FullBodyCase& body = GetParam();
    const std::vector<BodyLayer> layers = list_layers(config);

    DarknetForwardRequest request;
    request.stem = body.stem;
    request.run = body.directory;
    request.normalisation = DarknetNormalisation::Folded;
    request.observed_layers = list_compared_layers(layers);
    SA_ASSERT_OK_AND_ASSIGN(const DarknetForwardRun run,
                            SaTest::run_darknet_forward(*engine, request));
    ASSERT_EQ(run.model.row_of_layer.size(), layers.size());
    ASSERT_EQ(list_translated_heads(run.model), list_head_layers(layers));

    std::string first_outside;
    std::size_t later_outside = 0;
    for (const BodyLayer& layer : layers)
    {
        if (layer.is_head)
            continue;

        SCOPED_TRACE(layer.dump);
        SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor reference,
                                SaTest::load_darknet_reference(body.directory, layer.dump));
        const std::vector<float>* const ours =
            find_output(run, SaTest::name_observed_layer(layer.index));
        ASSERT_NE(ours, nullptr);

        const ::testing::AssertionResult result =
            within_tolerance(name_measurement(body, layers, layer, reference.extents), *ours,
                             reference.values, K_LAYER_ABSOLUTE, K_LAYER_RELATIVE);
        if (result)
            continue;

        if (first_outside.empty())
            first_outside =
                std::string(result.message()) + "\n" + name_reads(run.model, layers, layer);
        else
            ++later_outside;
    }

    EXPECT_TRUE(first_outside.empty())
        << "first layer outside tolerance: " << first_outside << "\n"
        << later_outside << " later layers are outside it too";
}

INSTANTIATE_TEST_SUITE_P(, Integration_DarknetLayers, ::testing::ValuesIn(K_FULL_BODY_CASES),
                         [](const ::testing::TestParamInfo<FullBodyCase>& info)
                         {
                             return std::string(info.param.directory);
                         });
