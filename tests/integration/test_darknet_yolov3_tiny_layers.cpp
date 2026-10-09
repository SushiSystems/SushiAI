/**************************************************************************/
/* test_darknet_yolov3_tiny_layers.cpp                                    */
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
 * @file test_darknet_yolov3_tiny_layers.cpp
 * @brief Tests every layer of the imported YOLOv3-tiny against the Darknet dump of its output.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_fold.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>
#include <common/darknet_reference.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    using SaTest::DarknetFile;
    using SaTest::DarknetReference;
    using SaTest::DarknetTensor;
    using Vision::Importers::DarknetConfig;
    using Vision::Importers::DarknetModel;
    using Vision::Importers::DarknetNormalisation;
    using Vision::Importers::DarknetTranslation;
    using Vision::Importers::DarknetWeights;

    /** @brief Names one layer of yolov3-tiny.cfg and the dump REPORT section 14 holds for it. */
    struct LayerDump
    {
        std::int64_t layer;    /**< The index of the layer among the sections of the file. */
        std::string_view type; /**< The section type the file gives the layer. */
        std::string_view name; /**< The reference name of the layer's output. */
        bool is_yolo;          /**< Whether the layer is a [yolo] section, not compared here. */
    };

    /** @brief Holds the published output of one layer, by layer index. */
    using LayerOutputs = std::map<std::int64_t, std::vector<float>>;

    /** @brief Holds a translated model and the output of every layer the test compares. */
    struct LayerRun
    {
        DarknetModel model; /**< The translation of yolov3-tiny.cfg, in the folded form. */
        LayerOutputs outputs; /**< The output of each non-[yolo] layer, row-major HWC. */
    };

    /** @brief Holds the largest difference of a comparison and how many values fail it. */
    struct Gap
    {
        double largest = 0.0;    /**< The largest absolute difference. */
        std::size_t at = 0;      /**< The element that carries it. */
        std::size_t outside = 0; /**< Elements beyond the absolute plus relative bound. */
    };

    /** @brief Holds the number of sections of yolov3-tiny.cfg. */
    constexpr std::size_t K_LAYER_COUNT = 24;

    /** @brief Holds the absolute bound on a layer value (SPEC 6.3, the head tolerance). */
    constexpr double K_LAYER_ABSOLUTE = 1e-3;

    /** @brief Holds the relative bound on a layer value (SPEC 6.3, the head tolerance). */
    constexpr double K_LAYER_RELATIVE = 1e-4;

    /** @brief Holds the file name of the model's sections in the data directory. */
    constexpr std::string_view K_CFG_FILE = "yolov3-tiny.cfg";

    /** @brief Holds the file name of the model's weights in the data directory. */
    constexpr std::string_view K_WEIGHTS_FILE = "yolov3-tiny.weights";

    /** @brief Holds the reference directory of the run whose dumps are compared. */
    constexpr std::string_view K_RUN = "yolov3_tiny";

    /** @brief Holds the name of the reference of the network input. */
    constexpr std::string_view K_INPUT = "input";

    /**
     * @brief Lists the 24 layers of yolov3-tiny.cfg in file order, as REPORT section 14 names
     *        their dumps.
     *
     * A [yolo] layer adds no row; its output is the host computation of SPEC 4.4, which
     * test_darknet_yolov3_tiny_forward.cpp compares with the `layer_<k>_yolo` dump.
     */
    constexpr LayerDump K_LAYERS[] = {
        {0, "convolutional", "layer_000_convolutional", false},
        {1, "maxpool", "layer_001_maxpool", false},
        {2, "convolutional", "layer_002_convolutional", false},
        {3, "maxpool", "layer_003_maxpool", false},
        {4, "convolutional", "layer_004_convolutional", false},
        {5, "maxpool", "layer_005_maxpool", false},
        {6, "convolutional", "layer_006_convolutional", false},
        {7, "maxpool", "layer_007_maxpool", false},
        {8, "convolutional", "layer_008_convolutional", false},
        {9, "maxpool", "layer_009_maxpool", false},
        {10, "convolutional", "layer_010_convolutional", false},
        {11, "maxpool", "layer_011_maxpool", false},
        {12, "convolutional", "layer_012_convolutional", false},
        {13, "convolutional", "layer_013_convolutional", false},
        {14, "convolutional", "layer_014_convolutional", false},
        {15, "convolutional", "layer_015_convolutional", false},
        {16, "yolo", "layer_016_yolo", true},
        {17, "route", "layer_017_route", false},
        {18, "convolutional", "layer_018_convolutional", false},
        {19, "upsample", "layer_019_upsample", false},
        {20, "route", "layer_020_route", false},
        {21, "convolutional", "layer_021_convolutional", false},
        {22, "convolutional", "layer_022_convolutional", false},
        {23, "yolo", "layer_023_yolo", true},
    };

    /** @brief Returns the name under which a run publishes the output of @p layer. */
    std::string layer_output_name(std::int64_t layer)
    {
        return "layer." + std::to_string(layer);
    }

    /** @brief Returns why the first file this suite reads is missing, or an empty string. */
    std::string first_missing_file()
    {
        for (const std::string_view name : {K_CFG_FILE, K_WEIGHTS_FILE})
        {
            const DarknetFile file = SaTest::find_darknet_file(name);
            if (!file.found)
                return file.reason;
        }

        const DarknetReference input = SaTest::find_darknet_reference(K_RUN, K_INPUT);
        if (!input.found)
            return input.reason;

        for (const LayerDump& layer : K_LAYERS)
        {
            if (layer.is_yolo)
                continue;

            const DarknetReference reference = SaTest::find_darknet_reference(K_RUN, layer.name);
            if (!reference.found)
                return reference.reason;
        }

        return std::string();
    }

    /** @brief Runs on an engine, and skips every case when a file of the suite is missing. */
    class Integration_DarknetYolov3TinyLayers : public SaTest::EngineTest
    {
        protected:
            /** @brief Builds the engine, then skips naming the first missing file. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                const std::string missing = first_missing_file();
                if (!missing.empty())
                    GTEST_SKIP() << missing;
            }
    };

    /** @brief Loads the reference tensor @p name of the run K_RUN. */
    Result<DarknetTensor> load_reference(std::string_view name)
    {
        return SaTest::load_darknet_tensor(SaTest::find_darknet_reference(K_RUN, name));
    }

    /** @brief Translates yolov3-tiny.cfg in the folded form and returns the translation. */
    Result<DarknetModel> translate_model()
    {
        const DarknetFile file = SaTest::find_darknet_file(K_CFG_FILE);
        SA_FAIL_IF(!file.found, StatusCode::FileNotFound, "{}", file.reason);
        SA_TRY_ASSIGN(const DarknetConfig config,
                      Vision::Importers::load_darknet_config(file.path));

        DarknetTranslation translation;
        translation.normalisation = DarknetNormalisation::Folded;
        return Vision::Importers::translate_darknet_model(config, translation);
    }

    /**
     * @brief Reads the file's weights for @p model with every normalised convolution folded.
     * @return The weights, or the first failure of the read.
     */
    Result<DarknetWeights> read_folded_weights(const DarknetModel& model)
    {
        const DarknetFile file = SaTest::find_darknet_file(K_WEIGHTS_FILE);
        SA_FAIL_IF(!file.found, StatusCode::FileNotFound, "{}", file.reason);
        SA_TRY_ASSIGN(DarknetWeights weights,
                      Vision::Importers::read_darknet_weights(file.path, model.convolutions));

        for (std::size_t i = 0; i < model.convolutions.size(); ++i)
        {
            if (model.convolutions[i].normalised)
                Vision::Importers::fold_darknet_normalisation(weights.convolutions[i]);
        }

        return Result<DarknetWeights>(std::move(weights));
    }

    /**
     * @brief Runs one evaluation step of the folded model over @p input and publishes the
     *        output of every non-[yolo] layer as a named model output.
     *
     * Each layer's output is the row that holds it in the graph (DarknetModel::row_of_layer).
     * The outputs are added to a copy of the spec, not to the translation.
     */
    Result<LayerRun> run_layers(SushiBLAS::Engine& engine, const DarknetTensor& input)
    {
        LayerRun run;
        SA_TRY_ASSIGN(run.model, translate_model());
        SA_FAIL_IF(run.model.row_of_layer.size() != K_LAYER_COUNT, StatusCode::Incompatible,
                   "the translation has {} rows of layer, not {}", run.model.row_of_layer.size(),
                   K_LAYER_COUNT);
        SA_TRY_ASSIGN(const DarknetWeights weights, read_folded_weights(run.model));

        Config::ModelSpec spec = run.model.spec;
        for (const LayerDump& layer : K_LAYERS)
        {
            if (layer.is_yolo)
                continue;

            Config::OutputSpec published;
            published.node = run.model.row_of_layer[static_cast<std::size_t>(layer.layer)];
            published.port = std::string(NN::K_SINGLE_PORT_NAME);
            published.name = layer_output_name(layer.layer);
            spec.outputs.push_back(published);
        }

        const Shape& sample = spec.input_shape;
        SA_FAIL_IF(sample.rank() != 3 || input.extents.size() != 3 ||
                       input.extents[0] != sample[0] || input.extents[1] != sample[1] ||
                       input.extents[2] != sample[2],
                   StatusCode::Incompatible, "the reference input does not have the shape {}",
                   sample.to_string());
        const Shape batch_shape{1, sample[0], sample[1], sample[2]};

        NN::NodeRegistry registry;
        SA_TRY(NN::register_layers(registry));
        SA_TRY_ASSIGN(NN::ModelGraph graph, NN::build_model(registry, spec));
        SA_TRY_ASSIGN(const Train::TracedStep traced,
                      Train::build_evaluation_graph(graph, batch_shape));
        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, graph));
        SA_TRY(Vision::Importers::store_darknet_weights(weights, run.model.convolutions,
                                                        run.model.targets, state));

        Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                      Train::EvaluationStep::make(engine, traced, state, pool));
        SA_TRY_ASSIGN(SushiBLAS::Tensor image, allocate_tensor(engine, batch_shape));
        std::memcpy(image.data_as<float>(), input.values.data(),
                    input.values.size() * sizeof(float));
        SA_TRY(step->run(image));

        for (const LayerDump& layer : K_LAYERS)
        {
            if (layer.is_yolo)
                continue;

            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor,
                          step->output(layer_output_name(layer.layer)));
            const float* const values = tensor->data_as<float>();
            run.outputs.emplace(layer.layer,
                                std::vector<float>(values, values + tensor->num_elements));
        }

        return Result<LayerRun>(std::move(run));
    }

    /** @brief Measures @p actual against @p reference under `absolute + relative * |reference|`. */
    Gap measure_gap(SushiRuntime::span<const float> actual,
                    SushiRuntime::span<const float> reference, double absolute, double relative)
    {
        Gap gap;
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            const double expected = static_cast<double>(reference[i]);
            const double difference = std::fabs(static_cast<double>(actual[i]) - expected);
            if (!(difference <= absolute + relative * std::fabs(expected)))
                ++gap.outside;
            if (!(difference <= gap.largest))
            {
                gap.largest = difference;
                gap.at = i;
            }
        }

        return gap;
    }

    /**
     * @brief Checks @p actual against @p reference and prints the largest difference measured.
     * @return Success, or a failure naming the count outside the bound and the worst element.
     */
    ::testing::AssertionResult within_tolerance(std::string_view what,
                                                SushiRuntime::span<const float> actual,
                                                SushiRuntime::span<const float> reference,
                                                double absolute, double relative)
    {
        if (actual.size() != reference.size())
            return ::testing::AssertionFailure() << what << ": " << actual.size()
                                                 << " values against " << reference.size();

        const Gap gap = measure_gap(actual, reference, absolute, relative);
        std::cout << "[ MEASURED ] " << what << ": largest difference " << gap.largest
                  << " at element " << gap.at << "; bound " << absolute << " + " << relative
                  << " * |reference|\n";
        if (gap.outside == 0)
            return ::testing::AssertionSuccess();

        return ::testing::AssertionFailure()
               << what << ": " << gap.outside << " of " << actual.size()
               << " values are outside " << absolute << " + " << relative
               << " * |reference|; worst at " << gap.at << " (" << actual[gap.at] << " against "
               << reference[gap.at] << ")";
    }

    /** @brief Returns the output of @p layer in @p run, or null when the run lacks it. */
    const std::vector<float>* output_of(const LayerRun& run, std::int64_t layer)
    {
        const auto found = run.outputs.find(layer);
        return found == run.outputs.end() ? nullptr : &found->second;
    }

} // namespace

/**
 * @brief Tests that the cfg has the 24 sections REPORT section 14 names, in that order, and
 *        that a one-layer route takes no row of its own.
 */
TEST_F(Integration_DarknetYolov3TinyLayers, TheSectionsAndRowsAreThoseOfTheDumps)
{
    const DarknetFile file = SaTest::find_darknet_file(K_CFG_FILE);
    ASSERT_TRUE(file.found) << file.reason;
    SA_ASSERT_OK_AND_ASSIGN(const DarknetConfig config,
                            Vision::Importers::load_darknet_config(file.path));
    ASSERT_EQ(config.layers.size(), K_LAYER_COUNT);
    for (const LayerDump& layer : K_LAYERS)
    {
        SCOPED_TRACE(std::string(layer.name));
        EXPECT_EQ(config.layers[static_cast<std::size_t>(layer.layer)].type,
                  std::string(layer.type));
    }

    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_model());
    ASSERT_EQ(model.row_of_layer.size(), K_LAYER_COUNT);
    // SPEC 2.1: a one-layer route has no row; layer 17 reads the row of layer 13.
    EXPECT_EQ(model.row_of_layer[17], model.row_of_layer[13]);
}

/**
 * @brief Tests that the output of every non-[yolo] layer of the folded model equals its Darknet
 *        dump within 1e-3 absolute plus 1e-4 relative, and reports the first layer above it.
 */
TEST_F(Integration_DarknetYolov3TinyLayers, EveryLayerMatchesItsDumpWithinTolerance)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor input, load_reference(K_INPUT));
    SA_ASSERT_OK_AND_ASSIGN(const LayerRun run, run_layers(*engine, input));

    std::string first_above;
    for (const LayerDump& layer : K_LAYERS)
    {
        if (layer.is_yolo)
            continue;

        const std::string what = std::string(layer.name);
        SCOPED_TRACE(what);
        SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor reference, load_reference(layer.name));
        const std::vector<float>* const ours = output_of(run, layer.layer);
        ASSERT_NE(ours, nullptr);

        const ::testing::AssertionResult result = within_tolerance(
            what, *ours, reference.values, K_LAYER_ABSOLUTE, K_LAYER_RELATIVE);
        EXPECT_TRUE(result);
        if (!result && first_above.empty())
            first_above = what;
    }

    EXPECT_TRUE(first_above.empty()) << "first layer above tolerance: " << first_above;
}
