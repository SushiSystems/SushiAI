/**************************************************************************/
/* test_yolo26_forward.cpp                                                */
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
 * @file test_yolo26_forward.cpp
 * @brief Tests the forward pass of the YOLO26n model file against the reference tensors.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
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
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/vision/detect/anchors.hpp>
#include <SushiAI/vision/detect/box_coding.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/detections_topk.hpp>
#include <SushiAI/vision/detect/level_rows.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/registration.hpp>
#include <common/reference_fixture.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_Yolo26Forward : public SaTest::EngineTest {};

    using SaTest::ReferenceFile;
    using SaTest::ReferenceTensor;
    using SaTest::near_reference;
    using Vision::Detect::FeatureLevel;
    using Vision::Detect::LevelPlacement;
    using Vision::Detect::PredictionShape;

    /** @brief Holds every published tensor of one forward pass, by its published name. */
    using ForwardOutputs = std::map<std::string, std::vector<float>, std::less<>>;

    /** @brief Names a host decode of `[B, A, 4]` distances about `[A, 2]` anchor points. */
    using BoxDecode = void (*)(SushiRuntime::span<const float>, SushiRuntime::span<const float>,
                               SushiRuntime::span<float>);

    /** @brief Holds the absolute bound on a feature map and on a raw head output. */
    constexpr float K_ACTIVATION_TOLERANCE = 1e-4f;

    /** @brief Holds the absolute bound on a decoded box in pixels: two float32 steps at 640. */
    constexpr float K_DECODE_TOLERANCE = 1.3e-4f;

    /** @brief Holds the absolute bound on a class probability. */
    constexpr float K_PROBABILITY_TOLERANCE = 1e-6f;

    /** @brief Holds the class count of the detect row of the model file. */
    constexpr std::int64_t K_CLASSES = 80;

    /** @brief Holds the coordinates of one box. */
    constexpr std::int64_t K_BOX_VALUES = 4;

    /** @brief Holds the values of one detection row: four corners, a score and a class. */
    constexpr std::size_t K_DETECTION_VALUES = 6;

    /** @brief Holds the column of a detection row that carries its score. */
    constexpr std::size_t K_SCORE_COLUMN = 4;

    /** @brief Holds the record prefix of weights.sacp that a ModelState name lacks. */
    constexpr std::string_view K_MODEL_PREFIX = "model.";

    /** @brief Holds the extents of "image" in input.sacp, batch first. */
    const Shape K_INPUT_SHAPE{1, 640, 640, 3};

    /** @brief Holds the rows whose outputs the head reads, in level order. */
    const std::vector<std::int64_t> K_FEATURE_ROWS{16, 19, 22};

    /** @brief Holds the levels MANIFEST.json records under "levels" for a 640 px input. */
    const std::vector<FeatureLevel> K_LEVELS{{80, 80, 8.0f}, {40, 40, 16.0f}, {20, 20, 32.0f}};

    /** @brief Holds the joined boxes and probabilities of one copy of the head. */
    struct DecodedBranch
    {
        std::vector<float> boxes;  /**< `[1, A, 4]` in pixels. */
        std::vector<float> scores; /**< `[1, A, C]` after the sigmoid. */
    };

    /** @brief Returns the published name of the feature map of @p level. */
    std::string feature_name(std::size_t level)
    {
        return "features.level" + std::to_string(level);
    }

    /** @brief Returns the published name of @p kind of @p branch at @p level. */
    std::string port_name(std::string_view branch, std::size_t level, std::string_view kind)
    {
        std::string name(branch);
        name += ".level" + std::to_string(level) + ".";
        name += kind;
        return name;
    }

    /** @brief Returns the logistic function of @p logit, computed in double. */
    float sigmoid(float logit)
    {
        return static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(logit))));
    }

    /** @brief Returns the repository root; SA_TEST_FIXTURE_DIR is `<root>/tests/fixtures`. */
    std::filesystem::path repository_root()
    {
        return std::filesystem::path(SaTest::fixture_directory()).parent_path().parent_path();
    }

    /** @brief Builds a registry from NN::register_layers, then Vision::register_nodes. */
    Result<NN::NodeRegistry> compose_registry()
    {
        NN::NodeRegistry registry;
        SA_TRY(NN::register_layers(registry));
        SA_TRY(Vision::register_nodes(registry));

        return Result<NN::NodeRegistry>(std::move(registry));
    }

    /** @brief Loads the YOLO26n model file and publishes K_FEATURE_ROWS beside the head. */
    Config::ModelSpec load_spec_with_features()
    {
        Config::ModelSpec spec = Config::load_model(repository_root() / "applications" /
                                                    "examples" / "yolo26" / "model.json");
        for (std::size_t level = 0; level < K_FEATURE_ROWS.size(); ++level)
        {
            Config::OutputSpec output;
            output.node = K_FEATURE_ROWS[level];
            output.port = std::string(NN::K_SINGLE_PORT_NAME);
            output.name = feature_name(level);
            spec.outputs.push_back(std::move(output));
        }

        return spec;
    }

    /**
     * @brief Runs one evaluation step of the model over @p image with the reference weights.
     * @param image The float32 elements of a K_INPUT_SHAPE tensor.
     * @return Every published tensor by name, or the first failure of the build or the run.
     */
    Result<ForwardOutputs> run_forward(SushiBLAS::Engine& engine, const std::vector<float>& image)
    {
        SA_TRY_ASSIGN(const NN::NodeRegistry registry, compose_registry());
        const Config::ModelSpec spec = load_spec_with_features();

        SA_TRY_ASSIGN(NN::ModelGraph model, NN::build_model(registry, spec));
        SA_TRY_ASSIGN(const Train::TracedStep traced,
                      Train::build_evaluation_graph(model, K_INPUT_SHAPE));
        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));

        const IO::RenameRule rules[] = {{K_MODEL_PREFIX, ""}};
        const std::filesystem::path weights = std::filesystem::path(SaTest::fixture_directory()) /
                                              SaTest::K_YOLO26_FIXTURE_FOLDER / "weights.sacp";
        SA_TRY(IO::import_weights(weights, state, rules));

        Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                      Train::EvaluationStep::make(engine, traced, state, pool));

        SA_TRY_ASSIGN(SushiBLAS::Tensor input, allocate_tensor(engine, K_INPUT_SHAPE));
        SA_FAIL_IF(static_cast<std::size_t>(input.num_elements) != image.size(),
                   StatusCode::Incompatible, "the image holds {} values, but {} needs {}",
                   image.size(), K_INPUT_SHAPE.to_string(), input.num_elements);
        std::memcpy(input.data_as<float>(), image.data(), image.size() * sizeof(float));
        SA_TRY(step->run(input));

        ForwardOutputs outputs;
        for (const NN::ModelOutputs::Entry& published : traced.outputs.all())
        {
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, step->output(published.first));
            const float* const values = tensor->data_as<float>();
            outputs.emplace(published.first,
                            std::vector<float>(values, values + tensor->num_elements));
        }

        return Result<ForwardOutputs>(std::move(outputs));
    }

    /**
     * @brief Returns the outputs of the one forward pass the cases of this file share.
     * @return The outputs of the first successful run_forward(), or the failure of this call.
     */
    Result<const ForwardOutputs*> shared_forward(SushiBLAS::Engine& engine,
                                                 const std::vector<float>& image)
    {
        static ForwardOutputs outputs;
        if (outputs.empty())
        {
            SA_TRY_ASSIGN(outputs, run_forward(engine, image));
        }

        return Result<const ForwardOutputs*>(&outputs);
    }

    /**
     * @brief Checks the output published as @p name against the record of the same name.
     * @return Success, or a failure naming what is missing or the first and the worst element.
     */
    ::testing::AssertionResult matches_reference(const ForwardOutputs& ours, ReferenceFile& file,
                                                 const std::string& name, float tolerance)
    {
        const auto found = ours.find(name);
        if (found == ours.end())
            return ::testing::AssertionFailure() << "the model publishes no '" << name << "'";

        const ReferenceTensor reference = file.read(name);
        if (!reference.found)
            return ::testing::AssertionFailure() << reference.reason;

        return near_reference(found->second, reference.floats(), tolerance);
    }

    /**
     * @brief Joins the three levels of @p kind of @p branch into `[1, A, channels]`.
     * @param joined Receives the rows, levels in order; resized here.
     * @return Success, or a failure naming the port that is missing or has another size.
     */
    ::testing::AssertionResult join_levels(const ForwardOutputs& ours, std::string_view branch,
                                           std::string_view kind, std::int64_t channels,
                                           std::vector<float>& joined)
    {
        const std::int64_t anchors = Vision::Detect::count_anchors(K_LEVELS);
        joined.assign(static_cast<std::size_t>(anchors * channels), 0.0f);

        std::int64_t offset = 0;
        for (std::size_t level = 0; level < K_LEVELS.size(); ++level)
        {
            const std::string name = port_name(branch, level, kind);
            const auto found = ours.find(name);
            if (found == ours.end())
                return ::testing::AssertionFailure() << "the model publishes no '" << name << "'";

            const std::int64_t cells = K_LEVELS[level].height * K_LEVELS[level].width;
            const std::size_t expected = static_cast<std::size_t>(cells * channels);
            if (found->second.size() != expected)
                return ::testing::AssertionFailure() << "'" << name << "' holds "
                                                     << found->second.size() << " values, not "
                                                     << expected;

            Vision::Detect::join_level(found->second,
                                       LevelPlacement{1, anchors, channels, offset, cells}, joined);
            offset += cells;
        }

        return ::testing::AssertionSuccess();
    }

    /**
     * @brief Decodes the raw outputs of @p branch on the host: boxes in pixels through
     *        @p decode, class probabilities through the sigmoid.
     * @return Success, or the failure of join_levels().
     */
    ::testing::AssertionResult decode_branch(const ForwardOutputs& ours, std::string_view branch,
                                             BoxDecode decode, DecodedBranch& decoded)
    {
        const ::testing::AssertionResult boxes =
            join_levels(ours, branch, "boxes", K_BOX_VALUES, decoded.boxes);
        if (!boxes)
            return boxes;

        const ::testing::AssertionResult scores =
            join_levels(ours, branch, "scores", K_CLASSES, decoded.scores);
        if (!scores)
            return scores;

        const std::size_t anchors =
            static_cast<std::size_t>(Vision::Detect::count_anchors(K_LEVELS));
        std::vector<float> points(anchors * Vision::Detect::K_POINT_VALUES);
        std::vector<float> strides(anchors);
        Vision::Detect::make_anchors(K_LEVELS, points, strides);

        decode(decoded.boxes, points, decoded.boxes);
        Vision::Detect::scale_boxes(decoded.boxes, strides, decoded.boxes);
        for (float& score : decoded.scores)
            score = sigmoid(score);

        return ::testing::AssertionSuccess();
    }

    /** @brief Returns column K_SCORE_COLUMN of `[k, 6]` detection rows. */
    std::vector<float> score_column(const std::vector<float>& detections)
    {
        std::vector<float> scores;
        scores.reserve(detections.size() / K_DETECTION_VALUES);
        for (std::size_t at = 0; at + K_DETECTION_VALUES <= detections.size();
             at += K_DETECTION_VALUES)
            scores.push_back(detections[at + K_SCORE_COLUMN]);

        return scores;
    }

} // namespace

/**
 * @brief Declares @p reference as forward.sacp and @p ours as the shared forward outputs, or
 *        skips the test naming the reference file that is missing.
 * @pre Used as statements in a TEST_F of Integration_Yolo26Forward.
 */
#define SA_RUN_FORWARD_OR_SKIP(ours, reference)                                        \
    SA_OPEN_REFERENCE_OR_SKIP(reference, "forward.sacp");                              \
    SA_OPEN_REFERENCE_OR_SKIP(reference##_weights, "weights.sacp");                    \
    SA_OPEN_REFERENCE_OR_SKIP(reference##_input, "input.sacp");                        \
    SA_READ_REFERENCE_OR_FAIL(reference##_image, reference##_input, "image");          \
    SA_ASSERT_OK_AND_ASSIGN(const ForwardOutputs* const ours,                          \
                            shared_forward(*engine, reference##_image.floats()))

/** @brief Tests that rows 16, 19 and 22 compute the reference's three feature maps. */
TEST_F(Integration_Yolo26Forward, TheFeatureRowsMatchTheReferenceLevels)
{
    SA_RUN_FORWARD_OR_SKIP(ours, forward);

    for (std::size_t level = 0; level < K_FEATURE_ROWS.size(); ++level)
    {
        const std::string name = feature_name(level);
        SCOPED_TRACE(name);
        EXPECT_TRUE(matches_reference(*ours, forward, name, K_ACTIVATION_TOLERANCE));
    }
}

/** @brief Tests that the twelve ports of the head compute the reference's raw outputs. */
TEST_F(Integration_Yolo26Forward, TheTwelvePortsMatchTheReferenceRawOutputs)
{
    SA_RUN_FORWARD_OR_SKIP(ours, forward);

    for (const std::string_view branch : {std::string_view("one2many"),
                                          std::string_view("one2one")})
    {
        for (std::size_t level = 0; level < K_LEVELS.size(); ++level)
        {
            for (const std::string_view kind : {std::string_view("boxes"),
                                                std::string_view("scores")})
            {
                const std::string name = port_name(branch, level, kind);
                SCOPED_TRACE(name);
                EXPECT_TRUE(matches_reference(*ours, forward, name, K_ACTIVATION_TOLERANCE));
            }
        }
    }
}

/**
 * @brief Tests that the host decode of our raw outputs gives the reference's decoded boxes and
 *        probabilities: one-to-many as centre and size, one-to-one as corners.
 */
TEST_F(Integration_Yolo26Forward, TheHostDecodeMatchesTheReferenceDecodes)
{
    SA_RUN_FORWARD_OR_SKIP(ours, forward);

    const std::pair<std::string_view, BoxDecode> branches[] = {
        {"one2many", &Vision::Detect::decode_xywh},
        {"one2one", &Vision::Detect::decode_xyxy},
    };
    for (const std::pair<std::string_view, BoxDecode>& branch : branches)
    {
        SCOPED_TRACE(std::string(branch.first));
        const std::string prefix = std::string(branch.first) + ".decoded.";
        SA_READ_REFERENCE_OR_FAIL(boxes, forward, prefix + "boxes");
        SA_READ_REFERENCE_OR_FAIL(scores, forward, prefix + "scores");

        DecodedBranch decoded;
        ASSERT_TRUE(decode_branch(*ours, branch.first, branch.second, decoded));

        EXPECT_TRUE(near_reference(decoded.boxes, boxes.floats(), K_DECODE_TOLERANCE));
        EXPECT_TRUE(near_reference(decoded.scores, scores.floats(), K_PROBABILITY_TOLERANCE));
    }
}

/**
 * @brief Tests that the top-k selection over our one-to-one decode gives the scores of the
 *        reference detections, best first.
 */
TEST_F(Integration_Yolo26Forward, TheTopScoresMatchTheReferenceDetections)
{
    SA_RUN_FORWARD_OR_SKIP(ours, forward);
    SA_READ_REFERENCE_OR_FAIL(reference, forward, "one2one.detections");

    DecodedBranch decoded;
    ASSERT_TRUE(decode_branch(*ours, "one2one", &Vision::Detect::decode_xyxy, decoded));

    const PredictionShape shape{1, Vision::Detect::count_anchors(K_LEVELS), K_CLASSES};
    const std::int64_t rows =
        Vision::Detect::count_top_detections(shape, Vision::Detect::K_MAX_DETECTIONS);
    std::vector<float> detections(static_cast<std::size_t>(rows) * K_DETECTION_VALUES);
    Vision::Detect::select_top_detections(decoded.boxes, decoded.scores, shape,
                                          Vision::Detect::K_MAX_DETECTIONS, detections);

    EXPECT_TRUE(near_reference(score_column(detections), score_column(reference.floats()),
                               K_PROBABILITY_TOLERANCE));
}
