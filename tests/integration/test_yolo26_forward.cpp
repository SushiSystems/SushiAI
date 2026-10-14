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
#include <fstream>
#include <functional>
#include <ios>
#include <map>
#include <memory>
#include <set>
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
#include <SushiAI/core/tensor_shape.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/detect/free/box_coding.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/detections_topk.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/registration.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_Yolo26Forward : public SaTest::EngineTest {};

    using SaTest::ReferenceFile;
    using SaTest::ReferenceTensor;
    using SaTest::near_reference;
    using Vision::Detect::Free::FeatureLevel;
    using Vision::Detect::Free::LevelPlacement;
    using Vision::Detect::PredictionShape;

    /** @brief Holds every published tensor of one forward pass, by its published name. */
    using ForwardOutputs = std::map<std::string, std::vector<float>, std::less<>>;

    /** @brief Names a host decode of `[B, A, 4]` distances about `[A, 2]` anchor points. */
    using BoxDecode = void (*)(SushiRuntime::span<const float>, SushiRuntime::span<const float>,
                               SushiRuntime::span<float>);

    /** @brief Names a load of weights.sushi into the state of one model. */
    using WeightLoad = Status (*)(NN::ModelState&);

    /** @brief Names what an observed row of the model file is to the model. */
    enum class RowUse : std::uint8_t
    {
        HeadLevel, /**< The row is one of the three feature maps the head reads. */
        Inner,     /**< The row is read by other rows only. */
    };

    /** @brief Pairs a row of the model file with the record of forward.sacp of its output. */
    struct ObservedRow
    {
        std::int64_t row;        /**< The index of the row in the model file. */
        std::string_view record; /**< The record name, which is also the published name. */
        RowUse use;              /**< What the row is to the model. */
    };

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

    /** @brief Holds the record prefix of weights.sushi that a ModelState name lacks. */
    constexpr std::string_view K_MODEL_PREFIX = "model.";

    /** @brief Holds the extents of "image" in input.sacp, batch first. */
    const Shape K_INPUT_SHAPE{1, 640, 640, 3};

    /** @brief Holds the rule that turns a record name of weights.sushi into a state name. */
    constexpr IO::RenameRule K_WEIGHT_RULES[] = {{K_MODEL_PREFIX, std::string_view()}};

    /** @brief Lists the rows the forward pass publishes beside the head, as MANIFEST.json has. */
    constexpr ObservedRow K_OBSERVED_ROWS[] = {
        {9, "features.row9", RowUse::Inner},        {10, "features.row10", RowUse::Inner},
        {16, "features.level0", RowUse::HeadLevel}, {19, "features.level1", RowUse::HeadLevel},
        {21, "features.row21", RowUse::Inner},      {22, "features.level2", RowUse::HeadLevel},
    };

    /** @brief Holds the row of the model file that is the c2psa block. */
    constexpr std::int64_t K_C2PSA_ROW = 10;

    /** @brief Holds the levels MANIFEST.json records under "levels" for a 640 px input. */
    const std::vector<FeatureLevel> K_LEVELS{{80, 80, 8.0f}, {40, 40, 16.0f}, {20, 20, 32.0f}};

    /** @brief Holds the joined boxes and probabilities of one copy of the head. */
    struct DecodedBranch
    {
        std::vector<float> boxes;  /**< `[1, A, 4]` in pixels. */
        std::vector<float> scores; /**< `[1, A, C]` after the sigmoid. */
    };

    /** @brief Returns the record K_OBSERVED_ROWS pairs with @p row, or an empty name. */
    constexpr std::string_view record_of(std::int64_t row) noexcept
    {
        for (const ObservedRow& observed : K_OBSERVED_ROWS)
        {
            if (observed.row == row)
                return observed.record;
        }

        return std::string_view();
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

    /** @brief Returns the path of weights.sushi in the fixture directory. */
    std::filesystem::path weights_path()
    {
        return std::filesystem::path(SaTest::fixture_directory()) / "yolo26" / "weights.sushi";
    }

    /** @brief Loads the YOLO26n model file as it is written. */
    Config::ModelSpec load_spec()
    {
        return Config::load_model(repository_root() / "applications" / "examples" / "yolo26" /
                                  "model.json");
    }

    /** @brief Loads the YOLO26n model file and publishes K_OBSERVED_ROWS beside the head. */
    Config::ModelSpec load_spec_with_features()
    {
        Config::ModelSpec spec = load_spec();
        for (const ObservedRow& observed : K_OBSERVED_ROWS)
        {
            Config::OutputSpec output;
            output.node = observed.row;
            output.port = std::string(NN::K_SINGLE_PORT_NAME);
            output.name = std::string(observed.record);
            spec.outputs.push_back(std::move(output));
        }

        return spec;
    }

    /**
     * @brief Builds a graph that holds row @p row of the model file alone, named by its index.
     * @param sample The per-sample shape the row reads.
     * @param published The name the row's single port is published under.
     * @return The graph, or the failure of the registry, of the row's factory or of the graph.
     */
    Result<NN::ModelGraph> build_row_alone(std::int64_t row, const Shape& sample,
                                           const std::string& published)
    {
        SA_TRY_ASSIGN(const NN::NodeRegistry registry, compose_registry());
        const Config::ModelSpec spec = load_spec();
        const Config::LayerSpec& layer = spec.layers.at(static_cast<std::size_t>(row));
        const std::string name = std::to_string(row);

        NN::ModelGraph model(sample);
        SA_TRY_ASSIGN(std::unique_ptr<NN::Node> node,
                      NN::make_node(registry, layer.type, NN::ShapeSpan(&sample, 1),
                                    layer.attributes, name));
        const NN::NodeRef input = model.input();
        SA_TRY_ASSIGN(const NN::NodeRef made,
                      model.add(name, std::move(node),
                                SushiRuntime::span<const NN::NodeRef>(&input, 1)));
        SA_TRY(model.output(published, made));

        return Result<NN::ModelGraph>(std::move(model));
    }

    /** @brief Loads weights.sushi into @p state, which every record must match exactly once. */
    Status import_every_weight(NN::ModelState& state)
    {
        return IO::import_weights(weights_path(), state, K_WEIGHT_RULES);
    }

    /**
     * @brief Loads the records of weights.sushi that rename to an entry of @p state and skips
     *        the others.
     * @return FileNotFound for an unreadable file; Incompatible for a record that repeats or
     *         differs from its entry in dtype or shape, and for an entry left unsupplied.
     */
    Status import_matching_weights(NN::ModelState& state)
    {
        const std::filesystem::path path = weights_path();
        std::ifstream in(path, std::ios::binary);
        SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                   path.string());
        SA_TRY_ASSIGN(IO::CheckpointReader reader, IO::CheckpointReader::open(in));

        std::set<std::string> supplied;
        while (reader.read_count() < reader.count())
        {
            SA_TRY_ASSIGN(const IO::TensorRecord record, reader.next());
            const std::string target = IO::renamed(record.name, K_WEIGHT_RULES);
            const Result<SushiBLAS::Tensor*> found = state.tensor(target);
            if (!found.has_value())
            {
                in.seekg(static_cast<std::streamoff>(record.payload_bytes()), std::ios::cur);
                continue;
            }

            SushiBLAS::Tensor* const tensor = found.value();
            SA_FAIL_IF(record.dtype != tensor->dtype || record.shape != shape_of(*tensor),
                       StatusCode::Incompatible, "'{}' holds {} {}, but the entry '{}' is {} {}",
                       record.name, dtype_name(record.dtype), record.shape.to_string(), target,
                       dtype_name(tensor->dtype), shape_of(*tensor).to_string());
            SA_FAIL_IF(!supplied.insert(target).second, StatusCode::Incompatible,
                       "'{}' supplies '{}' twice", path.string(), target);
            SA_TRY(reader.read_payload(record, tensor->data()));
        }

        for (const NN::StateEntry& entry : state.entries())
        {
            SA_FAIL_IF(supplied.count(entry.name) == 0, StatusCode::Incompatible,
                       "'{}' does not supply '{}'", path.string(), entry.name);
        }

        return Status();
    }

    /**
     * @brief Runs one evaluation step of @p model over @p values with the weights @p load reads.
     * @param batch_shape The extents of the input, batch first.
     * @param values The float32 elements of a @p batch_shape tensor.
     * @return Every published tensor by name, or the first failure of the trace or the run.
     */
    Result<ForwardOutputs> run_model(SushiBLAS::Engine& engine, NN::ModelGraph& model,
                                     const Shape& batch_shape, const std::vector<float>& values,
                                     WeightLoad load)
    {
        SA_TRY_ASSIGN(const Train::TracedStep traced,
                      Train::build_evaluation_graph(model, batch_shape));
        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        SA_TRY(load(state));

        Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                      Train::EvaluationStep::make(engine, traced, state, pool));

        SA_TRY_ASSIGN(SushiBLAS::Tensor input, allocate_tensor(engine, batch_shape));
        SA_FAIL_IF(static_cast<std::size_t>(input.num_elements) != values.size(),
                   StatusCode::Incompatible, "the input holds {} values, but {} needs {}",
                   values.size(), batch_shape.to_string(), input.num_elements);
        std::memcpy(input.data_as<float>(), values.data(), values.size() * sizeof(float));
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
     * @brief Runs one evaluation step of the whole model over @p image with every weight.
     * @param image The float32 elements of a K_INPUT_SHAPE tensor.
     * @return Every published tensor by name, or the first failure of the build or the run.
     */
    Result<ForwardOutputs> run_forward(SushiBLAS::Engine& engine, const std::vector<float>& image)
    {
        SA_TRY_ASSIGN(const NN::NodeRegistry registry, compose_registry());
        SA_TRY_ASSIGN(NN::ModelGraph model,
                      NN::build_model(registry, load_spec_with_features()));

        return run_model(engine, model, K_INPUT_SHAPE, image, &import_every_weight);
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

    /** @brief Expects every row of K_OBSERVED_ROWS that is a @p use to match its record. */
    void expect_rows_match(const ForwardOutputs& ours, ReferenceFile& file, RowUse use)
    {
        for (const ObservedRow& observed : K_OBSERVED_ROWS)
        {
            if (observed.use != use)
                continue;

            const std::string name(observed.record);
            SCOPED_TRACE(name);
            EXPECT_TRUE(matches_reference(ours, file, name, K_ACTIVATION_TOLERANCE));
        }
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
        const std::int64_t anchors = Vision::Detect::Free::count_anchors(K_LEVELS);
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

            Vision::Detect::Free::join_level(found->second,
                                             LevelPlacement{1, anchors, channels, offset, cells},
                                             joined);
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
            static_cast<std::size_t>(Vision::Detect::Free::count_anchors(K_LEVELS));
        std::vector<float> points(anchors * Vision::Detect::Free::K_POINT_VALUES);
        std::vector<float> strides(anchors);
        Vision::Detect::Free::make_anchors(K_LEVELS, points, strides);

        decode(decoded.boxes, points, decoded.boxes);
        Vision::Detect::Free::scale_boxes(decoded.boxes, strides, decoded.boxes);
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
    SA_OPEN_REFERENCE_OR_SKIP(reference, "yolo26/forward.sacp");                       \
    SA_OPEN_REFERENCE_OR_SKIP(reference##_weights, "yolo26/weights.sushi");            \
    SA_OPEN_REFERENCE_OR_SKIP(reference##_input, "yolo26/input.sacp");                 \
    SA_READ_REFERENCE_OR_FAIL(reference##_image, reference##_input, "image");          \
    SA_ASSERT_OK_AND_ASSIGN(const ForwardOutputs* const ours,                          \
                            shared_forward(*engine, reference##_image.floats()))

/** @brief Tests that rows 16, 19 and 22 compute the reference's three feature maps. */
TEST_F(Integration_Yolo26Forward, TheFeatureRowsMatchTheReferenceLevels)
{
    SA_RUN_FORWARD_OR_SKIP(ours, forward);

    expect_rows_match(*ours, forward, RowUse::HeadLevel);
}

/**
 * @brief Tests that rows 9, 10 and 21 of the whole-model pass compute the reference's records
 *        of them, which pins the input and the output of c2psa and the concat row 22 reads.
 */
TEST_F(Integration_Yolo26Forward, TheInnerRowsMatchTheReferenceRows)
{
    SA_RUN_FORWARD_OR_SKIP(ours, forward);

    expect_rows_match(*ours, forward, RowUse::Inner);
}

/**
 * @brief Tests that row 10 built alone, with its own weights and the reference's row 9 as its
 *        input, computes the reference's row 10, which pins c2psa apart from the rows above it.
 */
TEST_F(Integration_Yolo26Forward, C2PSAAloneMapsTheReferenceRow9ToRow10)
{
    SA_OPEN_REFERENCE_OR_SKIP(forward, "yolo26/forward.sacp");
    SA_OPEN_REFERENCE_OR_SKIP(weights, "yolo26/weights.sushi");
    SA_READ_REFERENCE_OR_FAIL(input, forward, record_of(K_C2PSA_ROW - 1));
    ASSERT_EQ(input.shape.rank(), 4);

    const std::string produced(record_of(K_C2PSA_ROW));
    const Shape sample{input.shape[1], input.shape[2], input.shape[3]};
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph block, build_row_alone(K_C2PSA_ROW, sample, produced));
    SA_ASSERT_OK_AND_ASSIGN(const ForwardOutputs ours,
                            run_model(*engine, block, input.shape, input.floats(),
                                      &import_matching_weights));

    EXPECT_TRUE(matches_reference(ours, forward, produced, K_ACTIVATION_TOLERANCE));
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
        {"one2many", &Vision::Detect::Free::decode_xywh},
        {"one2one", &Vision::Detect::Free::decode_xyxy},
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
    ASSERT_TRUE(decode_branch(*ours, "one2one", &Vision::Detect::Free::decode_xyxy, decoded));

    const PredictionShape shape{1, Vision::Detect::Free::count_anchors(K_LEVELS), K_CLASSES};
    const std::int64_t rows =
        Vision::Detect::count_top_detections(shape, Vision::Detect::K_MAX_DETECTIONS);
    std::vector<float> detections(static_cast<std::size_t>(rows) * K_DETECTION_VALUES);
    Vision::Detect::select_top_detections(decoded.boxes, decoded.scores, shape,
                                          Vision::Detect::K_MAX_DETECTIONS, detections);

    EXPECT_TRUE(near_reference(score_column(detections), score_column(reference.floats()),
                               K_PROBABILITY_TOLERANCE));
}
