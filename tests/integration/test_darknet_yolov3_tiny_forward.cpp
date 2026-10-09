/**************************************************************************/
/* test_darknet_yolov3_tiny_forward.cpp                                   */
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
 * @file test_darknet_yolov3_tiny_forward.cpp
 * @brief Tests the forward pass of the imported YOLOv3-tiny against the Darknet reference
 *        tensors: both heads, the [yolo] outputs and the detection rows.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
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
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/vision/detect/anchor/head_decode.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>
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
    using Vision::Detect::Anchor::HeadCoding;
    using Vision::Detect::Anchor::HeadShape;
    using Vision::Importers::DarknetConfig;
    using Vision::Importers::DarknetModel;
    using Vision::Importers::DarknetNormalisation;
    using Vision::Importers::DarknetTranslation;
    using Vision::Importers::DarknetWeights;

    /** @brief Holds every published tensor of one forward pass, by its published name. */
    using ForwardOutputs = std::map<std::string, std::vector<float>, std::less<>>;

    /** @brief Names one head of yolov3-tiny.cfg and the reference tensors of it. */
    struct HeadReference
    {
        std::int64_t layer;      /**< The index of the [yolo] section among the layers. */
        std::int64_t grid;       /**< The rows and the columns of the head's grid. */
        std::string_view output; /**< The name the model publishes the head under. */
        std::string_view input;  /**< The reference of the raw head. */
        std::string_view yolo;   /**< The reference of the [yolo] layer's own output. */
    };

    /** @brief Names one reference run and what REPORT section 6 states of its detections. */
    struct DetectionRun
    {
        std::string_view directory; /**< The folder of the run under `references`. */
        float thresh;               /**< The score threshold of the run. */
        std::int64_t rows;          /**< The detection rows the run kept. */
    };

    /** @brief Holds one detection as both sides are compared. */
    struct DetectionRow
    {
        float box[4];             /**< Centre x, centre y, width, height, as fractions. */
        float score;              /**< Objectness times the class probability. */
        std::int64_t class_index; /**< The class. */
    };

    /** @brief Holds the decoded rows of every head of one image, heads in file order. */
    struct DecodedHeads
    {
        std::vector<float> boxes;  /**< `[rows, 4]` centre and size as fractions. */
        std::vector<float> scores; /**< `[rows, K_CLASSES]` objectness times class. */
        std::int64_t rows = 0;     /**< Rows over both heads. */
    };

    /** @brief Holds the largest difference of a comparison and how many values fail it. */
    struct Gap
    {
        double largest = 0.0;    /**< The largest absolute difference. */
        std::size_t at = 0;      /**< The element that carries it. */
        std::size_t outside = 0; /**< Elements beyond the absolute plus relative bound. */
    };

    /** @brief Holds the absolute bound on a head value of the folded model (SPEC 6.3). */
    constexpr double K_HEAD_ABSOLUTE = 1e-3;

    /** @brief Holds the relative bound on a head value of the folded model (SPEC 6.3). */
    constexpr double K_HEAD_RELATIVE = 1e-4;

    /** @brief Holds the absolute bound on a box entry and a score of a detection (SPEC 6.3). */
    constexpr double K_ROW_ABSOLUTE = 1e-4;

    /** @brief Holds the distance from a threshold below which a score or a pair is named. */
    constexpr double K_THRESHOLD_MARGIN = 1e-6;

    /** @brief Holds the suppression threshold of both runs (REPORT 3.8). */
    constexpr float K_SUPPRESSION_THRESHOLD = 0.45f;

    /** @brief Holds the class count of yolov3-tiny.cfg. */
    constexpr std::int64_t K_CLASSES = 80;

    /** @brief Holds the anchor slots of each head of yolov3-tiny.cfg. */
    constexpr std::int64_t K_SLOTS = 3;

    /** @brief Holds the entries of one anchor slot: four of the box, objectness, classes. */
    constexpr std::int64_t K_SLOT_ENTRIES = 5 + K_CLASSES;

    /** @brief Holds the first of the two size entries of a slot, which carry no logistic. */
    constexpr std::int64_t K_FIRST_SIZE_ENTRY = 2;

    /** @brief Holds the last of the two size entries of a slot. */
    constexpr std::int64_t K_LAST_SIZE_ENTRY = 3;

    /** @brief Holds the side of the network input of yolov3-tiny.cfg in pixels. */
    constexpr std::int64_t K_NETWORK_SIDE = 416;

    /** @brief Holds the columns of a reference detection row. */
    constexpr std::int64_t K_REFERENCE_COLUMNS = 7;

    /** @brief Holds the column of a reference detection row that carries its score. */
    constexpr std::size_t K_REFERENCE_SCORE = 5;

    /** @brief Holds the column of a reference detection row that carries its class. */
    constexpr std::size_t K_REFERENCE_CLASS = 6;

    /** @brief Holds the file name of the model's sections in the data directory. */
    constexpr std::string_view K_CFG_FILE = "yolov3-tiny.cfg";

    /** @brief Holds the file name of the model's weights in the data directory. */
    constexpr std::string_view K_WEIGHTS_FILE = "yolov3-tiny.weights";

    /** @brief Holds the name of the reference of the network input. */
    constexpr std::string_view K_INPUT = "input";

    /** @brief Holds the name of the reference of the detection rows. */
    constexpr std::string_view K_DETECTIONS = "detections";

    /** @brief Lists the two heads as REPORT section 6 names their tensors. */
    constexpr HeadReference K_HEADS[] = {
        {16, 13, "head.0", "layer_015_yolo_input", "layer_016_yolo"},
        {23, 26, "head.1", "layer_022_yolo_input", "layer_023_yolo"},
    };

    /** @brief Lists the two runs with the thresholds and row counts REPORT section 6 states. */
    constexpr DetectionRun K_RUNS[] = {
        {"yolov3_tiny", 0.25f, 0},
        {"yolov3_tiny_thresh_0p01", 0.01f, 16},
    };

    /** @brief Returns why the first file this suite reads is missing, or an empty string. */
    std::string first_missing_file()
    {
        for (const std::string_view name : {K_CFG_FILE, K_WEIGHTS_FILE})
        {
            const DarknetFile file = SaTest::find_darknet_file(name);
            if (!file.found)
                return file.reason;
        }

        for (const DetectionRun& run : K_RUNS)
        {
            std::vector<std::string_view> names{K_INPUT, K_DETECTIONS};
            for (const HeadReference& head : K_HEADS)
            {
                names.push_back(head.input);
                names.push_back(head.yolo);
            }

            for (const std::string_view name : names)
            {
                const DarknetReference reference =
                    SaTest::find_darknet_reference(run.directory, name);
                if (!reference.found)
                    return reference.reason;
            }
        }

        return std::string();
    }

    /** @brief Runs on an engine, and skips every case when a file of the suite is missing. */
    class Integration_DarknetYolov3TinyForward : public SaTest::EngineTest
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

    /** @brief Holds a translated model and what one forward pass of it published. */
    struct ForwardRun
    {
        DarknetModel model;     /**< The translation of yolov3-tiny.cfg. */
        ForwardOutputs outputs; /**< Every published tensor by name. */
    };

    /** @brief Loads the reference tensor @p name of the run in @p directory. */
    Result<DarknetTensor> load_reference(std::string_view directory, std::string_view name)
    {
        return SaTest::load_darknet_tensor(SaTest::find_darknet_reference(directory, name));
    }

    /** @brief Returns the logistic function of @p logit, computed in double. */
    float logistic(float logit)
    {
        return static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(logit))));
    }

    /** @brief Reads the file's weights for @p model, folded when the model is. */
    Result<DarknetWeights> read_weights(const DarknetModel& model,
                                        DarknetNormalisation normalisation)
    {
        const DarknetFile file = SaTest::find_darknet_file(K_WEIGHTS_FILE);
        SA_FAIL_IF(!file.found, StatusCode::FileNotFound, "{}", file.reason);
        SA_TRY_ASSIGN(DarknetWeights weights,
                      Vision::Importers::read_darknet_weights(file.path, model.convolutions));
        if (normalisation == DarknetNormalisation::Separate)
            return Result<DarknetWeights>(std::move(weights));

        for (std::size_t i = 0; i < model.convolutions.size(); ++i)
        {
            if (model.convolutions[i].normalised)
                Vision::Importers::fold_darknet_normalisation(weights.convolutions[i]);
        }

        return Result<DarknetWeights>(std::move(weights));
    }

    /** @brief Translates yolov3-tiny.cfg from the data directory in the form @p normalisation. */
    Result<DarknetModel> translate_model(DarknetNormalisation normalisation)
    {
        const DarknetFile file = SaTest::find_darknet_file(K_CFG_FILE);
        SA_FAIL_IF(!file.found, StatusCode::FileNotFound, "{}", file.reason);
        SA_TRY_ASSIGN(const DarknetConfig config,
                      Vision::Importers::load_darknet_config(file.path));

        DarknetTranslation translation;
        translation.normalisation = normalisation;
        return Vision::Importers::translate_darknet_model(config, translation);
    }

    /**
     * @brief Runs one evaluation step of the imported model over the reference input.
     * @param input The `input` reference, `[H, W, C]`.
     * @return The model and every published tensor, or the first failure of the import, the
     *         build or the run.
     */
    Result<ForwardRun> run_forward(SushiBLAS::Engine& engine, DarknetNormalisation normalisation,
                                   const DarknetTensor& input)
    {
        ForwardRun run;
        SA_TRY_ASSIGN(run.model, translate_model(normalisation));
        SA_TRY_ASSIGN(const DarknetWeights weights, read_weights(run.model, normalisation));

        const Shape& sample = run.model.spec.input_shape;
        SA_FAIL_IF(sample.rank() != 3 || input.extents.size() != 3 ||
                       input.extents[0] != sample[0] || input.extents[1] != sample[1] ||
                       input.extents[2] != sample[2],
                   StatusCode::Incompatible, "the reference input does not have the shape {}",
                   sample.to_string());
        const Shape batch_shape{1, sample[0], sample[1], sample[2]};

        NN::NodeRegistry registry;
        SA_TRY(NN::register_layers(registry));
        SA_TRY_ASSIGN(NN::ModelGraph graph, NN::build_model(registry, run.model.spec));
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

        for (const NN::ModelOutputs::Entry& published : traced.outputs.all())
        {
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor, step->output(published.first));
            const float* const values = tensor->data_as<float>();
            run.outputs.emplace(published.first,
                                std::vector<float>(values, values + tensor->num_elements));
        }

        return Result<ForwardRun>(std::move(run));
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

    /** @brief Returns the output @p name of @p run, or null when the model publishes none. */
    const std::vector<float>* output_of(const ForwardRun& run, std::string_view name)
    {
        const auto found = run.outputs.find(name);
        return found == run.outputs.end() ? nullptr : &found->second;
    }

    /**
     * @brief Returns the [yolo] output SPEC 4.4 states for a logit head of scale 1: the
     *        logistic of every entry of a slot except its two sizes.
     */
    std::vector<float> activate_logit_head(const std::vector<float>& head)
    {
        std::vector<float> activated(head.size());
        for (std::size_t i = 0; i < head.size(); ++i)
        {
            const std::int64_t entry = static_cast<std::int64_t>(i) % K_SLOT_ENTRIES;
            const bool is_size = entry >= K_FIRST_SIZE_ENTRY && entry <= K_LAST_SIZE_ENTRY;
            activated[i] = is_size ? head[i] : logistic(head[i]);
        }

        return activated;
    }

    /**
     * @brief Decodes both heads of @p run through the library, heads in file order.
     * @return Success, or a failure naming the head that is missing or fails validation.
     */
    ::testing::AssertionResult decode_heads(const ForwardRun& run, DecodedHeads& decoded)
    {
        for (std::size_t k = 0; k < std::size(K_HEADS); ++k)
        {
            const std::vector<float>* const head = output_of(run, K_HEADS[k].output);
            if (head == nullptr)
                return ::testing::AssertionFailure() << "no '" << K_HEADS[k].output << "'";

            const std::int64_t index = static_cast<std::int64_t>(k);
            const Vision::Detect::Anchor::HeadOptions& options = run.model.heads[k].options;
            const Status valid =
                Vision::Detect::Anchor::validate_head(options, run.model.layout, index);
            if (!valid.is_ok())
                return ::testing::AssertionFailure() << valid.message();

            const HeadShape shape{1, K_HEADS[k].grid, K_HEADS[k].grid};
            const std::size_t rows = static_cast<std::size_t>(
                Vision::Detect::Anchor::count_head_rows(shape, run.model.layout, index));
            std::vector<float> boxes(rows * 4);
            std::vector<float> scores(rows * static_cast<std::size_t>(K_CLASSES));
            Vision::Detect::Anchor::decode_head(*head, shape, options, run.model.layout, index,
                                                boxes, scores);

            decoded.boxes.insert(decoded.boxes.end(), boxes.begin(), boxes.end());
            decoded.scores.insert(decoded.scores.end(), scores.begin(), scores.end());
            decoded.rows += static_cast<std::int64_t>(rows);
        }

        return ::testing::AssertionSuccess();
    }

    /**
     * @brief Selects the detections of @p decoded as the library does for an imported model:
     *        boxes in pixels, every class of a row, one threshold on the score.
     * @return The kept rows with their boxes back as centre and size in fractions.
     */
    std::vector<DetectionRow> select_rows(const ForwardRun& run, const DecodedHeads& decoded,
                                          float thresh, float suppression)
    {
        const float side = static_cast<float>(K_NETWORK_SIDE);
        std::vector<float> pixels(decoded.boxes);
        for (float& value : pixels)
            value *= side;

        Vision::Detect::NmsConfig config;
        config.confidence_threshold = thresh;
        config.iou_threshold = suppression;
        config.max_detections = decoded.rows * K_CLASSES;
        config.max_candidates = decoded.rows * K_CLASSES;
        config.criterion = run.model.criterion;
        config.distance_exponent = run.model.distance_exponent;

        std::vector<std::vector<float>> detections;
        Vision::Detect::select_nms_detections(
            pixels, decoded.scores, Vision::Detect::PredictionShape{1, decoded.rows, K_CLASSES},
            config, detections);

        std::vector<DetectionRow> rows;
        const std::vector<float>& image = detections.front();
        for (std::size_t at = 0; at + Vision::Detect::K_DETECTION_VALUES <= image.size();
             at += Vision::Detect::K_DETECTION_VALUES)
        {
            const float* const row = image.data() + at;
            DetectionRow kept{};
            kept.box[0] = (row[0] + row[2]) / (2.0f * side);
            kept.box[1] = (row[1] + row[3]) / (2.0f * side);
            kept.box[2] = (row[2] - row[0]) / side;
            kept.box[3] = (row[3] - row[1]) / side;
            kept.score = row[Vision::Detect::K_DETECTION_SCORE];
            kept.class_index = static_cast<std::int64_t>(row[Vision::Detect::K_DETECTION_CLASS]);
            rows.push_back(kept);
        }

        return rows;
    }

    /** @brief Returns the rows of a `[rows, 7]` reference `detections` tensor. */
    std::vector<DetectionRow> reference_rows(const DarknetTensor& detections)
    {
        const std::size_t columns = static_cast<std::size_t>(K_REFERENCE_COLUMNS);
        std::vector<DetectionRow> rows;
        for (std::size_t at = 0; at + columns <= detections.values.size(); at += columns)
        {
            const float* const row = detections.values.data() + at;
            DetectionRow kept{};
            std::copy(row, row + 4, kept.box);
            kept.score = row[K_REFERENCE_SCORE];
            kept.class_index = static_cast<std::int64_t>(row[K_REFERENCE_CLASS]);
            rows.push_back(kept);
        }

        return rows;
    }

    /** @brief Sorts @p rows by class ascending, then by score descending. */
    void sort_rows(std::vector<DetectionRow>& rows)
    {
        std::stable_sort(rows.begin(), rows.end(),
                         [](const DetectionRow& a, const DetectionRow& b)
                         {
                             if (a.class_index != b.class_index)
                                 return a.class_index < b.class_index;

                             return a.score > b.score;
                         });
    }

    /** @brief Returns the IoU of two centre-and-size boxes in double; 0 for an empty union. */
    double measure_iou(const float* a, const float* b)
    {
        const double left = std::max(a[0] - a[2] / 2.0, b[0] - b[2] / 2.0);
        const double right = std::min(a[0] + a[2] / 2.0, b[0] + b[2] / 2.0);
        const double top = std::max(a[1] - a[3] / 2.0, b[1] - b[3] / 2.0);
        const double bottom = std::min(a[1] + a[3] / 2.0, b[1] + b[3] / 2.0);
        const double overlap = std::max(0.0, right - left) * std::max(0.0, bottom - top);
        const double joined = static_cast<double>(a[2]) * a[3] +
                              static_cast<double>(b[2]) * b[3] - overlap;

        return joined > 0.0 ? overlap / joined : 0.0;
    }

    /**
     * @brief Names every score within K_THRESHOLD_MARGIN of @p thresh and every pair of rows
     *        of one class whose IoU is within it of @p suppression.
     * @return One line per score and per pair; empty when no decision is that close.
     */
    std::string name_close_decisions(const DecodedHeads& decoded, float thresh,
                                     float suppression)
    {
        const std::size_t classes = static_cast<std::size_t>(K_CLASSES);
        std::string names;
        std::vector<std::pair<std::size_t, std::size_t>> candidates;
        for (std::size_t at = 0; at < decoded.scores.size(); ++at)
        {
            const double score = static_cast<double>(decoded.scores[at]);
            const double distance = std::fabs(score - static_cast<double>(thresh));
            if (distance <= K_THRESHOLD_MARGIN)
                names += "row " + std::to_string(at / classes) + " class " +
                         std::to_string(at % classes) + " scores " + std::to_string(score) +
                         ", beside the threshold\n";
            if (score > static_cast<double>(thresh) - K_THRESHOLD_MARGIN)
                candidates.emplace_back(at / classes, at % classes);
        }

        for (std::size_t first = 0; first < candidates.size(); ++first)
        {
            for (std::size_t second = first + 1; second < candidates.size(); ++second)
            {
                if (candidates[first].second != candidates[second].second)
                    continue;

                const double iou = measure_iou(decoded.boxes.data() + candidates[first].first * 4,
                                               decoded.boxes.data() + candidates[second].first * 4);
                if (std::fabs(iou - static_cast<double>(suppression)) <= K_THRESHOLD_MARGIN)
                    names += "rows " + std::to_string(candidates[first].first) + " and " +
                             std::to_string(candidates[second].first) + " of class " +
                             std::to_string(candidates[first].second) + " overlap by " +
                             std::to_string(iou) + ", beside the suppression threshold\n";
            }
        }

        return names;
    }

    /** @brief Returns the real number @p key of @p description, or NaN when it has none. */
    float real_of(const Config::Value& description, std::string_view key)
    {
        if (!description.has(key))
            return std::nanf("");

        const Config::Value& value = description.at(key, key);
        const bool is_number =
            value.kind() == Config::Kind::Real || value.kind() == Config::Kind::Int;
        return is_number ? static_cast<float>(value.as_real(key)) : std::nanf("");
    }

} // namespace

/**
 * @brief Tests that both published heads of the folded model equal the raw heads Darknet
 *        dumped, in both runs, within 1e-3 absolute plus 1e-4 relative.
 */
TEST_F(Integration_DarknetYolov3TinyForward, TheFoldedHeadsMatchTheDumpedHeads)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor input,
                            load_reference(K_RUNS[0].directory, K_INPUT));
    SA_ASSERT_OK_AND_ASSIGN(const ForwardRun run,
                            run_forward(*engine, DarknetNormalisation::Folded, input));

    ASSERT_EQ(run.model.heads.size(), std::size(K_HEADS));
    for (const DetectionRun& reference_run : K_RUNS)
    {
        for (std::size_t k = 0; k < std::size(K_HEADS); ++k)
        {
            const HeadReference& head = K_HEADS[k];
            const std::string what =
                std::string(reference_run.directory) + "/" + std::string(head.input);
            SCOPED_TRACE(what);
            EXPECT_EQ(run.model.heads[k].layer, head.layer);
            EXPECT_EQ(run.model.heads[k].output, head.output);

            SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor reference,
                                    load_reference(reference_run.directory, head.input));
            EXPECT_EQ(reference.extents, (std::vector<std::int64_t>{head.grid, head.grid,
                                                                    K_SLOTS * K_SLOT_ENTRIES}));
            const std::vector<float>* const ours = output_of(run, head.output);
            ASSERT_NE(ours, nullptr);
            EXPECT_TRUE(within_tolerance(what, *ours, reference.values, K_HEAD_ABSOLUTE,
                                         K_HEAD_RELATIVE));
        }
    }
}

/**
 * @brief Tests that the [yolo] output computed on the host from our heads, the logistic of
 *        every entry but the sizes, equals Darknet's layer output within the head tolerance.
 *
 * The logistic has a slope of at most a quarter, so the bound of the head holds after it.
 */
TEST_F(Integration_DarknetYolov3TinyForward, TheHostYoloOutputsMatchTheDumpedLayers)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor input,
                            load_reference(K_RUNS[0].directory, K_INPUT));
    SA_ASSERT_OK_AND_ASSIGN(const ForwardRun run,
                            run_forward(*engine, DarknetNormalisation::Folded, input));

    ASSERT_EQ(run.model.heads.size(), std::size(K_HEADS));
    for (const DetectionRun& reference_run : K_RUNS)
    {
        for (std::size_t k = 0; k < std::size(K_HEADS); ++k)
        {
            const HeadReference& head = K_HEADS[k];
            const std::string what =
                std::string(reference_run.directory) + "/" + std::string(head.yolo);
            SCOPED_TRACE(what);
            ASSERT_EQ(run.model.heads[k].options.coding, HeadCoding::Logit);
            ASSERT_EQ(run.model.heads[k].options.scale_xy, 1.0f);
            ASSERT_EQ(run.model.heads[k].options.num_classes, K_CLASSES);

            SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor reference,
                                    load_reference(reference_run.directory, head.yolo));
            const std::vector<float>* const ours = output_of(run, head.output);
            ASSERT_NE(ours, nullptr);
            EXPECT_TRUE(within_tolerance(what, activate_logit_head(*ours), reference.values,
                                         K_HEAD_ABSOLUTE, K_HEAD_RELATIVE));
        }
    }
}

/**
 * @brief Tests that decode, threshold and suppression over our heads give Darknet's rows: none
 *        at 0.25 and the 16 of REPORT section 6 at 0.01, both with suppression at 0.45.
 *
 * Both sides are sorted by class, then by score. A row matches when its class is equal and its
 * box entries and score are within 1e-4. The reference's objectness column has no counterpart
 * in a library row and is not compared. Every decision within 1e-6 of a threshold is printed.
 */
TEST_F(Integration_DarknetYolov3TinyForward, TheDetectionsMatchTheDumpedRows)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor input,
                            load_reference(K_RUNS[0].directory, K_INPUT));
    SA_ASSERT_OK_AND_ASSIGN(const ForwardRun run,
                            run_forward(*engine, DarknetNormalisation::Folded, input));
    ASSERT_EQ(run.model.heads.size(), std::size(K_HEADS));
    ASSERT_EQ(run.model.network.height, K_NETWORK_SIDE);
    ASSERT_EQ(run.model.network.width, K_NETWORK_SIDE);
    ASSERT_EQ(run.model.criterion, Vision::Detect::SuppressionCriterion::Iou);

    DecodedHeads decoded;
    ASSERT_TRUE(decode_heads(run, decoded));

    for (const DetectionRun& reference_run : K_RUNS)
    {
        SCOPED_TRACE(std::string(reference_run.directory));
        SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor reference,
                                load_reference(reference_run.directory, K_DETECTIONS));
        ASSERT_EQ(reference.extents, (std::vector<std::int64_t>{reference_run.rows,
                                                                K_REFERENCE_COLUMNS}));
        const float thresh = real_of(reference.description, "thresh");
        const float suppression = real_of(reference.description, "nms");
        ASSERT_FLOAT_EQ(thresh, reference_run.thresh);
        ASSERT_FLOAT_EQ(suppression, K_SUPPRESSION_THRESHOLD);

        const std::string close = name_close_decisions(decoded, thresh, suppression);
        if (!close.empty())
            std::cout << "[ CLOSE    ] " << reference_run.directory << "\n" << close;

        std::vector<DetectionRow> ours = select_rows(run, decoded, thresh, suppression);
        std::vector<DetectionRow> theirs = reference_rows(reference);
        sort_rows(ours);
        sort_rows(theirs);
        ASSERT_EQ(ours.size(), theirs.size()) << close;

        double largest = 0.0;
        for (std::size_t row = 0; row < ours.size(); ++row)
        {
            SCOPED_TRACE("sorted row " + std::to_string(row));
            EXPECT_EQ(ours[row].class_index, theirs[row].class_index) << close;
            EXPECT_NEAR(ours[row].score, theirs[row].score, K_ROW_ABSOLUTE) << close;
            largest = std::max(largest, std::fabs(static_cast<double>(ours[row].score) -
                                                  static_cast<double>(theirs[row].score)));
            for (std::size_t entry = 0; entry < 4; ++entry)
            {
                EXPECT_NEAR(ours[row].box[entry], theirs[row].box[entry], K_ROW_ABSOLUTE)
                    << "box entry " << entry << "\n" << close;
                largest =
                    std::max(largest, std::fabs(static_cast<double>(ours[row].box[entry]) -
                                                static_cast<double>(theirs[row].box[entry])));
            }
        }
        std::cout << "[ MEASURED ] " << reference_run.directory << "/detections: "
                  << ours.size() << " rows, largest difference " << largest << "; bound "
                  << K_ROW_ABSOLUTE << "\n";
    }
}

/**
 * @brief Tests that the Separate model, with its own batch-norm rows, computes the heads of the
 *        Folded model, and prints the largest difference at each head.
 *
 * The bound is the head tolerance, 1e-3 absolute plus 1e-4 relative to the folded value.
 */
TEST_F(Integration_DarknetYolov3TinyForward, TheSeparateFormComputesTheFoldedHeads)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor input,
                            load_reference(K_RUNS[0].directory, K_INPUT));
    SA_ASSERT_OK_AND_ASSIGN(const ForwardRun folded,
                            run_forward(*engine, DarknetNormalisation::Folded, input));
    SA_ASSERT_OK_AND_ASSIGN(const ForwardRun separate,
                            run_forward(*engine, DarknetNormalisation::Separate, input));

    for (const HeadReference& head : K_HEADS)
    {
        const std::string what = "separate against folded, " + std::string(head.output);
        SCOPED_TRACE(what);
        const std::vector<float>* const from_folded = output_of(folded, head.output);
        const std::vector<float>* const from_separate = output_of(separate, head.output);
        ASSERT_NE(from_folded, nullptr);
        ASSERT_NE(from_separate, nullptr);
        EXPECT_TRUE(within_tolerance(what, *from_separate, *from_folded, K_HEAD_ABSOLUTE,
                                     K_HEAD_RELATIVE));
    }
}
