/**************************************************************************/
/* darknet_forward.cpp                                                    */
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
 * @file darknet_forward.cpp
 * @brief Implements the forward pass of a Darknet model and the checks against its dump.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/darknet_forward_cases/darknet_forward.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
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
#include <SushiAI/config/value.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node.hpp>
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
#include <SushiAI/darknet_reference/darknet_reference.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>

namespace SushiAI
{
    namespace Test
    {
        namespace
        {
            using Vision::Detect::SuppressionCriterion;
            using Vision::Detect::Anchor::HeadCoding;
            using Vision::Detect::Anchor::HeadOptions;
            using Vision::Detect::Anchor::HeadShape;
            using Vision::Importers::DarknetConfig;
            using Vision::Importers::DarknetModel;
            using Vision::Importers::DarknetNormalisation;
            using Vision::Importers::DarknetTranslation;
            using Vision::Importers::DarknetWeights;

            /** @brief Holds the origin a cfg text of a test is parsed under. */
            constexpr std::string_view K_TEXT_ORIGIN = "test.cfg";

            /** @brief Holds the digits of the layer index in a reference name. */
            constexpr std::size_t K_LAYER_DIGITS = 3;

            /** @brief Holds the entries of a slot before its classes: the box and objectness. */
            constexpr std::int64_t K_SLOT_HEAD_ENTRIES =
                Vision::Detect::Anchor::K_FIRST_CLASS_ENTRY;

            /** @brief Holds the entries of a box: two of its centre and two of its size. */
            constexpr std::size_t K_BOX_ENTRIES = 4;

            /** @brief Holds the columns of a reference detection row. */
            constexpr std::int64_t K_REFERENCE_COLUMNS = 7;

            /** @brief Holds the column of a reference detection row that carries its score. */
            constexpr std::size_t K_REFERENCE_SCORE = 5;

            /** @brief Holds the column of a reference detection row that carries its class. */
            constexpr std::size_t K_REFERENCE_CLASS = 6;

            /** @brief Holds the score threshold of a run taken without `-thresh` (REPORT 2). */
            constexpr float K_DEFAULT_THRESH = 0.25f;

            /** @brief Holds the score threshold of a `_thresh_0p01` run. */
            constexpr float K_LOW_THRESH = 0.01f;

            /** @brief Returns the translation of @p config in the form @p normalisation. */
            Result<DarknetModel> translate(const DarknetConfig& config,
                                           DarknetNormalisation normalisation)
            {
                DarknetTranslation translation;
                translation.normalisation = normalisation;
                return Vision::Importers::translate_darknet_model(config, translation);
            }

            /**
             * @brief Reads `<stem>.weights` for @p model, folding every normalised convolution
             *        when @p normalisation is Folded.
             * @return The weights; FileNotFound naming the file; or the failure of the read.
             */
            Result<DarknetWeights> read_weights(std::string_view stem, const DarknetModel& model,
                                                DarknetNormalisation normalisation)
            {
                const DarknetFile file = find_darknet_file(std::string(stem) + ".weights");
                SA_FAIL_IF(!file.found, StatusCode::FileNotFound, "{}", file.reason);
                SA_TRY_ASSIGN(
                    DarknetWeights weights,
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

            /**
             * @brief Returns the spec of @p model with the row of each layer of @p layers
             *        published under name_observed_layer.
             * @return The spec, or Incompatible naming a layer the file does not have.
             */
            Result<Config::ModelSpec> observe_layers(const DarknetModel& model,
                                                     const std::vector<std::int64_t>& layers)
            {
                const std::int64_t count = static_cast<std::int64_t>(model.row_of_layer.size());
                Config::ModelSpec spec = model.spec;
                for (const std::int64_t layer : layers)
                {
                    SA_FAIL_IF(layer < 0 || layer >= count, StatusCode::Incompatible,
                               "layer {} is observed, but the file has {} layers", layer, count);
                    Config::OutputSpec published;
                    published.node = model.row_of_layer[static_cast<std::size_t>(layer)];
                    published.port = std::string(NN::K_SINGLE_PORT_NAME);
                    published.name = name_observed_layer(layer);
                    spec.outputs.push_back(std::move(published));
                }

                return Result<Config::ModelSpec>(std::move(spec));
            }

            /**
             * @brief Returns the shape of a batch of one sample of @p spec.
             * @return The shape, or Incompatible when @p input is not one such sample.
             */
            Result<Shape> shape_batch(const Config::ModelSpec& spec, const DarknetTensor& input)
            {
                const Shape& sample = spec.input_shape;
                SA_FAIL_IF(sample.rank() != 3 || input.extents.size() != 3 ||
                               input.extents[0] != sample[0] || input.extents[1] != sample[1] ||
                               input.extents[2] != sample[2],
                           StatusCode::Incompatible,
                           "the reference input does not have the shape {}", sample.to_string());

                return Result<Shape>(Shape{1, sample[0], sample[1], sample[2]});
            }

            /**
             * @brief Builds @p spec, stores @p weights in its state, runs one evaluation step
             *        over @p input and copies every published tensor into @p run.
             */
            Status evaluate(SushiBLAS::Engine& engine, const Config::ModelSpec& spec,
                            const DarknetWeights& weights, const DarknetTensor& input,
                            DarknetForwardRun& run)
            {
                SA_TRY_ASSIGN(const Shape batch_shape, shape_batch(spec, input));

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

                for (const NN::ModelOutputs::Entry& published : traced.outputs.all())
                {
                    SA_TRY_ASSIGN(const SushiBLAS::Tensor* const tensor,
                                  step->output(published.first));
                    const float* const values = tensor->data_as<float>();
                    run.outputs.emplace(published.first,
                                        std::vector<float>(values, values + tensor->num_elements));
                }

                return Status();
            }
            /** @brief Names one head of a parity case and the reference tensors of it. */
            struct HeadReference
            {
                std::int64_t layer; /**< The index of the [yolo] section among the layers. */
                std::int64_t grid;  /**< The rows and the columns of the head's grid. */
                float scale;        /**< The scale_x_y the test states of the head. */
                std::string output; /**< The name the model publishes the head under. */
                std::string input;  /**< The reference of the raw head. */
                std::string yolo;   /**< The reference of the [yolo] layer's own output. */
            };

            /** @brief Holds one detection as both sides are compared. */
            struct DetectionRow
            {
                float box[K_BOX_ENTRIES]; /**< Centre x, centre y, width, height, as fractions. */
                float score;              /**< Objectness times the class probability. */
                std::int64_t class_index; /**< The class. */
            };

            /** @brief Holds the decoded rows of every head of one image, heads in file order. */
            struct DecodedHeads
            {
                std::vector<float> boxes;  /**< `[rows, 4]` centre and size as fractions. */
                std::vector<float> scores; /**< `[rows, classes]` objectness times class. */
                std::int64_t rows = 0;     /**< Rows over every head. */
                std::int64_t classes = 0;  /**< Scores of a row. */
            };

            /**
             * @brief Lists the heads of @p parity in file order, named as the references are.
             * @pre The case states one grid and one scale per [yolo] layer of the parity table;
             *      another count stops the process.
             */
            std::vector<HeadReference> list_heads(const DarknetParityCase& parity)
            {
                const DarknetParityModel model = find_darknet_parity_model(parity.stem);
                const std::size_t count = model.yolo_layers.size();
                SA_FATAL_IF(parity.grids.size() != count || parity.scales.size() != count,
                            "the parity table gives {} [yolo] layers of {}, and the test states "
                            "{} grids and {} scales",
                            count, parity.stem, parity.grids.size(), parity.scales.size());

                std::vector<HeadReference> heads;
                for (std::size_t k = 0; k < count; ++k)
                {
                    const std::int64_t layer = model.yolo_layers[k];
                    heads.push_back({layer, parity.grids[k], parity.scales[k],
                                     "head." + std::to_string(k),
                                     name_darknet_layer(layer - 1, "yolo_input"),
                                     name_darknet_layer(layer, "yolo")});
                }

                return heads;
            }

            /** @brief Runs the model of @p parity over the input of its first run. */
            Result<DarknetForwardRun> run_first_input(SushiBLAS::Engine& engine,
                                                      const DarknetParityCase& parity,
                                                      DarknetNormalisation normalisation)
            {
                const DarknetParityModel model = find_darknet_parity_model(parity.stem);
                DarknetForwardRequest request;
                request.stem = parity.stem;
                request.run = model.runs.front().directory;
                request.normalisation = normalisation;
                return run_darknet_forward(engine, request);
            }

            /**
             * @brief Decodes the heads @p heads of @p run through the library, in file order.
             * @return Success, or a failure naming the head that is missing or fails validation.
             */
            ::testing::AssertionResult decode_heads(const DarknetForwardRun& run,
                                                    const std::vector<HeadReference>& heads,
                                                    std::int64_t classes, DecodedHeads& decoded)
            {
                decoded.classes = classes;
                for (std::size_t k = 0; k < heads.size(); ++k)
                {
                    const std::vector<float>* const head = find_output(run, heads[k].output);
                    if (head == nullptr)
                        return ::testing::AssertionFailure() << "no '" << heads[k].output << "'";

                    const std::int64_t index = static_cast<std::int64_t>(k);
                    const HeadOptions& options = run.model.heads[k].options;
                    const Status valid =
                        Vision::Detect::Anchor::validate_head(options, run.model.layout, index);
                    if (!valid.is_ok())
                        return ::testing::AssertionFailure() << valid.message();
                    if (options.num_classes != classes)
                        return ::testing::AssertionFailure()
                               << heads[k].output << " has " << options.num_classes << " classes";

                    const HeadShape shape{1, heads[k].grid, heads[k].grid};
                    const std::size_t rows = static_cast<std::size_t>(
                        Vision::Detect::Anchor::count_head_rows(shape, run.model.layout, index));
                    std::vector<float> boxes(rows * K_BOX_ENTRIES);
                    std::vector<float> scores(rows * static_cast<std::size_t>(classes));
                    Vision::Detect::Anchor::decode_head(*head, shape, options, run.model.layout,
                                                        index, boxes, scores);

                    decoded.boxes.insert(decoded.boxes.end(), boxes.begin(), boxes.end());
                    decoded.scores.insert(decoded.scores.end(), scores.begin(), scores.end());
                    decoded.rows += static_cast<std::int64_t>(rows);
                }

                return ::testing::AssertionSuccess();
            }

            /** @brief Returns the row @p row of @p decoded as pixel corners x1 y1 x2 y2. */
            std::array<float, K_BOX_ENTRIES> measure_corners(const DarknetModel& model,
                                                             const DecodedHeads& decoded,
                                                             std::size_t row)
            {
                const float* const box = decoded.boxes.data() + row * K_BOX_ENTRIES;
                const float width = static_cast<float>(model.network.width);
                const float height = static_cast<float>(model.network.height);

                return {(box[0] - box[2] / 2.0f) * width, (box[1] - box[3] / 2.0f) * height,
                        (box[0] + box[2] / 2.0f) * width, (box[1] + box[3] / 2.0f) * height};
            }

            /**
             * @brief Selects the detections of @p decoded as the library does for an imported
             *        model: boxes in pixels, every class of a row, one threshold on the score,
             *        the criterion of the translation.
             * @return The kept rows with their boxes back as centre and size in fractions.
             */
            std::vector<DetectionRow> select_rows(const DarknetModel& model,
                                                  const DecodedHeads& decoded, float thresh,
                                                  float suppression)
            {
                const float width = static_cast<float>(model.network.width);
                const float height = static_cast<float>(model.network.height);
                std::vector<float> pixels(decoded.boxes);
                for (std::size_t at = 0; at < pixels.size(); ++at)
                    pixels[at] *= at % 2 == 0 ? width : height;

                Vision::Detect::NMSConfig config;
                config.confidence_threshold = thresh;
                config.iou_threshold = suppression;
                config.max_detections = decoded.rows * decoded.classes;
                config.max_candidates = decoded.rows * decoded.classes;
                config.criterion = model.criterion;
                config.distance_exponent = model.distance_exponent;

                std::vector<std::vector<float>> detections;
                Vision::Detect::select_nms_detections(
                    pixels, decoded.scores,
                    Vision::Detect::PredictionShape{1, decoded.rows, decoded.classes}, config,
                    detections);

                std::vector<DetectionRow> rows;
                const std::vector<float>& image = detections.front();
                for (std::size_t at = 0; at + Vision::Detect::K_DETECTION_VALUES <= image.size();
                     at += Vision::Detect::K_DETECTION_VALUES)
                {
                    const float* const row = image.data() + at;
                    DetectionRow kept{};
                    kept.box[0] = (row[0] + row[2]) / (2.0f * width);
                    kept.box[1] = (row[1] + row[3]) / (2.0f * height);
                    kept.box[2] = (row[2] - row[0]) / width;
                    kept.box[3] = (row[3] - row[1]) / height;
                    kept.score = row[Vision::Detect::K_DETECTION_SCORE];
                    kept.class_index =
                        static_cast<std::int64_t>(row[Vision::Detect::K_DETECTION_CLASS]);
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
                    std::copy(row, row + K_BOX_ENTRIES, kept.box);
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

            /** @brief Returns the absolute difference of @p a and @p b, taken in double. */
            double measure_distance(float a, float b)
            {
                return std::fabs(static_cast<double>(a) - static_cast<double>(b));
            }

            /**
             * @brief Names every score within @p margin of @p thresh and every pair of rows of
             *        one class whose suppression measure is within it of @p suppression.
             *
             * The measure is the library's, under the criterion of @p model, on pixel corners.
             * @return One line per score and per pair; empty when no decision is that close.
             */
            std::string name_close_decisions(const DarknetModel& model,
                                             const DecodedHeads& decoded, float thresh,
                                             float suppression, double margin)
            {
                const std::size_t classes = static_cast<std::size_t>(decoded.classes);
                std::string names;
                std::vector<std::pair<std::size_t, std::size_t>> candidates;
                for (std::size_t at = 0; at < decoded.scores.size(); ++at)
                {
                    const double score = static_cast<double>(decoded.scores[at]);
                    if (measure_distance(decoded.scores[at], thresh) <= margin)
                        names += "row " + std::to_string(at / classes) + " class " +
                                 std::to_string(at % classes) + " scores " +
                                 std::to_string(score) + ", beside the threshold\n";
                    if (score > static_cast<double>(thresh) - margin)
                        candidates.emplace_back(at / classes, at % classes);
                }

                for (std::size_t first = 0; first < candidates.size(); ++first)
                {
                    for (std::size_t second = first + 1; second < candidates.size(); ++second)
                    {
                        if (candidates[first].second != candidates[second].second)
                            continue;

                        const float measure = Vision::Detect::measure_suppression(
                            measure_corners(model, decoded, candidates[first].first),
                            measure_corners(model, decoded, candidates[second].first),
                            model.criterion, model.distance_exponent);
                        if (measure_distance(measure, suppression) <= margin)
                            names += "rows " + std::to_string(candidates[first].first) + " and " +
                                     std::to_string(candidates[second].first) + " of class " +
                                     std::to_string(candidates[first].second) + " measure " +
                                     std::to_string(measure) +
                                     ", beside the suppression threshold\n";
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

        std::vector<DarknetParityModel> list_darknet_parity_models()
        {
            const float low = K_LOW_THRESH;
            const float normal = K_DEFAULT_THRESH;
            return {
                {"yolov3-tiny", 416,
                 {{"yolov3_tiny", normal}, {"yolov3_tiny_thresh_0p01", low}}, {16, 23}},
                {"yolov3", 416, {{"yolov3", normal}, {"yolov3_thresh_0p01", low}}, {82, 94, 106}},
                {"yolov4-tiny", 416,
                 {{"yolov4_tiny", normal}, {"yolov4_tiny_thresh_0p01", low}}, {30, 37}},
                {"yolov4", 608, {{"yolov4_plain_pool", normal}, {"yolov4_plain_pool_thresh_0p01", low}}, {139, 150, 161}},
                {"yolov7-tiny", 416,
                 {{"yolov7_tiny_layers_plain_pool", normal},
                  {"yolov7_tiny_plain_pool_thresh_0p01", low}}, {90, 94, 98}},
                {"yolov7", 640, {{"yolov7_plain_pool_thresh_0p01", low}}, {134, 138, 142}},
            };
        }

        DarknetParityModel find_darknet_parity_model(std::string_view stem)
        {
            const std::vector<DarknetParityModel> models = list_darknet_parity_models();
            const auto found = std::find_if(models.begin(), models.end(),
                                            [stem](const DarknetParityModel& model)
                                            {
                                                return model.stem == stem;
                                            });
            SA_FATAL_IF(found == models.end(), "the parity table has no model '{}'", stem);

            return *found;
        }

        std::string name_darknet_layer(std::int64_t layer, std::string_view kind)
        {
            const std::string index = std::to_string(layer);
            const std::size_t zeros =
                index.size() < K_LAYER_DIGITS ? K_LAYER_DIGITS - index.size() : 0;

            return "layer_" + std::string(zeros, '0') + index + "_" + std::string(kind);
        }

        std::string name_observed_layer(std::int64_t layer)
        {
            return "layer." + std::to_string(layer);
        }

        std::string find_missing_darknet_file(std::string_view stem, std::string_view run,
                                              SushiRuntime::span<const std::string> names)
        {
            for (const std::string_view extension : {".cfg", ".weights"})
            {
                const DarknetFile file =
                    find_darknet_file(std::string(stem) + std::string(extension));
                if (!file.found)
                    return file.reason;
            }

            for (const std::string& name : names)
            {
                const DarknetReference reference = find_darknet_reference(run, name);
                if (!reference.found)
                    return reference.reason;
            }

            return std::string();
        }

        Result<DarknetTensor> load_darknet_reference(std::string_view run, std::string_view name)
        {
            return load_darknet_tensor(find_darknet_reference(run, name));
        }

        Result<DarknetModel> translate_darknet_text(std::string_view text,
                                                    DarknetNormalisation normalisation)
        {
            SA_TRY_ASSIGN(const DarknetConfig config,
                          Vision::Importers::parse_darknet_config(text, K_TEXT_ORIGIN));
            return translate(config, normalisation);
        }

        Result<DarknetConfig> load_darknet_config_file(std::string_view stem)
        {
            const DarknetFile file = find_darknet_file(std::string(stem) + ".cfg");
            SA_FAIL_IF(!file.found, StatusCode::FileNotFound, "{}", file.reason);

            return Vision::Importers::load_darknet_config(file.path);
        }

        Result<DarknetModel> translate_darknet_file(std::string_view stem,
                                                    DarknetNormalisation normalisation)
        {
            SA_TRY_ASSIGN(const DarknetConfig config, load_darknet_config_file(stem));
            return translate(config, normalisation);
        }

        std::vector<std::int64_t> sources_of_row(const DarknetModel& model, std::int64_t row)
        {
            const std::int64_t rows = static_cast<std::int64_t>(model.spec.layers.size());
            SA_FATAL_IF(row < 0 || row >= rows, "row {} is asked of a model of {} rows", row, rows);

            std::vector<std::int64_t> sources;
            for (const Config::SourceSpec& source :
                 model.spec.layers[static_cast<std::size_t>(row)].from)
                sources.push_back(source.node);

            return sources;
        }

        Result<DarknetForwardRun> run_darknet_forward(SushiBLAS::Engine& engine,
                                                      const DarknetForwardRequest& request)
        {
            SA_TRY_ASSIGN(const DarknetTensor input,
                          load_darknet_reference(request.run, K_DARKNET_INPUT));

            DarknetForwardRun run;
            SA_TRY_ASSIGN(run.model, translate_darknet_file(request.stem, request.normalisation));
            SA_TRY_ASSIGN(const DarknetWeights weights,
                          read_weights(request.stem, run.model, request.normalisation));
            SA_TRY_ASSIGN(const Config::ModelSpec spec,
                          observe_layers(run.model, request.observed_layers));
            SA_TRY(evaluate(engine, spec, weights, input, run));

            return Result<DarknetForwardRun>(std::move(run));
        }

        const std::vector<float>* find_output(const DarknetForwardRun& run, std::string_view name)
        {
            const auto found = run.outputs.find(name);
            return found == run.outputs.end() ? nullptr : &found->second;
        }

        ToleranceGap measure_gap(SushiRuntime::span<const float> actual,
                                 SushiRuntime::span<const float> reference, double absolute,
                                 double relative)
        {
            SA_FATAL_IF(actual.size() != reference.size(), "{} values are measured against {}",
                        actual.size(), reference.size());

            ToleranceGap gap;
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

        ::testing::AssertionResult within_tolerance(std::string_view what,
                                                    SushiRuntime::span<const float> actual,
                                                    SushiRuntime::span<const float> reference,
                                                    double absolute, double relative)
        {
            if (actual.size() != reference.size())
                return ::testing::AssertionFailure() << what << ": " << actual.size()
                                                     << " values against " << reference.size();

            const ToleranceGap gap = measure_gap(actual, reference, absolute, relative);
            std::cout << "[ MEASURED ] " << what << ": largest difference " << gap.largest
                      << " at element " << gap.at << "; bound " << absolute << " + " << relative
                      << " * |reference|\n";
            if (gap.outside == 0)
                return ::testing::AssertionSuccess();

            return ::testing::AssertionFailure()
                   << what << ": " << gap.outside << " of " << actual.size()
                   << " values are outside " << absolute << " + " << relative
                   << " * |reference|; worst at " << gap.at << " (" << actual[gap.at]
                   << " against " << reference[gap.at] << ")";
        }

        std::vector<float> compute_yolo_output(SushiRuntime::span<const float> head,
                                               const HeadOptions& options)
        {
            SA_FATAL_IF(options.num_classes < 1, "a head of {} classes is activated",
                        options.num_classes);
            const std::size_t entries = static_cast<std::size_t>(
                Vision::Detect::Anchor::K_FIRST_CLASS_ENTRY + options.num_classes);
            SA_FATAL_IF(head.size() % entries != 0, "{} values do not hold slots of {} entries",
                        head.size(), entries);

            const double scale = static_cast<double>(options.scale_xy);
            const bool is_logit = options.coding == HeadCoding::Logit;
            std::vector<float> output(head.size());
            for (std::size_t i = 0; i < head.size(); ++i)
            {
                const std::int64_t entry = static_cast<std::int64_t>(i % entries);
                const bool is_offset = entry == Vision::Detect::Anchor::K_X_ENTRY ||
                                       entry == Vision::Detect::Anchor::K_Y_ENTRY;
                const bool is_size = entry == Vision::Detect::Anchor::K_WIDTH_ENTRY ||
                                     entry == Vision::Detect::Anchor::K_HEIGHT_ENTRY;
                double value = static_cast<double>(head[i]);
                if (is_logit && !is_size)
                    value = 1.0 / (1.0 + std::exp(-value));
                if (is_offset)
                    value = value * scale - (scale - 1.0) / 2.0;
                output[i] = static_cast<float>(value);
            }

            return output;
        }

        std::string find_missing_parity_file(const DarknetParityCase& parity)
        {
            std::vector<std::string> names{std::string(K_DARKNET_INPUT),
                                           std::string(K_DARKNET_DETECTIONS)};
            for (const HeadReference& head : list_heads(parity))
            {
                names.push_back(head.input);
                names.push_back(head.yolo);
            }

            for (const DarknetParityRun& run : find_darknet_parity_model(parity.stem).runs)
            {
                const std::string missing =
                    find_missing_darknet_file(parity.stem, run.directory, names);
                if (!missing.empty())
                    return missing;
            }

            return std::string();
        }

        DarknetParityTest::DarknetParityTest(DarknetParityCase stated) : parity(std::move(stated))
        {

        }

        void DarknetParityTest::SetUp()
        {
            EngineTest::SetUp();
            const std::string missing = find_missing_parity_file(parity);
            if (!missing.empty())
                GTEST_SKIP() << missing;
        }

        void check_folded_heads(SushiBLAS::Engine& engine, const DarknetParityCase& parity)
        {
            const std::vector<HeadReference> heads = list_heads(parity);
            const DarknetParityModel model = find_darknet_parity_model(parity.stem);
            SA_ASSERT_OK_AND_ASSIGN(const DarknetForwardRun run,
                                    run_first_input(engine, parity, DarknetNormalisation::Folded));

            ASSERT_EQ(run.model.heads.size(), heads.size());
            for (const DarknetParityRun& reference_run : model.runs)
            {
                for (std::size_t k = 0; k < heads.size(); ++k)
                {
                    const HeadReference& head = heads[k];
                    const std::string what =
                        std::string(reference_run.directory) + "/" + head.input;
                    SCOPED_TRACE(what);
                    EXPECT_EQ(run.model.heads[k].layer, head.layer);
                    EXPECT_EQ(run.model.heads[k].output, head.output);

                    SA_ASSERT_OK_AND_ASSIGN(
                        const DarknetTensor reference,
                        load_darknet_reference(reference_run.directory, head.input));
                    EXPECT_EQ(reference.extents,
                              (std::vector<std::int64_t>{
                                  head.grid, head.grid,
                                  parity.slots * (K_SLOT_HEAD_ENTRIES + parity.classes)}));
                    const std::vector<float>* const ours = find_output(run, head.output);
                    ASSERT_NE(ours, nullptr);
                    EXPECT_TRUE(within_tolerance(what, *ours, reference.values,
                                                 parity.bounds.head_absolute,
                                                 parity.bounds.head_relative));
                }
            }
        }

        void check_yolo_outputs(SushiBLAS::Engine& engine, const DarknetParityCase& parity)
        {
            const std::vector<HeadReference> heads = list_heads(parity);
            const DarknetParityModel model = find_darknet_parity_model(parity.stem);
            SA_ASSERT_OK_AND_ASSIGN(const DarknetForwardRun run,
                                    run_first_input(engine, parity, DarknetNormalisation::Folded));

            ASSERT_EQ(run.model.heads.size(), heads.size());
            for (const DarknetParityRun& reference_run : model.runs)
            {
                for (std::size_t k = 0; k < heads.size(); ++k)
                {
                    const HeadReference& head = heads[k];
                    const HeadOptions& options = run.model.heads[k].options;
                    const std::string what =
                        std::string(reference_run.directory) + "/" + head.yolo;
                    SCOPED_TRACE(what);
                    ASSERT_EQ(options.coding, parity.coding);
                    ASSERT_EQ(options.scale_xy, head.scale);
                    ASSERT_EQ(options.num_classes, parity.classes);

                    SA_ASSERT_OK_AND_ASSIGN(
                        const DarknetTensor reference,
                        load_darknet_reference(reference_run.directory, head.yolo));
                    const std::vector<float>* const ours = find_output(run, head.output);
                    ASSERT_NE(ours, nullptr);
                    EXPECT_TRUE(within_tolerance(what, compute_yolo_output(*ours, options),
                                                 reference.values, parity.bounds.head_absolute,
                                                 parity.bounds.head_relative));
                }
            }
        }

        void check_detections(SushiBLAS::Engine& engine, const DarknetParityCase& parity)
        {
            const std::vector<HeadReference> heads = list_heads(parity);
            const DarknetParityModel model = find_darknet_parity_model(parity.stem);
            ASSERT_EQ(model.runs.size(), parity.run_rows.size());
            SA_ASSERT_OK_AND_ASSIGN(const DarknetForwardRun run,
                                    run_first_input(engine, parity, DarknetNormalisation::Folded));
            ASSERT_EQ(run.model.heads.size(), heads.size());
            ASSERT_EQ(run.model.network.height, model.side);
            ASSERT_EQ(run.model.network.width, model.side);
            ASSERT_EQ(run.model.criterion, parity.criterion);
            if (parity.criterion == SuppressionCriterion::DistanceIoU)
                ASSERT_EQ(run.model.distance_exponent, parity.distance_exponent);

            DecodedHeads decoded;
            ASSERT_TRUE(decode_heads(run, heads, parity.classes, decoded));

            for (std::size_t i = 0; i < model.runs.size(); ++i)
            {
                const std::string directory(model.runs[i].directory);
                SCOPED_TRACE(directory);
                SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor reference,
                                        load_darknet_reference(directory, K_DARKNET_DETECTIONS));
                ASSERT_EQ(reference.extents,
                          (std::vector<std::int64_t>{parity.run_rows[i], K_REFERENCE_COLUMNS}));
                const float thresh = real_of(reference.description, "thresh");
                const float suppression = real_of(reference.description, "nms");
                ASSERT_FLOAT_EQ(thresh, model.runs[i].thresh);
                ASSERT_FLOAT_EQ(suppression, parity.suppression);

                const std::string close = name_close_decisions(
                    run.model, decoded, thresh, suppression, parity.bounds.threshold_margin);
                if (!close.empty())
                    std::cout << "[ CLOSE    ] " << directory << "\n" << close;

                std::vector<DetectionRow> ours =
                    select_rows(run.model, decoded, thresh, suppression);
                std::vector<DetectionRow> theirs = reference_rows(reference);
                sort_rows(ours);
                sort_rows(theirs);
                ASSERT_EQ(ours.size(), theirs.size()) << close;

                const double bound = parity.bounds.row_absolute;
                double largest = 0.0;
                for (std::size_t row = 0; row < ours.size(); ++row)
                {
                    SCOPED_TRACE("sorted row " + std::to_string(row));
                    EXPECT_EQ(ours[row].class_index, theirs[row].class_index) << close;
                    EXPECT_NEAR(ours[row].score, theirs[row].score, bound) << close;
                    largest =
                        std::max(largest, measure_distance(ours[row].score, theirs[row].score));
                    for (std::size_t entry = 0; entry < K_BOX_ENTRIES; ++entry)
                    {
                        EXPECT_NEAR(ours[row].box[entry], theirs[row].box[entry], bound)
                            << "box entry " << entry << "\n" << close;
                        largest = std::max(largest, measure_distance(ours[row].box[entry],
                                                                     theirs[row].box[entry]));
                    }
                }
                std::cout << "[ MEASURED ] " << directory << "/detections: " << ours.size()
                          << " rows, largest difference " << largest << "; bound " << bound
                          << "\n";
            }
        }

        void check_separate_form(SushiBLAS::Engine& engine, const DarknetParityCase& parity)
        {
            const std::vector<HeadReference> heads = list_heads(parity);
            SA_ASSERT_OK_AND_ASSIGN(const DarknetForwardRun folded,
                                    run_first_input(engine, parity, DarknetNormalisation::Folded));
            SA_ASSERT_OK_AND_ASSIGN(
                const DarknetForwardRun separate,
                run_first_input(engine, parity, DarknetNormalisation::Separate));

            for (const HeadReference& head : heads)
            {
                const std::string what = "separate against folded, " + head.output;
                SCOPED_TRACE(what);
                const std::vector<float>* const from_folded = find_output(folded, head.output);
                const std::vector<float>* const from_separate = find_output(separate, head.output);
                ASSERT_NE(from_folded, nullptr);
                ASSERT_NE(from_separate, nullptr);
                EXPECT_TRUE(within_tolerance(what, *from_separate, *from_folded,
                                             parity.bounds.head_absolute,
                                             parity.bounds.head_relative));
            }
        }

    } // namespace Test
} // namespace SushiAI
