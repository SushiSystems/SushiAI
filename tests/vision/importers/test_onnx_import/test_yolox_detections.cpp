/**************************************************************************/
/* test_yolox_detections.cpp                                              */
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
 * @file test_yolox_detections.cpp
 * @brief Tests the detections of an imported YOLOX ONNX file against the reference runtime's own
 *        on letterboxed images.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/vision/codecs/decode_image.hpp>
#include <SushiAI/vision/darknet_import_cases/darknet_import.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/yolox/yolox_decode.hpp>
#include <SushiAI/vision/detect/yolox/yolox_task.hpp>
#include <SushiAI/vision/importers/onnx_import.hpp>
#include <SushiAI/vision/registration/registration.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <support/reference_detections.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Names the variable that holds the folder of the ONNX file and the references. */
    constexpr const char* K_DATA_VARIABLE = "SA_YOLOX_DATA_DIR";

    /** @brief Names the ONNX file in that folder. */
    constexpr const char* K_ONNX_FILE = "yolox_x.onnx";

    /** @brief Names the reference file below `references/`. */
    constexpr const char* K_REFERENCE_FILE = "yolox_x_onnx.json";

    using SushiAI::Test::Detection;
    using SushiAI::Test::K_BOX_ENTRIES;
    using SushiAI::Test::expect_same_detections;
    using SushiAI::Test::read_expected;

    /** @brief Holds the colour channels of an image and of the mean and the deviation. */
    constexpr std::size_t K_CHANNELS = 3;

    /** @brief Holds the strides of the three head levels, finest first. */
    constexpr std::int64_t K_STRIDES[] = {8, 16, 32};

    /** @brief Holds the roles of one head level: classes, boxes and objectness. */
    constexpr const char* K_ROLES[] = {"classes", "boxes", "objectness"};

    /** @brief Holds one image of the reference file and what the runtime detected on it. */
    struct ReferenceImage
    {
        std::filesystem::path file;      /**< The letterboxed PNG. */
        std::vector<Detection> expected; /**< The reference runtime's detections. */
    };

    /** @brief Holds the reference file: the preparation of the images and what was detected. */
    struct Reference
    {
        float mean[K_CHANNELS] = {};        /**< Per-channel mean, red first. */
        float deviation[K_CHANNELS] = {};   /**< Per-channel standard deviation, red first. */
        float score_threshold = 0.0f;       /**< A score must exceed it. */
        float suppression_threshold = 0.0f; /**< An IoU above it suppresses. */
        std::vector<ReferenceImage> images; /**< One entry per image. */
    };

    /** @brief Returns the data folder, or an empty path when the variable is unset. */
    std::filesystem::path data_folder()
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        const char* const value = std::getenv(K_DATA_VARIABLE);
#pragma clang diagnostic pop
        return value == nullptr ? std::filesystem::path() : std::filesystem::path(value);
    }

    /** @brief Reads the three numbers of the array @p key of @p document into @p out. */
    Status read_channels(const Config::Value& document, const char* key, float (&out)[K_CHANNELS])
    {
        SA_TRY_ASSIGN(const Config::Value& member, document.at(key, "reference"));
        SA_TRY_ASSIGN(const std::vector<Config::Value>& list, member.as_array(key));
        SA_FAIL_IF(list.size() != K_CHANNELS, StatusCode::InvalidFile, "'{}' holds {} values", key,
                   list.size());
        for (std::size_t c = 0; c < K_CHANNELS; ++c)
        {
            SA_TRY_ASSIGN(const double value, list[c].as_real(key));
            out[c] = static_cast<float>(value);
        }
        return Status();
    }

    /** @brief Reads the number @p key of @p document as a float. */
    Result<float> read_threshold(const Config::Value& document, const char* key)
    {
        SA_TRY_ASSIGN(const Config::Value& member, document.at(key, "reference"));
        SA_TRY_ASSIGN(const double value, member.as_real(key));
        return static_cast<float>(value);
    }

    /** @brief Reads `references/yolox_x_onnx.json` of @p folder. */
    Result<Reference> read_reference(const std::filesystem::path& folder)
    {
        const std::filesystem::path references = folder / "references";
        SA_TRY_ASSIGN(const Config::Value document,
                      Config::parse_json_file(references / K_REFERENCE_FILE));
        Reference reference;
        SA_TRY(read_channels(document, "mean", reference.mean));
        SA_TRY(read_channels(document, "std", reference.deviation));
        SA_TRY_ASSIGN(reference.score_threshold, read_threshold(document, "score_threshold"));
        SA_TRY_ASSIGN(reference.suppression_threshold,
                      read_threshold(document, "suppression_threshold"));

        SA_TRY_ASSIGN(const Config::Value& images, document.at("images", "reference"));
        SA_TRY_ASSIGN(const std::vector<Config::Value>& entries, images.as_array("images"));
        for (const Config::Value& entry : entries)
        {
            ReferenceImage image;
            SA_TRY_ASSIGN(const Config::Value& file, entry.at("file", "reference"));
            SA_TRY_ASSIGN(const std::string& name, file.as_string("file"));
            image.file = references / name;
            SA_TRY_ASSIGN(image.expected, read_expected(entry));
            reference.images.push_back(std::move(image));
        }
        return reference;
    }

    /** @brief Returns the name the ONNX file publishes the @p role of @p level under. */
    std::string port_name(std::size_t level, const char* role)
    {
        return "level" + std::to_string(level) + "." + role;
    }

    /** @brief Imports the ONNX file and runs it the way `sa inference` runs a YOLOX model. */
    class Large_YoloxOnnxDetections : public SushiAI::Test::EngineTest
    {
        protected:
            /** @brief Skips the test when the data folder or a file of it is missing. */
            void SetUp() override
            {
                SushiAI::Test::EngineTest::SetUp();
                folder_ = data_folder();
                if (folder_.empty())
                    GTEST_SKIP() << K_DATA_VARIABLE << " is not set";
                for (const std::filesystem::path& file :
                     {folder_ / K_ONNX_FILE, folder_ / "references" / K_REFERENCE_FILE})
                {
                    if (!std::filesystem::exists(file))
                        GTEST_SKIP() << file.string() << " is missing; run "
                                     << "tools/reference/export_yolox_onnx_reference.py";
                }
            }

            /** @brief Imports the file and prepares the compiled step. */
            Status prepare();

            /** @brief Returns the detections of @p image under the reference's preparation. */
            Result<std::vector<Detection>> detect(const std::filesystem::path& image,
                                                  const Reference& reference);

            std::filesystem::path folder_;                /**< The data folder. */
            Shape batch_shape_;                           /**< `[1, H, W, C]`. */
            std::unique_ptr<NN::ModelState> state_;       /**< The imported weights. */
            std::unique_ptr<Graph::TensorPool> pool_;     /**< The step's tensors. */
            std::unique_ptr<Train::EvaluationStep> step_; /**< The compiled step. */
    };

    Status Large_YoloxOnnxDetections::prepare()
    {
        const std::filesystem::path file = folder_ / K_ONNX_FILE;
        SA_TRY_ASSIGN(const IO::ImportedModel imported,
                      Vision::Importers::import_onnx_files(
                          SushiRuntime::span<const std::filesystem::path>(&file, 1)));
        const Shape& sample = imported.model.input_shape;
        batch_shape_ = Shape{1, sample[0], sample[1], sample[2]};

        SushiBLAS::Engine& engine = backend().engine();
        NN::NodeRegistry registry;
        SA_TRY(NN::register_layers(registry));
        SA_TRY(Vision::register_nodes(registry));
        SA_TRY_ASSIGN(NN::ModelGraph graph, NN::build_model(registry, imported.model));
        SA_TRY_ASSIGN(const Train::TracedStep traced,
                      Train::build_evaluation_graph(graph, batch_shape_));
        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, graph));
        state_ = std::make_unique<NN::ModelState>(std::move(state));
        SA_TRY(SushiAI::Test::fill_imported_state(
            SushiRuntime::span<const IO::ImportedTensor>(imported.weights.data(),
                                                         imported.weights.size()),
            *state_));

        pool_ = std::make_unique<Graph::TensorPool>(engine);
        SA_TRY_ASSIGN(step_, Train::EvaluationStep::make(backend(), traced, *state_, *pool_));
        return Status();
    }

    Result<std::vector<Detection>> Large_YoloxOnnxDetections::detect(
        const std::filesystem::path& image, const Reference& reference)
    {
        Vision::Codecs::DecodeOptions options;
        options.layout = Vision::Codecs::ChannelLayout::Color;
        SA_TRY_ASSIGN(const Vision::Imaging::Image pixels,
                      Vision::Codecs::load_image(image, options));
        SA_TRY_ASSIGN(SushiBLAS::Tensor input, allocate_tensor(backend().engine(), batch_shape_));
        const std::size_t count = static_cast<std::size_t>(input.num_elements);
        SA_FAIL_IF(static_cast<std::size_t>(pixels.size().height) * pixels.size().width *
                           pixels.channels() !=
                       count,
                   StatusCode::Incompatible, "{} is not the model's input size", image.string());
        float* const values = input.data_as<float>();
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t channel = i % K_CHANNELS;
            values[i] = (static_cast<float>(pixels.data()[i]) / 255.0f - reference.mean[channel]) /
                        reference.deviation[channel];
        }
        SA_TRY(step_->run(input));

        std::vector<Vision::Detect::Yolox::YoloxLevel> levels;
        std::int64_t classes = 0;
        for (std::size_t level = 0; level < std::size(K_STRIDES); ++level)
        {
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const class_logits,
                          step_->output(port_name(level, K_ROLES[0])));
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const boxes,
                          step_->output(port_name(level, K_ROLES[1])));
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const objectness,
                          step_->output(port_name(level, K_ROLES[2])));
            Vision::Detect::Yolox::YoloxLevel bound;
            bound.stride = static_cast<float>(K_STRIDES[level]);
            bound.height = batch_shape_[1] / K_STRIDES[level];
            bound.width = batch_shape_[2] / K_STRIDES[level];
            bound.classes = SushiRuntime::span<const float>(
                class_logits->data_as<float>(),
                static_cast<std::size_t>(class_logits->num_elements));
            bound.boxes = SushiRuntime::span<const float>(
                boxes->data_as<float>(), static_cast<std::size_t>(boxes->num_elements));
            bound.objectness = SushiRuntime::span<const float>(
                objectness->data_as<float>(), static_cast<std::size_t>(objectness->num_elements));
            levels.push_back(bound);
            classes = class_logits->num_elements / (bound.height * bound.width);
        }

        Vision::Detect::Yolox::YoloxTask task;
        task.score_threshold = reference.score_threshold;
        task.suppression.threshold = reference.suppression_threshold;
        std::vector<std::vector<float>> found;
        SA_TRY(Vision::Detect::Yolox::select_yolox_detections(
            SushiRuntime::span<const Vision::Detect::Yolox::YoloxLevel>(levels.data(),
                                                                         levels.size()),
            1, classes, task, found));

        std::vector<Detection> out;
        const std::vector<float>& rows = found.front();
        for (std::size_t at = 0; at + Vision::Detect::K_DETECTION_VALUES <= rows.size();
             at += Vision::Detect::K_DETECTION_VALUES)
        {
            const float* const row = rows.data() + at;
            Detection detection;
            detection.class_index =
                static_cast<std::int64_t>(row[Vision::Detect::K_DETECTION_CLASS]);
            detection.score = row[Vision::Detect::K_DETECTION_SCORE];
            for (std::size_t k = 0; k < K_BOX_ENTRIES; ++k)
                detection.corners[k] = row[k];
            out.push_back(detection);
        }
        return out;
    }
} // namespace

/**
 * @brief Tests that the imported ONNX file finds, on every reference image, the detections the
 *        reference runtime found: the same count and classes, each score within 2048 u and each
 *        corner within 2048 u of the larger input side.
 */
TEST_F(Large_YoloxOnnxDetections, TheDetectionsAreTheReferenceRuntimesOwn)
{
    const Status prepared = prepare();
    ASSERT_TRUE(prepared.is_ok()) << prepared.message();
    SA_ASSERT_OK_AND_ASSIGN(const Reference reference, read_reference(folder_));
    ASSERT_FALSE(reference.images.empty());

    const double side = static_cast<double>(std::max(batch_shape_[1], batch_shape_[2]));
    for (const ReferenceImage& image : reference.images)
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::vector<Detection> found, detect(image.file, reference));
        expect_same_detections(found, image.expected, side, image.file.filename().string());
    }
}
