/**************************************************************************/
/* test_detections.cpp                                                    */
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
 * @file test_detections.cpp
 * @brief Tests the detections of imported Ultralytics checkpoints against Ultralytics' own on
 *        coco128 images.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/vision/codecs/decode_image.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/importers/ultralytics_import.hpp>
#include <SushiAI/vision/registration/registration.hpp>
#include <SushiAI/vision/training/free_batch_decoder.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>
#include <SushiAI/vision/darknet_import_cases/darknet_import.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <support/reference_detections.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Names the variable that holds the folder of the checkpoints and the references. */
    constexpr const char* K_DATA_VARIABLE = "SA_ULTRALYTICS_DATA_DIR";

    using SushiAI::Test::Detection;
    using SushiAI::Test::K_BOX_ENTRIES;
    using SushiAI::Test::expect_same_detections;
    using SushiAI::Test::read_expected;

    /** @brief Holds one image of a reference file and the detections Ultralytics made on it. */
    struct ReferenceImage
    {
        std::filesystem::path file;       /**< The square PNG. */
        std::vector<Detection> expected; /**< Ultralytics' detections. */
    };

    /** @brief Names one released checkpoint. */
    struct CheckpointRow
    {
        const char* name; /**< The row's name. */
        const char* stem; /**< The stem of `<stem>.pt` and `references/<stem>.json`. */
    };

    /** @brief Prints @p row by its name, so a failure names the checkpoint. */
    void PrintTo(const CheckpointRow& row, std::ostream* out)
    {
        *out << row.stem;
    }

    /** @brief Lists the checkpoints whose detections are compared. */
    const CheckpointRow K_CHECKPOINTS[] = {
        {"Yolo26n", "yolo26n"},
        {"Yolov8n", "yolov8n"},
        {"Yolo11n", "yolo11n"},
        {"Yolov5nu", "yolov5nu"},
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

    /** @brief Reads `references/<stem>.json` of @p folder. */
    Result<std::vector<ReferenceImage>> read_reference(const std::filesystem::path& folder,
                                                       const std::string& stem)
    {
        const std::filesystem::path references = folder / "references";
        SA_TRY_ASSIGN(const Config::Value document,
                      Config::parse_json_file(references / (stem + ".json")));
        SA_TRY_ASSIGN(const Config::Value& images, document.at("images", stem));
        SA_TRY_ASSIGN(const std::vector<Config::Value>& entries, images.as_array("images"));
        std::vector<ReferenceImage> out;
        for (const Config::Value& entry : entries)
        {
            ReferenceImage image;
            SA_TRY_ASSIGN(const Config::Value& file, entry.at("file", stem));
            SA_TRY_ASSIGN(const std::string& name, file.as_string("file"));
            image.file = references / name;
            SA_TRY_ASSIGN(image.expected, read_expected(entry));
            out.push_back(std::move(image));
        }
        return out;
    }

    /**
     * @brief Imports a checkpoint and runs it the way `sa inference` runs an anchor-free model:
     *        one-to-many outputs, decoded, suppressed by best class at the task's threshold.
     */
    class Large_UltralyticsDetections
        : public SushiAI::Test::EngineTest,
          public ::testing::WithParamInterface<CheckpointRow>
    {
        protected:
            /** @brief Skips the test when the data folder or a file of the row is missing. */
            void SetUp() override
            {
                SushiAI::Test::EngineTest::SetUp();
                folder_ = data_folder();
                if (folder_.empty())
                    GTEST_SKIP() << K_DATA_VARIABLE << " is not set";
                const std::string stem = GetParam().stem;
                for (const std::filesystem::path& file :
                     {folder_ / (stem + ".pt"), folder_ / "references" / (stem + ".json")})
                {
                    if (!std::filesystem::exists(file))
                        GTEST_SKIP() << file.string() << " is missing; run "
                                     << "tools/reference/export_ultralytics_detections.py";
                }
            }

            /** @brief Returns the detections of @p image above the task's score, clipped. */
            Result<std::vector<Detection>> detect(const std::filesystem::path& image);

            /** @brief Imports the row's checkpoint and prepares the step and the decoder. */
            Status prepare();

            /** @brief Expects, on every image of @p images, the detections Ultralytics made. */
            void expect_ultralytics_detections(const std::vector<ReferenceImage>& images);

            std::filesystem::path folder_;                       /**< The data folder. */
            Config::ModelSpec spec_;                             /**< The imported model. */
            Vision::Detect::Free::FreeTask task_;                /**< Its task section. */
            Vision::Training::FreeModelFacts facts_;             /**< Its levels. */
            std::unique_ptr<NN::ModelState> state_;              /**< Its weights. */
            std::unique_ptr<Graph::TensorPool> pool_;            /**< The step's tensors. */
            std::unique_ptr<Train::EvaluationStep> step_;        /**< The compiled step. */
            std::unique_ptr<Vision::Training::FreeBatchDecoder> decoder_; /**< The decoder. */
            Shape batch_shape_;                                  /**< `[1, H, W, C]`. */
    };

    Status Large_UltralyticsDetections::prepare()
    {
        const std::filesystem::path checkpoint = folder_ / (std::string(GetParam().stem) + ".pt");
        SA_TRY_ASSIGN(const IO::ImportedModel imported,
                      Vision::Importers::import_ultralytics_files(
                          SushiRuntime::span<const std::filesystem::path>(&checkpoint, 1)));
        spec_ = imported.model;
        SA_TRY_ASSIGN(task_, Vision::Detect::Free::read_free_task(spec_.task));
        const Shape& sample = spec_.input_shape;
        batch_shape_ = Shape{1, sample[0], sample[1], sample[2]};

        SushiBLAS::Engine& engine = backend().engine();
        NN::NodeRegistry registry;
        SA_TRY(NN::register_layers(registry));
        SA_TRY(Vision::register_nodes(registry));
        SA_TRY_ASSIGN(NN::ModelGraph graph, NN::build_model(registry, spec_));
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
        std::vector<Train::OutputShape> shapes;
        for (const NN::ModelOutputs::Entry& published : traced.outputs.all())
            shapes.push_back(
                Train::OutputShape{published.first, traced.graph.value(published.second).shape});
        SA_TRY_ASSIGN(facts_, Vision::Training::derive_free_model_facts(sample, shapes));
        SA_TRY_ASSIGN(Vision::Training::FreeBatchDecoder decoder,
                      Vision::Training::FreeBatchDecoder::make(facts_, 1));
        decoder_ = std::make_unique<Vision::Training::FreeBatchDecoder>(std::move(decoder));
        return Status();
    }

    /**
     * @brief Copies the elements of @p tensor into a new entry of @p copies.
     * @return A view of the entry, valid while @p copies holds it, or the failure of the copy.
     */
    Result<SushiRuntime::span<const float>> host_copy_of(const SushiBLAS::Tensor& tensor,
                                                         std::vector<std::vector<float>>& copies)
    {
        SA_TRY_ASSIGN(std::vector<float> values, ::SushiAI::read_elements<float>(tensor));
        copies.push_back(std::move(values));
        return SushiRuntime::span<const float>(copies.back().data(), copies.back().size());
    }

    Result<std::vector<Detection>> Large_UltralyticsDetections::detect(
        const std::filesystem::path& image)
    {
        Vision::Codecs::DecodeOptions options;
        options.layout = Vision::Codecs::ChannelLayout::Color;
        SA_TRY_ASSIGN(const Vision::Imaging::Image pixels,
                      Vision::Codecs::load_image(image, options));
        SA_TRY_ASSIGN(SushiBLAS::Tensor input, allocate_tensor(backend().engine(), batch_shape_));
        const std::size_t count = static_cast<std::size_t>(input.num_elements);
        SA_FAIL_IF(static_cast<std::size_t>(pixels.size().height) * pixels.size().width *
                           pixels.channels() != count,
                   StatusCode::Incompatible, "{} is not the model's input size", image.string());
        std::vector<float> values(count);
        for (std::size_t i = 0; i < count; ++i)
            values[i] = static_cast<float>(pixels.data()[i]) / 255.0f;
        SA_TRY(::SushiAI::write_elements(input, values));
        SA_TRY(step_->run(input));

        std::vector<Vision::Training::FreeLevelLogits> levels;
        std::vector<std::vector<float>> copies;
        for (const Vision::Training::FreeLevelFacts& level : facts_.levels)
        {
            Vision::Training::FreeLevelLogits bound;
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const boxes,
                          step_->output(level.one_to_many.boxes));
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const scores,
                          step_->output(level.one_to_many.scores));
            SA_TRY_ASSIGN(bound.boxes, host_copy_of(*boxes, copies));
            SA_TRY_ASSIGN(bound.scores, host_copy_of(*scores, copies));
            levels.push_back(bound);
        }
        const std::size_t anchors = static_cast<std::size_t>(decoder_->anchors());
        std::vector<float> boxes(anchors * K_BOX_ENTRIES);
        std::vector<float> scores(anchors * static_cast<std::size_t>(facts_.classes));
        SA_TRY(decoder_->decode(SushiRuntime::span<const Vision::Training::FreeLevelLogits>(
                                    levels.data(), levels.size()),
                                boxes, scores));

        Vision::Detect::NMSConfig config;
        config.iou_threshold = task_.suppression_threshold;
        config.label_mode = Vision::Detect::LabelMode::BestClass;
        config.criterion = task_.criterion;
        config.distance_exponent = task_.distance_exponent;
        std::vector<std::vector<float>> found;
        SA_TRY(Vision::Detect::select_nms_detections(
            boxes, scores,
            Vision::Detect::PredictionShape{1, static_cast<std::int64_t>(anchors),
                                            facts_.classes},
            config, found));

        // Ultralytics clips every corner to the image; KI-22 records that sa inference does not.
        const double sides[4] = {static_cast<double>(batch_shape_[2]),
                                 static_cast<double>(batch_shape_[1]),
                                 static_cast<double>(batch_shape_[2]),
                                 static_cast<double>(batch_shape_[1])};
        std::vector<Detection> out;
        const std::vector<float>& rows = found.front();
        for (std::size_t at = 0; at + Vision::Detect::K_DETECTION_VALUES <= rows.size();
             at += Vision::Detect::K_DETECTION_VALUES)
        {
            const float* const row = rows.data() + at;
            if (row[Vision::Detect::K_DETECTION_SCORE] <= task_.score_threshold)
                break;
            Detection detection;
            detection.class_index =
                static_cast<std::int64_t>(row[Vision::Detect::K_DETECTION_CLASS]);
            detection.score = row[Vision::Detect::K_DETECTION_SCORE];
            for (std::size_t k = 0; k < K_BOX_ENTRIES; ++k)
                detection.corners[k] = std::clamp(static_cast<double>(row[k]), 0.0, sides[k]);
            out.push_back(detection);
        }
        return out;
    }

    void Large_UltralyticsDetections::expect_ultralytics_detections(
        const std::vector<ReferenceImage>& images)
    {
        for (const ReferenceImage& image : images)
        {
            const Result<std::vector<Detection>> found = detect(image.file);
            ASSERT_TRUE(found.has_value()) << found.status().message();
            expect_same_detections(found.value(), image.expected,
                                   static_cast<double>(batch_shape_[1]),
                                   image.file.filename().string());
        }
    }
} // namespace

/**
 * @brief Tests that the imported checkpoint finds, on every reference image, the detections
 *        Ultralytics found: the same count and classes, each score within 2048 u and each
 *        corner within 2048 u of the input side.
 */
TEST_P(Large_UltralyticsDetections, TheDetectionsAreUltralyticsOwn)
{
    const Status prepared = prepare();
    ASSERT_TRUE(prepared.is_ok()) << prepared.message();
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<ReferenceImage> images,
                            read_reference(folder_, GetParam().stem));
    ASSERT_FALSE(images.empty());

    expect_ultralytics_detections(images);
}

INSTANTIATE_TEST_SUITE_P(Released, Large_UltralyticsDetections,
                         ::testing::ValuesIn(K_CHECKPOINTS),
                         [](const ::testing::TestParamInfo<CheckpointRow>& info)
                         { return std::string(info.param.name); });
