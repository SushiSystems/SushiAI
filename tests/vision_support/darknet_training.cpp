/**************************************************************************/
/* darknet_training.cpp                                                   */
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
 * @file darknet_training.cpp
 * @brief Implements the loaders of a head's dumped training pass and the training suite the
 *        six model tests run against those dumps.
 * @author Mustafa Garip
 */

#include <vision_support/darknet_training.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
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
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_training.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>
#include <SushiAI/vision/training/anchor_objective.hpp>
#include <SushiAI/vision/training/anchor_training_step.hpp>
#include <vision_support/darknet_forward.hpp>
#include <vision_support/darknet_reference.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

namespace SushiAI
{
    namespace Test
    {
        namespace
        {
            using Vision::Detect::GroundTruthBatch;
            using Vision::Detect::GroundTruthRows;

            /** @brief Holds the bytes of one stored element. */
            constexpr std::size_t K_ELEMENT_BYTES = 4;

            /** @brief Holds the largest element count the loader accepts. */
            constexpr std::int64_t K_MAX_ELEMENTS = std::int64_t{1} << 31;

            /** @brief Holds the values of a box in a truth row. */
            constexpr std::size_t K_BOX_VALUES = 4;

            /** @brief Holds the column names of a truth row's box, in the order a row is fed. */
            constexpr std::string_view K_BOX_COLUMNS[K_BOX_VALUES] = {"x_center", "y_center",
                                                                      "width", "height"};

            /** @brief Returns the string @p key of @p root, or an empty one when it has none. */
            std::string text_of(const Config::Value& root, std::string_view key,
                                std::string_view origin)
            {
                if (!root.has(key) || root.at(key, origin).kind() != Config::Kind::String)
                    return std::string();

                return root.at(key, origin).as_string(origin);
            }

            /**
             * @brief Reads the `shape` of @p root into @p extents.
             * @return Success, or InvalidFile for a missing shape, an entry that is not a whole
             *         number above zero, or a product above K_MAX_ELEMENTS.
             */
            Status read_extents(const Config::Value& root, std::string_view origin,
                                std::vector<std::int64_t>& extents)
            {
                SA_FAIL_IF(!root.has("shape") ||
                               root.at("shape", origin).kind() != Config::Kind::Array,
                           StatusCode::InvalidFile, "{} has no array 'shape'", origin);

                std::int64_t count = 1;
                for (const Config::Value& entry : root.at("shape", origin).as_array(origin))
                {
                    SA_FAIL_IF(entry.kind() != Config::Kind::Int, StatusCode::InvalidFile,
                               "'shape' of {} holds a value that is not a whole number", origin);
                    const std::int64_t extent = entry.as_int(origin);
                    SA_FAIL_IF(extent < 1 || extent > K_MAX_ELEMENTS / count,
                               StatusCode::InvalidFile, "'shape' of {} holds the extent {}",
                               origin, extent);
                    count *= extent;
                    extents.push_back(extent);
                }

                return Status();
            }

            /**
             * @brief Reads @p count little-endian float32 values, the whole of the file @p path.
             * @return The values; FileNotFound when the file cannot be opened; InvalidFile when
             *         it does not hold exactly @p count values.
             */
            Result<std::vector<float>> read_floats(const std::filesystem::path& path,
                                                   std::size_t count)
            {
                std::ifstream in(path, std::ios::binary | std::ios::ate);
                SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                           path.string());
                const std::streamoff bytes = in.tellg();
                SA_FAIL_IF(bytes < 0 ||
                               static_cast<std::size_t>(bytes) != count * K_ELEMENT_BYTES,
                           StatusCode::InvalidFile,
                           "'{}' holds {} bytes, but its shape names {} values of {} bytes",
                           path.string(), bytes, count, K_ELEMENT_BYTES);

                std::vector<unsigned char> raw(count * K_ELEMENT_BYTES);
                in.seekg(0);
                in.read(reinterpret_cast<char*>(raw.data()),
                        static_cast<std::streamsize>(raw.size()));
                SA_FAIL_IF(!in && !raw.empty(), StatusCode::InvalidFile,
                           "'{}' ended before its {} bytes", path.string(), raw.size());

                std::vector<float> values(count);
                for (std::size_t i = 0; i < count; ++i)
                {
                    const unsigned char* const at = raw.data() + i * K_ELEMENT_BYTES;
                    const std::uint32_t bits =
                        static_cast<std::uint32_t>(at[0]) | static_cast<std::uint32_t>(at[1]) << 8 |
                        static_cast<std::uint32_t>(at[2]) << 16 |
                        static_cast<std::uint32_t>(at[3]) << 24;
                    std::memcpy(&values[i], &bits, sizeof(bits));
                }

                return Result<std::vector<float>>(std::move(values));
            }

            /** @brief Returns the product of @p extents. */
            std::size_t count_elements(const std::vector<std::int64_t>& extents) noexcept
            {
                std::size_t count = 1;
                for (const std::int64_t extent : extents)
                    count *= static_cast<std::size_t>(extent);

                return count;
            }

            /** @brief Returns the whole numbers nearest to @p values. */
            std::vector<std::int64_t> round_values(const std::vector<float>& values)
            {
                std::vector<std::int64_t> rounded;
                rounded.reserve(values.size());
                for (const float value : values)
                    rounded.push_back(static_cast<std::int64_t>(std::llround(value)));

                return rounded;
            }

            /**
             * @brief Loads the assignment maps @p kind of a head into @p assigned.
             * @return Success, or Incompatible when the maps do not hold one value per slot.
             */
            Status load_assignment(std::string_view run, std::int64_t yolo_layer,
                                   std::string_view kind, const DarknetTensor& input,
                                   std::vector<std::int64_t>& assigned)
            {
                const std::string name = name_darknet_layer(yolo_layer, kind);
                SA_TRY_ASSIGN(const DarknetTensor maps, load_darknet_maps(run, name));
                SA_FAIL_IF(maps.extents[0] != input.extents[0] ||
                               maps.extents[1] != input.extents[1] ||
                               maps.extents[2] != input.extents[2],
                           StatusCode::Incompatible,
                           "{} of {} does not have the images and the grid of its head", name,
                           run);
                assigned = round_values(maps.values);
                return Status();
            }

            /** @brief Holds where each value of a truth row is; -1 for a column a run lacks. */
            struct TruthColumns
            {
                std::int64_t image = -1;                             /**< The image's index. */
                std::int64_t class_index = -1;                       /**< The class. */
                std::int64_t box[K_BOX_VALUES] = {-1, -1, -1, -1};   /**< Centre, then size. */
            };

            /**
             * @brief Reads the positions of the columns from the `columns` of @p description.
             * @return The positions, or InvalidFile for a box or a class column that is absent.
             */
            Result<TruthColumns> read_truth_columns(const Config::Value& description,
                                                    std::string_view origin)
            {
                const std::string listed = text_of(description, "columns", origin);
                TruthColumns columns;
                std::int64_t index = 0;
                std::size_t begin = 0;
                while (begin < listed.size())
                {
                    std::size_t end = listed.find(' ', begin);
                    end = end == std::string::npos ? listed.size() : end;
                    const std::string_view word(listed.data() + begin, end - begin);
                    if (word == "image")
                        columns.image = index;
                    if (word == "class")
                        columns.class_index = index;
                    for (std::size_t i = 0; i < K_BOX_VALUES; ++i)
                    {
                        if (word == K_BOX_COLUMNS[i])
                            columns.box[i] = index;
                    }

                    index += word.empty() ? 0 : 1;
                    begin = end + 1;
                }

                bool complete = columns.class_index >= 0;
                for (const std::int64_t position : columns.box)
                    complete = complete && position >= 0;
                SA_FAIL_IF(!complete, StatusCode::InvalidFile,
                           "'columns' of {} is '{}', which lacks a box or a class column", origin,
                           listed);

                return Result<TruthColumns>(columns);
            }

            /** @brief Returns whether @p value is a whole number in [0, @p limit). */
            bool is_index_below(float value, std::int64_t limit) noexcept
            {
                return value >= 0.0f && value < static_cast<float>(limit) &&
                       std::floor(value) == value;
            }

            /**
             * @brief Reads the reference tensor @p name of the run folder @p run as it is stored.
             * @return The tensor with the extents and the element order of the file; FileNotFound
             *         naming a missing file; InvalidFile for a `dtype` other than float32, a
             *         `byte_order` other than little, a malformed shape or a `.bin` of another
             *         size.
             */
            Result<DarknetTensor> load_stored(std::string_view run, std::string_view name)
            {
                const DarknetReference reference = find_darknet_reference(run, name);
                SA_FAIL_IF(!reference.found, StatusCode::FileNotFound, "{}", reference.reason);

                const std::string origin = reference.description.string();
                DarknetTensor tensor;
                tensor.description = Config::parse_json_file(reference.description);
                SA_FAIL_IF(text_of(tensor.description, "dtype", origin) != "float32" ||
                               text_of(tensor.description, "byte_order", origin) != "little",
                           StatusCode::InvalidFile, "{} does not describe little-endian float32",
                           origin);
                SA_TRY(read_extents(tensor.description, origin, tensor.extents));
                SA_TRY_ASSIGN(tensor.values,
                              read_floats(reference.payload, count_elements(tensor.extents)));

                return Result<DarknetTensor>(std::move(tensor));
            }
        } // namespace

        Result<DarknetTensor> load_darknet_maps(std::string_view run, std::string_view name)
        {
            SA_TRY_ASSIGN(DarknetTensor tensor, load_stored(run, name));

            const std::string order = text_of(tensor.description, "order", name);
            const bool single = order == "CHW" && tensor.extents.size() == 3;
            const bool batched =
                (order == "NCHW" || order == "OIHW") && tensor.extents.size() == 4;
            SA_FAIL_IF(!single && !batched, StatusCode::Unsupported,
                       "{} of {} orders {} axes as '{}', and the loader moves CHW, NCHW and OIHW",
                       name, run, tensor.extents.size(), order);
            if (single)
                tensor.extents.insert(tensor.extents.begin(), 1);

            const std::size_t images = static_cast<std::size_t>(tensor.extents[0]);
            const std::size_t channels = static_cast<std::size_t>(tensor.extents[1]);
            const std::size_t positions =
                static_cast<std::size_t>(tensor.extents[2] * tensor.extents[3]);
            std::vector<float> moved(tensor.values.size());
            for (std::size_t image = 0; image < images; ++image)
            {
                const std::size_t base = image * channels * positions;
                for (std::size_t channel = 0; channel < channels; ++channel)
                {
                    const float* const from = tensor.values.data() + base + channel * positions;
                    for (std::size_t position = 0; position < positions; ++position)
                        moved[base + position * channels + channel] = from[position];
                }
            }

            tensor.values = std::move(moved);
            tensor.extents = {tensor.extents[0], tensor.extents[2], tensor.extents[3],
                              tensor.extents[1]};
            return Result<DarknetTensor>(std::move(tensor));
        }

        Result<DarknetHeadDump> load_darknet_head_dump(std::string_view run,
                                                       std::int64_t yolo_layer)
        {
            const std::string delta_name = name_darknet_layer(yolo_layer, "yolo_delta");
            SA_TRY_ASSIGN(DarknetTensor input,
                          load_darknet_maps(run, name_darknet_layer(yolo_layer - 1,
                                                                    "yolo_input")));
            SA_TRY_ASSIGN(DarknetTensor delta, load_darknet_maps(run, delta_name));
            SA_FAIL_IF(delta.extents != input.extents, StatusCode::Incompatible,
                       "{} of {} does not have the extents of its head", delta_name, run);

            DarknetHeadDump dump;
            dump.shape = {input.extents[0], input.extents[1], input.extents[2]};
            if (delta.description.has("cost"))
            {
                dump.cost = delta.description.at("cost", delta_name).as_real(delta_name);
            }
            else
            {
                SA_TRY_ASSIGN(const DarknetTensor cost,
                              load_darknet_reference(
                                  run, name_darknet_layer(yolo_layer, "yolo_cost")));
                SA_FAIL_IF(cost.values.size() != 1, StatusCode::Incompatible,
                           "the stored cost of layer {} of {} holds {} values", yolo_layer, run,
                           cost.values.size());
                dump.cost = static_cast<double>(cost.values.front());
            }

            const std::string row_name = name_darknet_layer(yolo_layer, "yolo_assigned_row");
            dump.has_assignments = find_darknet_reference(run, row_name).found;
            if (dump.has_assignments)
            {
                SA_TRY(load_assignment(run, yolo_layer, "yolo_assigned_row", input,
                                       dump.assigned_row));
                SA_TRY(load_assignment(run, yolo_layer, "yolo_assigned_class", input,
                                       dump.assigned_class));
            }

            dump.input = std::move(input.values);
            dump.delta = std::move(delta.values);
            dump.settings = std::move(delta.description);
            return Result<DarknetHeadDump>(std::move(dump));
        }

        Result<GroundTruthBatch> load_darknet_truth(std::string_view run, std::int64_t batch,
                                                    Vision::Detect::Anchor::NetworkSize network)
        {
            SA_TRY_ASSIGN(const DarknetTensor truth,
                          load_darknet_reference(run, K_DARKNET_TRUTH));
            SA_FAIL_IF(truth.extents.size() != 2 || batch < 1, StatusCode::Incompatible,
                       "the truth of {} has {} axes and is asked for {} images", run,
                       truth.extents.size(), batch);
            SA_TRY_ASSIGN(const TruthColumns columns,
                          read_truth_columns(truth.description, run));

            const std::size_t rows = static_cast<std::size_t>(truth.extents[0]);
            const std::size_t width = static_cast<std::size_t>(truth.extents[1]);
            std::vector<std::int64_t> images(rows, 0);
            std::vector<std::int64_t> classes(rows, 0);
            std::vector<float> boxes(rows * K_BOX_VALUES, 0.0f);
            for (std::size_t row = 0; row < rows; ++row)
            {
                const float* const values = truth.values.data() + row * width;
                const float image = columns.image < 0 ? 0.0f : values[columns.image];
                const float label = values[columns.class_index];
                SA_FAIL_IF(!is_index_below(image, batch) ||
                               !is_index_below(label, K_MAX_ELEMENTS),
                           StatusCode::Incompatible,
                           "row {} of the truth of {} names image {} of {} and class {}", row,
                           run, image, batch, label);

                images[row] = static_cast<std::int64_t>(image);
                classes[row] = static_cast<std::int64_t>(label);
                for (std::size_t i = 0; i < K_BOX_VALUES; ++i)
                    boxes[row * K_BOX_VALUES + i] = values[columns.box[i]];
            }

            GroundTruthRows listed;
            listed.image_indices = SushiRuntime::span<const std::int64_t>(images.data(), rows);
            listed.classes = SushiRuntime::span<const std::int64_t>(classes.data(), rows);
            listed.boxes = SushiRuntime::span<const float>(boxes.data(), boxes.size());

            // Every row names an image of the batch, so the padding has nothing to refuse.
            GroundTruthBatch padded;
            Vision::Detect::pad_ground_truth(listed, batch, static_cast<float>(network.width),
                                             static_cast<float>(network.height), padded);
            return Result<GroundTruthBatch>(std::move(padded));
        }

        ::testing::AssertionResult delta_within(std::string_view what,
                                                SushiRuntime::span<const float> actual,
                                                SushiRuntime::span<const float> reference,
                                                std::int64_t slot_entries,
                                                const DarknetDeltaBounds& bounds)
        {
            const std::size_t entries = static_cast<std::size_t>(slot_entries);
            if (slot_entries < 1 || actual.size() != reference.size() ||
                actual.size() % entries != 0)
                return ::testing::AssertionFailure()
                       << what << ": " << actual.size() << " values against " << reference.size()
                       << ", in slots of " << slot_entries;

            const std::size_t box_entries =
                static_cast<std::size_t>(Vision::Detect::Anchor::K_OBJECTNESS_ENTRY);
            std::vector<float> box_actual;
            std::vector<float> box_reference;
            std::vector<float> other_actual;
            std::vector<float> other_reference;
            for (std::size_t i = 0; i < actual.size(); ++i)
            {
                const bool is_box = i % entries < box_entries;
                (is_box ? box_actual : other_actual).push_back(actual[i]);
                (is_box ? box_reference : other_reference).push_back(reference[i]);
            }

            const ToleranceGap box = measure_gap(box_actual, box_reference, bounds.box_absolute,
                                                 bounds.box_relative);
            const ToleranceGap other =
                measure_gap(other_actual, other_reference, bounds.other_absolute, 0.0);
            std::cout << "[ MEASURED ] " << what << ": largest box difference " << box.largest
                      << ", bound " << bounds.box_absolute << " + " << bounds.box_relative
                      << " * |reference|; largest other difference " << other.largest
                      << ", bound " << bounds.other_absolute << "\n";
            if (box.outside == 0 && other.outside == 0)
                return ::testing::AssertionSuccess();

            return ::testing::AssertionFailure()
                   << what << ": " << box.outside << " of " << box_actual.size()
                   << " box entries are outside their bound, the worst by " << box.largest
                   << " (" << box_actual[box.at] << " against " << box_reference[box.at]
                   << "); " << other.outside << " of " << other_actual.size()
                   << " other entries are outside theirs, the worst by " << other.largest << " ("
                   << other_actual[other.at] << " against " << other_reference[other.at] << ")";
        }

        namespace
        {
            using Vision::Detect::Anchor::BoxLoss;
            using Vision::Detect::Anchor::HeadCost;
            using Vision::Detect::Anchor::HeadDelta;
            using Vision::Detect::Anchor::HeadOptions;
            using Vision::Detect::Anchor::HeadTrainingOptions;
            using Vision::Detect::Anchor::ObjectnessTarget;
            using Vision::Importers::DarknetConfig;
            using Vision::Importers::DarknetHead;
            using Vision::Importers::DarknetModel;
            using Vision::Importers::DarknetTraining;
            using Vision::Importers::DarknetWeightTarget;
            using Vision::Training::AnchorHead;
            using Vision::Training::AnchorObjective;
            using Vision::Training::AnchorTrainingStep;

            /** @brief Holds the name of the first batch norm's rolling mean before a step. */
            constexpr std::string_view K_ROLLING_MEAN_BEFORE = "rolling_mean_before";

            /** @brief Holds the name of that rolling mean after the step. */
            constexpr std::string_view K_ROLLING_MEAN_AFTER = "rolling_mean_after";

            /** @brief Holds the name of the first batch norm's rolling variance before a step. */
            constexpr std::string_view K_ROLLING_VARIANCE_BEFORE = "rolling_variance_before";

            /** @brief Holds the name of that rolling variance after the step. */
            constexpr std::string_view K_ROLLING_VARIANCE_AFTER = "rolling_variance_after";

            /** @brief Holds the key that names the layer whose updates a step run dumped. */
            constexpr std::string_view K_FIRST_LAYER_KEY = "first_batch_norm_layer";

            /** @brief Names a loader of one reference tensor of a run folder. */
            using TensorLoader = Result<DarknetTensor> (*)(std::string_view run,
                                                           std::string_view name);

            /** @brief Holds the translated cfg and the training options read from it. */
            struct ImportedTraining
            {
                DarknetModel model;       /**< The Separate form of the model. */
                DarknetTraining training; /**< The solver and each head's options. */
            };

            /** @brief Imports `<stem>.cfg` with batch norm kept as layers. */
            Result<ImportedTraining> import_training(std::string_view stem)
            {
                SA_TRY_ASSIGN(const DarknetConfig config, load_darknet_config_file(stem));

                ImportedTraining imported;
                SA_TRY_ASSIGN(imported.model,
                              Vision::Importers::translate_darknet_model(config, {}));
                SA_TRY_ASSIGN(imported.training,
                              Vision::Importers::read_darknet_training(config));
                SA_FAIL_IF(imported.training.heads.size() != imported.model.heads.size(),
                           StatusCode::Incompatible,
                           "{}.cfg gives {} heads and training options for {}", stem,
                           imported.model.heads.size(), imported.training.heads.size());
                return Result<ImportedTraining>(std::move(imported));
            }

            /**
             * @brief Reads the layer whose updates and statistics the step run dumped.
             * @return The layer, or InvalidFile when @p solver does not name one.
             */
            Result<std::int64_t> read_first_layer(const DarknetTensor& solver)
            {
                const Config::Value& description = solver.description;
                SA_FAIL_IF(!description.has(K_FIRST_LAYER_KEY) ||
                               description.at(K_FIRST_LAYER_KEY, K_DARKNET_NETWORK_COST).kind() !=
                                   Config::Kind::Int,
                           StatusCode::InvalidFile, "{} names no whole '{}'",
                           K_DARKNET_NETWORK_COST, K_FIRST_LAYER_KEY);

                return Result<std::int64_t>(
                    description.at(K_FIRST_LAYER_KEY, K_DARKNET_NETWORK_COST)
                        .as_int(K_DARKNET_NETWORK_COST));
            }

            /** @brief Lists the references of the host run that the suite reads. */
            std::vector<std::string> list_host_names(const std::vector<std::int64_t>& layers)
            {
                std::vector<std::string> names{std::string(K_DARKNET_TRUTH)};
                for (const std::int64_t layer : layers)
                {
                    names.push_back(name_darknet_layer(layer - 1, "yolo_input"));
                    names.push_back(name_darknet_layer(layer, "yolo_delta"));
                    names.push_back(name_darknet_layer(layer, "yolo_assigned_row"));
                    names.push_back(name_darknet_layer(layer, "yolo_assigned_class"));
                }

                return names;
            }

            /** @brief Lists the references of the step run that the suite reads by fixed name. */
            std::vector<std::string> list_step_names(const std::vector<std::int64_t>& layers)
            {
                std::vector<std::string> names;
                for (const std::string_view name :
                     {K_DARKNET_TRUTH, K_DARKNET_INPUT, K_DARKNET_NETWORK_COST,
                      K_ROLLING_MEAN_BEFORE, K_ROLLING_MEAN_AFTER, K_ROLLING_VARIANCE_BEFORE,
                      K_ROLLING_VARIANCE_AFTER})
                    names.emplace_back(name);
                for (const std::int64_t layer : layers)
                {
                    names.push_back(name_darknet_layer(layer - 1, "yolo_input"));
                    names.push_back(name_darknet_layer(layer, "yolo_delta"));
                    names.push_back(name_darknet_layer(layer, "yolo"));
                    names.push_back(name_darknet_layer(layer, "yolo_cost"));
                    names.push_back(name_darknet_layer(layer - 1, "weight_updates"));
                    names.push_back(name_darknet_layer(layer - 1, "bias_updates"));
                }

                return names;
            }

            /** @brief Returns the float32 elements of @p tensor. */
            std::vector<float> read_elements(const SushiBLAS::Tensor& tensor)
            {
                const float* const values = tensor.data_as<float>();
                return std::vector<float>(values, values + tensor.num_elements);
            }

            /** @brief Returns @p values with every NaN replaced by 0. */
            std::vector<float> clear_nans(std::vector<float> values)
            {
                for (float& value : values)
                    value = std::isnan(value) ? 0.0f : value;

                return values;
            }

            /** @brief Returns the largest magnitude of @p values; 0 for none. */
            double measure_peak(const std::vector<float>& values) noexcept
            {
                double peak = 0.0;
                for (const float value : values)
                    peak = std::max(peak, std::fabs(static_cast<double>(value)));

                return peak;
            }

            /**
             * @brief Checks that two assignment maps are equal.
             * @return Success, or a failure naming the count of slots that differ and the first.
             */
            ::testing::AssertionResult assignments_equal(
                std::string_view what, const std::vector<std::int64_t>& actual,
                const std::vector<std::int64_t>& reference)
            {
                if (actual.size() != reference.size())
                    return ::testing::AssertionFailure()
                           << what << ": " << actual.size() << " slots against "
                           << reference.size();

                std::size_t differing = 0;
                std::size_t first = 0;
                for (std::size_t i = 0; i < actual.size(); ++i)
                {
                    if (actual[i] == reference[i])
                        continue;

                    first = differing == 0 ? i : first;
                    ++differing;
                }
                if (differing == 0)
                    return ::testing::AssertionSuccess();

                return ::testing::AssertionFailure()
                       << what << ": " << differing << " of " << actual.size()
                       << " slots differ, the first at " << first << " (" << actual[first]
                       << " against " << reference[first] << ")";
            }

            /**
             * @brief Checks the library's delta of every head @p run dumped against the dump:
             *        the assignments where the run has them, the delta and the stored cost.
             */
            void check_dumped_deltas(const DarknetTrainingCase& stated, std::string_view run)
            {
                SA_ASSERT_OK_AND_ASSIGN(const ImportedTraining imported,
                                        import_training(stated.stem));
                const DarknetModel& model = imported.model;

                for (std::size_t k = 0; k < model.heads.size(); ++k)
                {
                    const std::string what = std::string(run) + " " + model.heads[k].output;
                    SCOPED_TRACE(what);
                    const std::int64_t index = static_cast<std::int64_t>(k);
                    const HeadOptions& options = model.heads[k].options;
                    const HeadTrainingOptions& training = imported.training.heads[k];
                    SA_ASSERT_OK_AND_ASSIGN(
                        const DarknetHeadDump dump,
                        load_darknet_head_dump(run, model.heads[k].layer));
                    SA_ASSERT_OK_AND_ASSIGN(
                        const Vision::Detect::GroundTruthBatch truth,
                        load_darknet_truth(run, dump.shape.batch, model.network));

                    const Status valid_head =
                        Vision::Detect::Anchor::validate_head(options, model.layout, index);
                    ASSERT_TRUE(valid_head.is_ok()) << valid_head.message();
                    const Status valid_training =
                        Vision::Detect::Anchor::validate_head_training(training);
                    ASSERT_TRUE(valid_training.is_ok()) << valid_training.message();
                    const std::int64_t channels =
                        Vision::Detect::Anchor::count_head_channels(options, model.layout, index);
                    ASSERT_EQ(dump.input.size(),
                              static_cast<std::size_t>(dump.shape.batch * dump.shape.height *
                                                       dump.shape.width * channels));

                    HeadDelta computed;
                    ASSERT_TRUE(::SushiAI::Test::status_is_ok(
                        Vision::Detect::Anchor::compute_head_delta(
                            dump.input, dump.shape, options, model.layout, index, training,
                            truth, model.network, computed)));

                    if (dump.has_assignments)
                    {
                        EXPECT_TRUE(assignments_equal(what + " assigned rows",
                                                      computed.assigned_row, dump.assigned_row));
                        EXPECT_TRUE(assignments_equal(what + " assigned classes",
                                                      computed.assigned_class,
                                                      dump.assigned_class));
                    }
                    EXPECT_TRUE(delta_within(
                        what + " delta", computed.delta, dump.delta,
                        Vision::Detect::Anchor::K_FIRST_CLASS_ENTRY + options.num_classes,
                        stated.bounds.host_delta));

                    const HeadCost cost = Vision::Detect::Anchor::measure_head_cost(
                        computed, dump.shape, options, training);
                    EXPECT_GT(dump.cost, 0.0);
                    EXPECT_NEAR(cost.stored, dump.cost,
                                stated.bounds.host_cost_relative * dump.cost);
                }
            }

            /** @brief Holds an imported model, its two graphs and one step over a dumped batch. */
            struct StepRun
            {
                ImportedTraining imported;                 /**< The cfg's translation. */
                DarknetTensor solver;                      /**< The network cost reference. */
                std::vector<DarknetHeadDump> dumps;        /**< Each head's dump. */
                Vision::Detect::GroundTruthBatch truth;    /**< The dumped truth rows. */
                std::int64_t batch = 0;                    /**< The images of the step. */
                std::int64_t first_layer = 0;              /**< The layer of the first updates. */
                std::optional<NN::ModelGraph> graph;       /**< The built model. */
                std::optional<AnchorObjective> objective;  /**< The heads' objective. */
                Train::TracedStep probe;                   /**< The forward-only graph. */
                Train::TrainingGraph built;                /**< The training graph. */
                std::optional<NN::ModelState> state;       /**< The file's weights. */
                std::unique_ptr<Graph::TensorPool> probe_pool;    /**< Forward-only values. */
                std::unique_ptr<Graph::TensorPool> training_pool; /**< Training values. */
                std::vector<Optim::ParameterSlot> slots;   /**< Each gradient's tensor. */
                std::unique_ptr<Optim::SGD> optimizer;     /**< Plain gradient descent. */
                std::unique_ptr<AnchorTrainingStep> step;  /**< The step under test. */
                std::optional<SushiBLAS::Tensor> inputs;   /**< The dumped batch. */
                HeadCost cost;                             /**< What the step returned. */
                bool is_ready = false;                     /**< Whether prepare_step() ended. */
            };

            /**
             * @brief Builds the model of @p stated with the file's weights and one step over
             *        the batch of its step run; sets `is_ready` when nothing failed.
             */
            void prepare_step(SushiBLAS::Engine& engine, const DarknetTrainingCase& stated,
                              StepRun& run)
            {
                const std::string_view folder = stated.step_run;
                SA_ASSERT_OK_AND_ASSIGN(run.imported, import_training(stated.stem));
                const DarknetModel& model = run.imported.model;

                SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor input,
                                        load_darknet_maps(folder, K_DARKNET_INPUT));
                SA_ASSERT_OK_AND_ASSIGN(
                    run.solver, load_darknet_reference(folder, K_DARKNET_NETWORK_COST));
                SA_ASSERT_OK_AND_ASSIGN(run.first_layer, read_first_layer(run.solver));
                run.batch = input.extents[0];
                const Shape batch_shape{input.extents[0], input.extents[1], input.extents[2],
                                        input.extents[3]};

                std::vector<AnchorHead> heads(model.heads.size());
                for (std::size_t k = 0; k < model.heads.size(); ++k)
                {
                    SA_ASSERT_OK_AND_ASSIGN(
                        DarknetHeadDump dump,
                        load_darknet_head_dump(folder, model.heads[k].layer));
                    ASSERT_EQ(dump.shape.batch, run.batch);
                    heads[k].output = model.heads[k].output;
                    heads[k].shape = dump.shape;
                    heads[k].options = model.heads[k].options;
                    heads[k].training = run.imported.training.heads[k];
                    run.dumps.push_back(std::move(dump));
                }
                SA_ASSERT_OK_AND_ASSIGN(run.truth,
                                        load_darknet_truth(folder, run.batch, model.network));

                NN::NodeRegistry registry;
                const Status registered = NN::register_layers(registry);
                ASSERT_TRUE(registered.is_ok()) << registered.message();
                SA_ASSERT_OK_AND_ASSIGN(run.graph, NN::build_model(registry, model.spec));
                SA_ASSERT_OK_AND_ASSIGN(
                    run.objective,
                    AnchorObjective::make(model.network, model.layout, std::move(heads)));
                SA_ASSERT_OK_AND_ASSIGN(run.probe,
                                        Train::build_probe_graph(*run.graph, batch_shape));
                SA_ASSERT_OK_AND_ASSIGN(
                    run.built,
                    Train::build_training_graph(*run.graph, *run.objective, batch_shape));
                SA_ASSERT_OK_AND_ASSIGN(run.state, NN::ModelState::create(engine, *run.graph));

                const DarknetFile file =
                    find_darknet_file(std::string(stated.stem) + ".weights");
                ASSERT_TRUE(file.found) << file.reason;
                SA_ASSERT_OK_AND_ASSIGN(
                    const Vision::Importers::DarknetWeights weights,
                    Vision::Importers::read_darknet_weights(file.path, model.convolutions));
                const Status stored = Vision::Importers::store_darknet_weights(
                    weights, model.convolutions, model.targets, *run.state);
                ASSERT_TRUE(stored.is_ok()) << stored.message();

                run.probe_pool = std::make_unique<Graph::TensorPool>(engine);
                run.training_pool = std::make_unique<Graph::TensorPool>(engine);
                SA_ASSERT_OK_AND_ASSIGN(
                    run.slots, Train::bind_parameters(run.built, *run.state, *run.training_pool));
                SA_ASSERT_OK_AND_ASSIGN(
                    std::vector<Train::BufferUpdateSlot> updates,
                    Train::bind_buffer_updates(run.built.updates, *run.state,
                                               *run.training_pool));

                // Momentum and decay stay 0, so the step adds nothing to a gradient it reads.
                Optim::SGDOptions options;
                options.learning_rate = run.solver.description
                                            .at("learning_rate", K_DARKNET_NETWORK_COST)
                                            .as_real(K_DARKNET_NETWORK_COST);
                SA_ASSERT_OK_AND_ASSIGN(run.optimizer,
                                        Optim::SGD::make(engine, run.slots, options));
                SA_ASSERT_OK_AND_ASSIGN(
                    run.step,
                    AnchorTrainingStep::make(engine, *run.objective, run.probe, *run.probe_pool,
                                             run.built, *run.training_pool, *run.state,
                                             *run.optimizer, std::move(updates)));

                SA_ASSERT_OK_AND_ASSIGN(run.inputs, allocate_tensor(engine, batch_shape));
                ASSERT_EQ(static_cast<std::size_t>(run.inputs->num_elements),
                          input.values.size());
                std::memcpy(run.inputs->data_as<float>(), input.values.data(),
                            input.values.size() * sizeof(float));
                run.is_ready = true;
            }

            /** @brief Runs the one step over the dumped batch and truth; keeps its cost. */
            void run_step(StepRun& run)
            {
                run.is_ready = false;
                SA_ASSERT_OK_AND_ASSIGN(run.cost, run.step->run(*run.inputs, run.truth));
                run.is_ready = true;
            }

            /** @brief Returns the weight targets of the convolution at layer @p layer, or null. */
            const DarknetWeightTarget* find_targets(const StepRun& run, std::int64_t layer)
            {
                const DarknetModel& model = run.imported.model;
                for (std::size_t i = 0; i < model.convolutions.size(); ++i)
                {
                    if (model.convolutions[i].layer == layer)
                        return &model.targets[i];
                }

                return nullptr;
            }

            /**
             * @brief Reads the gradient of the parameter @p name as the update a run dumps.
             * @return The gradient times minus the batch: the root is minus the mean over the
             *         batch of delta times head, and a dumped update is the sum of the descent
             *         direction. Empty, with a failure recorded, for no such parameter.
             */
            std::vector<float> read_update(const StepRun& run, const std::string& name)
            {
                for (const Optim::ParameterSlot& slot : run.slots)
                {
                    if (slot.name != name)
                        continue;

                    std::vector<float> update = read_elements(slot.gradient);
                    for (float& value : update)
                        value = static_cast<float>(-static_cast<double>(run.batch) * value);
                    return update;
                }

                ADD_FAILURE() << "the training graph has no parameter '" << name << "'";
                return {};
            }

            /** @brief Returns the state entry @p name; empty, with a failure, when absent. */
            std::vector<float> read_state(StepRun& run, const std::string& name)
            {
                const Result<SushiBLAS::Tensor*> found = run.state->tensor(name);
                SA_EXPECT_OK(found);
                return found.has_value() ? read_elements(*found.value()) : std::vector<float>{};
            }

            /**
             * @brief Checks the update of @p parameter against the reference @p name.
             * @param load Loads the reference in the layout the parameter is stored in.
             * @param share The absolute bound, as a share of the reference's largest magnitude.
             */
            void check_update(const StepRun& run, const DarknetTrainingCase& stated,
                              const std::string& parameter, const std::string& name,
                              TensorLoader load, double share)
            {
                SCOPED_TRACE(name);
                ASSERT_FALSE(parameter.empty()) << "the importer names no entry for " << name;

                SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor expected,
                                        load(stated.step_run, name));
                const double peak = measure_peak(expected.values);
                EXPECT_GT(peak, 0.0) << name << " is zero in the dump and pins nothing";
                EXPECT_TRUE(within_tolerance(name, read_update(run, parameter), expected.values,
                                             share * peak, stated.bounds.update_relative));
            }

            /** @brief Returns the number @p key of @p settings, widened to float. */
            float read_setting(const Config::Value& settings, std::string_view key)
            {
                return static_cast<float>(settings.at(key, key).as_real(key));
            }
        } // namespace

        std::string find_missing_training_file(const DarknetTrainingCase& stated)
        {
            const std::vector<std::int64_t> layers =
                find_darknet_parity_model(stated.stem).yolo_layers;
            const std::vector<std::string> host = list_host_names(layers);
            const std::vector<std::string> step = list_step_names(layers);

            std::string missing = find_missing_darknet_file(stated.stem, stated.host_run, host);
            if (missing.empty())
                missing = find_missing_darknet_file(stated.stem, stated.step_run, step);
            return missing;
        }

        DarknetTrainingTest::DarknetTrainingTest(DarknetTrainingCase stated)
            : training(std::move(stated))
        {

        }

        void DarknetTrainingTest::SetUp()
        {
            EngineTest::SetUp();
            const std::string missing = find_missing_training_file(training);
            if (!missing.empty())
                GTEST_SKIP() << missing;
        }

        void check_training_settings(const DarknetTrainingCase& stated)
        {
            SA_ASSERT_OK_AND_ASSIGN(const ImportedTraining imported,
                                    import_training(stated.stem));
            const DarknetModel& model = imported.model;

            for (std::size_t k = 0; k < model.heads.size(); ++k)
            {
                SCOPED_TRACE(model.heads[k].output);
                const HeadTrainingOptions& training = imported.training.heads[k];
                SA_ASSERT_OK_AND_ASSIGN(
                    const DarknetHeadDump dump,
                    load_darknet_head_dump(stated.host_run, model.heads[k].layer));
                const Config::Value& settings = dump.settings;
                for (const std::string_view key :
                     {"iou_normalizer", "obj_normalizer", "ignore_thresh", "truth_thresh",
                      "iou_thresh", "objectness_smooth", "max_delta"})
                    ASSERT_TRUE(settings.has(key)) << "the delta's description has no " << key;

                EXPECT_EQ(training.box_loss, stated.box_loss);
                EXPECT_EQ(training.objectness_target, stated.objectness_target);
                EXPECT_EQ(read_setting(settings, "objectness_smooth") != 0.0f,
                          training.objectness_target == ObjectnessTarget::BestIoU);
                EXPECT_FLOAT_EQ(training.box_weight, read_setting(settings, "iou_normalizer"));
                EXPECT_FLOAT_EQ(training.object_weight,
                                read_setting(settings, "obj_normalizer"));
                EXPECT_FLOAT_EQ(training.thresholds.ignore_threshold,
                                read_setting(settings, "ignore_thresh"));
                EXPECT_FLOAT_EQ(training.thresholds.match_threshold,
                                read_setting(settings, "iou_thresh"));
                EXPECT_FLOAT_EQ(read_setting(settings, "truth_thresh"), 1.0f);

                // A run without a bound dumps the largest float; the library spells that 0.
                const float dumped = read_setting(settings, "max_delta");
                const bool is_bounded = dumped < std::numeric_limits<float>::max();
                EXPECT_FLOAT_EQ(training.max_delta, is_bounded ? dumped : 0.0f);
            }
        }

        void check_host_deltas(const DarknetTrainingCase& stated)
        {
            check_dumped_deltas(stated, stated.host_run);
        }

        void check_two_image_deltas(const DarknetTrainingCase& stated)
        {
            check_dumped_deltas(stated, stated.step_run);
        }

        void check_step_heads(SushiBLAS::Engine& engine, const DarknetTrainingCase& stated)
        {
            StepRun run;
            prepare_step(engine, stated, run);
            ASSERT_TRUE(run.is_ready);
            run_step(run);
            ASSERT_TRUE(run.is_ready);

            for (std::size_t k = 0; k < run.dumps.size(); ++k)
            {
                const DarknetHead& head = run.imported.model.heads[k];
                SCOPED_TRACE(head.output);
                SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const found,
                                        run.step->evaluation().output(head.output));
                const std::vector<float> computed = read_elements(*found);
                EXPECT_TRUE(within_tolerance(head.output + " under batch statistics", computed,
                                             run.dumps[k].input, stated.bounds.head_absolute,
                                             stated.bounds.head_relative));

                SA_ASSERT_OK_AND_ASSIGN(
                    const DarknetTensor output,
                    load_darknet_maps(stated.step_run, name_darknet_layer(head.layer, "yolo")));
                ASSERT_EQ(computed.size(), output.values.size());
                EXPECT_TRUE(within_tolerance(head.output + " [yolo] output",
                                             compute_yolo_output(computed, head.options),
                                             clear_nans(output.values),
                                             stated.bounds.yolo_absolute,
                                             stated.bounds.head_relative));
            }
        }

        void check_step_deltas(SushiBLAS::Engine& engine, const DarknetTrainingCase& stated)
        {
            StepRun run;
            prepare_step(engine, stated, run);
            ASSERT_TRUE(run.is_ready);
            run_step(run);
            ASSERT_TRUE(run.is_ready);

            const SushiRuntime::span<const HeadDelta> deltas = run.step->deltas();
            ASSERT_EQ(deltas.size(), run.dumps.size());

            double layer_costs = 0.0;
            for (std::size_t k = 0; k < run.dumps.size(); ++k)
            {
                const DarknetHead& head = run.imported.model.heads[k];
                EXPECT_TRUE(delta_within(
                    head.output + " delta of the step", deltas[k].delta, run.dumps[k].delta,
                    Vision::Detect::Anchor::K_FIRST_CLASS_ENTRY + head.options.num_classes,
                    stated.bounds.step_delta));
                layer_costs += run.dumps[k].cost;
            }

            // A run's network cost is the plain mean of its layers' stored costs.
            ASSERT_EQ(run.solver.values.size(), 1u);
            const double network_cost = static_cast<double>(run.solver.values.front());
            const double heads = static_cast<double>(run.dumps.size());
            const double relative = stated.bounds.step_cost_relative;
            EXPECT_GT(layer_costs, 0.0);
            EXPECT_NEAR(run.cost.stored, layer_costs, relative * layer_costs);
            EXPECT_NEAR(run.cost.stored / heads, network_cost, relative * network_cost);
        }

        void check_head_updates(SushiBLAS::Engine& engine, const DarknetTrainingCase& stated)
        {
            StepRun run;
            prepare_step(engine, stated, run);
            ASSERT_TRUE(run.is_ready);
            run_step(run);
            ASSERT_TRUE(run.is_ready);

            for (const DarknetHead& head : run.imported.model.heads)
            {
                const std::int64_t layer = head.layer - 1;
                const DarknetWeightTarget* const targets = find_targets(run, layer);
                ASSERT_NE(targets, nullptr) << "the cfg has no convolution at layer " << layer;
                check_update(run, stated, targets->weight,
                             name_darknet_layer(layer, "weight_updates"), &load_darknet_maps,
                             stated.bounds.head_update_share);
                check_update(run, stated, targets->bias,
                             name_darknet_layer(layer, "bias_updates"), &load_darknet_reference,
                             stated.bounds.head_update_share);
            }
        }

        void check_first_gradients(SushiBLAS::Engine& engine, const DarknetTrainingCase& stated)
        {
            StepRun run;
            prepare_step(engine, stated, run);
            ASSERT_TRUE(run.is_ready);
            run_step(run);
            ASSERT_TRUE(run.is_ready);

            const std::int64_t layer = run.first_layer;
            const DarknetWeightTarget* const targets = find_targets(run, layer);
            ASSERT_NE(targets, nullptr) << "the cfg has no convolution at layer " << layer;

            // The dumped updates of this layer are no reference; see tests/integration/README.md.
            for (const std::string* const name :
                 {&targets->weight, &targets->scale, &targets->bias})
            {
                SCOPED_TRACE(*name);
                ASSERT_FALSE(name->empty()) << "the importer names no entry of layer " << layer;

                const std::vector<float> update = read_update(run, *name);
                EXPECT_TRUE(std::all_of(update.begin(), update.end(),
                                        [](float value) { return std::isfinite(value); }))
                    << "a gradient of layer " << layer << " is not finite";
                EXPECT_GT(measure_peak(update), 0.0)
                    << "layer " << layer << " received no gradient";
            }
        }

        namespace
        {
            /** @brief Holds each channel's rolling variance after one step, summed two ways. */
            struct HostRollingVariance
            {
                std::vector<float> exact;      /**< From a batch variance summed in double. */
                std::vector<float> sequential; /**< From one summed term by term in float32. */
            };

            /**
             * @brief Computes what the rolling variance of the first convolution becomes after
             *        one step over the dumped batch, from @p weight and @p before on the host.
             * @return Both sums, or Incompatible when the convolution of @p target does not read
             *         the model input with a square kernel.
             */
            Result<HostRollingVariance> measure_host_rolling_variance(
                const StepRun& run, const DarknetWeightTarget& target,
                const std::vector<float>& weight, const std::vector<float>& before)
            {
                const std::size_t row = static_cast<std::size_t>(
                    std::stoll(target.weight.substr(0, target.weight.find('.'))));
                const Config::ModelSpec& spec = run.imported.model.spec;
                SA_FAIL_IF(row >= spec.layers.size() || spec.layers[row].from.size() != 1 ||
                               spec.layers[row].from[0].node != Config::K_MODEL_INPUT,
                           StatusCode::Incompatible,
                           "'{}' is not a convolution over the model input", target.weight);

                const Config::Value& attributes = spec.layers[row].attributes;
                const std::int64_t kernel = attributes.at("kernel", "conv2d").as_int("kernel");
                const std::int64_t stride = attributes.at("stride", "conv2d").as_int("stride");
                const std::int64_t padding = attributes.at("padding", "conv2d").as_int("padding");

                const SushiBLAS::Tensor& input = *run.inputs;
                const std::int64_t images = input.shape[0];
                const std::int64_t height = input.shape[1];
                const std::int64_t width = input.shape[2];
                const std::int64_t channels = input.shape[3];
                const std::int64_t filters = static_cast<std::int64_t>(before.size());
                const std::int64_t out_height = (height + 2 * padding - kernel) / stride + 1;
                const std::int64_t out_width = (width + 2 * padding - kernel) / stride + 1;
                const std::int64_t count = images * out_height * out_width;
                SA_FAIL_IF(static_cast<std::int64_t>(weight.size()) !=
                               filters * kernel * kernel * channels,
                           StatusCode::Incompatible,
                           "'{}' holds {} values, and {} filters of {} by {} over {} channels "
                           "need {}",
                           target.weight, weight.size(), filters, kernel, kernel, channels,
                           filters * kernel * kernel * channels);

                const float* const pixels = input.data_as<float>();
                const double momentum = NN::NormOptions{}.momentum;
                HostRollingVariance result;
                std::vector<float> output(static_cast<std::size_t>(count));
                for (std::int64_t f = 0; f < filters; ++f)
                {
                    // The weights are stored filter, row, column, channel.
                    const float* const taps = weight.data() + f * kernel * kernel * channels;
                    std::size_t at = 0;
                    for (std::int64_t n = 0; n < images; ++n)
                    for (std::int64_t y = 0; y < out_height; ++y)
                    for (std::int64_t x = 0; x < out_width; ++x)
                    {
                        float sum = 0.0f;
                        for (std::int64_t ky = 0; ky < kernel; ++ky)
                        for (std::int64_t kx = 0; kx < kernel; ++kx)
                        {
                            const std::int64_t in_y = y * stride + ky - padding;
                            const std::int64_t in_x = x * stride + kx - padding;
                            if (in_y < 0 || in_y >= height || in_x < 0 || in_x >= width)
                                continue;

                            const float* const pixel =
                                pixels + ((n * height + in_y) * width + in_x) * channels;
                            const float* const tap = taps + (ky * kernel + kx) * channels;
                            for (std::int64_t c = 0; c < channels; ++c)
                                sum += pixel[c] * tap[c];
                        }
                        output[at++] = sum;
                    }

                    double exact_mean = 0.0;
                    float sequential_mean = 0.0f;
                    for (const float value : output)
                    {
                        exact_mean += static_cast<double>(value);
                        sequential_mean += value;
                    }
                    exact_mean /= static_cast<double>(count);
                    sequential_mean /= static_cast<float>(count);

                    double exact = 0.0;
                    float sequential = 0.0f;
                    for (const float value : output)
                    {
                        const double centred = static_cast<double>(value) - exact_mean;
                        exact += centred * centred;
                        sequential += (value - sequential_mean) * (value - sequential_mean);
                    }
                    exact /= static_cast<double>(count - 1);
                    sequential /= static_cast<float>(count - 1);

                    const double running = static_cast<double>(before[static_cast<std::size_t>(f)]);
                    result.exact.push_back(static_cast<float>(
                        NN::updated_statistic(running, exact, momentum, 1.0)));
                    result.sequential.push_back(static_cast<float>(NN::updated_statistic(
                        running, static_cast<double>(sequential), momentum, 1.0)));
                }

                return Result<HostRollingVariance>(std::move(result));
            }
        } // namespace

        void check_rolling_statistics(SushiBLAS::Engine& engine,                                      const DarknetTrainingCase& stated)
        {
            StepRun run;
            prepare_step(engine, stated, run);
            ASSERT_TRUE(run.is_ready);

            const std::string_view folder = stated.step_run;
            const DarknetWeightTarget* const targets = find_targets(run, run.first_layer);
            ASSERT_NE(targets, nullptr)
                << "the cfg has no convolution at layer " << run.first_layer;
            SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor mean_before,
                                    load_darknet_reference(folder, K_ROLLING_MEAN_BEFORE));
            SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor variance_before,
                                    load_darknet_reference(folder, K_ROLLING_VARIANCE_BEFORE));
            SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor mean_after,
                                    load_darknet_reference(folder, K_ROLLING_MEAN_AFTER));
            SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor variance_after,
                                    load_darknet_reference(folder, K_ROLLING_VARIANCE_AFTER));

            // Before the step both sides hold the float32 values of the same weights file.
            EXPECT_TRUE(within_tolerance("rolling mean before", read_state(run, targets->mean),
                                         mean_before.values, 0.0, 0.0));
            EXPECT_TRUE(within_tolerance("rolling variance before",
                                         read_state(run, targets->variance),
                                         variance_before.values, 0.0, 0.0));

            // The step moves the weights, so the host reads them first.
            SA_ASSERT_OK_AND_ASSIGN(
                const HostRollingVariance host,
                measure_host_rolling_variance(run, *targets, read_state(run, targets->weight),
                                              variance_before.values));

            run_step(run);
            ASSERT_TRUE(run.is_ready);

            const double bound = stated.bounds.rolling_absolute;
            EXPECT_TRUE(within_tolerance("rolling mean after", read_state(run, targets->mean),
                                         mean_after.values, bound, 0.0));

            // The two variances differ by more than the bound; see tests/integration/README.md.
            EXPECT_TRUE(within_tolerance("rolling variance after, against the sum in double",
                                         read_state(run, targets->variance), host.exact, bound,
                                         0.0));
            EXPECT_TRUE(within_tolerance(
                "the reference's rolling variance, against the sum term by term in float32",
                variance_after.values, host.sequential, bound, 0.0));
        }

        void check_one_batch_overfit(SushiBLAS::Engine& engine,
                                     const DarknetTrainingCase& stated)
        {
            ASSERT_GT(stated.overfit_steps, 1) << "the case states no steps to compare";
            ASSERT_GT(stated.overfit_cost_share, 0.0) << "the case states no share to reach";

            StepRun run;
            prepare_step(engine, stated, run);
            ASSERT_TRUE(run.is_ready);

            double first = 0.0;
            double last = 0.0;
            for (std::int64_t step = 0; step < stated.overfit_steps; ++step)
            {
                run_step(run);
                ASSERT_TRUE(run.is_ready) << "step " << step << " did not run";
                ASSERT_TRUE(std::isfinite(run.cost.total))
                    << "the summed cost of step " << step << " is " << run.cost.total;

                first = step == 0 ? run.cost.total : first;
                last = run.cost.total;
            }

            const double bound = stated.overfit_cost_share * first;
            std::cout << "[ MEASURED ] " << stated.step_run << ": summed cost " << first
                      << " at step 0, " << last << " at step " << stated.overfit_steps - 1
                      << ", bound " << bound << "\n";
            EXPECT_GT(first, 0.0) << "the first step has no cost to bring down";
            EXPECT_LT(last, bound);
        }

    } // namespace Test
} // namespace SushiAI
