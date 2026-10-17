/**************************************************************************/
/* task_table.cpp                                                         */
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
 * @file task_table.cpp
 * @brief Implements the table of the tasks a data section may name.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/task_table.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/binary_collator.hpp>
#include <SushiAI/vision/datasets/classification_collator.hpp>
#include <SushiAI/vision/datasets/detection_collator.hpp>
#include <SushiAI/vision/datasets/multi_label_collator.hpp>
#include <SushiAI/vision/datasets/regression_collator.hpp>
#include <SushiAI/vision/datasets/segmentation_collator.hpp>

#include "section_count.hpp"
#include "section_keys.hpp"
#include "task_dataset.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names the key that counts the object slots of one image. */
        constexpr std::string_view K_MAX_OBJECTS_KEY = "max_objects";

        /** @brief Names the key of the class the binary task writes as 1. */
        constexpr std::string_view K_POSITIVE_KEY = "positive";

        /** @brief Names the key of the value a pixel of no class has in the role mask. */
        constexpr std::string_view K_IGNORE_INDEX_KEY = "ignore_index";

        /** @brief Lists the keys the detection task reads beside the shared ones. */
        constexpr std::string_view K_DETECTION_KEYS[] = {K_MAX_OBJECTS_KEY};

        /** @brief Lists the keys the binary task reads beside the shared ones. */
        constexpr std::string_view K_BINARY_KEYS[] = {K_POSITIVE_KEY};

        /** @brief Names the key of the class a pixel no polygon holds takes. */
        constexpr std::string_view K_BACKGROUND_KEY = "background";

        /** @brief Lists the keys the segmentation task reads beside the shared ones. */
        constexpr std::string_view K_SEGMENTATION_KEYS[] = {K_IGNORE_INDEX_KEY,
                                                           K_BACKGROUND_KEY};

        /**
         * @brief Builds the detection collator; the section must state "max_objects".
         * @return The collator; InvalidDescription for a "max_objects" below 1.
         * @throws Error If "max_objects" is absent or no whole number.
         */
        [[nodiscard]] Result<std::unique_ptr<const Collator>> make_detection_collator(
            const Config::Value& section, std::string_view context,
            const CollatorRequest& request)
        {
            SA_TRY_ASSIGN(const std::int32_t max_objects,
                          Detail::read_section_count(section.at(K_MAX_OBJECTS_KEY, context),
                                                     K_MAX_OBJECTS_KEY, context, 1));
            SA_TRY_ASSIGN(std::unique_ptr<DetectionCollator> collator,
                          DetectionCollator::make(request.class_count, max_objects));

            return std::unique_ptr<const Collator>(std::move(collator));
        }

        /** @brief Builds the classification collator, which reads no key of its own. */
        [[nodiscard]] Result<std::unique_ptr<const Collator>> make_classification_collator(
            const Config::Value&, std::string_view, const CollatorRequest& request)
        {
            SA_TRY_ASSIGN(std::unique_ptr<ClassificationCollator> collator,
                          ClassificationCollator::make(request.class_count));

            return std::unique_ptr<const Collator>(std::move(collator));
        }

        /**
         * @brief Reads "positive": a class index, or a class name of the list.
         * @return The index, the second class when the key is absent; InvalidDescription for
         *         a name the list lacks, with the names it has, and for an index below 0.
         * @throws Error If "positive" is neither a string nor a whole number.
         */
        [[nodiscard]] Result<std::int32_t> read_positive_class(const Config::Value& section,
                                                               std::string_view context,
                                                               const SampleList& list)
        {
            if (!section.has(K_POSITIVE_KEY))
                return K_BINARY_CLASSES - 1;

            const Config::Value& stated = section.at(K_POSITIVE_KEY, context);
            if (stated.kind() != Config::Kind::String)
                return Detail::read_section_count(stated, K_POSITIVE_KEY, context, 0);

            const std::string& name = stated.as_string(K_POSITIVE_KEY);
            const auto found = std::find(list.class_names.begin(), list.class_names.end(), name);

            std::string listed;
            for (const std::string& known : list.class_names)
                listed.append(listed.empty() ? "'" : ", '").append(known).append("'");

            SA_FAIL_IF(found == list.class_names.end(), StatusCode::InvalidDescription,
                       "{} is '{}', which is no class of the samples; they have: {}",
                       Detail::named(context, K_POSITIVE_KEY), name, listed);

            return static_cast<std::int32_t>(found - list.class_names.begin());
        }

        /**
         * @brief Builds the binary collator over a set of two classes.
         * @return The collator; InvalidDescription for another class count and for a
         *         "positive" that names neither class.
         * @throws Error If "positive" is neither a string nor a whole number.
         */
        [[nodiscard]] Result<std::unique_ptr<const Collator>> make_binary_collator(
            const Config::Value& section, std::string_view context,
            const CollatorRequest& request)
        {
            SA_FAIL_IF(request.class_count != K_BINARY_CLASSES, StatusCode::InvalidDescription,
                       "{} is '{}', which reads {} classes; the samples have {}",
                       Detail::named(context, Detail::K_TASK_KEY), K_TASK_BINARY,
                       K_BINARY_CLASSES, request.class_count);

            SA_TRY_ASSIGN(const std::int32_t positive,
                          read_positive_class(section, context, request.list));
            SA_TRY_ASSIGN(std::unique_ptr<BinaryCollator> collator,
                          BinaryCollator::make(positive));

            return std::unique_ptr<const Collator>(std::move(collator));
        }

        /** @brief Builds the multi-label collator, which reads no key of its own. */
        [[nodiscard]] Result<std::unique_ptr<const Collator>> make_multi_label_collator(
            const Config::Value&, std::string_view, const CollatorRequest& request)
        {
            SA_TRY_ASSIGN(std::unique_ptr<MultiLabelCollator> collator,
                          MultiLabelCollator::make(request.class_count));

            return std::unique_ptr<const Collator>(std::move(collator));
        }

        /** @brief Builds the regression collator over the columns of the list's table. */
        [[nodiscard]] Result<std::unique_ptr<const Collator>> make_regression_collator(
            const Config::Value&, std::string_view, const CollatorRequest& request)
        {
            SA_TRY_ASSIGN(std::unique_ptr<RegressionCollator> collator,
                          RegressionCollator::make(request.list.table.feature_count,
                                                   request.list.table.target_count));

            return std::unique_ptr<const Collator>(std::move(collator));
        }

        /** @brief Reports whether a list has samples and none of them names a mask file. */
        [[nodiscard]] bool has_polygons_only(const SampleList& list) noexcept
        {
            return !list.samples.empty() &&
                   std::none_of(list.samples.begin(), list.samples.end(),
                                [](const Sample& sample) { return sample.mask.has_value(); });
        }

        /**
         * @brief Builds the segmentation collator; "ignore_index" defaults to 255, and a
         *        list whose samples name no mask file has its polygons painted.
         * @return The collator; InvalidDescription for an "ignore_index" no int32 holds,
         *         and what SegmentationCollator::make refuses.
         * @throws Error If "ignore_index" or "background" is no whole number.
         */
        [[nodiscard]] Result<std::unique_ptr<const Collator>> make_segmentation_collator(
            const Config::Value& section, std::string_view context,
            const CollatorRequest& request)
        {
            const std::string named = Detail::named(context, K_IGNORE_INDEX_KEY);
            const Config::Value fallback = Config::Value::integer(K_DEFAULT_IGNORE_INDEX);
            const std::int64_t ignore = section.at_or(K_IGNORE_INDEX_KEY, fallback).as_int(named);

            SA_FAIL_IF(ignore < std::numeric_limits<std::int32_t>::min() ||
                           ignore > std::numeric_limits<std::int32_t>::max(),
                       StatusCode::InvalidDescription, "{} is {}; it must fit 32 bits", named,
                       ignore);

            SegmentationOptions options;
            options.class_count = request.class_count;
            options.size = request.input;
            options.ignore_index = static_cast<std::int32_t>(ignore);
            options.origin = has_polygons_only(request.list) ? MaskOrigin::Polygons
                                                             : MaskOrigin::File;

            if (section.has(K_BACKGROUND_KEY))
            {
                SA_TRY_ASSIGN(const std::int32_t background,
                              Detail::read_section_count(section.at(K_BACKGROUND_KEY, context),
                                                         K_BACKGROUND_KEY, context, 0));
                options.background = background;
            }

            SA_TRY_ASSIGN(std::unique_ptr<SegmentationCollator> collator,
                          SegmentationCollator::make(options));

            return std::unique_ptr<const Collator>(std::move(collator));
        }

        /** @brief Returns the keys of a row as the span a TaskEntry holds. */
        template <std::size_t Count>
        [[nodiscard]] SushiRuntime::span<const std::string_view> keys_of(
            const std::string_view (&keys)[Count]) noexcept
        {
            return SushiRuntime::span<const std::string_view>(keys, Count);
        }
    } // namespace

    SushiRuntime::span<const TaskEntry> task_table() noexcept
    {
        // The rows hold spans, so they are built on the first call and not at compile time.
        static const TaskEntry K_TASKS[] = {
            TaskEntry{K_TASK_DETECTION, SampleFit::Letterbox, keys_of(K_DETECTION_KEYS),
                      &make_detection_collator, &Detail::build_image_dataset},
            TaskEntry{K_TASK_CLASSIFICATION, SampleFit::Resize, {},
                      &make_classification_collator, &Detail::build_image_dataset},
            TaskEntry{K_TASK_BINARY, SampleFit::Resize, keys_of(K_BINARY_KEYS),
                      &make_binary_collator, &Detail::build_image_dataset},
            TaskEntry{K_TASK_MULTI_LABEL, SampleFit::Resize, {}, &make_multi_label_collator,
                      &Detail::build_image_dataset},
            TaskEntry{K_TASK_REGRESSION, SampleFit::Resize, {}, &make_regression_collator,
                      &Detail::build_table_dataset},
            TaskEntry{K_TASK_SEGMENTATION, SampleFit::Letterbox, keys_of(K_SEGMENTATION_KEYS),
                      &make_segmentation_collator, &Detail::build_image_dataset},
        };

        return SushiRuntime::span<const TaskEntry>(K_TASKS, std::size(K_TASKS));
    }

    const TaskEntry* find_task(std::string_view name) noexcept
    {
        for (const TaskEntry& row : task_table())
        {
            if (row.name == name)
                return &row;
        }

        return nullptr;
    }
} // namespace SushiAI::Vision::Datasets
