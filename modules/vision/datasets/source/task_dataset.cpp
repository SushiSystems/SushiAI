/**************************************************************************/
/* task_dataset.cpp                                                       */
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
 * @file task_dataset.cpp
 * @brief Implements the two dataset builders the rows of the task table point at.
 * @author Mustafa Garip
 */

#include "task_dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/fit_pipeline.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/pipeline_table.hpp>
#include <SushiAI/vision/datasets/reader_dataset.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_cache.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>
#include <SushiAI/vision/datasets/table_dataset.hpp>
#include <SushiAI/vision/imaging/image.hpp>

#include "section_count.hpp"
#include "section_keys.hpp"

namespace SushiAI::Vision::Datasets::Detail
{
    namespace
    {
        /** @brief Counts the entries of "input": the height, then the width. */
        constexpr std::size_t K_INPUT_ENTRIES = 2;

        /** @brief Holds the samples a reader listed and the class count of their collator. */
        struct ListedSamples
        {
            SampleList list;              /**< Holds what the reader returned. */
            std::int32_t class_count = 0; /**< Counts the classes the collator is built with. */
        };

        /**
         * @brief Returns the class count the collator is built with.
         * @return The count the section states, else the count of the list; InvalidDescription
         *         when the section states fewer classes than the list holds.
         */
        [[nodiscard]] Result<std::int32_t> resolve_class_count(const ReaderRequest& request,
                                                               const SampleList& list,
                                                               std::string_view context)
        {
            if (request.class_count == 0)
                return list.class_count;

            SA_FAIL_IF(request.class_count < list.class_count, StatusCode::InvalidDescription,
                       "{} is {}, but the samples of '{}' name {} classes",
                       named(context, K_CLASSES_KEY), request.class_count,
                       request.root.generic_string(), list.class_count);

            return request.class_count;
        }

        /** @brief Calls the reader of a row and resolves the class count of what it lists. */
        [[nodiscard]] Result<ListedSamples> read_samples(const ReaderEntry& reader,
                                                         const ReaderRequest& request,
                                                         std::string_view context)
        {
            ListedSamples listed;
            SA_TRY_ASSIGN(listed.list, reader.read(request));
            SA_TRY_ASSIGN(listed.class_count,
                          resolve_class_count(request, listed.list, context));
            return Result<ListedSamples>(std::move(listed));
        }

        /** @brief Reads "input" as [height, width], each at least 1. */
        [[nodiscard]] Result<Imaging::ImageSize> read_input_size(const Config::Value& section,
                                                                 std::string_view context)
        {
            const std::vector<Config::Value>& entries =
                section.at(K_INPUT_KEY, context).as_array(named(context, K_INPUT_KEY));
            SA_FAIL_IF(entries.size() != K_INPUT_ENTRIES, StatusCode::InvalidDescription,
                       "{} holds {} entries; it is [height, width]",
                       named(context, K_INPUT_KEY), entries.size());

            Imaging::ImageSize size;
            SA_TRY_ASSIGN(size.height, read_section_count(entries[0], "input[0]", context, 1));
            SA_TRY_ASSIGN(size.width, read_section_count(entries[1], "input[1]", context, 1));
            return size;
        }

        /** @brief Reads "batch_size", or returns @p fallback when the section states none. */
        [[nodiscard]] std::int64_t read_batch_size(const Config::Value& section,
                                                   std::string_view context,
                                                   std::int64_t fallback)
        {
            const Config::Value batch_size = Config::Value::integer(fallback);
            return section.at_or(K_BATCH_SIZE_KEY, batch_size)
                .as_int(named(context, K_BATCH_SIZE_KEY));
        }

        /** @brief Pairs the name a section gives a fit with that fit. */
        struct FitName
        {
            std::string_view name; /**< Holds the name the section writes. */
            SampleFit fit;         /**< Holds the fit the name selects. */
        };

        /** @brief Lists the fits a section may name. */
        constexpr FitName K_FIT_NAMES[] = {
            {Config::K_FIT_RESIZE, SampleFit::Resize},
            {Config::K_FIT_LETTERBOX, SampleFit::Letterbox},
        };

        /**
         * @brief Reads "fit"; @p fallback when the section states none.
         * @return The fit; InvalidDescription, with the names of the fits, for a name the list
         *         lacks.
         */
        [[nodiscard]] Result<SampleFit> read_fit(const Config::Value& section,
                                                 std::string_view context, SampleFit fallback)
        {
            if (!section.has(K_FIT_KEY))
                return fallback;

            const std::string& name =
                section.at(K_FIT_KEY, context).as_string(named(context, K_FIT_KEY));

            const FitName* found = nullptr;
            std::string listed;
            for (const FitName& known : K_FIT_NAMES)
            {
                if (known.name == name)
                    found = &known;

                listed.append(listed.empty() ? "'" : ", '").append(known.name).append("'");
            }

            SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                       "{} is '{}', which is no fit; a fit is one of {}",
                       named(context, K_FIT_KEY), name, listed);
            return found->fit;
        }

        /** @brief Reads the input size, the channels and the batch size; the task gives the fit. */
        [[nodiscard]] Result<ReaderDatasetOptions> read_image_options(
            const Config::Value& section, std::string_view context, const TaskEntry& task)
        {
            ReaderDatasetOptions options;
            SA_TRY_ASSIGN(options.fit.fit, read_fit(section, context, task.fit));
            SA_TRY_ASSIGN(options.fit.size, read_input_size(section, context));

            const Config::Value channels = Config::Value::integer(options.channels);
            SA_TRY_ASSIGN(options.channels,
                          read_section_count(section.at_or(K_CHANNELS_KEY, channels),
                                             K_CHANNELS_KEY, context, 1));

            options.batch_size = read_batch_size(section, context, options.batch_size);

            const Config::Value shuffle = Config::Value::boolean(options.shuffle);
            options.shuffle =
                section.at_or(K_SHUFFLE_KEY, shuffle).as_bool(named(context, K_SHUFFLE_KEY));

            // Reinterpreting the signed integer keeps every 64-bit seed a file can state.
            const Config::Value seed = Config::Value::integer(0);
            options.seed = static_cast<std::uint64_t>(
                section.at_or(K_SEED_KEY, seed).as_int(named(context, K_SEED_KEY)));
            return options;
        }

        /**
         * @brief Reads "pipeline" and returns its row; the first row when it is absent.
         * @return The row, never null; InvalidDescription, with the names of the rows, for
         *         a name the table lacks.
         */
        [[nodiscard]] Result<const PipelineEntry*> read_pipeline_row(
            const Config::Value& section, std::string_view context)
        {
            const SushiRuntime::span<const PipelineEntry> rows = pipeline_table();
            if (!section.has(K_PIPELINE_KEY))
                return rows.data();

            const std::string& name = section.at(K_PIPELINE_KEY, context)
                                          .as_string(named(context, K_PIPELINE_KEY));
            const PipelineEntry* const row = find_pipeline(name);

            std::string listed;
            for (const PipelineEntry& known : rows)
                listed.append(listed.empty() ? "'" : ", '").append(known.name).append("'");

            SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                       "{} is '{}', which is no pipeline; this build has: {}",
                       named(context, K_PIPELINE_KEY), name, listed);
            return row;
        }

        /**
         * @brief Reads "cache" and "cache_side".
         * @return The options, or nothing for a section without "cache"; InvalidDescription
         *         for a "cache_side" below 0 or stated without "cache".
         */
        [[nodiscard]] Result<std::optional<SampleCacheOptions>> read_cache_options(
            const Config::Value& section, std::string_view context, std::int32_t channels)
        {
            SA_FAIL_IF(!section.has(K_CACHE_KEY) && section.has(K_CACHE_SIDE_KEY),
                       StatusCode::InvalidDescription,
                       "{} is stated without {}, the directory it applies to",
                       named(context, K_CACHE_SIDE_KEY), named(context, K_CACHE_KEY));

            if (!section.has(K_CACHE_KEY))
                return std::optional<SampleCacheOptions>();

            SampleCacheOptions options;
            options.directory =
                section.at(K_CACHE_KEY, context).as_string(named(context, K_CACHE_KEY));
            options.layout = FitPipeline::layout_of(channels);

            const Config::Value side = Config::Value::integer(options.longest_side);
            SA_TRY_ASSIGN(options.longest_side,
                          read_section_count(section.at_or(K_CACHE_SIDE_KEY, side),
                                             K_CACHE_SIDE_KEY, context, 0));
            return std::optional<SampleCacheOptions>(std::move(options));
        }

        /** @brief Writes the cache files a list lacks and points the list at the cache. */
        [[nodiscard]] Status cache_samples(const SampleCacheOptions& cache, SampleList& list)
        {
            SA_TRY(write_sample_cache(list, cache));
            SA_TRY_ASSIGN(list, use_sample_cache(std::move(list), cache));
            return Status();
        }
    } // namespace

    Result<std::unique_ptr<Data::Dataset>> build_image_dataset(
        SushiBLAS::Engine& engine, const Config::Value& section, std::string_view context,
        const TaskEntry& task, const ReaderEntry& reader, const ReaderRequest& request)
    {
        SA_TRY_ASSIGN(ReaderDatasetOptions options, read_image_options(section, context, task));
        options.name = std::string(reader.name);

        SA_TRY_ASSIGN(const PipelineEntry* const pipeline_row,
                      read_pipeline_row(section, context));
        SA_TRY_ASSIGN(const std::optional<SampleCacheOptions> cache,
                      read_cache_options(section, context, options.channels));
        SA_TRY_ASSIGN(std::unique_ptr<const SamplePipeline> pipeline,
                      pipeline_row->make(
                          PipelineRequest{options.fit, options.channels, options.seed}));

        SA_TRY_ASSIGN(ListedSamples listed, read_samples(reader, request, context));
        SA_FAIL_IF(!listed.list.table.values.empty(), StatusCode::InvalidDescription,
                   "{} is '{}', which reads images; the reader '{}' lists rows of numbers",
                   named(context, K_TASK_KEY), task.name, reader.name);

        if (cache.has_value())
            SA_TRY(cache_samples(*cache, listed.list));

        SA_TRY_ASSIGN(std::unique_ptr<const Collator> collator,
                      task.make_collator(section, context,
                                         CollatorRequest{listed.list, listed.class_count,
                                                         options.fit.size}));
        SA_TRY_ASSIGN(std::unique_ptr<ReaderDataset> dataset,
                      ReaderDataset::make(engine, std::move(listed.list), std::move(collator),
                                          std::move(pipeline), std::move(options)));

        return std::unique_ptr<Data::Dataset>(std::move(dataset));
    }

    Result<std::unique_ptr<Data::Dataset>> build_table_dataset(
        SushiBLAS::Engine& engine, const Config::Value& section, std::string_view context,
        const TaskEntry& task, const ReaderEntry& reader, const ReaderRequest& request)
    {
        for (std::string_view key : K_IMAGE_KEYS)
        {
            SA_FAIL_IF(section.has(key), StatusCode::InvalidDescription,
                       "{} is stated, and {} is '{}', which reads rows of numbers and no image",
                       named(context, key), named(context, K_TASK_KEY), task.name);
        }

        TableDatasetOptions options;
        options.batch_size = read_batch_size(section, context, options.batch_size);
        options.name = std::string(reader.name);

        SA_TRY_ASSIGN(ListedSamples listed, read_samples(reader, request, context));
        SA_FAIL_IF(listed.list.table.values.empty(), StatusCode::InvalidDescription,
                   "{} is '{}', which reads rows of numbers; the reader '{}' lists none",
                   named(context, K_TASK_KEY), task.name, reader.name);

        SA_TRY_ASSIGN(std::unique_ptr<const Collator> collator,
                      task.make_collator(section, context,
                                         CollatorRequest{listed.list, listed.class_count,
                                                         Imaging::ImageSize{}}));
        SA_TRY_ASSIGN(std::unique_ptr<TableDataset> dataset,
                      TableDataset::make(engine, std::move(listed.list), std::move(collator),
                                         std::move(options)));

        return std::unique_ptr<Data::Dataset>(std::move(dataset));
    }
} // namespace SushiAI::Vision::Datasets::Detail
