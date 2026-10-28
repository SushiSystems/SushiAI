/**************************************************************************/
/* reader_section.cpp                                                     */
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
 * @file reader_section.cpp
 * @brief Implements the factory that builds a reader dataset from a data section.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/reader_section.hpp>

#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/threaded_loader.hpp>
#include <SushiAI/vision/datasets/reader_table.hpp>
#include <SushiAI/vision/datasets/task_table.hpp>

#include "section_count.hpp"
#include "section_keys.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Lists the keys every section may hold, whatever its reader and its task. */
        constexpr std::string_view K_SHARED_KEYS[] = {
            Detail::K_READER_KEY, Detail::K_TASK_KEY, Detail::K_ROOT_KEY,
            Detail::K_SPLIT_KEY, Detail::K_BATCH_SIZE_KEY, Detail::K_INPUT_KEY,
            Detail::K_CHANNELS_KEY, Detail::K_CLASSES_KEY, Detail::K_FIT_KEY, Detail::K_CACHE_KEY,
            Detail::K_CACHE_SIDE_KEY, Detail::K_SHUFFLE_KEY, Detail::K_SEED_KEY,
            Detail::K_PIPELINE_KEY, Detail::K_WORKERS_KEY, Detail::K_PREFETCH_KEY,
        };

        /** @brief Joins the names of @p rows into "'a', 'b'" for a diagnostic. */
        template <typename Entry>
        [[nodiscard]] std::string list_names(SushiRuntime::span<const Entry> rows)
        {
            std::string listed;
            for (const Entry& row : rows)
            {
                if (!listed.empty())
                    listed.append(", ");

                listed.append("'").append(row.name).append("'");
            }

            return listed;
        }

        /**
         * @brief Refuses a key that no section, nor the reader, nor the task of this one reads.
         * @return InvalidDescription when the section holds such a key; the message lists the
         *         keys allowed.
         */
        [[nodiscard]] Status reject_unread_keys(const Config::Value& section,
                                                std::string_view context,
                                                const ReaderEntry& reader, const TaskEntry& task)
        {
            std::vector<std::string_view> allowed(std::begin(K_SHARED_KEYS),
                                                  std::end(K_SHARED_KEYS));
            allowed.insert(allowed.end(), reader.keys.begin(), reader.keys.end());
            allowed.insert(allowed.end(), task.keys.begin(), task.keys.end());

            return section.reject_unknown_keys(
                context,
                SushiRuntime::span<const std::string_view>(allowed.data(), allowed.size()));
        }

        /** @brief Reads the root, the split and the class count, and keeps the section. */
        [[nodiscard]] Result<ReaderRequest> read_request(const Config::Value& section,
                                                         std::string_view context)
        {
            ReaderRequest request;
            request.section = section;
            request.context = std::string(context);
            SA_TRY_ASSIGN(const Config::Value& stated_root,
                          section.at(Detail::K_ROOT_KEY, context));
            SA_TRY_ASSIGN(request.root,
                          stated_root.as_string(Detail::named(context, Detail::K_ROOT_KEY)));

            const Config::Value split = Config::Value::string(request.split);
            SA_TRY_ASSIGN(request.split,
                          section.at_or(Detail::K_SPLIT_KEY, split)
                              .as_string(Detail::named(context, Detail::K_SPLIT_KEY)));

            const Config::Value classes = Config::Value::integer(request.class_count);
            SA_TRY_ASSIGN(request.class_count,
                          Detail::read_section_count(
                              section.at_or(Detail::K_CLASSES_KEY, classes),
                              Detail::K_CLASSES_KEY, context, 0));
            return request;
        }

        /**
         * @brief Reads "workers" and "prefetch", which defaults to the worker count.
         * @return The options, or nothing for no worker; InvalidDescription for a count
         *         out of range and for "prefetch" stated with no worker.
         */
        [[nodiscard]] Result<std::optional<Data::LoaderOptions>> read_loader_options(
            const Config::Value& section, std::string_view context)
        {
            const Config::Value none = Config::Value::integer(0);
            SA_TRY_ASSIGN(const std::int32_t workers,
                          Detail::read_section_count(
                              section.at_or(Detail::K_WORKERS_KEY, none),
                              Detail::K_WORKERS_KEY, context, 0));
            SA_FAIL_IF(workers == 0 && section.has(Detail::K_PREFETCH_KEY),
                       StatusCode::InvalidDescription,
                       "{} is stated and {} is 0; batches are read ahead by worker threads",
                       Detail::named(context, Detail::K_PREFETCH_KEY),
                       Detail::named(context, Detail::K_WORKERS_KEY));

            if (workers == 0)
                return std::optional<Data::LoaderOptions>();

            Data::LoaderOptions options;
            options.workers = workers;

            const Config::Value fallback = Config::Value::integer(workers);
            SA_TRY_ASSIGN(options.prefetch,
                          Detail::read_section_count(
                              section.at_or(Detail::K_PREFETCH_KEY, fallback),
                              Detail::K_PREFETCH_KEY, context, 1));
            return std::optional<Data::LoaderOptions>(options);
        }
    } // namespace

    Result<std::unique_ptr<Data::Dataset>> make_reader_dataset(SushiBLAS::Engine& engine,
                                                               const Config::Value& section,
                                                               std::string_view context) noexcept
    {
        SA_TRY_ASSIGN(const Config::Value& stated_reader,
                      section.at(Detail::K_READER_KEY, context));
        SA_TRY_ASSIGN(const std::string& reader_name,
                      stated_reader.as_string(Detail::named(context, Detail::K_READER_KEY)));
        const ReaderEntry* const reader = find_reader(reader_name);
        SA_FAIL_IF(reader == nullptr, StatusCode::InvalidDescription,
                   "{} is '{}', which is no reader; this build has: {}",
                   Detail::named(context, Detail::K_READER_KEY), reader_name,
                   list_names(reader_table()));

        SA_TRY_ASSIGN(const Config::Value& stated_task, section.at(Detail::K_TASK_KEY, context));
        SA_TRY_ASSIGN(const std::string& task_name,
                      stated_task.as_string(Detail::named(context, Detail::K_TASK_KEY)));
        const TaskEntry* const task = find_task(task_name);
        SA_FAIL_IF(task == nullptr, StatusCode::InvalidDescription,
                   "{} is '{}', which is no task; this build has: {}",
                   Detail::named(context, Detail::K_TASK_KEY), task_name,
                   list_names(task_table()));

        SA_TRY(reject_unread_keys(section, context, *reader, *task));

        SA_TRY_ASSIGN(const ReaderRequest request, read_request(section, context));
        SA_TRY_ASSIGN(const std::optional<Data::LoaderOptions> loader,
                      read_loader_options(section, context));
        SA_TRY_ASSIGN(std::unique_ptr<Data::Dataset> dataset,
                      task->build(engine, section, context, *task, *reader, request));

        if (!loader.has_value())
            return Result<std::unique_ptr<Data::Dataset>>(std::move(dataset));

        SA_TRY_ASSIGN(std::unique_ptr<Data::ThreadedLoader> threaded,
                      Data::ThreadedLoader::make(std::move(dataset), *loader));
        return std::unique_ptr<Data::Dataset>(std::move(threaded));
    }
} // namespace SushiAI::Vision::Datasets
