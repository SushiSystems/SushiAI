/**************************************************************************/
/* table_dataset.cpp                                                      */
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
 * @file table_dataset.cpp
 * @brief Implements the dataset that collates each batch of a tabular sample list.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/table_dataset.hpp>

#include <utility>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_DATASET = "TableDataset";
    } // namespace

    Result<std::unique_ptr<TableDataset>> TableDataset::make(
        SushiBLAS::Engine& engine, SampleList list, std::unique_ptr<const Collator> collator,
        TableDatasetOptions options) noexcept
    {
        SA_FAIL_IF(collator == nullptr, StatusCode::InvalidDescription,
                   "{}: '{}' has no collator", K_DATASET, options.name);
        SA_FAIL_IF(list.table.feature_count < 1, StatusCode::InvalidDescription,
                   "{}: '{}' has a table of {} features; its reader lists images or nothing",
                   K_DATASET, options.name, list.table.feature_count);
        SA_FAIL_IF(list.table.target_count < 0 ||
                       list.table.values.size() != list.samples.size() * list.table.row_width(),
                   StatusCode::InvalidDescription,
                   "{}: '{}' has {} samples and {} numbers; a row holds {}", K_DATASET,
                   options.name, list.samples.size(), list.table.values.size(),
                   list.table.row_width());
        SA_FAIL_IF(options.batch_size < 1, StatusCode::InvalidDescription,
                   "{}: '{}' has a batch size of {}; it must be positive", K_DATASET,
                   options.name, options.batch_size);
        SA_FAIL_IF(list.samples.size() < static_cast<std::size_t>(options.batch_size),
                   StatusCode::InvalidDescription,
                   "{}: '{}' has {} samples, fewer than one batch of {}", K_DATASET, options.name,
                   list.samples.size(), options.batch_size);

        return Detail::at_boundary(K_DATASET, [&]() -> Result<std::unique_ptr<TableDataset>>
        {
            return std::unique_ptr<TableDataset>(new TableDataset(
                engine, std::move(list), std::move(collator), std::move(options)));
        });
    }

    TableDataset::TableDataset(SushiBLAS::Engine& engine, SampleList list,
                               std::unique_ptr<const Collator> collator,
                               TableDatasetOptions options) noexcept
        : engine_(&engine), list_(std::move(list)), collator_(std::move(collator)),
          options_(std::move(options))
    {

    }

    std::int64_t TableDataset::batch_size() const noexcept
    {
        return options_.batch_size;
    }

    Shape TableDataset::sample_shape() const
    {
        return Shape{list_.table.feature_count};
    }

    const std::vector<Data::TargetDescription>& TableDataset::targets() const noexcept
    {
        return collator_->targets();
    }

    std::size_t TableDataset::batch_count() const noexcept
    {
        return list_.samples.size() / static_cast<std::size_t>(options_.batch_size);
    }

    std::string_view TableDataset::name() const noexcept
    {
        return options_.name;
    }

    Status TableDataset::read(std::size_t index, Data::Batch& batch) const
    {
        SA_FATAL_IF(index >= batch_count(),
                    "{}: '{}' has {} batches; batch {} is outside them", K_DATASET,
                    options_.name, batch_count(), index);

        return read_rows(index, Data::RowRange{0, options_.batch_size}, batch);
    }

    Status TableDataset::read_rows(std::size_t index, Data::RowRange range,
                                   Data::Batch& batch) const
    {
        SA_TRY(Data::check_rows(*this, index, range));

        return Detail::at_boundary(K_DATASET, [&]() -> Status
        {
            const std::size_t rows = static_cast<std::size_t>(range.count);
            const std::size_t first = index * static_cast<std::size_t>(options_.batch_size)
                                      + static_cast<std::size_t>(range.first);
            std::vector<FittedSample> fitted(rows);

            for (std::size_t row = 0; row < rows; ++row)
            {
                const std::size_t sample = first + row;

                fitted[row].file = list_.samples[sample].file;
                fitted[row].features = list_.table.features(sample);
                fitted[row].targets = list_.table.targets(sample);
            }

            SA_TRY(Data::prepare_batch(*engine_, range.count, Data::describe_input(*this),
                                       collator_->targets(), batch));
            return collator_->collate(
                list_, SushiRuntime::span<const FittedSample>(fitted.data(), fitted.size()),
                batch);
        });
    }
} // namespace SushiAI::Vision::Datasets
