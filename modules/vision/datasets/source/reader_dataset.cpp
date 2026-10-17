/**************************************************************************/
/* reader_dataset.cpp                                                     */
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
 * @file reader_dataset.cpp
 * @brief Implements the dataset that builds each batch of a sample list when it is asked for.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/reader_dataset.hpp>

#include <filesystem>
#include <utility>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch_permutation.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/vision/datasets/fit_pipeline.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_DATASET = "ReaderDataset";
    } // namespace

    Result<std::unique_ptr<ReaderDataset>> ReaderDataset::make(
        SushiBLAS::Engine& engine, SampleList list, std::unique_ptr<const Collator> collator,
        ReaderDatasetOptions options)
    {
        // A size no pipeline takes is left for the other form to refuse by the dataset's name.
        std::unique_ptr<const SamplePipeline> pipeline;
        if (collator != nullptr && Imaging::is_valid_size(options.fit.size))
        {
            SA_TRY_ASSIGN(std::unique_ptr<FitPipeline> fit,
                          FitPipeline::make(options.fit, options.channels));
            pipeline = std::move(fit);
        }

        return make(engine, std::move(list), std::move(collator), std::move(pipeline),
                    std::move(options));
    }

    Result<std::unique_ptr<ReaderDataset>> ReaderDataset::make(
        SushiBLAS::Engine& engine, SampleList list, std::unique_ptr<const Collator> collator,
        std::unique_ptr<const SamplePipeline> pipeline, ReaderDatasetOptions options)
    {
        SA_FAIL_IF(collator == nullptr, StatusCode::InvalidDescription,
                   "{}: '{}' has no collator", K_DATASET, options.name);
        SA_FAIL_IF(!Imaging::is_valid_size(options.fit.size), StatusCode::InvalidDescription,
                   "{}: '{}' has an input of {} by {} pixels, outside the bounds of an image",
                   K_DATASET, options.name, options.fit.size.height, options.fit.size.width);
        SA_FAIL_IF(!Imaging::is_valid_channel_count(options.channels),
                   StatusCode::InvalidDescription,
                   "{}: '{}' has {} channels; an image has 1 to {}", K_DATASET, options.name,
                   options.channels, Imaging::K_MAX_CHANNELS);
        SA_FAIL_IF(options.batch_size < 1, StatusCode::InvalidDescription,
                   "{}: '{}' has a batch size of {}; it must be positive", K_DATASET,
                   options.name, options.batch_size);
        SA_FAIL_IF(list.samples.size() < static_cast<std::size_t>(options.batch_size),
                   StatusCode::InvalidDescription,
                   "{}: '{}' has {} samples, fewer than one batch of {}", K_DATASET, options.name,
                   list.samples.size(), options.batch_size);

        SA_FAIL_IF(pipeline == nullptr, StatusCode::InvalidDescription,
                   "{}: '{}' has no pipeline", K_DATASET, options.name);

        return Detail::at_boundary(K_DATASET, [&]() -> Result<std::unique_ptr<ReaderDataset>>
        {
            return std::unique_ptr<ReaderDataset>(
                new ReaderDataset(engine, std::move(list), std::move(collator),
                                  std::move(pipeline), std::move(options)));
        });
    }

    ReaderDataset::ReaderDataset(SushiBLAS::Engine& engine, SampleList list,
                                 std::unique_ptr<const Collator> collator,
                                 std::unique_ptr<const SamplePipeline> pipeline,
                                 ReaderDatasetOptions options) noexcept
        : engine_(&engine), list_(std::move(list)), collator_(std::move(collator)),
          pipeline_(std::move(pipeline)), options_(std::move(options))
    {

    }

    std::int64_t ReaderDataset::batch_size() const noexcept
    {
        return options_.batch_size;
    }

    Shape ReaderDataset::sample_shape() const
    {
        return Shape{options_.fit.size.height, options_.fit.size.width, options_.channels};
    }

    const std::vector<Data::TargetDescription>& ReaderDataset::targets() const noexcept
    {
        return collator_->targets();
    }

    std::size_t ReaderDataset::batch_count() const noexcept
    {
        return list_.samples.size() / static_cast<std::size_t>(options_.batch_size);
    }

    std::string_view ReaderDataset::name() const noexcept
    {
        return options_.name;
    }

    Status ReaderDataset::read(std::size_t index, Data::Batch& batch) const
    {
        SA_FATAL_IF(index >= batch_count(),
                    "{}: '{}' has {} batches; batch {} is outside them", K_DATASET,
                    options_.name, batch_count(), index);

        return Detail::at_boundary(K_DATASET, [&]() -> Status
        {
            const std::size_t rows = static_cast<std::size_t>(options_.batch_size);
            const std::int64_t epoch = epoch_.load(std::memory_order_relaxed);
            const Data::EpochPermutation order(options_.seed, epoch, list_.samples.size());

            std::vector<std::size_t> samples(rows);
            for (std::size_t row = 0; row < rows; ++row)
            {
                const std::size_t position = index * rows + row;
                samples[row] = options_.shuffle ? static_cast<std::size_t>(order.at(position))
                                                : position;
            }

            std::vector<FittedSample> fitted(rows);
            SA_TRY(pipeline_->produce(
                list_, SushiRuntime::span<const std::size_t>(samples.data(), samples.size()),
                epoch, SushiRuntime::span<FittedSample>(fitted.data(), fitted.size())));

            for (std::size_t row = 0; row < rows; ++row)
            {
                SA_FAIL_IF(fitted[row].image.channels() != options_.channels,
                           StatusCode::Incompatible,
                           "'{}': {}: the image has {} channels; '{}' reads {}",
                           Detail::display_of(list_.files[list_.samples[samples[row]].file]),
                           K_DATASET, fitted[row].image.channels(), options_.name,
                           options_.channels);
            }

            SA_TRY(Data::prepare_batch(*engine_, options_.batch_size,
                                       Data::describe_input(*this), collator_->targets(),
                                       batch));
            return collator_->collate(
                list_, SushiRuntime::span<const FittedSample>(fitted.data(), fitted.size()),
                batch);
        });
    }

    void ReaderDataset::begin_epoch(std::int64_t epoch) const noexcept
    {
        SA_FATAL_IF(epoch < 0, "{}: '{}' was given the epoch {}; epochs count from 0",
                    K_DATASET, options_.name, epoch);

        epoch_.store(epoch, std::memory_order_relaxed);
    }
} // namespace SushiAI::Vision::Datasets
