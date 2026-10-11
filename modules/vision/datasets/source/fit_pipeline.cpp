/**************************************************************************/
/* fit_pipeline.cpp                                                       */
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
 * @file fit_pipeline.cpp
 * @brief Implements the pipeline that fits each sample to the input size and draws nothing.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/fit_pipeline.hpp>

#include <filesystem>

#include <SushiBLAS/core/format.hpp>

#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/load_sample_image.hpp>
#include <SushiAI/vision/datasets/load_sample_mask.hpp>
#include <SushiAI/vision/imaging/gray.hpp>
#include <SushiAI/vision/imaging/image.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_PIPELINE = "FitPipeline";

        /** @brief Returns the decoding that yields an image of a channel count. */
        [[nodiscard]] Codecs::DecodeOptions decoding_of(std::int32_t channels) noexcept
        {
            Codecs::DecodeOptions options;
            options.layout = FitPipeline::layout_of(channels);
            return options;
        }

        /**
         * @brief Loads the mask of a sample that has one and fits it as its image was fitted.
         * @param image Holds the size of the sample's image before the fit.
         * @return Success for a sample with no mask; else the failure of load_sample_mask
         *         or of fit_mask, the second opening with the mask file.
         */
        [[nodiscard]] Status fit_mask_of(const SampleList& list, std::size_t sample,
                                         Imaging::ImageSize image, const SampleFitOptions& fit,
                                         Imaging::Image& loaded, FittedSample& fitted)
        {
            const Sample& record = list.samples[sample];
            if (!record.mask.has_value())
                return Status();

            SA_TRY(load_sample_mask(list, sample, loaded));
            return SA_ANNOTATE(fit_mask(loaded.view(), image, fit, fitted.mask), "'{}'",
                               Detail::display_of(list.files[*record.mask]));
        }
    } // namespace

    Result<std::unique_ptr<FitPipeline>> FitPipeline::make(const SampleFitOptions& fit,
                                                           std::int32_t channels)
    {
        SA_FAIL_IF(!Imaging::is_valid_size(fit.size), StatusCode::InvalidDescription,
                   "{}: an input of {} by {} pixels is outside the bounds of an image",
                   K_PIPELINE, fit.size.height, fit.size.width);

        return Detail::at_boundary(K_PIPELINE, [&]() -> Result<std::unique_ptr<FitPipeline>>
        {
            return std::unique_ptr<FitPipeline>(new FitPipeline(fit, channels));
        });
    }

    FitPipeline::FitPipeline(const SampleFitOptions& fit, std::int32_t channels) noexcept
        : fit_(fit), decoding_(decoding_of(channels))
    {

    }

    Codecs::ChannelLayout FitPipeline::layout_of(std::int32_t channels) noexcept
    {
        return channels == Imaging::K_GRAY_CHANNELS ? Codecs::ChannelLayout::Gray
                                                    : Codecs::ChannelLayout::Color;
    }

    std::string_view FitPipeline::name() const noexcept
    {
        return K_PIPELINE_NONE;
    }

    Status FitPipeline::produce(const SampleList& list,
                                SushiRuntime::span<const std::size_t> samples, std::int64_t,
                                SushiRuntime::span<FittedSample> fitted) const
    {
        SA_FAIL_IF(samples.size() != fitted.size(), StatusCode::InvalidDescription,
                   "{}: {} samples were asked for into {} fitted samples", K_PIPELINE,
                   samples.size(), fitted.size());

        return Detail::at_boundary(K_PIPELINE, [&]() -> Status
        {
            Imaging::Image loaded;
            Imaging::Image mask;

            for (std::size_t row = 0; row < samples.size(); ++row)
            {
                SA_TRY(load_sample_image(list, samples[row], decoding_, loaded));

                // load_sample_image checked that the sample and its file are in the list.
                const Sample& record = list.samples[samples[row]];
                const std::filesystem::path& file = list.files[record.file];

                SA_TRY(SA_ANNOTATE(fit_sample(record, loaded.view(), fit_, fitted[row]), "'{}'",
                                   Detail::display_of(file)));
                SA_TRY(fit_mask_of(list, samples[row], loaded.size(), fit_, mask, fitted[row]));
            }

            return Status();
        });
    }
} // namespace SushiAI::Vision::Datasets
