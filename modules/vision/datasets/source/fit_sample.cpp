/**************************************************************************/
/* fit_sample.cpp                                                         */
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
 * @file fit_sample.cpp
 * @brief Implements the fit of one loaded sample to the input size of a model.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/fit_sample.hpp>

#include <cstddef>
#include <iterator>
#include <string_view>


#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/resize.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_FITTER = "fit_sample";

        /** @brief Names a function that writes the fitted image and returns the moved labels. */
        using Fit = Imaging::Labels (*)(const Sample& sample, Imaging::ImageView source,
                                        const SampleFitOptions& options,
                                        Imaging::Image& destination);

        /** @brief Scales each axis to the input size; the labels scale with their axis. */
        Imaging::Labels fit_by_resize(const Sample& sample, Imaging::ImageView source,
                                      const SampleFitOptions& options,
                                      Imaging::Image& destination)
        {
            Imaging::resize_bilinear(source, options.size, destination);
            return Imaging::to_pixel_labels(sample.objects, options.size);
        }

        /** @brief Letterboxes the image and moves the labels by the same plan. */
        Imaging::Labels fit_by_letterbox(const Sample& sample, Imaging::ImageView source,
                                         const SampleFitOptions& options,
                                         Imaging::Image& destination)
        {
            const Imaging::LetterboxPlan plan =
                Imaging::plan_letterbox(options.letterbox, source.size(), options.size);

            Imaging::apply_letterbox(source, plan, destination);
            return Imaging::apply_letterbox(
                Imaging::to_pixel_labels(sample.objects, source.size()), plan);
        }

        /** @brief Holds the function of each SampleFit, at the index of its enumerator. */
        constexpr Fit K_FITS[] = {fit_by_resize, fit_by_letterbox};
    } // namespace

    Status fit_sample(const Sample& sample, Imaging::ImageView source,
                      const SampleFitOptions& options, FittedSample& fitted)
    {
        const std::size_t fit = static_cast<std::size_t>(options.fit);

        SA_FAIL_IF(fit >= std::size(K_FITS), StatusCode::InvalidDescription,
                   "{}: {} names no fit", K_FITTER, fit);
        SA_FAIL_IF(!Imaging::is_valid_size(options.size), StatusCode::InvalidDescription,
                   "{}: an input of {} by {} pixels is outside the bounds of an image", K_FITTER,
                   options.size.height, options.size.width);

        return Detail::at_boundary(K_FITTER, [&]()
        {
            fitted.objects = K_FITS[fit](sample, source, options, fitted.image);
            Imaging::clip_labels(fitted.objects, options.size);
            Imaging::remove_empty_boxes(fitted.objects);

            fitted.file = sample.file;
            fitted.class_index = sample.class_index;
            return Status();
        });
    }
} // namespace SushiAI::Vision::Datasets
