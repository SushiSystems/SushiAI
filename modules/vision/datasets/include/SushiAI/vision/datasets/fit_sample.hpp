/**************************************************************************/
/* fit_sample.hpp                                                         */
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
 * @file fit_sample.hpp
 * @brief Declares the fit of one loaded sample to the input size of a model.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/letterbox.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names how an image of any size becomes an image of the input size. */
    enum class SampleFit : std::uint8_t
    {
        Resize,    /**< Each axis is scaled on its own, so the aspect ratio may change. */
        Letterbox, /**< Both axes take one scale and the rest of the input is padding. */
    };

    /** @brief Holds what the caller chooses about a fit. */
    struct SampleFitOptions
    {
        /** @brief Holds the input height and width, each in [1, Imaging::K_MAX_IMAGE_SIDE]. */
        Imaging::ImageSize size{};

        /** @brief Selects the fit. */
        SampleFit fit = SampleFit::Resize;

        /** @brief Holds the settings of the letterbox; SampleFit::Resize does not read them. */
        Imaging::LetterboxSettings letterbox{};
    };

    /**
     * @brief Fits one sample's image to the input size and moves its objects with it.
     *
     * The objects become pixel boxes clipped to the fitted image; one left with no area is
     * removed. README.md, section fit_sample.hpp, lists the failures.
     * @param source The sample's loaded pixels; must not be the pixels of @p fitted.
     */
    [[nodiscard]] Status fit_sample(const Sample& sample, Imaging::ImageView source,
                                    const SampleFitOptions& options, FittedSample& fitted);
} // namespace SushiAI::Vision::Datasets
