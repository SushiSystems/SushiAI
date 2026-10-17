/**************************************************************************/
/* fit_mask.cpp                                                           */
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
 * @file fit_mask.cpp
 * @brief Implements the fit of one sample's mask to the input size by nearest neighbour.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/fit_mask.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string_view>


#include <SushiAI/vision/imaging/letterbox.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_FITTER = "fit_mask";

        /** @brief Holds the channel count of a mask. */
        constexpr std::int32_t K_MASK_CHANNELS = 1;

        /** @brief Names a function that writes the fitted mask of one SampleFit. */
        using MaskFit = void (*)(Imaging::ImageView mask, const SampleFitOptions& options,
                                 Imaging::Image& fitted);

        /** @brief Returns the source index the centre of destination index @p at falls in. */
        [[nodiscard]] constexpr std::int32_t nearest_index(std::int32_t at, std::int32_t source,
                                                           std::int32_t destination) noexcept
        {
            const std::int64_t index = (2 * static_cast<std::int64_t>(at) + 1) * source /
                                       (2 * static_cast<std::int64_t>(destination));
            return static_cast<std::int32_t>(std::min<std::int64_t>(index, source - 1));
        }

        /**
         * @brief Writes a mask resized to @p size into a region of a one-channel image.
         * @param top Holds the row of @p fitted the first resized row is written to.
         * @param left Holds the column of @p fitted the first resized column is written to.
         * @pre The region lies inside @p fitted.
         */
        void resize_nearest(Imaging::ImageView mask, Imaging::ImageSize size, std::int32_t top,
                            std::int32_t left, Imaging::Image& fitted) noexcept
        {
            for (std::int32_t y = 0; y < size.height; ++y)
            {
                const std::uint8_t* const in = mask.row(nearest_index(y, mask.height(),
                                                                      size.height));
                std::uint8_t* const out = fitted.row(top + y) + left;

                for (std::int32_t x = 0; x < size.width; ++x)
                    out[x] = in[nearest_index(x, mask.width(), size.width)];
            }
        }

        /** @brief Scales each axis of the mask to the input size. */
        void fit_by_resize(Imaging::ImageView mask, const SampleFitOptions& options,
                           Imaging::Image& fitted)
        {
            fitted.reset(options.size, K_MASK_CHANNELS);
            resize_nearest(mask, options.size, 0, 0, fitted);
        }

        /** @brief Places the mask where the letterbox places its image; the rest is void. */
        void fit_by_letterbox(Imaging::ImageView mask, const SampleFitOptions& options,
                              Imaging::Image& fitted)
        {
            const Imaging::LetterboxPlan plan =
                Imaging::plan_letterbox(options.letterbox, mask.size(), options.size);

            fitted.reset(options.size, K_MASK_CHANNELS);
            fitted.fill(K_MASK_VOID);
            resize_nearest(mask, plan.resized, plan.top, plan.left, fitted);
        }

        /** @brief Holds the function of each SampleFit, at the index of its enumerator. */
        constexpr MaskFit K_FITS[] = {fit_by_resize, fit_by_letterbox};
    } // namespace

    Status fit_mask(Imaging::ImageView mask, Imaging::ImageSize image,
                    const SampleFitOptions& options, Imaging::Image& fitted)
    {
        const std::size_t fit = static_cast<std::size_t>(options.fit);

        SA_FAIL_IF(fit >= std::size(K_FITS), StatusCode::InvalidDescription,
                   "{}: {} names no fit", K_FITTER, fit);
        SA_FAIL_IF(!Imaging::is_valid_size(options.size), StatusCode::InvalidDescription,
                   "{}: an input of {} by {} pixels is outside the bounds of an image", K_FITTER,
                   options.size.height, options.size.width);
        SA_FAIL_IF(mask.channels() != K_MASK_CHANNELS, StatusCode::Incompatible,
                   "{}: the mask has {} channels; a mask holds one class index per pixel",
                   K_FITTER, mask.channels());
        SA_FAIL_IF(mask.size() != image, StatusCode::Incompatible,
                   "{}: the mask is {} by {} pixels; its image is {} by {}", K_FITTER,
                   mask.height(), mask.width(), image.height, image.width);

        return Detail::at_boundary(K_FITTER, [&]()
        {
            K_FITS[fit](mask, options, fitted);
            return Status();
        });
    }
} // namespace SushiAI::Vision::Datasets
