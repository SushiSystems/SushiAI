/**************************************************************************/
/* rectangle.hpp                                                          */
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
 * @file rectangle.hpp
 * @brief Declares the batch order and the batch shapes of rectangular validation.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <SushiAI/imaging/image.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Imaging
{
    /** @brief Holds what the caller chooses about rectangular batches. */
    struct RectangleSettings
    {
        /** @brief Holds the long side an image is loaded at, in pixels (cfg/default.yaml:16). */
        std::int32_t image_size = 640;

        /** @brief Counts the images of a batch: twice the training batch (trainer.py:317). */
        std::int32_t batch_size = 32;

        /** @brief Holds the stride every batch side is a multiple of (detect/train.py:75). */
        std::int32_t stride = 32;

        /** @brief Holds the padding, in strides, added before the ceiling (data/build.py:341). */
        double pad = 0.5;
    };

    /** @brief Holds the order rectangular validation visits images in and every batch's shape. */
    struct RectangularBatches
    {
        /** @brief Counts the images of every batch but the last, which may hold fewer. */
        std::size_t batch_size = 0;

        /** @brief Holds every dataset index once, by rising height over width. */
        std::vector<std::size_t> order;

        /** @brief Holds the letterbox target of each batch; batch b is shapes[b]. */
        std::vector<ImageSize> shapes;

        /**
         * @brief Returns the target of the image at a position of order.
         * @param position In [0, order.size()); not checked.
         */
        [[nodiscard]] ImageSize shape_at(std::size_t position) const noexcept
        {
            return shapes[position / batch_size];
        }
    };

    /**
     * @brief Throws Error unless the sizes are positive and the pad is finite and not negative.
     *
     * image_size must pass is_valid_side; batch_size and stride must be above zero.
     */
    void validate_rectangle(const RectangleSettings& settings);

    /**
     * @brief Sorts images by aspect ratio and gives each batch one shape, as set_rectangle.
     *
     * The sort is stable on height / width in double. A batch's unit shape is (largest, 1)
     * when its largest ratio is below 1, (1, 1 / smallest) when its smallest is above 1, else
     * (1, 1); a side is ceil(unit * image_size / stride + pad) * stride (data/base.py:405-427).
     * @param sizes The stored size of each image, by dataset index; none may be invalid.
     */
    [[nodiscard]] RectangularBatches plan_rectangular_batches(const std::vector<ImageSize>& sizes,
                                                              const RectangleSettings& settings);
} // namespace SushiAI::Imaging
