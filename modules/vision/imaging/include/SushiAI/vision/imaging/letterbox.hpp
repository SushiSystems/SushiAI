/**************************************************************************/
/* letterbox.hpp                                                          */
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
 * @file letterbox.hpp
 * @brief Declares the letterbox transform: an aspect-preserving resize, then centred padding.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds what the caller chooses about a letterbox. */
    struct LetterboxSettings
    {
        /** @brief Allows a ratio above 1; the validation pipeline of the reference passes false. */
        bool scale_up = false;

        /** @brief Holds the value of every padding byte (data/augment.py:1709). */
        std::uint8_t padding_value = 114;
    };

    /** @brief Holds everything apply_letterbox needs; nothing in it is random. */
    struct LetterboxPlan
    {
        /** @brief Holds the size of the image the plan was made for. */
        ImageSize source{};

        /** @brief Holds the size of the result. */
        ImageSize target{};

        /** @brief Holds the one scale applied to both axes. */
        double ratio = 1.0;

        /** @brief Holds the size after the resize and before the padding. */
        ImageSize resized{};

        /** @brief Counts the padding rows above the image. */
        std::int32_t top = 0;

        /** @brief Counts the padding rows below the image. */
        std::int32_t bottom = 0;

        /** @brief Counts the padding columns left of the image. */
        std::int32_t left = 0;

        /** @brief Counts the padding columns right of the image. */
        std::int32_t right = 0;

        /** @brief Holds the value of every padding byte. */
        std::uint8_t padding_value = 114;
    };

    /**
     * @brief Accepts every letterbox setting, since both fields are valid at any value.
     *
     * It exists so that a pipeline validates each transform's settings the same way.
     */
    void validate_letterbox(const LetterboxSettings& settings) noexcept;

    /**
     * @brief Computes the ratio, the resized size and the four paddings of a letterbox.
     *
     * In double: ratio = min(target.height / source.height, target.width / source.width),
     * capped at 1 unless scale_up. Each resized side is side * ratio rounded half to even.
     * The padding of an axis is split in two; an odd one puts its extra pixel below or right.
     * @return The plan, or InvalidDescription unless both sizes are valid and both sides positive.
     */
    [[nodiscard]] Result<LetterboxPlan> plan_letterbox(const LetterboxSettings& settings,
                                                       ImageSize source, ImageSize target);

    /**
     * @brief Resizes an image to plan.resized and pads it to plan.target.
     *
     * The resize is the kernel of resize_bilinear and is skipped when plan.resized equals
     * the source size. The resized image starts at row plan.top and column plan.left; every
     * other byte is plan.padding_value.
     * @pre source.size() is plan.source and the plan tiles its target; check_source passes.
     */
    void apply_letterbox(ImageView source, const LetterboxPlan& plan, Image& destination);

    /**
     * @brief Moves boxes and polygons into the letterboxed image.
     *
     * With r = float(plan.ratio): x becomes x * r + plan.left and y becomes y * r + plan.top,
     * in float, multiply first. Classes and order are kept; nothing is clipped or removed.
     */
    [[nodiscard]] Labels apply_letterbox(const Labels& labels, const LetterboxPlan& plan);
} // namespace SushiAI::Vision::Imaging
