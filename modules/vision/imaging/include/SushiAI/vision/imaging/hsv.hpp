/**************************************************************************/
/* hsv.hpp                                                                */
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
 * @file hsv.hpp
 * @brief Declares the random hue, saturation and value jitter of an RGB image.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstdint>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/random.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds the number of 8-bit hue steps in a full turn; one step is two degrees. */
    inline constexpr std::int32_t K_HUE_STEPS = 180;

    /** @brief Holds one 8-bit RGB pixel. */
    struct RGBPixel
    {
        /** @brief Holds the red level. */
        std::uint8_t red = 0;

        /** @brief Holds the green level. */
        std::uint8_t green = 0;

        /** @brief Holds the blue level. */
        std::uint8_t blue = 0;
    };

    /** @brief Holds one 8-bit HSV pixel: hue in [0, K_HUE_STEPS), the others in [0, 255]. */
    struct HSVPixel
    {
        /** @brief Holds the hue in steps of two degrees. */
        std::uint8_t hue = 0;

        /** @brief Holds the saturation, 255 being full. */
        std::uint8_t saturation = 0;

        /** @brief Holds the value, the largest of the three RGB levels. */
        std::uint8_t value = 0;
    };

    /** @brief Holds what the caller chooses about the jitter: the half-range of each gain. */
    struct HSVSettings
    {
        /** @brief Holds the hue half-range as a fraction of a full turn (cfg/default.yaml:115). */
        float hue_gain = 0.015f;

        /** @brief Holds the saturation half-range (cfg/default.yaml:116). */
        float saturation_gain = 0.7f;

        /** @brief Holds the value half-range (cfg/default.yaml:117). */
        float value_gain = 0.4f;
    };

    /** @brief Holds everything apply_hsv needs: the three drawn gains. */
    struct HSVPlan
    {
        /** @brief Holds the hue shift as a fraction of a full turn, in [-hue_gain, hue_gain). */
        float hue = 0.0f;

        /** @brief Holds the saturation gain; saturation is multiplied by 1 + this. */
        float saturation = 0.0f;

        /** @brief Holds the value gain; value is multiplied by 1 + this. */
        float value = 0.0f;
    };

    /** @brief Holds the three lookup tables a plan turns into, indexed by the 8-bit input. */
    struct HSVTables
    {
        /** @brief Maps a hue to its shifted hue; entries from K_HUE_STEPS up are never read. */
        std::array<std::uint8_t, 256> hue{};

        /** @brief Maps a saturation to its scaled saturation. */
        std::array<std::uint8_t, 256> saturation{};

        /** @brief Maps a value to its scaled value. */
        std::array<std::uint8_t, 256> value{};
    };

    /** @brief Returns InvalidDescription unless each gain is finite and not negative. */
    [[nodiscard]] Status validate_hsv(const HSVSettings& settings);

    /**
     * @brief Draws the three gains of one jitter.
     *
     * Three draws, in the order hue, saturation, value: each is
     * random.uniform(-1.0f, 1.0f) multiplied in float by its gain. The draws are made even
     * when a gain is zero.
     * @return The plan, or what validate_hsv returns, before any draw.
     */
    [[nodiscard]] Result<HSVPlan> plan_hsv(const HSVSettings& settings, Random& random);

    /**
     * @brief Reports whether a plan changes no pixel: all three of its gains are zero.
     *
     * Three zero settings always give such a plan, which is the case the reference skips
     * (data/augment.py:1514).
     */
    [[nodiscard]] bool is_hsv_identity(const HSVPlan& plan) noexcept;

    /**
     * @brief Builds the lookup tables of a plan.
     *
     * For input x in double: hue[x] is (x + plan.hue * 180) reduced into [0, 180) by a
     * floored modulo; saturation[x] is x * (1 + plan.saturation) and value[x] is
     * x * (1 + plan.value), each clamped to [0, 255]. All three truncate toward zero, and
     * saturation[0] is 0.
     */
    [[nodiscard]] HSVTables make_hsv_tables(const HSVPlan& plan) noexcept;

    /**
     * @brief Converts an RGB pixel to 8-bit HSV in integer arithmetic.
     *
     * Value is the largest level, saturation is 255 * (largest - smallest) / largest, and
     * hue is the hexcone angle in steps of two degrees, each rounded half up. A grey pixel
     * has hue 0 and a black one saturation 0. README.md, section hsv.hpp, gives the formulas.
     */
    [[nodiscard]] HSVPixel to_hsv(RGBPixel pixel) noexcept;

    /**
     * @brief Converts an 8-bit HSV pixel to RGB in integer arithmetic.
     *
     * The inverse hexcone over six sectors of 30 hue steps, each level rounded half up.
     * README.md, section hsv.hpp, gives the formulas.
     * @pre pixel.hue < K_HUE_STEPS.
     */
    [[nodiscard]] RGBPixel to_rgb(HSVPixel pixel) noexcept;

    /**
     * @brief Jitters the colour of an image.
     *
     * Each pixel of a three-channel image goes through to_hsv, the tables of
     * make_hsv_tables(plan), then to_rgb. The pixels are copied unchanged when the image has
     * another channel count or when is_hsv_identity(plan). No label moves.
     * @param destination Reset to the source's size and channels; must pass check_source.
     */
    void apply_hsv(ImageView source, const HSVPlan& plan, Image& destination);
} // namespace SushiAI::Vision::Imaging
