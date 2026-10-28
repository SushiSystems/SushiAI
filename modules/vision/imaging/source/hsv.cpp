/**************************************************************************/
/* hsv.cpp                                                                */
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
 * @file hsv.cpp
 * @brief Implements the HSV jitter: its draws, its lookup tables and the integer hexcone.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/hsv.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Holds the largest 8-bit level. */
        constexpr std::int32_t K_FULL_LEVEL = 255;

        /** @brief Holds the hue steps of one of the six hexcone sectors. */
        constexpr std::int32_t K_SECTOR_STEPS = K_HUE_STEPS / 6;

        /** @brief Holds the divisor of the two sector interpolations, 255 * 30. */
        constexpr std::int32_t K_SECTOR_SCALE = K_FULL_LEVEL * K_SECTOR_STEPS;

        /** @brief Reports whether a gain is a finite number that is not negative. */
        [[nodiscard]] bool is_gain(float gain) noexcept
        {
            return std::isfinite(gain) && gain >= 0.0f;
        }

        /** @brief Reduces a hue into [0, K_HUE_STEPS) by a floored modulo and truncates it. */
        [[nodiscard]] std::uint8_t wrap_hue(double hue) noexcept
        {
            double wrapped = std::fmod(hue, static_cast<double>(K_HUE_STEPS));
            if (wrapped < 0.0)
                wrapped += K_HUE_STEPS;

            // A tiny negative hue rounds up to K_HUE_STEPS, and a NaN compares false.
            if (!(wrapped < K_HUE_STEPS))
                wrapped = 0.0;

            return static_cast<std::uint8_t>(wrapped);
        }

        /** @brief Clamps a level to [0, 255] and truncates it toward zero; a NaN gives 0. */
        [[nodiscard]] std::uint8_t clamp_level(double level) noexcept
        {
            if (!(level > 0.0))
                return 0;

            if (level >= K_FULL_LEVEL)
                return static_cast<std::uint8_t>(K_FULL_LEVEL);

            return static_cast<std::uint8_t>(level);
        }
    } // namespace

    Status validate_hsv(const HSVSettings& settings)
    {
        SA_FAIL_IF(!is_gain(settings.hue_gain) || !is_gain(settings.saturation_gain) ||
                       !is_gain(settings.value_gain),
                   StatusCode::InvalidDescription,
                   "validate_hsv: every gain must be finite and not negative; got hue {}, "
                   "saturation {}, value {}",
                   settings.hue_gain, settings.saturation_gain, settings.value_gain);
        return Status();
    }

    Result<HSVPlan> plan_hsv(const HSVSettings& settings, Random& random)
    {
        SA_TRY(validate_hsv(settings));

        HSVPlan plan;
        plan.hue = random.uniform(-1.0f, 1.0f) * settings.hue_gain;
        plan.saturation = random.uniform(-1.0f, 1.0f) * settings.saturation_gain;
        plan.value = random.uniform(-1.0f, 1.0f) * settings.value_gain;
        return Result<HSVPlan>(plan);
    }

    bool is_hsv_identity(const HSVPlan& plan) noexcept
    {
        return plan.hue == 0.0f && plan.saturation == 0.0f && plan.value == 0.0f;
    }

    HSVTables make_hsv_tables(const HSVPlan& plan) noexcept
    {
        const double hue_shift = static_cast<double>(plan.hue) * K_HUE_STEPS;
        const double saturation_scale = static_cast<double>(plan.saturation) + 1.0;
        const double value_scale = static_cast<double>(plan.value) + 1.0;

        HSVTables tables;
        for (std::size_t x = 0; x < tables.hue.size(); ++x)
        {
            const auto level = static_cast<double>(x);
            tables.hue[x] = wrap_hue(level + hue_shift);
            tables.saturation[x] = clamp_level(level * saturation_scale);
            tables.value[x] = clamp_level(level * value_scale);
        }

        tables.saturation[0] = 0;
        return tables;
    }

    HSVPixel to_hsv(RGBPixel pixel) noexcept
    {
        const std::int32_t red = pixel.red;
        const std::int32_t green = pixel.green;
        const std::int32_t blue = pixel.blue;
        const std::int32_t largest = std::max(red, std::max(green, blue));
        const std::int32_t spread = largest - std::min(red, std::min(green, blue));

        HSVPixel result;
        result.value = static_cast<std::uint8_t>(largest);

        if (largest != 0)
            result.saturation =
                static_cast<std::uint8_t>((K_FULL_LEVEL * spread + largest / 2) / largest);

        if (spread == 0)
            return result;

        std::int32_t numerator = 0;
        if (largest == red)
            numerator = K_SECTOR_STEPS * (green - blue);
        else if (largest == green)
            numerator = K_SECTOR_STEPS * (blue - red) + 2 * K_SECTOR_STEPS * spread;
        else
            numerator = K_SECTOR_STEPS * (red - green) + 4 * K_SECTOR_STEPS * spread;

        if (numerator < 0)
            numerator += K_HUE_STEPS * spread;

        std::int32_t hue = (2 * numerator + spread) / (2 * spread);
        if (hue >= K_HUE_STEPS)
            hue -= K_HUE_STEPS;

        result.hue = static_cast<std::uint8_t>(hue);
        return result;
    }

    RGBPixel to_rgb(HSVPixel pixel) noexcept
    {
        const std::int32_t hue = pixel.hue % K_HUE_STEPS;
        const std::int32_t saturation = pixel.saturation;
        const std::int32_t value = pixel.value;
        const std::int32_t sector = hue / K_SECTOR_STEPS;
        const std::int32_t offset = hue - K_SECTOR_STEPS * sector;

        const auto low = static_cast<std::uint8_t>(
            (value * (K_FULL_LEVEL - saturation) + K_FULL_LEVEL / 2) / K_FULL_LEVEL);
        const auto falling = static_cast<std::uint8_t>(
            (value * (K_SECTOR_SCALE - saturation * offset) + K_SECTOR_SCALE / 2) /
            K_SECTOR_SCALE);
        const auto rising = static_cast<std::uint8_t>(
            (value * (K_SECTOR_SCALE - saturation * (K_SECTOR_STEPS - offset)) +
             K_SECTOR_SCALE / 2) /
            K_SECTOR_SCALE);
        const std::uint8_t high = pixel.value;

        switch (sector)
        {
            case 0:
                return RGBPixel{high, rising, low};
            case 1:
                return RGBPixel{falling, high, low};
            case 2:
                return RGBPixel{low, high, rising};
            case 3:
                return RGBPixel{low, falling, high};
            case 4:
                return RGBPixel{rising, low, high};
            default:
                return RGBPixel{high, low, falling};
        }
    }

    void apply_hsv(ImageView source, const HSVPlan& plan, Image& destination)
    {
        check_source(source, destination, "apply_hsv");

        destination.reset(source.size(), source.channels());

        if (source.channels() != K_RGB_CHANNELS || is_hsv_identity(plan))
        {
            std::memcpy(destination.data(), source.data(), source.byte_count());
            return;
        }

        const HSVTables tables = make_hsv_tables(plan);
        const std::uint8_t* in = source.data();
        const std::uint8_t* const end = in + source.byte_count();
        std::uint8_t* out = destination.data();

        for (; in != end; in += K_RGB_CHANNELS, out += K_RGB_CHANNELS)
        {
            const HSVPixel hsv = to_hsv(RGBPixel{in[0], in[1], in[2]});
            const RGBPixel rgb = to_rgb(HSVPixel{tables.hue[hsv.hue],
                                                 tables.saturation[hsv.saturation],
                                                 tables.value[hsv.value]});
            out[0] = rgb.red;
            out[1] = rgb.green;
            out[2] = rgb.blue;
        }
    }
} // namespace SushiAI::Vision::Imaging
