/**************************************************************************/
/* free_loss_gains.cpp                                                    */
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

/**
 * @file free_loss_gains.cpp
 * @brief Implements the gains of the anchor-free detection loss and their reader.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_loss_gains.hpp>

#include <cmath>
#include <string>

#include <SushiAI/config/section_reader.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Names the section in a diagnostic; every refusal starts with it. */
        constexpr std::string_view K_SECTION = "objective";

        /** @brief Pairs one key with the gain it states. */
        struct GainField
        {
            std::string_view key;          /**< The key in the section. */
            float FreeLossGains::*gain;    /**< The gain the key states. */
        };

        /** @brief Lists every gain, in the order of K_FREE_LOSS_GAIN_KEYS. */
        constexpr GainField K_FIELDS[] = {
            {K_FREE_LOSS_GAIN_KEYS[0], &FreeLossGains::box_gain},
            {K_FREE_LOSS_GAIN_KEYS[1], &FreeLossGains::class_gain},
            {K_FREE_LOSS_GAIN_KEYS[2], &FreeLossGains::distribution_gain},
            {K_FREE_LOSS_GAIN_KEYS[3], &FreeLossGains::l1_gain}};
    } // namespace

    Status validate_free_loss_gains(const FreeLossGains& gains) noexcept
    {
        for (const GainField& field : K_FIELDS)
        {
            const float gain = gains.*field.gain;
            SA_FAIL_IF(!std::isfinite(gain) || gain < 0.0f, StatusCode::InvalidDescription,
                       "{} is {}; a gain is finite and not negative",
                       Config::member_of(K_SECTION, field.key), gain);
        }

        return Status();
    }

    Result<FreeLossGains> read_free_loss_gains(const Config::Value& section) noexcept
    {
        const Config::SectionReader reader(section, K_SECTION);
        SA_TRY(reader.check_object());

        FreeLossGains result;
        for (const GainField& field : K_FIELDS)
        {
            if (!reader.has(field.key))
                continue;

            SA_TRY_ASSIGN(const Config::Value* const member, reader.find(field.key));
            SA_TRY_ASSIGN(result.*field.gain,
                          Config::read_float32(*member, reader.member(field.key)));
        }

        SA_TRY(validate_free_loss_gains(result));
        return result;
    }

} // namespace SushiAI::Vision::Training
