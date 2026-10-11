/**************************************************************************/
/* sampler_section.hpp                                                    */
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
 * @file sampler_section.hpp
 * @brief Declares the factory that builds a sampler from the "sampler" object of a task.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/language/generation/sampler.hpp>

namespace SushiAI::Language::Generation
{
    /** @brief Holds the temperature of a drawing sampler whose section states none. */
    inline constexpr double K_DEFAULT_TEMPERATURE = 1.0;

    /**
     * @brief Builds the sampler "type" names: greedy, temperature, top_k or nucleus.
     * @param section An object. "temperature" is optional for the three that draw, "k" is
     *        required by top_k and "p" by nucleus; a key the type does not read is refused.
     * @param context What to call the section in a diagnostic.
     * @return The sampler, or InvalidDescription naming the key or the type that does not
     *         fit; an unknown type lists the four.
     */
    [[nodiscard]] Result<std::unique_ptr<Sampler>> make_sampler(const Config::Value& section,
                                                                std::string_view context);

} // namespace SushiAI::Language::Generation
