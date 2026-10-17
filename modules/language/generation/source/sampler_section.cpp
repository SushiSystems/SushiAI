/**************************************************************************/
/* sampler_section.cpp                                                    */
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
 * @file sampler_section.cpp
 * @brief Implements make_sampler over a table of the four sampler types.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/sampler_section.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/nucleus_sampler.hpp>
#include <SushiAI/language/generation/temperature_sampler.hpp>
#include <SushiAI/language/generation/top_k_sampler.hpp>
#include "section.hpp"

namespace SushiAI::Language::Generation
{
    namespace
    {
        /** @brief Names the key that holds the sampler's type. */
        constexpr std::string_view K_TYPE_KEY = "type";

        /** @brief Names the key that holds the temperature of a drawing sampler. */
        constexpr std::string_view K_TEMPERATURE_KEY = "temperature";

        /** @brief Names the key that holds the tokens top_k keeps. */
        constexpr std::string_view K_K_KEY = "k";

        /** @brief Names the key that holds the mass nucleus keeps. */
        constexpr std::string_view K_P_KEY = "p";

        /** @brief Lists the keys of a "greedy" section. */
        constexpr std::string_view K_GREEDY_KEYS[] = {K_TYPE_KEY};

        /** @brief Lists the keys of a "temperature" section. */
        constexpr std::string_view K_TEMPERATURE_KEYS[] = {K_TYPE_KEY, K_TEMPERATURE_KEY};

        /** @brief Lists the keys of a "top_k" section. */
        constexpr std::string_view K_TOP_K_KEYS[] = {K_TYPE_KEY, K_TEMPERATURE_KEY, K_K_KEY};

        /** @brief Lists the keys of a "nucleus" section. */
        constexpr std::string_view K_NUCLEUS_KEYS[] = {K_TYPE_KEY, K_TEMPERATURE_KEY, K_P_KEY};

        /** @brief Builds one sampler from a section whose keys were checked. */
        using SamplerFactory =
            Result<std::unique_ptr<Sampler>> (*)(const Config::SectionReader& reader);

        /** @brief Holds one row of the sampler table. */
        struct SamplerEntry
        {
            std::string_view type; /**< The "type" a section spells. */
            Config::KeySpan keys;  /**< Every key a section of the type may hold. */
            SamplerFactory make;   /**< What builds the sampler. */
        };

        /** @brief Builds a GreedySampler, which reads no key. */
        [[nodiscard]] Result<std::unique_ptr<Sampler>> make_greedy(
            const Config::SectionReader&)
        {
            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, []()
            {
                return std::unique_ptr<Sampler>(new GreedySampler());
            });
        }

        /** @brief Builds a TemperatureSampler from the optional "temperature". */
        [[nodiscard]] Result<std::unique_ptr<Sampler>> make_temperature(
            const Config::SectionReader& reader)
        {
            SA_TRY_ASSIGN(const double temperature,
                          Detail::read_number_or(reader, K_TEMPERATURE_KEY,
                                                 K_DEFAULT_TEMPERATURE));
            SA_TRY_ASSIGN(std::unique_ptr<TemperatureSampler> sampler,
                          TemperatureSampler::make(temperature));

            return std::unique_ptr<Sampler>(std::move(sampler));
        }

        /** @brief Builds a TopKSampler from "k" and the optional "temperature". */
        [[nodiscard]] Result<std::unique_ptr<Sampler>> make_top_k(
            const Config::SectionReader& reader)
        {
            SA_TRY_ASSIGN(const std::int64_t k, Detail::read_whole(reader, K_K_KEY));
            SA_TRY_ASSIGN(const double temperature,
                          Detail::read_number_or(reader, K_TEMPERATURE_KEY,
                                                 K_DEFAULT_TEMPERATURE));
            SA_TRY_ASSIGN(std::unique_ptr<TopKSampler> sampler,
                          TopKSampler::make(k, temperature));

            return std::unique_ptr<Sampler>(std::move(sampler));
        }

        /** @brief Builds a NucleusSampler from "p" and the optional "temperature". */
        [[nodiscard]] Result<std::unique_ptr<Sampler>> make_nucleus(
            const Config::SectionReader& reader)
        {
            SA_TRY_ASSIGN(const double mass, Detail::read_number(reader, K_P_KEY));
            SA_TRY_ASSIGN(const double temperature,
                          Detail::read_number_or(reader, K_TEMPERATURE_KEY,
                                                 K_DEFAULT_TEMPERATURE));
            SA_TRY_ASSIGN(std::unique_ptr<NucleusSampler> sampler,
                          NucleusSampler::make(mass, temperature));

            return std::unique_ptr<Sampler>(std::move(sampler));
        }

        /** @brief Lists the four samplers a section may name. */
        const SamplerEntry K_SAMPLERS[] = {
            {K_GREEDY_SAMPLER, Config::KeySpan(K_GREEDY_KEYS), &make_greedy},
            {K_TEMPERATURE_SAMPLER, Config::KeySpan(K_TEMPERATURE_KEYS), &make_temperature},
            {K_TOP_K_SAMPLER, Config::KeySpan(K_TOP_K_KEYS), &make_top_k},
            {K_NUCLEUS_SAMPLER, Config::KeySpan(K_NUCLEUS_KEYS), &make_nucleus},
        };

        /** @brief Joins the types of the table into "'a', 'b'" for a diagnostic. */
        [[nodiscard]] std::string list_samplers()
        {
            std::string listed;
            for (const SamplerEntry& entry : K_SAMPLERS)
            {
                if (!listed.empty())
                    listed.append(", ");

                listed.append("'").append(entry.type).append("'");
            }

            return listed;
        }
    } // namespace

    Result<std::unique_ptr<Sampler>> make_sampler(const Config::Value& section,
                                                  std::string_view context)
    {
        const Config::SectionReader reader(section, context);
        SA_TRY(reader.check_object());
        SA_TRY_ASSIGN(const std::string type, Detail::read_text(reader, K_TYPE_KEY));

        for (const SamplerEntry& entry : K_SAMPLERS)
        {
            if (entry.type != type)
                continue;

            SA_TRY(reader.check_keys(entry.keys));
            return entry.make(reader);
        }

        return Status::from_message(StatusCode::InvalidDescription, [&]()
        {
            return std::string(context) + " names the sampler '" + type
                 + "', which this build does not have; it has: " + list_samplers();
        });
    }

} // namespace SushiAI::Language::Generation
