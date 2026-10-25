/**************************************************************************/
/* free_objective_factory.cpp                                             */
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
 * @file free_objective_factory.cpp
 * @brief Implements the factory that builds an anchor-free objective from the files of a run.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_objective_factory.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <string_view>
#include <utility>

#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the name a refusal gives the objective section. */
        constexpr std::string_view K_SECTION = "objective";

        /** @brief Holds the key that names the objective's type. */
        constexpr std::string_view K_TYPE_KEY = "type";

        /** @brief Returns "type" and every gain key, the keys the section may hold. */
        [[nodiscard]] std::array<std::string_view, K_FREE_LOSS_GAIN_KEYS.size() + 1>
        list_keys() noexcept
        {
            std::array<std::string_view, K_FREE_LOSS_GAIN_KEYS.size() + 1> keys{};
            keys[0] = K_TYPE_KEY;
            for (std::size_t index = 0; index < K_FREE_LOSS_GAIN_KEYS.size(); ++index)
                keys[index + 1] = K_FREE_LOSS_GAIN_KEYS[index];
            return keys;
        }
    } // namespace

    Result<FreeObjective> build_free_objective(const Train::ObjectiveContext& given,
                                               const Config::Value& attributes)
    {
        SA_FAIL_IF(given.task.is_null(), StatusCode::InvalidDescription,
                   "the model file has no 'task' section, which states the assigner an '{}' "
                   "objective trains with",
                   K_FREE_OBJECTIVE_TYPE);

        const Config::SectionReader reader(attributes, K_SECTION);
        SA_TRY(reader.check_object());
        const auto keys = list_keys();
        SA_TRY(reader.check_keys(Config::KeySpan(keys.data(), keys.size())));
        SA_TRY(reader.check_type(K_TYPE_KEY, K_FREE_OBJECTIVE_TYPE, "an anchor-free objective"));

        SA_TRY_ASSIGN(Detect::Free::FreeTask task, Detect::Free::read_free_task(given.task));
        SA_TRY_ASSIGN(const FreeLossGains gains, read_free_loss_gains(attributes));
        SA_TRY_ASSIGN(FreeModelFacts facts,
                      derive_free_model_facts(given.input_sample, given.outputs));

        SA_FAIL_IF(given.classes != 0 && given.classes != facts.classes,
                   StatusCode::Incompatible,
                   "the data spans {} classes, but the model's score ports hold {}",
                   given.classes, facts.classes);

        return FreeObjective::make(std::move(facts), std::move(task), given.batch, gains);
    }

    Result<std::unique_ptr<Train::Objective>> make_free_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes,
        std::string_view context)
    {
        Result<FreeObjective> built = build_free_objective(given, attributes);
        SA_TRY(SA_ANNOTATE(built.status(), "{}", context));

        return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<Train::Objective>(
                std::make_unique<FreeObjective>(std::move(built).value()));
        });
    }

} // namespace SushiAI::Vision::Training
