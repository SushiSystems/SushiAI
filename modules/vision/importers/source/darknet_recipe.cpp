/**************************************************************************/
/* darknet_recipe.cpp                                                     */
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
 * @file darknet_recipe.cpp
 * @brief Implements the writing of a .cfg's training settings as training-file sections.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/darknet_recipe.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiBLAS/core/format.hpp>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/optim/step_schedule_section.hpp>
#include <SushiAI/vision/detect/anchor/head_training_section.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_training.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Holds the section type of a head. */
            constexpr std::string_view K_HEAD_SECTION = "yolo";

            /** @brief Holds the optimizer type the solver of a .cfg is written as. */
            constexpr std::string_view K_OPTIMIZER_TYPE = "sgd";

            /** @brief Holds the character the JSON writer ends its text with. */
            constexpr char K_LINE_END = '\n';

            /** @brief Names a key a .cfg may spell and why no training file holds it. */
            struct UnplacedKey
            {
                /** @brief Holds the key as the .cfg spells it. */
                std::string_view key;

                /** @brief Holds the reason a note gives. */
                std::string_view reason;
            };

            /** @brief Lists the keys of the first section that reach no training file. */
            constexpr UnplacedKey K_UNPLACED_NETWORK_KEYS[] = {
                {"batch", "the batch size is 'batch_size' of the data block"},
                {"subdivisions", "a training file has no key that splits a batch"},
                {"max_batches", "a training file counts 'epochs', not iterations"},
                {"angle", "the reader augments no sample"},
                {"saturation", "the reader augments no sample"},
                {"exposure", "the reader augments no sample"},
                {"hue", "the reader augments no sample"},
                {"mosaic", "the reader augments no sample"},
            };

            /** @brief Lists the keys of a head section that reach no training file. */
            constexpr UnplacedKey K_UNPLACED_HEAD_KEYS[] = {
                {"jitter", "the reader augments no sample"},
                {"random", "the model file fixes the input size"},
                {"resize", "the reader augments no sample"},
                {"cls_normalizer", "no option of the anchor objective scales a class delta"},
            };

            /** @brief Returns the head sections of @p config, in file order. */
            [[nodiscard]] std::vector<const DarknetSection*> list_head_sections(
                const DarknetConfig& config)
            {
                std::vector<const DarknetSection*> heads;
                for (const DarknetSection& section : config.layers)
                {
                    if (section.type == K_HEAD_SECTION)
                        heads.push_back(&section);
                }

                return heads;
            }

            /** @brief Appends a note for each option of @p section that @p unplaced lists. */
            template<std::size_t N>
            void note_unplaced(const DarknetSection& section, const UnplacedKey (&unplaced)[N],
                               std::vector<std::string>& notes)
            {
                for (const DarknetOption& option : section.options)
                {
                    for (const UnplacedKey& row : unplaced)
                    {
                        if (option.key == row.key)
                            notes.push_back(SushiBLAS::Core::format(
                                "line {}: {}={} of [{}] is not carried over: {}", option.line,
                                option.key, option.value, section.type, row.reason));
                    }
                }
            }

            /** @brief Returns the JSON spelling of @p value on one line, without a line end. */
            [[nodiscard]] Result<std::string> spell_value(const Config::Value& value)
            {
                SA_TRY_ASSIGN(std::string text, Config::write_json(value, 0));
                while (!text.empty() && text.back() == K_LINE_END)
                    text.pop_back();

                return text;
            }

            /**
             * @brief Checks that two objective sections state the same value under every key.
             * @param first The section of head @p first_head; @p other that of @p other_head.
             * @return InvalidDescription naming both heads, their lines and the first key that
             *         differs, or the failure of the JSON writer.
             */
            [[nodiscard]] Status require_same_options(const Config::Value& first,
                                                      const Config::Value& other,
                                                      const DarknetSection& first_head,
                                                      const DarknetSection& other_head,
                                                      std::size_t other_index)
            {
                const Config::Value absent;
                for (const std::string& key : first.keys())
                {
                    SA_TRY_ASSIGN(const std::string wanted, spell_value(first.at_or(key, absent)));
                    SA_TRY_ASSIGN(const std::string found, spell_value(other.at_or(key, absent)));
                    SA_FAIL_IF(wanted != found, StatusCode::InvalidDescription,
                               "head 0 (line {}) and head {} (line {}) differ in '{}': {} and "
                               "{}; the objective section of a training file holds one set of "
                               "options for every head",
                               first_head.line, other_index, other_head.line, key, wanted,
                               found);
                }

                return Status();
            }

            /**
             * @brief Writes the options every head shares as the objective section.
             * @return The section; InvalidDescription for a file without a head, for heads
             *         that differ, or for what write_head_training_section refuses.
             */
            [[nodiscard]] Result<Config::Value> write_objective(
                const DarknetTraining& training,
                const std::vector<const DarknetSection*>& heads)
            {
                SA_FAIL_IF(training.heads.empty(), StatusCode::InvalidDescription,
                           "the configuration has no [{}] section, so it states no objective",
                           K_HEAD_SECTION);
                SA_FAIL_IF(training.heads.size() != heads.size(), StatusCode::InvalidDescription,
                           "the training settings hold {} heads, but the configuration has {} "
                           "[{}] sections",
                           training.heads.size(), heads.size(), K_HEAD_SECTION);

                SA_TRY_ASSIGN(Config::Value first, Detect::Anchor::write_head_training_section(
                                                       training.heads.front()));
                for (std::size_t k = 1; k < training.heads.size(); ++k)
                {
                    SA_TRY_ASSIGN(const Config::Value other,
                                  Detect::Anchor::write_head_training_section(training.heads[k]));
                    SA_TRY(require_same_options(first, other, *heads.front(), *heads[k], k));
                }

                return first;
            }

            /** @brief Writes the rate, burn-in, steps and scales of @p solver as a schedule. */
            [[nodiscard]] Result<Config::Value> write_schedule(const DarknetSolver& solver)
            {
                SA_TRY_ASSIGN(const Optim::StepSchedule schedule,
                              SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                           [&]()
                {
                    Optim::StepSchedule stated;
                    stated.base_rate = solver.learning_rate;
                    stated.warmup_iterations = solver.burn_in;
                    stated.steps = solver.steps;
                    stated.scales = solver.scales;
                    return stated;
                }));

                return Optim::write_step_schedule_section(schedule);
            }

            /** @brief Returns the optimizer section: "sgd" with the solver's momentum and decay. */
            [[nodiscard]] Config::Value write_optimizer(const DarknetSolver& solver)
            {
                return Config::Value::object(
                    {"type", "momentum", "weight_decay"},
                    {Config::Value::string(std::string(K_OPTIMIZER_TYPE)),
                     Config::Value::real(solver.momentum), Config::Value::real(solver.decay)});
            }

        } // namespace

        Result<DarknetRecipe> write_darknet_recipe(const DarknetConfig& config,
                                                   const DarknetTraining& training)
        {
            SA_TRY_ASSIGN(const std::vector<const DarknetSection*> heads,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&]()
            {
                return list_head_sections(config);
            }));

            SA_TRY_ASSIGN(Config::Value objective, write_objective(training, heads));
            SA_TRY_ASSIGN(Config::Value schedule, write_schedule(training.solver));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                DarknetRecipe recipe;
                recipe.training = Config::Value::object(
                    {"objective", "optimizer", "schedule"},
                    {std::move(objective), write_optimizer(training.solver),
                     std::move(schedule)});

                note_unplaced(config.net, K_UNPLACED_NETWORK_KEYS, recipe.notes);
                for (const DarknetSection* const head : heads)
                    note_unplaced(*head, K_UNPLACED_HEAD_KEYS, recipe.notes);

                return recipe;
            });
        }

    } // namespace Vision::Importers
} // namespace SushiAI
