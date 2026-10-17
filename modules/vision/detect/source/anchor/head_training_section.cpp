/**************************************************************************/
/* head_training_section.cpp                                              */
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
 * @file head_training_section.cpp
 * @brief Implements the reader and the writer of the anchor objective section.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/head_training_section.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/iou_gradient.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            using Config::Kind;
            using Config::Value;

            /** @brief Names the section in a diagnostic; every refusal starts with it. */
            constexpr std::string_view K_SECTION = "objective";

            /** @brief Names the key that states which objective the section describes. */
            constexpr std::string_view K_TYPE_KEY = "type";

            /** @brief Pairs one enumerator with the string a section states for it. */
            template<typename Enumerator>
            struct ChoiceName
            {
                Enumerator value;      /**< The enumerator. */
                std::string_view name; /**< Its string in the section. */
            };

            /** @brief Lists every box loss; the reader and the writer both look it up. */
            constexpr ChoiceName<BoxLoss> K_BOX_LOSS_NAMES[] = {
                {BoxLoss::SquaredError, "squared_error"},
                {BoxLoss::CompleteIoU, "complete_iou"}};

            /** @brief Lists every box gradient; the reader and the writer both look it up. */
            constexpr ChoiceName<BoxGradient> K_BOX_GRADIENT_NAMES[] = {
                {BoxGradient::Analytic, "analytic"},
                {BoxGradient::DarknetCompatible, "darknet_compatible"}};

            /** @brief Lists every objectness target; the reader and the writer look it up. */
            constexpr ChoiceName<ObjectnessTarget> K_OBJECTNESS_TARGET_NAMES[] = {
                {ObjectnessTarget::One, "one"},
                {ObjectnessTarget::BestIoU, "best_iou"}};

            /** @brief Returns the strings of @p table, quoted and separated by commas. */
            template<typename Enumerator, std::size_t N>
            std::string list_names(const ChoiceName<Enumerator> (&table)[N])
            {
                std::array<std::string_view, N> names{};
                for (std::size_t index = 0; index < N; ++index)
                    names[index] = table[index].name;
                return Config::list_quoted(Config::KeySpan(names.data(), names.size()));
            }

            /** @brief Returns the text that names the member @p key of the section. */
            std::string member_of(std::string_view key)
            {
                return Config::member_of(K_SECTION, key);
            }

            /**
             * @brief Reads the number @p member, named @p where, into the float @p field.
             * @return Success, or InvalidDescription for a value that is not a number or
             *         that float32 cannot hold as a finite value.
             */
            Status read_number(const Value& member, std::string_view where, float& field)
            {
                SA_TRY_ASSIGN(field, Config::read_float32(member, where));
                return Status();
            }

            /** @brief Builds the real of equal value to @p field. */
            Value number_value(float field)
            {
                return Value::real(static_cast<double>(field));
            }

            /**
             * @brief Reads the string @p member, named @p where, into the enumerator @p field.
             * @return Success, or InvalidDescription for a value that is not a string or a
             *         string outside @p table, with the strings of the table.
             */
            template<typename Enumerator, std::size_t N>
            Status read_choice(const Value& member, std::string_view where,
                               const ChoiceName<Enumerator> (&table)[N], Enumerator& field)
            {
                SA_TRY_ASSIGN(const std::string* const text, Config::read_string(member, where));

                const std::string& stated = *text;
                const ChoiceName<Enumerator>* found = nullptr;
                for (const ChoiceName<Enumerator>& row : table)
                {
                    if (row.name == stated)
                        found = &row;
                }

                SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                           "{} is '{}'; it is one of {}", where, stated, list_names(table));
                field = found->value;
                return Status();
            }

            /**
             * @brief Builds the string @p table holds for the enumerator @p field.
             * @return The string, or InvalidDescription naming @p key for an enumerator
             *         outside @p table.
             */
            template<typename Enumerator, std::size_t N>
            Result<Value> choice_value(const ChoiceName<Enumerator> (&table)[N],
                                       Enumerator field, std::string_view key)
            {
                const ChoiceName<Enumerator>* found = nullptr;
                for (const ChoiceName<Enumerator>& row : table)
                {
                    if (row.value == field)
                        found = &row;
                }

                SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                           "{} holds the value {}, which has no string; the strings are {}",
                           member_of(key), static_cast<std::int64_t>(field), list_names(table));
                return Value::string(std::string(found->name));
            }

            /** @brief Holds how one key of the section is read into and written from a field. */
            struct SectionField
            {
                /** @brief Holds the key, which is the name of the field it states. */
                std::string_view key;

                /** @brief Reads the member, named by the second argument, into the options. */
                Status (*read)(const Value&, std::string_view, HeadTrainingOptions&);

                /** @brief Builds the member from the options; the second argument is the key. */
                Result<Value> (*write)(const HeadTrainingOptions&, std::string_view);
            };

            /** @brief Lists every key besides "type", in the order the writer states them. */
            const SectionField K_FIELDS[] = {
                {"match_threshold",
                 [](const Value& member, std::string_view where, HeadTrainingOptions& options)
                 {
                     return read_number(member, where, options.thresholds.match_threshold);
                 },
                 [](const HeadTrainingOptions& options, std::string_view) -> Result<Value>
                 {
                     return number_value(options.thresholds.match_threshold);
                 }},
                {"ignore_threshold",
                 [](const Value& member, std::string_view where, HeadTrainingOptions& options)
                 {
                     return read_number(member, where, options.thresholds.ignore_threshold);
                 },
                 [](const HeadTrainingOptions& options, std::string_view) -> Result<Value>
                 {
                     return number_value(options.thresholds.ignore_threshold);
                 }},
                {"box_loss",
                 [](const Value& member, std::string_view where, HeadTrainingOptions& options)
                 {
                     return read_choice(member, where, K_BOX_LOSS_NAMES, options.box_loss);
                 },
                 [](const HeadTrainingOptions& options, std::string_view key) -> Result<Value>
                 {
                     return choice_value(K_BOX_LOSS_NAMES, options.box_loss, key);
                 }},
                {"box_gradient",
                 [](const Value& member, std::string_view where, HeadTrainingOptions& options)
                 {
                     return read_choice(member, where, K_BOX_GRADIENT_NAMES,
                                        options.box_gradient);
                 },
                 [](const HeadTrainingOptions& options, std::string_view key) -> Result<Value>
                 {
                     return choice_value(K_BOX_GRADIENT_NAMES, options.box_gradient, key);
                 }},
                {"objectness_target",
                 [](const Value& member, std::string_view where, HeadTrainingOptions& options)
                 {
                     return read_choice(member, where, K_OBJECTNESS_TARGET_NAMES,
                                        options.objectness_target);
                 },
                 [](const HeadTrainingOptions& options, std::string_view key) -> Result<Value>
                 {
                     return choice_value(K_OBJECTNESS_TARGET_NAMES, options.objectness_target,
                                         key);
                 }},
                {"box_weight",
                 [](const Value& member, std::string_view where, HeadTrainingOptions& options)
                 {
                     return read_number(member, where, options.box_weight);
                 },
                 [](const HeadTrainingOptions& options, std::string_view) -> Result<Value>
                 {
                     return number_value(options.box_weight);
                 }},
                {"object_weight",
                 [](const Value& member, std::string_view where, HeadTrainingOptions& options)
                 {
                     return read_number(member, where, options.object_weight);
                 },
                 [](const HeadTrainingOptions& options, std::string_view) -> Result<Value>
                 {
                     return number_value(options.object_weight);
                 }},
                {"max_delta",
                 [](const Value& member, std::string_view where, HeadTrainingOptions& options)
                 {
                     return read_number(member, where, options.max_delta);
                 },
                 [](const HeadTrainingOptions& options, std::string_view) -> Result<Value>
                 {
                     return number_value(options.max_delta);
                 }},
                {"class_gate",
                 [](const Value& member, std::string_view where, HeadTrainingOptions& options)
                 {
                     return read_number(member, where, options.class_gate);
                 },
                 [](const HeadTrainingOptions& options, std::string_view) -> Result<Value>
                 {
                     return number_value(options.class_gate);
                 }}};

            /** @brief Returns "type" and the key of every field, in the order a section lists. */
            std::array<std::string_view, std::size(K_FIELDS) + 1> list_keys()
            {
                std::array<std::string_view, std::size(K_FIELDS) + 1> keys{};
                keys[0] = K_TYPE_KEY;
                for (std::size_t index = 0; index < std::size(K_FIELDS); ++index)
                    keys[index + 1] = K_FIELDS[index].key;
                return keys;
            }

            /** @brief Returns the failure of validate_head_training, named as the section's. */
            Status validate_section(const HeadTrainingOptions& options)
            {
                return SA_ANNOTATE(validate_head_training(options), "{}", K_SECTION);
            }
        } // namespace

        Result<HeadTrainingOptions> read_head_training_section(const Config::Value& section)
        {
            const Config::SectionReader reader(section, K_SECTION);
            const auto keys = list_keys();
            SA_TRY(reader.check_object());
            SA_TRY(reader.check_keys(Config::KeySpan(keys.data(), keys.size())));
            SA_TRY(reader.check_type(K_TYPE_KEY, K_ANCHOR_OBJECTIVE_TYPE, "an anchor objective"));

            HeadTrainingOptions result;
            for (const SectionField& field : K_FIELDS)
            {
                if (!section.has(field.key))
                    continue;

                const std::string where = member_of(field.key);
                SA_TRY(field.read(section.at(field.key, where), where, result));
            }

            SA_TRY(validate_section(result));
            return result;
        }

        Result<Config::Value> write_head_training_section(const HeadTrainingOptions& options)
        {
            SA_TRY(validate_section(options));

            std::vector<std::string> keys;
            std::vector<Value> values;
            keys.reserve(std::size(K_FIELDS) + 1);
            values.reserve(std::size(K_FIELDS) + 1);

            keys.emplace_back(K_TYPE_KEY);
            values.push_back(Value::string(std::string(K_ANCHOR_OBJECTIVE_TYPE)));
            for (const SectionField& field : K_FIELDS)
            {
                SA_TRY_ASSIGN(Value value, field.write(options, field.key));
                keys.emplace_back(field.key);
                values.push_back(std::move(value));
            }
            return Value::object(std::move(keys), std::move(values));
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
