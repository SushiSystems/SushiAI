/**************************************************************************/
/* node_attributes.cpp                                                    */
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
 * @file node_attributes.cpp
 * @brief Implements the status-returning readers a node factory reads a model-file row with.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/node_attributes.hpp>

#include <cstddef>
#include <string>

namespace SushiAI
{
    namespace NN
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Names the rank of a per-sample [H, W, C] feature map. */
                constexpr int32_t K_FEATURE_MAP_RANK = 3;

                /** @brief Reports whether @p key is one of @p allowed. */
                [[nodiscard]] bool is_allowed(std::string_view key, KeySpan allowed) noexcept
                {
                    for (std::string_view candidate : allowed)
                    {
                        if (candidate == key)
                            return true;
                    }

                    return false;
                }

                /** @brief Holds the row reference that means the row before. */
                constexpr int64_t K_PREVIOUS_ROW = -1;

                /** @brief Names the key of the object form of a row reference. */
                constexpr std::string_view K_ROW_NODE_KEY = "node";

                /** @brief Lists every key the object form of a row reference accepts. */
                constexpr std::string_view K_ROW_KEYS[] = {K_ROW_NODE_KEY};

                /** @brief Returns the value under @p key once it is known to be a whole number. */
                [[nodiscard]] Result<int64_t> checked_positive_int(const Config::Value& value,
                                                                   std::string_view key,
                                                                   std::string_view context)
                {
                    SA_FAIL_IF(value.kind() != Config::Kind::Int, StatusCode::InvalidDescription,
                               "{} '{}' must be a whole number, but it is {}", context, key,
                               Config::kind_name(value.kind()));

                    const int64_t number = value.as_int(key);
                    SA_FAIL_IF(number <= 0, StatusCode::InvalidDescription,
                               "{} '{}' must be positive, but it is {}", context, key, number);

                    return number;
                }

                /**
                 * @brief Reads the number under @p key, or @p fallback when the key is absent.
                 * @return InvalidDescription when the key holds anything but a number.
                 */
                [[nodiscard]] Result<double> read_real_or(const Config::Value& attributes,
                                                          std::string_view key,
                                                          std::string_view context,
                                                          double fallback)
                {
                    if (!attributes.has(key))
                        return fallback;

                    const Config::Value& value = attributes.at(key, context);
                    SA_FAIL_IF(value.kind() != Config::Kind::Int &&
                                   value.kind() != Config::Kind::Real,
                               StatusCode::InvalidDescription,
                               "{} '{}' must be a number, but it is {}", context, key,
                               Config::kind_name(value.kind()));

                    return value.as_real(key);
                }

            } // namespace

            std::string list_names(KeySpan names)
            {
                std::string listed;
                for (std::size_t index = 0; index < names.size(); ++index)
                {
                    if (index != 0)
                        listed.append(", ");

                    listed.append(names[index]);
                }

                return listed;
            }

            Result<Shape> read_feature_map(NN::ShapeSpan inputs, std::string_view context)
            {
                SA_FAIL_IF(inputs.size() != 1, StatusCode::InvalidDescription,
                           "{} reads one input, but it was given {}", context, inputs.size());
                SA_FAIL_IF(inputs[0].rank() != K_FEATURE_MAP_RANK, StatusCode::InvalidDescription,
                           "{} was given a per-sample input of {}, but it reads a rank-3 "
                           "[H, W, C] input",
                           context, inputs[0].to_string());

                return inputs[0];
            }

            Status reject_unknown_keys(const Config::Value& attributes, KeySpan allowed,
                                       std::string_view context)
            {
                for (const std::string& key : attributes.keys())
                {
                    SA_FAIL_IF(!is_allowed(key, allowed), StatusCode::InvalidDescription,
                               "{} has the unknown key '{}'; the keys it accepts are: {}",
                               context, key, list_names(allowed));
                }

                return Status();
            }

            Result<int64_t> read_positive_int(const Config::Value& attributes,
                                              std::string_view key, std::string_view context)
            {
                SA_FAIL_IF(!attributes.has(key), StatusCode::InvalidDescription,
                           "{} needs the key '{}', which is absent", context, key);

                return checked_positive_int(attributes.at(key, context), key, context);
            }

            Result<int64_t> read_positive_int_or(const Config::Value& attributes,
                                                 std::string_view key, std::string_view context,
                                                 int64_t fallback)
            {
                if (!attributes.has(key))
                    return fallback;

                return checked_positive_int(attributes.at(key, context), key, context);
            }

            Result<bool> read_bool_or(const Config::Value& attributes, std::string_view key,
                                      std::string_view context, bool fallback)
            {
                if (!attributes.has(key))
                    return fallback;

                const Config::Value& value = attributes.at(key, context);
                SA_FAIL_IF(value.kind() != Config::Kind::Bool, StatusCode::InvalidDescription,
                           "{} '{}' must be true or false, but it is {}", context, key,
                           Config::kind_name(value.kind()));

                return value.as_bool(key);
            }

            Result<std::size_t> read_row_above(const Config::Value& attributes,
                                               std::string_view key, std::string_view context,
                                               std::size_t rows_above)
            {
                SA_FAIL_IF(!attributes.has(key), StatusCode::InvalidDescription,
                           "{} needs the key '{}', which is absent", context, key);

                const Config::Value& stated = attributes.at(key, context);
                const bool is_object = stated.kind() == Config::Kind::Object;
                if (is_object)
                    SA_TRY(reject_unknown_keys(stated, KeySpan(K_ROW_KEYS), context));

                SA_FAIL_IF(is_object && !stated.has(K_ROW_NODE_KEY),
                           StatusCode::InvalidDescription,
                           "{} '{}' is an object without '{}', the row it names", context, key,
                           K_ROW_NODE_KEY);

                const Config::Value& number =
                    is_object ? stated.at(K_ROW_NODE_KEY, context) : stated;
                SA_FAIL_IF(number.kind() != Config::Kind::Int, StatusCode::InvalidDescription,
                           "{} '{}' must be a row number, {} or an object carrying '{}', but "
                           "it is {}",
                           context, key, K_PREVIOUS_ROW, K_ROW_NODE_KEY,
                           Config::kind_name(number.kind()));

                const int64_t above = static_cast<int64_t>(rows_above);
                const int64_t named = number.as_int(key);
                const int64_t row = named == K_PREVIOUS_ROW ? above - 1 : named;
                SA_FAIL_IF(row < 0 || row >= above, StatusCode::InvalidDescription,
                           "{} '{}' names row {}, which is not above it; {} rows are above, "
                           "and {} names the one before",
                           context, key, named, above, K_PREVIOUS_ROW);

                return static_cast<std::size_t>(row);
            }

            Result<double> read_real(const Config::Value& attributes, std::string_view key,
                                     std::string_view context)
            {
                SA_FAIL_IF(!attributes.has(key), StatusCode::InvalidDescription,
                           "{} needs the key '{}', which is absent", context, key);

                return read_real_or(attributes, key, context, 0.0);
            }

            Result<double> read_layer_norm_epsilon(const Config::Value& attributes,
                                                   std::string_view context)
            {
                SA_TRY_ASSIGN(const double epsilon,
                              read_real_or(attributes, K_LAYER_NORM_EPSILON_KEY, context,
                                           NN::K_LAYER_NORM_EPSILON));
                SA_FAIL_IF(!(epsilon > 0.0), StatusCode::InvalidDescription,
                           "{} '{}' must be positive, but it is {}", context,
                           K_LAYER_NORM_EPSILON_KEY, epsilon);

                return epsilon;
            }

            Result<NN::NormOptions> read_norm_options(const Config::Value& attributes,
                                                      std::string_view context)
            {
                NN::NormOptions options{};
                SA_TRY_ASSIGN(const double epsilon,
                              read_real_or(attributes, K_NORM_EPSILON_KEY, context,
                                           options.epsilon));
                SA_TRY_ASSIGN(const double momentum,
                              read_real_or(attributes, K_NORM_MOMENTUM_KEY, context,
                                           options.momentum));

                SA_FAIL_IF(!(epsilon > 0.0), StatusCode::InvalidDescription,
                           "{} '{}' must be positive, but it is {}", context, K_NORM_EPSILON_KEY,
                           epsilon);
                SA_FAIL_IF(!(momentum >= 0.0 && momentum <= 1.0), StatusCode::InvalidDescription,
                           "{} '{}' must be in [0, 1], but it is {}", context,
                           K_NORM_MOMENTUM_KEY, momentum);

                options.epsilon = epsilon;
                options.momentum = momentum;
                return options;
            }

        } // namespace Detail
    } // namespace NN
} // namespace SushiAI
