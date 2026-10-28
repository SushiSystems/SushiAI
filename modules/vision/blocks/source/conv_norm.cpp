/**************************************************************************/
/* conv_norm.cpp                                                          */
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
 * @file conv_norm.cpp
 * @brief Implements the factory that builds a Conv node from a model-file row.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/conv_norm.hpp>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            /** @brief Names the key that holds a row's zero padding on each side. */
            constexpr std::string_view K_PADDING_KEY = "padding";

            /** @brief Lists every key a "conv" row accepts. */
            constexpr std::string_view K_CONV_KEYS[] = {
                "filters", "kernel", "stride", K_PADDING_KEY, Detail::K_NORM_EPSILON_KEY,
                Detail::K_NORM_MOMENTUM_KEY};

            /** @brief Holds the kernel and the stride of a row that states neither. */
            constexpr int64_t K_DEFAULT_WINDOW = 1;

            /**
             * @brief Reads the whole number under "padding", or kernel / 2 when it is absent.
             * @return InvalidDescription when the key holds anything but a whole number of zero
             *         or more.
             */
            [[nodiscard]] Result<int64_t> read_padding_or_same(const Config::Value& attributes,
                                                               int64_t kernel,
                                                               std::string_view context)
            {
                if (!attributes.has(K_PADDING_KEY))
                    return kernel / 2;

                SA_TRY_ASSIGN(const Config::Value& value, attributes.at(K_PADDING_KEY, context));
                SA_FAIL_IF(value.kind() != Config::Kind::Int, StatusCode::InvalidDescription,
                           "{} '{}' must be a whole number, but it is {}", context, K_PADDING_KEY,
                           Config::kind_name(value.kind()));

                SA_TRY_ASSIGN(const int64_t padding, value.as_int(K_PADDING_KEY));
                SA_FAIL_IF(padding < 0, StatusCode::InvalidDescription,
                           "{} '{}' must be zero or more, but it is {}", context, K_PADDING_KEY,
                           padding);

                return padding;
            }

        } // namespace

        Result<std::unique_ptr<NN::Node>> make_conv_node(NN::ShapeSpan inputs,
                                                         const Config::Value& attributes,
                                                         std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_CONV_KEYS), context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_feature_map(inputs, context));
            SA_TRY_ASSIGN(const int64_t filters,
                          Detail::read_positive_int(attributes, "filters", context));
            SA_TRY_ASSIGN(const int64_t kernel,
                          Detail::read_positive_int_or(attributes, "kernel", context,
                                                       K_DEFAULT_WINDOW));
            SA_TRY_ASSIGN(const int64_t stride,
                          Detail::read_positive_int_or(attributes, "stride", context,
                                                       K_DEFAULT_WINDOW));
            SA_TRY_ASSIGN(const int64_t padding,
                          read_padding_or_same(attributes, kernel, context));
            SA_TRY_ASSIGN(const NN::NormOptions norm,
                          Detail::read_norm_options(attributes, context));

            return Detail::wrap_block(Conv(input[2], filters, kernel, stride, padding, norm),
                                      input);
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
