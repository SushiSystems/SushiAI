/**************************************************************************/
/* c3k2.cpp                                                               */
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
 * @file c3k2.cpp
 * @brief Implements the factories that build a C3k2 node over each inner block.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/c3k2.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "c3k2" row accepts. */
            constexpr std::string_view K_C3K2_KEYS[] = {
                "filters", "hidden", "repeats", "inner", "shortcut",
                Detail::K_NORM_EPSILON_KEY, Detail::K_NORM_MOMENTUM_KEY};

            /** @brief Names the inner block make_c3k2_of_bottleneck() builds. */
            constexpr std::string_view K_INNER_BOTTLENECK = "bottleneck";

            /** @brief Names the inner block make_c3k2_of_c3k() builds. */
            constexpr std::string_view K_INNER_C3K = "c3k";

            /** @brief Names the inner block make_c3k2_of_bottleneck_psa() builds. */
            constexpr std::string_view K_INNER_BOTTLENECK_PSA = "bottleneck_psa";

            /** @brief Names the inner block of a row that states none. */
            constexpr std::string_view K_DEFAULT_INNER = K_INNER_BOTTLENECK;

            /** @brief Holds the smallest hidden width, which every inner block halves. */
            constexpr int64_t K_SMALLEST_HIDDEN = 2;

            /** @brief Holds the bottleneck count of a C3k used as an inner block. */
            constexpr int64_t K_INNER_C3K_REPEATS = 2;

            /** @brief Holds what a c3k2 row states, read and checked. */
            struct C3k2Row
            {
                Shape input{};          /**< The per-sample [H, W, C] shape the block reads. */
                int64_t filters = 0;    /**< The output width. */
                int64_t hidden = 0;     /**< The width of each half of the stem. */
                int64_t repeats = 0;    /**< The number of inner blocks. */
                bool shortcut = true;   /**< Whether the inner blocks add their input. */
                NN::NormOptions norm{}; /**< The constants of every batch normalisation. */
            };

            /**
             * @brief Reads the name under "inner", or @p fallback when the key is absent.
             * @return InvalidDescription when the key holds anything but a string.
             */
            [[nodiscard]] Result<std::string_view> read_inner_or(const Config::Value& attributes,
                                                                 std::string_view context,
                                                                 std::string_view fallback)
            {
                if (!attributes.has("inner"))
                    return fallback;

                SA_TRY_ASSIGN(const Config::Value& stated, attributes.at("inner", context));
                SA_FAIL_IF(stated.kind() != Config::Kind::String, StatusCode::InvalidDescription,
                           "{} 'inner' must be a string, but it is {}", context,
                           Config::kind_name(stated.kind()));

                SA_TRY_ASSIGN(const std::string& name, stated.as_string("inner"));
                return std::string_view(name);
            }

            /**
             * @brief Reads the keys every inner kind shares, or reports the one that is wrong.
             * @param kind The inner block the caller builds; a stated "inner" must name it.
             */
            [[nodiscard]] Result<C3k2Row> read_row(NN::ShapeSpan inputs,
                                                   const Config::Value& attributes,
                                                   std::string_view context,
                                                   std::string_view kind)
            {
                SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_C3K2_KEYS),
                                                   context));
                SA_TRY_ASSIGN(const std::string_view inner,
                              read_inner_or(attributes, context, kind));
                SA_FAIL_IF(inner != kind, StatusCode::InvalidDescription,
                           "{} 'inner' is '{}', but this factory builds '{}'", context, inner,
                           kind);

                C3k2Row row;
                SA_TRY_ASSIGN(row.input, Detail::read_feature_map(inputs, context));
                SA_TRY_ASSIGN(row.filters,
                              Detail::read_positive_int(attributes, "filters", context));
                SA_TRY_ASSIGN(row.hidden,
                              Detail::read_positive_int(attributes, "hidden", context));
                SA_TRY_ASSIGN(row.repeats,
                              Detail::read_positive_int_or(attributes, "repeats", context,
                                                           Detail::K_DEFAULT_REPEATS));
                SA_TRY_ASSIGN(row.shortcut,
                              Detail::read_bool_or(attributes, "shortcut", context,
                                                   Detail::K_DEFAULT_SHORTCUT));
                SA_TRY_ASSIGN(row.norm, Detail::read_norm_options(attributes, context));

                SA_FAIL_IF(row.hidden < K_SMALLEST_HIDDEN, StatusCode::InvalidDescription,
                           "{} 'hidden' must be at least {}, because an inner block halves it; "
                           "it is {}",
                           context, K_SMALLEST_HIDDEN, row.hidden);

                return row;
            }

            /** @brief Builds the Bottleneck one inner slot of @p row holds. */
            [[nodiscard]] Bottleneck bottleneck_of(const C3k2Row& row)
            {
                return Bottleneck(row.hidden, row.hidden, row.hidden / 2, row.shortcut, row.norm);
            }

            /** @brief Builds the C3k one inner slot of @p row holds. */
            [[nodiscard]] C3k c3k_of(const C3k2Row& row)
            {
                return C3k(row.hidden, row.hidden, K_INNER_C3K_REPEATS, row.shortcut, row.norm);
            }

            /** @brief Builds the Bottleneck and the PSABlock one inner slot of @p row holds. */
            [[nodiscard]] NN::Sequential<Bottleneck, PSABlock> bottleneck_psa_of(
                const C3k2Row& row)
            {
                return NN::Sequential<Bottleneck, PSABlock>(
                    bottleneck_of(row),
                    PSABlock(row.hidden, Detail::default_head_count(row.hidden), row.norm));
            }

            /**
             * @brief Builds a C3k2 node from a row that has been read and checked.
             * @param make_inner Called once per inner slot of the row.
             */
            template<typename Inner>
            [[nodiscard]] Result<std::unique_ptr<NN::Node>> build_c3k2_of(
                const C3k2Row& row, Inner (*make_inner)(const C3k2Row&))
            {
                NN::ModuleList<Inner> inner;
                for (int64_t index = 0; index < row.repeats; ++index)
                    inner.push_back(make_inner(row));

                return Detail::wrap_block(
                    C3k2<Inner>(row.input[2], row.filters, row.hidden, std::move(inner), row.norm),
                    row.input);
            }

            /**
             * @brief Builds a C3k2 node whose inner blocks @p make_inner produces.
             * @param kind The name K_C3K2_INNER gives this inner block.
             * @param make_inner Called once per inner slot of the row.
             */
            template<typename Inner>
            [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_c3k2_of(
                NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context,
                std::string_view kind, Inner (*make_inner)(const C3k2Row&))
            {
                SA_TRY_ASSIGN(const C3k2Row row, read_row(inputs, attributes, context, kind));

                return build_c3k2_of(row, make_inner);
            }

            /** @brief Joins the names of K_C3K2_INNER into "a, b" for a diagnostic. */
            [[nodiscard]] std::string list_inner_names()
            {
                std::array<std::string_view, K_C3K2_INNER.size()> names{};
                for (std::size_t index = 0; index < names.size(); ++index)
                    names[index] = K_C3K2_INNER[index].name;

                return Detail::list_names(Detail::KeySpan(names.data(), names.size()));
            }

            /** @brief Returns the row of K_C3K2_INNER named @p name, or null when none is. */
            [[nodiscard]] const C3k2InnerEntry* find_inner(std::string_view name) noexcept
            {
                for (const C3k2InnerEntry& entry : K_C3K2_INNER)
                {
                    if (entry.name == name)
                        return &entry;
                }

                return nullptr;
            }

        } // namespace

        Result<std::unique_ptr<NN::Node>> make_c3k2_of_bottleneck(NN::ShapeSpan inputs,
                                                                  const Config::Value& attributes,
                                                                  std::string_view context)
        {
            return make_c3k2_of<Bottleneck>(inputs, attributes, context, K_INNER_BOTTLENECK,
                                            &bottleneck_of);
        }

        Result<std::unique_ptr<NN::Node>> make_c3k2_of_c3k(NN::ShapeSpan inputs,
                                                           const Config::Value& attributes,
                                                           std::string_view context)
        {
            return make_c3k2_of<C3k>(inputs, attributes, context, K_INNER_C3K, &c3k_of);
        }

        Result<std::unique_ptr<NN::Node>> make_c3k2_of_bottleneck_psa(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context)
        {
            SA_TRY_ASSIGN(const C3k2Row row,
                          read_row(inputs, attributes, context, K_INNER_BOTTLENECK_PSA));

            // PSABlock throws on a width or head count its attention refuses; refuse them here.
            const int64_t heads = Detail::default_head_count(row.hidden);
            SA_FAIL_IF(row.hidden % heads != 0 ||
                           row.hidden / heads < Detail::K_SMALLEST_HEAD_WIDTH,
                       StatusCode::InvalidDescription,
                       "{} 'hidden' must split into {} parts of at least {} channels; it is {}",
                       context, heads, Detail::K_SMALLEST_HEAD_WIDTH, row.hidden);

            return build_c3k2_of(row, &bottleneck_psa_of);
        }

        Result<std::unique_ptr<NN::Node>> make_c3k2_node(NN::ShapeSpan inputs,
                                                         const Config::Value& attributes,
                                                         std::string_view context)
        {
            SA_TRY_ASSIGN(const std::string_view inner,
                          read_inner_or(attributes, context, K_DEFAULT_INNER));

            const C3k2InnerEntry* found = find_inner(inner);
            SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                       "{} 'inner' is '{}', which names no inner block; the inner blocks are: {}",
                       context, inner, list_inner_names());

            return found->make(inputs, attributes, context);
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
