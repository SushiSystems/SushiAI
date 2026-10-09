/**************************************************************************/
/* import_weights.hpp                                                     */
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
 * @file import_weights.hpp
 * @brief Declares the strict checkpoint loader that fills a ModelState by renamed records.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/nn/model_state.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Rewrites a record name that starts with from_prefix into to_prefix. */
        struct RenameRule
        {
            /** @brief The prefix a record name must start with for the rule to apply. */
            std::string_view from_prefix;

            /** @brief The text that replaces from_prefix in a matching name. */
            std::string_view to_prefix;
        };

        /**
         * @brief Returns name under the first rule whose from_prefix it starts with, else name.
         * @param name The record name to rename.
         * @param rules The rules, tried in the order given; the first match wins.
         * @return The renamed name, or a copy of name when no rule matches.
         */
        [[nodiscard]] std::string renamed(std::string_view name,
                                          SushiRuntime::span<const RenameRule> rules);

        /**
         * @brief Loads an SACP file into state, renaming each record first; strict both ways.
         * @param path The checkpoint file to read.
         * @param state The state whose every entry the file must supply exactly once.
         * @param rules The rules applied to each record name before the lookup.
         * @return FileNotFound for an unreadable file; Incompatible for a record that matches no
         *         entry, repeats, or differs in dtype or shape, and for an entry left unsupplied.
         */
        [[nodiscard]] Status import_weights(const std::filesystem::path& path,
                                            NN::ModelState& state,
                                            SushiRuntime::span<const RenameRule> rules);

    } // namespace IO
} // namespace SushiAI
