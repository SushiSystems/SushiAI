/**************************************************************************/
/* import_weights.cpp                                                     */
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
 * @file import_weights.cpp
 * @brief Defines the strict checkpoint loader and the prefix rules it applies to record names.
 * @author Mustafa Garip
 */

#include <SushiAI/io/import_weights.hpp>

#include <fstream>
#include <ios>
#include <string>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/nn/model_state.hpp>
#include "model_records.hpp"

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /**
             * @brief Returns whether @p name begins with @p prefix.
             * @param name The text to test.
             * @param prefix The text it must begin with; the empty prefix matches every name.
             * @return True when name starts with prefix.
             */
            bool starts_with(std::string_view name, std::string_view prefix) noexcept
            {
                return name.size() >= prefix.size() && name.substr(0, prefix.size()) == prefix;
            }
        } // namespace

        std::string renamed(std::string_view name, SushiRuntime::span<const RenameRule> rules)
        {
            for (const RenameRule& rule : rules)
            {
                if (!starts_with(name, rule.from_prefix))
                    continue;

                std::string out;
                out.reserve(rule.to_prefix.size() + name.size() - rule.from_prefix.size());
                out.append(rule.to_prefix);
                out.append(name.substr(rule.from_prefix.size()));
                return out;
            }
            return std::string(name);
        }

        Status import_weights(const std::filesystem::path& path, NN::ModelState& state,
                              SushiRuntime::span<const RenameRule> rules)
        {
            std::ifstream in(path, std::ios::binary);
            SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                       path.string());

            SA_TRY_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));

            // A run-state section follows the model records and is left unread.
            return Detail::read_model_records(reader, path, state, rules);
        }

    } // namespace IO
} // namespace SushiAI
