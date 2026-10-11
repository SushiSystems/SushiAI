/**************************************************************************/
/* model_outputs.cpp                                                      */
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
 * @file model_outputs.cpp
 * @brief Implements ModelOutputs: publishing a named value and finding it again.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/model_outputs.hpp>

#include <algorithm>
#include <cstddef>

#include <SushiAI/core/check.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Returns the entry of @p entries named @p name, or the end iterator. */
            std::vector<ModelOutputs::Entry>::const_iterator find_entry(
                const std::vector<ModelOutputs::Entry>& entries, std::string_view name) noexcept
            {
                return std::find_if(entries.begin(), entries.end(),
                                    [name](const ModelOutputs::Entry& entry)
                {
                    return entry.first == name;
                });
            }

            /** @brief Joins the names of @p entries with ", ", or returns "none" for no entry. */
            std::string list_names(const std::vector<ModelOutputs::Entry>& entries)
            {
                if (entries.empty())
                    return "none";

                std::string names;
                for (std::size_t index = 0; index < entries.size(); ++index)
                {
                    if (index != 0)
                        names += ", ";

                    names += entries[index].first;
                }

                return names;
            }

        } // namespace

        Status ModelOutputs::add(std::string name, Graph::ValueID id)
        {
            SA_FATAL_IF(id == Graph::K_NO_VALUE,
                        "a model output named '{}' was added with no value", name);
            SA_FAIL_IF(name.empty(), StatusCode::InvalidDescription,
                       "a model output needs a name; the outputs so far: {}",
                       list_names(entries_));
            SA_FAIL_IF(find_entry(entries_, name) != entries_.end(),
                       StatusCode::InvalidDescription,
                       "the model already has an output named '{}'; the outputs so far: {}",
                       name, list_names(entries_));

            entries_.emplace_back(std::move(name), id);
            return Status();
        }

        Result<Graph::ValueID> ModelOutputs::at(std::string_view name) const noexcept
        {
            const auto found = find_entry(entries_, name);
            SA_FAIL_IF(found == entries_.end(), StatusCode::InvalidDescription,
                       "the model has no output named '{}'; its outputs: {}", name,
                       list_names(entries_));

            return found->second;
        }

        Result<Graph::ValueID> ModelOutputs::only() const noexcept
        {
            SA_FAIL_IF(entries_.size() != 1, StatusCode::InvalidDescription,
                       "exactly one model output was expected; the model has {}: {}",
                       entries_.size(), list_names(entries_));

            return entries_.front().second;
        }

        const std::vector<ModelOutputs::Entry>& ModelOutputs::all() const noexcept
        {
            return entries_;
        }

    } // namespace NN
} // namespace SushiAI
