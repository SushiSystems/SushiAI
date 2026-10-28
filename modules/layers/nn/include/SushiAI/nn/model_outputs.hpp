/**************************************************************************/
/* model_outputs.hpp                                                      */
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
 * @file model_outputs.hpp
 * @brief Declares ModelOutputs, the named values a model hands a head or an objective.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @brief Holds the values a model produced, by name, in the order they were published.
         *
         * A name is unique within one set. The set stores graph ids and owns no graph.
         */
        class ModelOutputs final
        {
            public:
                /** @brief Pairs one output's name with its value in the graph. */
                using Entry = std::pair<std::string, Graph::ValueID>;

                /**
                 * @brief Appends @p id under @p name.
                 * @param name Must be non-empty and not yet in the set.
                 * @param id A value of the traced graph; Graph::NO_VALUE is fatal.
                 * @return InvalidDescription for an empty or a duplicate name; the set is
                 *         unchanged on failure.
                 */
                [[nodiscard]] Status add(std::string name, Graph::ValueID id) noexcept;

                /**
                 * @brief Returns the value published under @p name.
                 * @return InvalidDescription listing the names that exist when @p name is absent.
                 */
                [[nodiscard]] Result<Graph::ValueID> at(std::string_view name) const noexcept;

                /**
                 * @brief Returns the value of a set that holds exactly one.
                 * @return InvalidDescription stating the count and the names otherwise.
                 */
                [[nodiscard]] Result<Graph::ValueID> only() const noexcept;

                /** @brief Returns every output in the order it was added. */
                [[nodiscard]] const std::vector<Entry>& all() const noexcept;

            private:
                /** @brief Holds the outputs in the order they were added; names are unique. */
                std::vector<Entry> entries_;
        };

    } // namespace NN
} // namespace SushiAI
