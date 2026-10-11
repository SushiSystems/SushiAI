/**************************************************************************/
/* host_table.hpp                                                         */
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
 * @file host_table.hpp
 * @brief Declares the leaf a step fills once from numbers the host computed, and its sink.
 * @author Mustafa Garip
 */

#pragma once

#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Describes a leaf whose numbers the host computed when the graph was traced. */
        struct HostTable
        {
            /** @brief Names the leaf in the traced graph. */
            Graph::ValueID leaf = Graph::K_NO_VALUE;

            /** @brief Holds one number per element of the leaf, the last axis fastest. */
            std::vector<double> values;
        };

        /** @brief Receives the host tables a trace declares. */
        class HostTableSink
        {
            public:
                /** @brief Releases the sink through its base. */
                virtual ~HostTableSink() = default;

                /** @brief Takes one table, in the order the trace declares them. */
                virtual void add(HostTable table) = 0;
        };

        /** @brief Collects the host tables of one trace, in the order they were declared. */
        class HostTableList final : public HostTableSink
        {
            public:
                /** @brief Appends @p table. */
                void add(HostTable table) override
                {
                    tables_.push_back(std::move(table));
                }

                /**
                 * @brief Returns every table, in the order it was added.
                 * @return A view that the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const HostTable> tables() const noexcept
                {
                    return SushiRuntime::span<const HostTable>(tables_.data(), tables_.size());
                }

            private:
                /** @brief Holds the tables in the order they were added. */
                std::vector<HostTable> tables_;
        };

    } // namespace NN
} // namespace SushiAI
