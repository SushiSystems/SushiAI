/**************************************************************************/
/* scope_trace.hpp                                                        */
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
 * @file scope_trace.hpp
 * @brief Declares ScopeTrace, which splits a trace into segments as scopes open and close.
 * @author Mustafa Garip
 * @see include/SushiAI/eager/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <SushiAI/eager/segment.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @brief Splits the nodes traced into one graph into segments, by the scope open at each.
         * @see include/SushiAI/eager/README.md
         */
        class ScopeTrace final
        {
            public:
                /**
                 * @brief Starts a trace over @p graph with no scope open.
                 * @param graph The graph being traced; must outlive this object.
                 */
                explicit ScopeTrace(const Graph::Graph& graph) noexcept;

                /**
                 * @brief Opens a scope nested in the current one.
                 * @param name Scope name; not empty and without '/' or '#'.
                 */
                void open(std::string_view name);

                /**
                 * @brief Closes the innermost open scope.
                 * @pre At least one scope is open.
                 */
                void close();

                /**
                 * @brief Ends the trace and returns its segments in node order.
                 * @pre Every opened scope has been closed.
                 */
                [[nodiscard]] std::vector<Segment> seal();

                /** @brief Returns how many scopes are open. */
                std::size_t depth() const noexcept;

            private:
                /** @brief Ends the running segment at the graph's current node count. */
                void end_segment();

                /** @brief Returns the open scopes joined by '/'. */
                std::string current_path() const;

                /// @brief The graph whose node count marks segment boundaries.
                const Graph::Graph* graph_;

                /// @brief Names of the open scopes, outermost first.
                std::vector<std::string> open_;

                /// @brief Segments ended so far, in node order.
                std::vector<Segment> segments_;

                /// @brief First node of the running segment.
                Graph::NodeId segment_begin_ = 0;

                /// @brief Segments emitted so far, per scope path.
                std::unordered_map<std::string, std::uint32_t> occurrences_;

                /// @brief Whether seal() has returned.
                bool sealed_ = false;
        };

    } // namespace Eager
} // namespace SushiAI
