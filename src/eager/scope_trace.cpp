/**************************************************************************/
/* scope_trace.cpp                                                        */
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
 * @file scope_trace.cpp
 * @brief Implements ScopeTrace, which splits a trace into segments as scopes open and close.
 * @author Mustafa Garip
 * @see include/SushiAI/eager/README.md
 */

#include <SushiAI/eager/scope_trace.hpp>

#include <utility>

#include <SushiBLAS/core/format.hpp>

#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Eager
    {
        namespace
        {
            /** @brief The character that joins scope names into a path. */
            constexpr char K_PATH_SEPARATOR = '/';

            /** @brief The character that separates a name from its occurrence in a region name. */
            constexpr char K_OCCURRENCE_SEPARATOR = '#';

            /**
             * @brief Throws when @p sealed is true.
             * @param call The member function being called, for the message.
             */
            void require_unsealed(bool sealed, std::string_view call)
            {
                SA_THROW_IF(sealed,
                            "ScopeTrace::{} was called on a sealed trace. seal() ends the trace "
                            "and hands its segments over once; start a new ScopeTrace for the "
                            "next trace",
                            std::string(call));
            }
        } // namespace

        ScopeTrace::ScopeTrace(const Graph::Graph& graph) noexcept : graph_(&graph)
        {

        }

        void ScopeTrace::open(std::string_view name)
        {
            require_unsealed(sealed_, "open()");
            SA_THROW_IF(name.empty(), "ScopeTrace::open() was given an empty scope name");
            SA_THROW_IF(name.find(K_PATH_SEPARATOR) != std::string_view::npos ||
                            name.find(K_OCCURRENCE_SEPARATOR) != std::string_view::npos,
                        "ScopeTrace::open() was given the scope name \"{}\"; a scope name may not "
                        "contain '{}' or '{}'",
                        std::string(name), K_PATH_SEPARATOR, K_OCCURRENCE_SEPARATOR);

            end_segment();
            open_.emplace_back(name);
        }

        void ScopeTrace::close()
        {
            require_unsealed(sealed_, "close()");
            SA_THROW_IF(open_.empty(), "ScopeTrace::close() was called with no scope open");

            end_segment();
            open_.pop_back();
        }

        std::vector<Segment> ScopeTrace::seal()
        {
            require_unsealed(sealed_, "seal()");
            SA_THROW_IF(!open_.empty(),
                        "ScopeTrace::seal() was called with {} scope(s) still open; the innermost "
                        "is \"{}\"",
                        open_.size(), current_path());

            end_segment();

            sealed_ = true;
            occurrences_.clear();
            return std::move(segments_);
        }

        std::size_t ScopeTrace::depth() const noexcept
        {
            return open_.size();
        }

        void ScopeTrace::end_segment()
        {
            const Graph::NodeId end = static_cast<Graph::NodeId>(graph_->node_count());
            if (end == segment_begin_)
                return;

            Segment segment;
            segment.name = current_path();
            segment.occurrence = occurrences_[segment.name]++;
            segment.begin = segment_begin_;
            segment.end = end;

            segments_.push_back(std::move(segment));
            segment_begin_ = end;
        }

        std::string ScopeTrace::current_path() const
        {
            std::string path;
            for (const std::string& name : open_)
            {
                if (!path.empty())
                    path += K_PATH_SEPARATOR;

                path += name;
            }

            return path;
        }

    } // namespace Eager
} // namespace SushiAI
