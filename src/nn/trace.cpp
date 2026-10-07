/**************************************************************************/
/* trace.cpp                                                              */
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
 * @file trace.cpp
 * @brief Implements Trace and the two buffer-update sinks: the list and the discarding one.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/trace.hpp>

#include <cstddef>
#include <utility>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Drops every update it is given. */
            class DiscardingSink final : public BufferUpdateSink
            {
                public:
                    /** @brief Ignores the update. */
                    void add(const BufferUpdate&) override
                    {

                    }
            };

        } // namespace

        void BufferUpdateList::add(const BufferUpdate& update)
        {
            updates_.push_back(update);
            sources_.push_back(update.source);
        }

        SushiRuntime::span<const BufferUpdate> BufferUpdateList::updates() const noexcept
        {
            return SushiRuntime::span<const BufferUpdate>(updates_.data(), updates_.size());
        }

        SushiRuntime::span<const Graph::ValueId> BufferUpdateList::sources() const noexcept
        {
            return SushiRuntime::span<const Graph::ValueId>(sources_.data(), sources_.size());
        }

        BufferUpdateSink& discard_updates() noexcept
        {
            static DiscardingSink sink;
            return sink;
        }

        Trace::Trace(const Graph::GraphBuilder& builder, BufferUpdateSink* updates,
                     TraceMode mode) noexcept
            : builder_(&builder), updates_(updates), mode_(mode)
        {

        }

        Trace Trace::training(const Graph::GraphBuilder& builder,
                              BufferUpdateSink& updates) noexcept
        {
            return Trace(builder, &updates, TraceMode::Training);
        }

        Trace Trace::evaluation(const Graph::GraphBuilder& builder) noexcept
        {
            return Trace(builder, nullptr, TraceMode::Evaluation);
        }

        const Graph::GraphBuilder& Trace::builder() const noexcept
        {
            return *builder_;
        }

        TraceMode Trace::mode() const noexcept
        {
            return mode_;
        }

        void Trace::fail(Status failure) const noexcept
        {
            SA_FATAL_IF(failure.is_ok(), "Trace::fail was given a success; it records a failure");

            const Status refused = builder_->refuse(std::move(failure));
            if (failure_.is_ok())
                failure_ = refused;
        }

        const Status& Trace::status() const noexcept
        {
            return failure_;
        }

        Graph::ValueId Trace::value_or_fail(const Result<Graph::ValueId>& traced,
                                            Graph::ValueId fallback) const noexcept
        {
            if (traced.has_value())
                return traced.value();

            fail(traced.status());
            return fallback;
        }

        Status Trace::update(Buffer& target, Graph::ValueId source, double momentum,
                             double source_scale) const
        {
            const std::size_t value_count = builder_->graph().value_count();
            SA_FATAL_IF(!target.declared(),
                        "a buffer update names a buffer that define() has not given a shape");
            SA_FATAL_IF(source == Graph::NO_VALUE || source >= value_count,
                        "a buffer update reads value {} of a graph that holds {} values",
                        source, value_count);
            SA_FAIL_IF(mode_ == TraceMode::Evaluation, StatusCode::InvalidDescription,
                       "a buffer update was requested in an evaluation trace; a layer requests "
                       "updates in its training form only");

            updates_->add(BufferUpdate{target.state(), source, momentum, source_scale});
            return Status();
        }

    } // namespace NN
} // namespace SushiAI
