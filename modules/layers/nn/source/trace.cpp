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
 * @brief Implements Trace and the lists and the discarding sink its forms hand over.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/trace.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Holds the seed of a trace that has no random site. */
            constexpr std::uint64_t K_NO_SEED = 0;

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

        SushiRuntime::span<const Graph::ValueID> BufferUpdateList::sources() const noexcept
        {
            return SushiRuntime::span<const Graph::ValueID>(sources_.data(), sources_.size());
        }

        Graph::ValueID FedLeafList::find(std::string_view name) const noexcept
        {
            for (const FedLeaf& leaf : leaves_)
            {
                if (leaf.name == name)
                    return leaf.value;
            }

            return Graph::K_NO_VALUE;
        }

        void FedLeafList::add(const FedLeaf& leaf)
        {
            leaves_.push_back(leaf);
        }

        SushiRuntime::span<const FedLeaf> FedLeafList::leaves() const noexcept
        {
            return SushiRuntime::span<const FedLeaf>(leaves_.data(), leaves_.size());
        }

        void CarriedValueList::add(const CarriedValue& carried)
        {
            values_.push_back(carried);
            sources_.push_back(carried.next);
        }

        SushiRuntime::span<const CarriedValue> CarriedValueList::values() const noexcept
        {
            return SushiRuntime::span<const CarriedValue>(values_.data(), values_.size());
        }

        SushiRuntime::span<const Graph::ValueID> CarriedValueList::sources() const noexcept
        {
            return SushiRuntime::span<const Graph::ValueID>(sources_.data(), sources_.size());
        }

        BufferUpdateSink& discard_updates() noexcept
        {
            static DiscardingSink sink;
            return sink;
        }

        Trace::Trace(const Graph::GraphBuilder& builder, BufferUpdateSink* updates,
                     FedLeafSink* fed, CarriedValueSink* carried, TraceMode mode,
                     std::uint64_t seed) noexcept
            : builder_(&builder), updates_(updates), fed_(fed), carried_(carried), mode_(mode),
              seed_(seed)
        {

        }

        Trace Trace::training(const Graph::GraphBuilder& builder, BufferUpdateSink& updates,
                              std::uint64_t seed) noexcept
        {
            return Trace(builder, &updates, nullptr, nullptr, TraceMode::Training, seed);
        }

        Trace Trace::evaluation(const Graph::GraphBuilder& builder) noexcept
        {
            return Trace(builder, nullptr, nullptr, nullptr, TraceMode::Evaluation,
                         K_NO_SEED);
        }

        Trace Trace::carrying(const Graph::GraphBuilder& builder, FedLeafSink& fed,
                              CarriedValueSink& carried) noexcept
        {
            return Trace(builder, nullptr, &fed, &carried, TraceMode::Evaluation, K_NO_SEED);
        }

        const Graph::GraphBuilder& Trace::builder() const noexcept
        {
            return *builder_;
        }

        TraceMode Trace::mode() const noexcept
        {
            return mode_;
        }

        std::uint64_t Trace::seed() const noexcept
        {
            return seed_;
        }

        std::uint64_t Trace::claim_random_site() const noexcept
        {
            return random_sites_++;
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

        Graph::ValueID Trace::value_or_fail(const Result<Graph::ValueID>& traced,
                                            Graph::ValueID fallback) const noexcept
        {
            if (traced.has_value())
                return traced.value();

            fail(traced.status());
            return fallback;
        }

        Status Trace::update(Buffer& target, Graph::ValueID source, double momentum,
                             double source_scale) const
        {
            const std::size_t value_count = builder_->graph().value_count();
            SA_FATAL_IF(!target.declared(),
                        "a buffer update names a buffer that define() has not given a shape");
            SA_FATAL_IF(source == Graph::K_NO_VALUE || source >= value_count,
                        "a buffer update reads value {} of a graph that holds {} values",
                        source, value_count);
            SA_FAIL_IF(mode_ == TraceMode::Evaluation, StatusCode::InvalidDescription,
                       "a buffer update was requested in an evaluation trace; a layer requests "
                       "updates in its training form only");

            updates_->add(BufferUpdate{target.state(), source, momentum, source_scale});
            return Status();
        }

        bool Trace::is_carrying() const noexcept
        {
            return carried_ != nullptr;
        }

        Result<Graph::ValueID> Trace::feed(std::string_view name, Shape shape, DType dtype) const
        {
            SA_FAIL_IF(fed_ == nullptr, StatusCode::InvalidDescription,
                       "the leaf '{}' was declared in a trace that feeds the model input alone; "
                       "a layer declares a fed leaf in a carrying trace only",
                       name);

            const Graph::ValueID held = fed_->find(name);
            if (held == Graph::K_NO_VALUE)
            {
                return Detail::bridge_throwing([this, name, &shape, dtype]()
                {
                    const Graph::ValueID declared = builder_->input(shape, name, dtype);
                    fed_->add(FedLeaf{name, declared});
                    return declared;
                });
            }

            const Graph::Value& value = builder_->graph().value(held);
            SA_FAIL_IF(value.shape != shape || value.dtype != dtype,
                       StatusCode::InvalidDescription,
                       "the fed leaf '{}' was declared as {} {} and is now declared as {} {}",
                       name, dtype_name(value.dtype), value.shape.to_string(), dtype_name(dtype),
                       shape.to_string());

            return held;
        }

        Result<Graph::ValueID> Trace::carried_leaf(Shape shape, std::string_view debug_name,
                                                   DType dtype) const
        {
            SA_FAIL_IF(carried_ == nullptr, StatusCode::InvalidDescription,
                       "the carried value '{}' was declared in a trace whose step keeps "
                       "nothing between runs; a layer carries a value in a carrying trace only",
                       debug_name);

            return Detail::bridge_throwing([this, &shape, debug_name, dtype]()
            {
                return builder_->input(shape, debug_name, dtype);
            });
        }

        Status Trace::carry(Graph::ValueID leaf, Graph::ValueID next, double initial_value) const
        {
            const Graph::Graph& graph = builder_->graph();
            const std::size_t value_count = graph.value_count();
            SA_FATAL_IF(leaf >= value_count || next >= value_count,
                        "a carried value names leaf {} and next value {} of a graph that holds "
                        "{} values",
                        leaf, next, value_count);
            SA_FATAL_IF(graph.value(leaf).lifetime != Graph::Lifetime::Input,
                        "a carried value names value {} as its leaf, which carried_leaf() did "
                        "not declare",
                        leaf);
            SA_FAIL_IF(carried_ == nullptr, StatusCode::InvalidDescription,
                       "a value was carried in a trace whose step keeps nothing between runs; "
                       "a layer carries a value in a carrying trace only");

            const Graph::Value& held = graph.value(leaf);
            const Graph::Value& taken = graph.value(next);
            SA_FAIL_IF(held.shape != taken.shape || held.dtype != taken.dtype,
                       StatusCode::InvalidDescription,
                       "a carried leaf is {} {}, but the value it takes after a run is {} {}",
                       dtype_name(held.dtype), held.shape.to_string(), dtype_name(taken.dtype),
                       taken.shape.to_string());

            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                carried_->add(CarriedValue{leaf, next, initial_value});
                return Status();
            });
        }

        void Trace::collect_tables(HostTableSink& tables) noexcept
        {
            tables_ = &tables;
        }

        Result<Graph::ValueID> Trace::host_table(Shape shape, std::string_view debug_name,
                                                 DType dtype, std::vector<double> values) const
        {
            SA_FAIL_IF(tables_ == nullptr, StatusCode::InvalidDescription,
                       "the host table '{}' was declared in a trace no step fills tables for; "
                       "a graph builder of Train names the sink with collect_tables()",
                       debug_name);
            SA_FAIL_IF(dtype != DType::Float32 && dtype != DType::Float64,
                       StatusCode::InvalidDescription,
                       "the host table '{}' is declared as {}; a step fills a float32 or a "
                       "float64 table",
                       debug_name, dtype_name(dtype));
            SA_FAIL_IF(static_cast<std::int64_t>(values.size()) != shape.num_elements(),
                       StatusCode::InvalidDescription,
                       "the host table '{}' is {}, which holds {} elements, but it was given "
                       "{} values",
                       debug_name, shape.to_string(), shape.num_elements(), values.size());

            return Detail::bridge_throwing([this, &shape, debug_name, dtype, &values]()
            {
                const Graph::ValueID leaf = builder_->buffer(shape, debug_name, dtype);
                tables_->add(HostTable{leaf, std::move(values)});
                return leaf;
            });
        }

    } // namespace NN
} // namespace SushiAI
