/**************************************************************************/
/* model_state.cpp                                                        */
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
 * @file model_state.cpp
 * @brief Implements ModelState, the tensors a model owns across the graphs it is traced into.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/model_state.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/seam.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace Detail
        {
            StateEntry describe_parameter(const std::string& name, const Parameter& parameter)
            {
                StateEntry entry;
                entry.name = name;
                entry.kind = StateKind::Parameter;
                entry.state = parameter.state();
                entry.shape = parameter.shape();
                entry.dtype = parameter.dtype();
                entry.init = parameter.init();
                entry.fan_in = parameter.fan_in();
                entry.fan_out = parameter.fan_out();
                return entry;
            }

            StateEntry describe_buffer(const std::string& name, const Buffer& buffer)
            {
                StateEntry entry;
                entry.name = name;
                entry.kind = StateKind::Buffer;
                entry.state = buffer.state();
                entry.shape = buffer.shape();
                entry.dtype = buffer.dtype();
                entry.init = ParameterInit::constant_value(buffer.initial_value());
                return entry;
            }

        } // namespace Detail

        Result<ModelState> ModelState::from_entries(SushiBLAS::Engine& engine,
                                                    std::vector<StateEntry> entries)
        {
            ModelState state;
            SA_TRY(state.index_entries(entries));

            std::vector<SushiBLAS::Tensor> tensors;
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                tensors.reserve(entries.size());
                for (const StateEntry& entry : entries)
                {
                    SA_TRY_ASSIGN(SushiBLAS::Tensor tensor,
                                  allocate_tensor(engine, entry.shape, entry.dtype));
                    tensors.push_back(std::move(tensor));
                }

                return Status();
            }));

            state.entries_ = std::move(entries);
            state.tensors_ = std::move(tensors);
            return std::move(state);
        }

        ModelState::ModelState(ModelState&& other) noexcept
            : entries_(std::move(other.entries_)),
              tensors_(std::move(other.tensors_)),
              by_name_(std::move(other.by_name_)),
              by_state_(std::move(other.by_state_))
        {

        }

        ModelState& ModelState::operator=(ModelState&& other) noexcept
        {
            if (this != &other)
            {
                entries_ = std::move(other.entries_);
                tensors_ = std::move(other.tensors_);
                by_name_ = std::move(other.by_name_);
                by_state_ = std::move(other.by_state_);
            }

            return *this;
        }

        ModelState::~ModelState() = default;

        Status ModelState::attach(const Graph::Graph& graph, Graph::TensorPool& pool) const
        {
            std::vector<std::size_t> positions;
            SA_TRY(match_leaves(graph, positions));

            const std::vector<std::pair<Graph::StateId, Graph::ValueId>>& leaves =
                graph.state_leaves();
            for (std::size_t i = 0; i < leaves.size(); ++i)
                SA_TRY(pool.bind(leaves[i].second, tensors_[positions[i]]));

            return Status();
        }

        SushiRuntime::span<const StateEntry> ModelState::entries() const noexcept
        {
            return SushiRuntime::span<const StateEntry>(entries_.data(), entries_.size());
        }

        Result<SushiBLAS::Tensor*> ModelState::tensor(std::string_view name)
        {
            const auto found = SushiAI::Detail::guard_value(
                SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return by_name_.find(std::string(name));
            });
            if (!found.has_value())
                return found.status();

            const auto position = found.value();
            SA_FAIL_IF(position == by_name_.end(), StatusCode::InvalidDescription,
                       "No state entry is named '{}'; the state holds {} entries", name,
                       entries_.size());
            return &tensors_[position->second];
        }

        Result<SushiBLAS::Tensor*> ModelState::tensor(Graph::StateId state)
        {
            const auto position = by_state_.find(state);
            SA_FAIL_IF(position == by_state_.end(), StatusCode::InvalidDescription,
                       "No state entry holds identity {}; the state holds {} entries",
                       static_cast<std::uint64_t>(state), entries_.size());
            return &tensors_[position->second];
        }

        Status ModelState::index_entries(const std::vector<StateEntry>& entries)
        {
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                by_name_.reserve(entries.size());
                by_state_.reserve(entries.size());
                for (std::size_t position = 0; position < entries.size(); ++position)
                {
                    const StateEntry& entry = entries[position];
                    SA_FATAL_IF(entry.state == Graph::K_NO_STATE,
                                "State entry '{}' has no identity; declare it before it is "
                                "listed",
                                entry.name);
                    SA_FAIL_IF(!by_name_.emplace(entry.name, position).second,
                               StatusCode::InvalidDescription,
                               "Two state entries are named '{}'", entry.name);
                    SA_FAIL_IF(!by_state_.emplace(entry.state, position).second,
                               StatusCode::InvalidDescription,
                               "State entry '{}' repeats the identity of an earlier entry",
                               entry.name);
                }

                return Status();
            });
        }

        Status ModelState::match_leaves(const Graph::Graph& graph,
                                        std::vector<std::size_t>& positions) const
        {
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                const std::vector<std::pair<Graph::StateId, Graph::ValueId>>& leaves =
                    graph.state_leaves();
                positions.clear();
                positions.reserve(leaves.size());
                for (const std::pair<Graph::StateId, Graph::ValueId>& leaf : leaves)
                {
                    const auto found = by_state_.find(leaf.first);
                    SA_FAIL_IF(found == by_state_.end(), StatusCode::Incompatible,
                               "The graph has a leaf for state {}, which this state does not hold",
                               static_cast<std::uint64_t>(leaf.first));

                    const StateEntry& entry = entries_[found->second];
                    const Graph::Value& value = graph.value(leaf.second);
                    SA_FAIL_IF(value.shape != entry.shape || value.dtype != entry.dtype,
                               StatusCode::Incompatible,
                               "State '{}' is {} {} in this state and {} {} in the graph",
                               entry.name, entry.shape.to_string(), dtype_name(entry.dtype),
                               value.shape.to_string(), dtype_name(value.dtype));
                    positions.push_back(found->second);
                }

                return Status();
            });
        }

    } // namespace NN
} // namespace SushiAI
