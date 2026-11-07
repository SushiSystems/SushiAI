/**************************************************************************/
/* in_memory_region_partitioner.hpp                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file in_memory_region_partitioner.hpp
 * @brief Declares InMemoryRegionPartitioner, a RegionPartitioner that returns a scripted cut.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_partitioner.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @class InMemoryRegionPartitioner
         * @brief Returns the partition it was given for every graph and journals what each call
         *        was handed; touches no device, file, thread or clock.
         *
         * It does not read the graph, so the partition it returns keeps the promise of the
         * interface only when the caller scripted one that fits the graph it passes.
         */
        class InMemoryRegionPartitioner final : public RegionPartitioner
        {
            public:
                /** @brief Holds what one call of assign() was handed. */
                struct Call
                {
                    /** @brief How many segments the call received. */
                    std::size_t segment_count = 0;

                    /** @brief The forward node count of the provenance the call received. */
                    std::size_t forward_node_count = 0;

                    /** @brief The node count of the graph the call received. */
                    std::size_t node_count = 0;

                    /** @brief The segments the call received, in order. */
                    std::vector<Segment> segments;
                };

                /**
                 * @brief Builds a partitioner that returns @p partition.
                 * @param partition The cut to return; empty by default.
                 */
                explicit InMemoryRegionPartitioner(Partition partition = {})
                    : partition_(std::move(partition))
                {

                }

                /**
                 * @brief Builds a partitioner that returns what @p inner returns for each call.
                 * @param inner Cuts each graph; must outlive the partitioner.
                 */
                explicit InMemoryRegionPartitioner(const RegionPartitioner& inner)
                    : inner_(&inner)
                {
                }

                /**
                 * @brief Makes every later assign() throw std::invalid_argument with @p message,
                 *        as a partitioner does on a graph it cannot cut.
                 */
                void script_refusal(std::string message)
                {
                    refusal_ = std::move(message);
                    refuses_ = true;
                }

                /**
                 * @brief Journals the call, then throws when a refusal is scripted, returns the
                 *        cut of the inner partitioner when one was given and a copy of the
                 *        scripted partition otherwise.
                 */
                [[nodiscard]] Partition assign(
                    const Graph::Graph& graph,
                    SushiRuntime::span<const Segment> segments,
                    const Autograd::Provenance& provenance) const override
                {
                    calls_.push_back(Call{segments.size(), provenance.forward_node_count,
                                          graph.node_count(),
                                          std::vector<Segment>(segments.begin(), segments.end())});
                    if (refuses_)
                        throw std::invalid_argument(refusal_);

                    return inner_ != nullptr ? inner_->assign(graph, segments, provenance)
                                             : partition_;
                }

                /** @brief Returns what each call of assign() was handed, first call first. */
                [[nodiscard]] const std::vector<Call>& calls() const noexcept
                {
                    return calls_;
                }

            private:
                /// @brief Holds the partition every call returns when no inner one cuts.
                Partition partition_;

                /// @brief Cuts each graph in place of partition_, or null.
                const RegionPartitioner* inner_ = nullptr;

                /// @brief Holds the message of the scripted refusal.
                std::string refusal_;

                /// @brief Whether a refusal is scripted.
                bool refuses_ = false;

                /// @brief Holds the journal of assign().
                mutable std::vector<Call> calls_;
        };

    } // namespace Eager
} // namespace SushiAI
