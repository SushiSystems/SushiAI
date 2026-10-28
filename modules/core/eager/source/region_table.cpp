/**************************************************************************/
/* region_table.cpp                                                       */
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
 * @file region_table.cpp
 * @brief Implements RegionTable, which compares the regions of a step with the last step's.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/region_table.hpp>

#include <algorithm>
#include <utility>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace Eager
    {
        namespace
        {
            /**
             * @brief Returns the storage the caller placed each buffer in, by buffer.
             * @pre No two values of one buffer are placed in different storage.
             */
            [[nodiscard]] std::vector<StorageID> place_presets(
                const BoundaryMap& boundaries, SushiRuntime::span<const StorageID> preset)
            {
                std::vector<StorageID> storage(preset.size(), K_NO_STORAGE);
                for (std::size_t value = 0; value < preset.size(); ++value)
                {
                    if (preset[value] == K_NO_STORAGE)
                        continue;

                    StorageID& placed = storage[boundaries.buffer_of[value]];
                    SA_FATAL_IF(placed != K_NO_STORAGE && placed != preset[value],
                                "Value {} is placed in storage {}, but it shares its buffer "
                                "with value {}, which is placed in storage {}",
                                value, preset[value], boundaries.buffer_of[value], placed);
                    placed = preset[value];
                }

                return storage;
            }

            /** @brief Names a run of entries of a ReaderList. */
            struct ReaderRange
            {
                std::size_t begin = 0; /**< The first entry. */
                std::size_t count = 0; /**< How many entries follow it. */
            };

            /** @brief Lists, for each buffer the ledger places, the regions that read it. */
            class ReaderList final
            {
                public:
                    /**
                     * @brief Collects the readers of every buffer that has no storage yet.
                     * @param ids The id of each region, in the order of @p boundaries.
                     * @param storage The storage of each preset buffer, by buffer.
                     */
                    ReaderList(const std::vector<RegionID>& ids, const BoundaryMap& boundaries,
                               const std::vector<StorageID>& storage)
                    {
                        std::vector<Found> found;
                        for (std::size_t region = 0; region < boundaries.regions.size(); ++region)
                        {
                            const std::vector<Graph::ValueID>& imports =
                                boundaries.regions[region].imports;
                            for (std::size_t ordinal = 0; ordinal < imports.size(); ++ordinal)
                            {
                                const Graph::ValueID buffer =
                                    boundaries.buffer_of[imports[ordinal]];
                                if (storage[buffer] != K_NO_STORAGE)
                                    continue;

                                found.push_back(
                                    Found{buffer, static_cast<std::uint32_t>(region),
                                          ReaderSlot{ids[region],
                                                     static_cast<std::uint32_t>(ordinal)}});
                            }
                        }

                        // A stable sort keeps the readers of one buffer in region order.
                        std::stable_sort(found.begin(), found.end(),
                                         [](const Found& a, const Found& b)
                                         {
                                             return a.buffer < b.buffer;
                                         });

                        buffers_.reserve(found.size());
                        regions_.reserve(found.size());
                        slots_.reserve(found.size());
                        for (const Found& entry : found)
                        {
                            buffers_.push_back(entry.buffer);
                            regions_.push_back(entry.region);
                            slots_.push_back(entry.slot);
                        }
                    }

                    /** @brief Returns the entries that read @p buffer, in region order. */
                    [[nodiscard]] ReaderRange range_of(Graph::ValueID buffer) const
                    {
                        const auto range =
                            std::equal_range(buffers_.begin(), buffers_.end(), buffer);
                        return ReaderRange{
                            static_cast<std::size_t>(range.first - buffers_.begin()),
                            static_cast<std::size_t>(range.second - range.first)};
                    }

                    /** @brief Returns the readers of @p range as the ledger takes them. */
                    [[nodiscard]] SushiRuntime::span<const ReaderSlot> slots(
                        const ReaderRange& range) const noexcept
                    {
                        return SushiRuntime::span<const ReaderSlot>(slots_.data() + range.begin,
                                                                    range.count);
                    }

                    /** @brief Returns the reader of entry @p at. */
                    [[nodiscard]] const ReaderSlot& slot(std::size_t at) const noexcept
                    {
                        return slots_[at];
                    }

                    /** @brief Returns the position in Partition::regions of the reader of @p at. */
                    [[nodiscard]] std::uint32_t region(std::size_t at) const noexcept
                    {
                        return regions_[at];
                    }

                private:
                    /** @brief Holds one reader before the entries are sorted by buffer. */
                    struct Found
                    {
                        Graph::ValueID buffer = Graph::K_NO_VALUE; /**< The buffer read. */
                        std::uint32_t region = 0;                  /**< The reader's position. */
                        ReaderSlot slot;                           /**< The reader's own name. */
                    };

                    /// @brief The buffer of each entry, ascending.
                    std::vector<Graph::ValueID> buffers_;

                    /// @brief The position in Partition::regions of each entry's reader.
                    std::vector<std::uint32_t> regions_;

                    /// @brief The reader of each entry.
                    std::vector<ReaderSlot> slots_;
            };

            /**
             * @brief Computes the signature of a region with its storage left out, on demand.
             * @see modules/core/eager/README.md
             */
            class StructureProbe final
            {
                public:
                    /** @brief Binds the probe to one step; the three arguments outlive it. */
                    StructureProbe(const Graph::Graph& graph, const Partition& partition,
                                   const BoundaryMap& boundaries)
                        : graph_(&graph), partition_(&partition), boundaries_(&boundaries),
                          ordinal_(graph.value_count(), 0),
                          known_(partition.regions.size(), false),
                          structures_(partition.regions.size())
                    {

                    }

                    /** @brief Returns the structure of the region at @p region. */
                    [[nodiscard]] const RegionSignature& of(std::size_t region)
                    {
                        if (known_[region])
                            return structures_[region];

                        // An import is named by its place among the imports, not by storage.
                        const std::vector<Graph::ValueID>& imports =
                            boundaries_->regions[region].imports;
                        for (std::size_t at = 0; at < imports.size(); ++at)
                            ordinal_[imports[at]] = at + 1;

                        structures_[region] = signature_of(
                            *graph_, *partition_, partition_->regions[region].key,
                            SushiRuntime::span<const std::uint64_t>(ordinal_.data(),
                                                                    ordinal_.size()));

                        for (Graph::ValueID value : imports)
                            ordinal_[value] = 0;

                        known_[region] = true;
                        return structures_[region];
                    }

                private:
                    /// @brief The step's graph.
                    const Graph::Graph* graph_;

                    /// @brief The step's partition.
                    const Partition* partition_;

                    /// @brief The step's boundaries.
                    const BoundaryMap* boundaries_;

                    /// @brief One past the import ordinal of each value, zero between calls.
                    std::vector<std::uint64_t> ordinal_;

                    /// @brief Whether each region's structure has been computed.
                    std::vector<bool> known_;

                    /// @brief The structure of each region that was asked for.
                    std::vector<RegionSignature> structures_;
            };

        } // namespace

        RegionTable::RegionTable(StorageIDSource& ids) noexcept : slots_(ids)
        {

        }

        StepPlan RegionTable::plan(const Graph::Graph& graph, const Partition& partition,
                                   const BoundaryMap& boundaries,
                                   SushiRuntime::span<const StorageID> preset) noexcept
        {
            const std::size_t region_count = partition.regions.size();
            SA_FATAL_IF(boundaries.regions.size() != region_count
                            || boundaries.buffer_of.size() != graph.value_count(),
                        "The boundary map describes {} regions and {} values, but the partition "
                        "has {} regions and the graph {} values",
                        boundaries.regions.size(), boundaries.buffer_of.size(), region_count,
                        graph.value_count());
            SA_FATAL_IF(preset.size() != graph.value_count(),
                        "The caller placed {} values, but the graph holds {}", preset.size(),
                        graph.value_count());

            ++step_;

            StepPlan plan;
            plan.storage_of_value.assign(graph.value_count(), K_NO_STORAGE);
            plan.regions.reserve(region_count);

            std::vector<StorageID> storage = place_presets(boundaries, preset);

            std::vector<RegionID> ids(region_count);
            for (std::size_t region = 0; region < region_count; ++region)
                ids[region] = names_.intern(partition.regions[region]);
            if (entries_.size() < names_.size())
                entries_.resize(names_.size());

            const ReaderList readers(ids, boundaries, storage);
            StructureProbe structures(graph, partition, boundaries);

            // A reader is trusted when its own structure is the one the last step recorded.
            std::vector<signed char> trust(region_count, -1);
            const auto is_trusted = [&](std::uint32_t reader)
            {
                if (trust[reader] < 0)
                {
                    const Entry& known = entries_[ids[reader]];
                    trust[reader] = known.live && known.structure == structures.of(reader) ? 1 : 0;
                }

                return trust[reader] == 1;
            };

            slots_.begin_step();
            std::vector<ReaderSlot> trusted;

            // Regions are in ascending key order, so a producer is placed before its readers.
            for (std::size_t region = 0; region < region_count; ++region)
            {
                const RegionBoundary& boundary = boundaries.regions[region];
                const RegionKey key = partition.regions[region].key;
                const RegionID id = ids[region];

                for (Graph::ValueID value : boundary.imports)
                {
                    const StorageID placed = storage[boundaries.buffer_of[value]];
                    SA_FATAL_IF(placed == K_NO_STORAGE,
                                "Region '{}' reads value {} ('{}'), which has no storage; a "
                                "leaf is fed through the step or bound as state",
                                names_.label_of(id), value, graph.value(value).debug_name);
                    plan.storage_of_value[value] = placed;
                }

                RegionSignature signature = signature_of(
                    graph, partition, key,
                    SushiRuntime::span<const StorageID>(plan.storage_of_value.data(),
                                                        plan.storage_of_value.size()));

                Entry& entry = entries_[id];
                const bool same_structure = entry.live && entry.signature == signature
                                            && entry.export_positions == boundary.export_positions;
                const std::size_t export_count = boundary.exports.size();

                if (same_structure)
                {
                    for (std::size_t at = 0; at < export_count; ++at)
                    {
                        const Graph::ValueID buffer = boundaries.buffer_of[boundary.exports[at]];
                        if (storage[buffer] == K_NO_STORAGE)
                            storage[buffer] = slots_.keep(
                                id, boundary.export_positions[at],
                                readers.slots(readers.range_of(buffer)));
                    }
                }

                // Slots a trusted reader names are claimed before any is taken by position.
                for (std::size_t at = 0; at < export_count; ++at)
                {
                    const Graph::ValueID buffer = boundaries.buffer_of[boundary.exports[at]];
                    if (storage[buffer] != K_NO_STORAGE)
                        continue;

                    const ReaderRange range = readers.range_of(buffer);
                    trusted.clear();
                    for (std::size_t i = range.begin; i < range.begin + range.count; ++i)
                    {
                        if (is_trusted(readers.region(i)))
                            trusted.push_back(readers.slot(i));
                    }
                    if (trusted.empty())
                        continue;

                    const Graph::Value& described = graph.value(boundary.exports[at]);
                    storage[buffer] = slots_.match_readers(
                        id, boundary.export_positions[at], described.shape, described.dtype,
                        SushiRuntime::span<const ReaderSlot>(trusted.data(), trusted.size()),
                        readers.slots(range));
                }

                std::vector<StorageID> export_storage(export_count, K_NO_STORAGE);
                for (std::size_t at = 0; at < export_count; ++at)
                {
                    const Graph::ValueID value = boundary.exports[at];
                    const Graph::ValueID buffer = boundaries.buffer_of[value];
                    if (storage[buffer] == K_NO_STORAGE)
                    {
                        const Graph::Value& described = graph.value(value);
                        storage[buffer] = slots_.place(
                            id, boundary.export_positions[at], described.shape, described.dtype,
                            readers.slots(readers.range_of(buffer)));
                    }

                    plan.storage_of_value[value] = storage[buffer];
                    export_storage[at] = storage[buffer];
                }

                const bool unchanged = same_structure && entry.export_storage == export_storage;

                RegionOrder order;
                order.id = id;
                order.key = key;
                order.previous_key = entry.live ? entry.key : K_NO_REGION;
                if (!unchanged)
                    order.action = RegionAction::Build;
                else
                    order.action = entry.key != key ? RegionAction::Record : RegionAction::Keep;
                plan.regions.push_back(order);

                // An equal signature implies an equal structure, so it is computed on a change.
                if (!same_structure)
                    entry.structure = structures.of(region);

                entry.live = true;
                entry.step = step_;
                entry.key = key;
                entry.signature = std::move(signature);
                entry.export_positions = boundary.export_positions;
                entry.export_storage = std::move(export_storage);
            }

            for (RegionID id : live_)
            {
                Entry& entry = entries_[id];
                if (entry.step == step_)
                    continue;

                plan.dropped.push_back(entry.key);
                entry = Entry{};
            }

            live_ = std::move(ids);
            plan.released = slots_.end_step();
            plan.created.assign(slots_.created().begin(), slots_.created().end());
            return plan;
        }

        std::vector<StorageID> RegionTable::forget() noexcept
        {
            entries_.assign(names_.size(), Entry{});
            live_.clear();
            return slots_.forget();
        }

    } // namespace Eager
} // namespace SushiAI
