/**************************************************************************/
/* region_signature.hpp                                                   */
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
 * @file region_signature.hpp
 * @brief Declares RegionSignature, the structure of one region as a comparable value.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/ops/ops.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @brief Holds what two steps must agree on for a region to be reused.
         * @see modules/core/eager/README.md
         */
        struct RegionSignature
        {
            std::uint64_t hash = 0; /**< Hash of the region's structure. */
            std::vector<OpID> ops;  /**< The region's operations, in node order. */

            /** @brief Reports whether both the hash and the operations agree. */
            bool operator==(const RegionSignature& other) const noexcept
            {
                return hash == other.hash && ops == other.ops;
            }

            /** @brief Reports whether the hash or the operations differ. */
            bool operator!=(const RegionSignature& other) const noexcept
            {
                return !(*this == other);
            }
        };

        /**
         * @brief Computes the signature of one region of @p partition.
         * @param region A key listed in @p partition.
         * @param external_identity One stable identity per value id; read for every value the
         *        region reads and does not produce.
         * @see modules/core/eager/README.md
         */
        [[nodiscard]] RegionSignature signature_of(
            const Graph::Graph& graph,
            const Partition& partition,
            RegionKey region,
            SushiRuntime::span<const std::uint64_t> external_identity) noexcept;

    } // namespace Eager
} // namespace SushiAI
