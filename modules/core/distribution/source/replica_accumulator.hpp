/**************************************************************************/
/* replica_accumulator.hpp                                                */
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
 * @file replica_accumulator.hpp
 * @brief Declares ReplicaAccumulator, the mean of a replica's results over its micro-batches.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiBLAS/cluster/bound_pack.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /**
         * @brief Sums the result tensors of each micro-batch on the host, in micro-batch
         *        order, and writes their mean in the layout of the result pack.
         *
         * One micro-batch is copied byte for byte. Half sums in float and rounds once.
         */
        class ReplicaAccumulator final
        {
            public:
                /** @brief Holds the sum of the tensors @p layout lays out, empty. */
                explicit ReplicaAccumulator(SushiBLAS::PackLayout layout);

                /** @brief Forgets every micro-batch added. */
                void reset() noexcept;

                /**
                 * @brief Adds one micro-batch's results.
                 * @pre @p results matches the layout entry by entry, each contiguous, and no
                 *      replay that writes them is running.
                 */
                void add(SushiRuntime::span<const SushiBLAS::Tensor> results);

                /**
                 * @brief Writes the mean of the micro-batches added into @p target.
                 * @param target Bound to the layout; zero micro-batches write zeros.
                 * @return Success, or Unsupported for an integer tensor over more than one
                 *         micro-batch, whose mean is not an integer.
                 */
                [[nodiscard]] Status write_mean(const SushiBLAS::BoundPack& target) const;

                /** @brief Returns how many micro-batches were added since reset(). */
                [[nodiscard]] std::size_t count() const noexcept
                {
                    return count_;
                }

            private:
                /** @brief Copies @p source into the sum of entry @p index. */
                void copy_entry(std::size_t index, const void* source);

                /** @brief Adds @p source to the sum of entry @p index. */
                void add_entry(std::size_t index, const void* source);

                /// @brief Where each result lies in the sum and in the result pack.
                SushiBLAS::PackLayout layout_;

                /// @brief The sum in the pack's layout; a Half entry holds its first micro-batch.
                std::vector<std::byte> sum_;

                /// @brief The sum of the Half segment, element for element, in float.
                std::vector<float> half_sum_;

                /// @brief The micro-batches added since reset().
                std::size_t count_ = 0;
        };

    } // namespace Distribution
} // namespace SushiAI
