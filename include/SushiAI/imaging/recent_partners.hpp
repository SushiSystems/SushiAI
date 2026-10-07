/**************************************************************************/
/* recent_partners.hpp                                                    */
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
 * @file recent_partners.hpp
 * @brief Declares the partner source that draws from the images a pipeline loaded last.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include <SushiAI/imaging/partner_source.hpp>
#include <SushiAI/imaging/random.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Imaging
{
    /** @brief Holds the number of batches whose images the buffer spans (data/base.py:168). */
    inline constexpr std::size_t K_RECENT_BATCHES = 8;

    /** @brief Holds the largest length the buffer is given (data/base.py:168). */
    inline constexpr std::size_t K_RECENT_LIMIT = 1000;

    /**
     * @brief Returns the reference's buffer length for a dataset and a batch size.
     * @return min(dataset_size, batch_size * K_RECENT_BATCHES, K_RECENT_LIMIT).
     * @pre batch_size is above zero.
     */
    [[nodiscard]] std::size_t recent_capacity(std::size_t dataset_size,
                                              std::int32_t batch_size) noexcept;

    /**
     * @brief Chooses partners from a buffer of the indices its pipeline loaded most recently.
     *
     * This is the reference's default rule (data/augment.py:499-500, data/base.py:306-312).
     * A sample's partners depend on the samples its pipeline made before it, so two pipelines
     * that share a dataset give one sample different partners.
     */
    class RecentPartners final : public PartnerSource
    {
        public:
            /**
             * @brief Constructs a source with an empty buffer.
             * @param dataset_size The image count, in [1, 2^32 - 1]; otherwise throws Error.
             * @param batch_size The training batch size, above zero; otherwise throws Error.
             */
            RecentPartners(std::size_t dataset_size, std::int32_t batch_size);

            /**
             * @brief Records the sample as loaded, then draws three entries of the buffer.
             *
             * An index already in the buffer is not added again. After an addition, the
             * oldest entry leaves when the buffer holds more than one entry and at least
             * capacity() of them. Each partner is recent()[random.index(recent().size())].
             */
            [[nodiscard]] MosaicPartners choose(std::size_t index, Random& random) override;

            /** @brief Returns the buffer, oldest index first. */
            [[nodiscard]] const std::deque<std::size_t>& recent() const noexcept { return recent_; }

            /** @brief Returns recent_capacity of the constructor's arguments. */
            [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

        private:
            /// @brief Holds the length at which an addition evicts the oldest entry.
            std::size_t capacity_ = 0;

            /// @brief Holds the buffered indices, oldest first, each at most once.
            std::deque<std::size_t> recent_;

            /// @brief Tells, per dataset index, whether recent_ holds it.
            std::vector<bool> is_recent_;
    };
} // namespace SushiAI::Imaging
