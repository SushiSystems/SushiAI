/**************************************************************************/
/* recent_partners.cpp                                                    */
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
 * @file recent_partners.cpp
 * @brief Implements the partner source that draws from the images a pipeline loaded last.
 * @author Mustafa Garip
 */

#include <SushiAI/imaging/recent_partners.hpp>

#include <algorithm>
#include <limits>

#include <SushiAI/core/common.hpp>

namespace SushiAI::Imaging
{
    namespace
    {
        /** @brief Returns the buffer length after checking the two arguments it comes from. */
        [[nodiscard]] std::size_t checked_capacity(std::size_t dataset_size,
                                                   std::int32_t batch_size)
        {
            SA_THROW_IF(dataset_size == 0 ||
                            dataset_size > std::numeric_limits<std::uint32_t>::max(),
                        "RecentPartners: a dataset size of {} is outside [1, {}]", dataset_size,
                        std::numeric_limits<std::uint32_t>::max());
            SA_THROW_IF(batch_size <= 0, "RecentPartners: a batch size of {} is not above zero",
                        batch_size);

            return recent_capacity(dataset_size, batch_size);
        }
    } // namespace

    std::size_t recent_capacity(std::size_t dataset_size, std::int32_t batch_size) noexcept
    {
        const std::size_t spanned = static_cast<std::size_t>(batch_size) * K_RECENT_BATCHES;
        return std::min({dataset_size, spanned, K_RECENT_LIMIT});
    }

    RecentPartners::RecentPartners(std::size_t dataset_size, std::int32_t batch_size)
        : capacity_(checked_capacity(dataset_size, batch_size)), is_recent_(dataset_size, false)
    {

    }

    MosaicPartners RecentPartners::choose(std::size_t index, Random& random)
    {
        SA_THROW_IF(index >= is_recent_.size(),
                    "RecentPartners::choose: sample index {} is outside a dataset of {} images",
                    index, is_recent_.size());

        if (!is_recent_[index])
        {
            recent_.push_back(index);
            is_recent_[index] = true;

            if (recent_.size() > 1 && recent_.size() >= capacity_)
            {
                is_recent_[recent_.front()] = false;
                recent_.pop_front();
            }
        }

        // recent_ holds at least the index just recorded, and at most K_RECENT_LIMIT entries.
        const std::uint32_t count = static_cast<std::uint32_t>(recent_.size());

        MosaicPartners partners{};
        for (std::size_t& partner : partners)
            partner = recent_[random.index(count)];

        return partners;
    }
} // namespace SushiAI::Imaging
