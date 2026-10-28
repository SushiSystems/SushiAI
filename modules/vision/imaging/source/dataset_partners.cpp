/**************************************************************************/
/* dataset_partners.cpp                                                   */
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
 * @file dataset_partners.cpp
 * @brief Implements the partner source that draws from every image of the dataset.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/dataset_partners.hpp>

#include <limits>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Returns a dataset size as the 32-bit count Random::index takes. */
        [[nodiscard]] Result<std::uint32_t> checked_dataset_size(std::size_t dataset_size)
        {
            SA_FAIL_IF(dataset_size == 0 ||
                           dataset_size > std::numeric_limits<std::uint32_t>::max(),
                       StatusCode::InvalidDescription,
                       "DatasetPartners: a dataset size of {} is outside [1, {}]", dataset_size,
                       std::numeric_limits<std::uint32_t>::max());

            return static_cast<std::uint32_t>(dataset_size);
        }
    } // namespace

    DatasetPartners::DatasetPartners(std::uint32_t dataset_size) noexcept
        : dataset_size_(dataset_size)
    {

    }

    Result<std::unique_ptr<DatasetPartners>> DatasetPartners::make(std::size_t dataset_size) noexcept
    {
        SA_TRY_ASSIGN(const std::uint32_t checked, checked_dataset_size(dataset_size));

        std::unique_ptr<DatasetPartners> partners(new DatasetPartners(checked));
        return Result<std::unique_ptr<DatasetPartners>>(std::move(partners));
    }

    MosaicPartners DatasetPartners::choose(std::size_t index, Random& random)
    {
        SA_FATAL_IF(index >= dataset_size_,
                    "DatasetPartners::choose: sample index {} is outside a dataset of {} images",
                    index, dataset_size_);

        MosaicPartners partners{};
        for (std::size_t& partner : partners)
            partner = random.index(dataset_size_);

        return partners;
    }
} // namespace SushiAI::Vision::Imaging
