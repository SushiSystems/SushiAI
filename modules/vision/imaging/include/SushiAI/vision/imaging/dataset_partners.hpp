/**************************************************************************/
/* dataset_partners.hpp                                                   */
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
 * @file dataset_partners.hpp
 * @brief Declares the partner source that draws from every image of the dataset.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/partner_source.hpp>
#include <SushiAI/vision/imaging/random.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /**
     * @brief Chooses partners from the whole dataset, each index equally likely.
     *
     * This is the reference's rule when its images are cached in memory
     * (data/augment.py:501-502). It keeps no state, so a sample's partners depend on the
     * sample's generator alone.
     */
    class DatasetPartners final : public PartnerSource
    {
        public:
            /**
             * @brief Builds a source over a dataset.
             * @param dataset_size The image count.
             * @return The source, or InvalidDescription unless the count is in [1, 2^32 - 1].
             */
            [[nodiscard]] static Result<std::unique_ptr<DatasetPartners>> make(
                std::size_t dataset_size) noexcept;

            /**
             * @brief Returns three random.index(dataset size) draws, in draw order.
             * @pre @p index is below the dataset size.
             */
            [[nodiscard]] MosaicPartners choose(std::size_t index, Random& random) override;

        private:
            /** @brief Stores a dataset size that make() has checked. */
            explicit DatasetPartners(std::uint32_t dataset_size) noexcept;

            /// @brief Counts the images of the dataset.
            std::uint32_t dataset_size_ = 0;
    };
} // namespace SushiAI::Vision::Imaging
