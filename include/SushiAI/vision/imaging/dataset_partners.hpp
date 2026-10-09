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

#include <cstddef>
#include <cstdint>

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
             * @brief Constructs a source over a dataset.
             * @param dataset_size The image count, in [1, 2^32 - 1]; otherwise throws Error.
             */
            explicit DatasetPartners(std::size_t dataset_size);

            /** @brief Returns three random.index(dataset size) draws, in draw order. */
            [[nodiscard]] MosaicPartners choose(std::size_t index, Random& random) override;

        private:
            /// @brief Counts the images of the dataset.
            std::uint32_t dataset_size_ = 0;
    };
} // namespace SushiAI::Vision::Imaging
