/**************************************************************************/
/* partner_source.hpp                                                     */
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
 * @file partner_source.hpp
 * @brief Declares the seam that chooses the three images a mosaic adds to a sample.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>

#include <SushiAI/imaging/random.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Imaging
{
    /** @brief Holds the number of images a mosaic adds to the sample's own. */
    inline constexpr std::size_t K_MOSAIC_PARTNERS = 3;

    /** @brief Names the dataset indices of the partners of one mosaic, in draw order. */
    using MosaicPartners = std::array<std::size_t, K_MOSAIC_PARTNERS>;

    /**
     * @brief Chooses the partners of a mosaic.
     *
     * A training pipeline owns one source and asks it once per sample, so a source may keep
     * state between samples. The reference has two rules (data/augment.py:499-502), and each
     * is one implementation: RecentPartners and DatasetPartners.
     */
    class PartnerSource
    {
        public:
            /** @brief Destroys the source through a base pointer. */
            virtual ~PartnerSource() = default;

            /**
             * @brief Returns the three partners of one sample from exactly three draws.
             *
             * A partner may be the sample itself, and one image may be chosen twice.
             * @param index The sample's own dataset index, already loaded by the caller.
             */
            [[nodiscard]] virtual MosaicPartners choose(std::size_t index, Random& random) = 0;
    };
} // namespace SushiAI::Imaging
