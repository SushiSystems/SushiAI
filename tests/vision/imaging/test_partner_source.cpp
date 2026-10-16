/**************************************************************************/
/* test_partner_source.cpp                                                */
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
 * @file test_partner_source.cpp
 * @brief Tests the partner count and a fake partner source used through the interface.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <gtest/gtest.h>
#include <SushiAI/vision/imaging/partner_source.hpp>
#include <SushiAI/vision/imaging/random.hpp>

using namespace SushiAI::Vision::Imaging;

namespace
{
    /** @brief Holds the seed of every generator below. */
    constexpr std::uint64_t K_SEED = 20261016;

    /** @brief Holds the images of the pool the fake draws from. */
    constexpr std::uint32_t K_POOL = 10;

    /** @brief Draws each partner as one index of the pool and flags its own destruction. */
    class DrawingPartners final : public PartnerSource
    {
        public:
            /** @brief Starts a source that sets @p destroyed when it is destroyed. */
            explicit DrawingPartners(bool& destroyed) noexcept : destroyed_(destroyed)
            {

            }

            /** @brief Sets the flag the constructor was given. */
            ~DrawingPartners() override
            {
                destroyed_ = true;
            }

            DrawingPartners(const DrawingPartners&) = delete;
            DrawingPartners& operator=(const DrawingPartners&) = delete;
            DrawingPartners(DrawingPartners&&) = delete;
            DrawingPartners& operator=(DrawingPartners&&) = delete;

            /** @brief Returns three draws of an index below K_POOL, in draw order. */
            [[nodiscard]] MosaicPartners choose(std::size_t index, Random& random) override
            {
                static_cast<void>(index);
                MosaicPartners partners{};
                for (std::size_t& partner : partners)
                    partner = random.index(K_POOL);
                return partners;
            }

        private:
            /// @brief Holds the flag the destructor sets.
            bool& destroyed_;
    };
} // namespace

/** @brief Tests that a mosaic adds three partners to the sample's own image. */
TEST(Small_ImagingPartnerSource, AMosaicHasThreePartners)
{
    EXPECT_EQ(K_MOSAIC_PARTNERS, 3u);
    EXPECT_EQ(std::tuple_size<MosaicPartners>::value, K_MOSAIC_PARTNERS);
}

/** @brief Tests that a source asked through the interface returns its draws in draw order. */
TEST(Small_ImagingPartnerSource, ASourceAskedThroughTheInterfaceReturnsItsDraws)
{
    bool destroyed = false;
    const std::unique_ptr<PartnerSource> source = std::make_unique<DrawingPartners>(destroyed);
    Random random(K_SEED);
    Random expected(K_SEED);

    const MosaicPartners partners = source->choose(4, random);

    for (std::size_t i = 0; i < K_MOSAIC_PARTNERS; ++i)
        EXPECT_EQ(partners[i], expected.index(K_POOL)) << "partner " << i;
    EXPECT_TRUE(random == expected);
}

/** @brief Tests that destroying a source through the interface runs its own destructor. */
TEST(Small_ImagingPartnerSource, DestroyingThroughTheInterfaceRunsTheSourcesDestructor)
{
    bool destroyed = false;
    std::unique_ptr<PartnerSource> source = std::make_unique<DrawingPartners>(destroyed);

    source.reset();

    EXPECT_TRUE(destroyed);
}
