/**************************************************************************/
/* test_slot_ledger.cpp                                                   */
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
 * @file test_slot_ledger.cpp
 * @brief Tests which slot SlotLedger gives a shared buffer from one step to the next.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/region_names.hpp>
#include <SushiAI/eager/slot_ledger.hpp>
#include <SushiAI/eager/storage_id.hpp>

using namespace SushiAI;
using Eager::K_NO_STORAGE;
using Eager::ReaderSlot;
using Eager::RegionID;
using Eager::SlotLedger;
using Eager::StorageID;
using Eager::ValuePosition;

namespace
{
    /** @brief Holds the id of the region that writes the buffer in every case. */
    constexpr RegionID K_PRODUCER = 1;

    /** @brief Holds the id of the region that reads the buffer in every case. */
    constexpr RegionID K_READER = 2;

    /** @brief Holds the extents of the buffer unless a case states others. */
    const Shape K_SHAPE{8, 4};

    /** @brief Views @p readers as the span the ledger takes. */
    [[nodiscard]] SushiRuntime::span<const ReaderSlot> span_of(
        const std::vector<ReaderSlot>& readers)
    {
        return SushiRuntime::span<const ReaderSlot>(readers.data(), readers.size());
    }

    /** @brief Claims the slot a reader names, then the one at @p position, or creates one. */
    [[nodiscard]] StorageID match(SlotLedger& ledger, RegionID producer, ValuePosition position,
                                  const Shape& shape, DType dtype,
                                  const std::vector<ReaderSlot>& readers)
    {
        const StorageID read = ledger.match_readers(producer, position, shape, dtype,
                                                    span_of(readers), span_of(readers));
        return read != K_NO_STORAGE ? read
                                    : ledger.place(producer, position, shape, dtype,
                                                   span_of(readers));
    }

    /** @brief Runs one step that creates a slot at @p position, read as @p readers say. */
    [[nodiscard]] StorageID first_step(SlotLedger& ledger, ValuePosition position,
                                       const std::vector<ReaderSlot>& readers)
    {
        ledger.begin_step();
        const StorageID slot =
            match(ledger, K_PRODUCER, position, K_SHAPE, DType::Float32, readers);
        EXPECT_TRUE(ledger.end_step().empty());
        return slot;
    }
} // namespace

TEST(Unit_SlotLedger, TheFirstMatchCreatesASlot)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);

    ledger.begin_step();
    const StorageID slot = match(ledger, K_PRODUCER, ValuePosition{2, 0},
                                 K_SHAPE, DType::Float32, {{K_READER, 0}});

    EXPECT_NE(slot, K_NO_STORAGE);
    ASSERT_EQ(ledger.created().size(), 1u);
    EXPECT_EQ(ledger.created()[0].id, slot);
    EXPECT_EQ(ledger.created()[0].shape, K_SHAPE);
    EXPECT_EQ(ledger.created()[0].dtype, DType::Float32);
}

TEST(Unit_SlotLedger, AnUnchangedProducerKeepsItsSlot)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    EXPECT_EQ(ledger.keep(K_PRODUCER, ValuePosition{2, 0}, span_of({{K_READER, 0}})), slot);
    EXPECT_TRUE(ledger.created().empty());
    EXPECT_TRUE(ledger.end_step().empty());
}

TEST(Unit_SlotLedger, KeepFindsNoSlotAtAnotherPosition)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    (void)first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    EXPECT_EQ(ledger.keep(K_PRODUCER, ValuePosition{3, 0}, span_of({{K_READER, 0}})),
              K_NO_STORAGE);
}

TEST(Unit_SlotLedger, AChangedProducerReusesTheSlotItsReaderRead)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    // The producer grew, so the value moved from node 2 to node 5; the reader did not change.
    ledger.begin_step();
    EXPECT_EQ(match(ledger, K_PRODUCER, ValuePosition{5, 0},
                    K_SHAPE, DType::Float32, {{K_READER, 0}}),
              slot);
    EXPECT_TRUE(ledger.created().empty());
    EXPECT_TRUE(ledger.end_step().empty());

    ledger.begin_step();
    EXPECT_EQ(ledger.keep(K_PRODUCER, ValuePosition{5, 0}, span_of({{K_READER, 0}})), slot);
}

TEST(Unit_SlotLedger, AReaderThatIsNotTrustedNamesNoSlot)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    (void)first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    EXPECT_EQ(ledger.match_readers(K_PRODUCER, ValuePosition{5, 0}, K_SHAPE, DType::Float32,
                                   span_of({}), span_of({{K_READER, 0}})),
              K_NO_STORAGE);
}

TEST(Unit_SlotLedger, PlaceTakesTheSlotWrittenAtTheSamePosition)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    EXPECT_EQ(ledger.place(K_PRODUCER, ValuePosition{2, 0}, K_SHAPE, DType::Float32,
                           span_of({})),
              slot);
    EXPECT_NE(ledger.place(K_PRODUCER, ValuePosition{2, 0}, K_SHAPE, DType::Float32,
                           span_of({})),
              slot);
}

TEST(Unit_SlotLedger, AChangedShapeGetsANewSlotAndReleasesTheOld)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    const StorageID wider = match(ledger, K_PRODUCER, ValuePosition{2, 0},
                                  Shape{8, 16}, DType::Float32, {{K_READER, 0}});
    EXPECT_NE(wider, slot);
    EXPECT_EQ(ledger.end_step(), std::vector<StorageID>{slot});
}

TEST(Unit_SlotLedger, AChangedElementTypeGetsANewSlot)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    EXPECT_NE(match(ledger, K_PRODUCER, ValuePosition{2, 0},
                    K_SHAPE, DType::Float64, {{K_READER, 0}}),
              slot);
}

TEST(Unit_SlotLedger, ASlotIsClaimedOnce)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    const StorageID first = match(ledger, K_PRODUCER, ValuePosition{2, 0},
                                  K_SHAPE, DType::Float32, {{K_READER, 0}});
    const StorageID second = match(ledger, K_PRODUCER, ValuePosition{3, 0},
                                   K_SHAPE, DType::Float32, {{K_READER, 0}});

    EXPECT_EQ(first, slot);
    EXPECT_NE(second, slot);
    EXPECT_EQ(ledger.keep(K_PRODUCER, ValuePosition{2, 0}, span_of({})), K_NO_STORAGE);
}

TEST(Unit_SlotLedger, ASlotOfAnotherProducerIsNotTaken)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    // Another region now feeds the reader's first import; the first producer still exists.
    constexpr RegionID K_OTHER = 7;
    ledger.begin_step();
    EXPECT_NE(match(ledger, K_OTHER, ValuePosition{0, 0}, K_SHAPE, DType::Float32, {{K_READER, 0}}),
              slot);
    EXPECT_EQ(ledger.keep(K_PRODUCER, ValuePosition{2, 0}, span_of({})), slot);
}

TEST(Unit_SlotLedger, AnUnclaimedSlotIsReleasedAtTheEndOfTheStep)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    EXPECT_EQ(ledger.end_step(), std::vector<StorageID>{slot});
    EXPECT_EQ(ledger.size(), 0u);
}

TEST(Unit_SlotLedger, ForgetReturnsEverySlotOnce)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID kept = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    // The second step is left open, as a step that failed after planning leaves it.
    ledger.begin_step();
    (void)ledger.keep(K_PRODUCER, ValuePosition{2, 0}, span_of({{K_READER, 0}}));
    const StorageID created = match(ledger, K_PRODUCER, ValuePosition{4, 0},
                                    K_SHAPE, DType::Float32, {{K_READER, 1}});

    std::vector<StorageID> released = ledger.forget();
    std::sort(released.begin(), released.end());
    EXPECT_EQ(released, (std::vector<StorageID>{kept, created}));
    EXPECT_EQ(ledger.size(), 0u);
}
