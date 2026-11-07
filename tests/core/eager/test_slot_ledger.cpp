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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/


/**
 * @file test_slot_ledger.cpp
 * @brief Tests which slot SlotLedger gives a shared buffer from one step to the next.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
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

    /** @brief Names how a row claims the slot written at a position. */
    enum class Claimer : std::uint8_t
    {
        Place, /**< place() takes the slot written at the position. */
        Match, /**< match_readers() or place() takes the slot a reader names. */
    };

    /** @brief Holds one way to claim a slot and the position the second claim asks for. */
    struct ClaimRow
    {
        Claimer claimer;         /**< How both claims go through the ledger. */
        ValuePosition second_at; /**< The position the second claim asks for. */
    };

    /** @brief Claims the slot written at @p position the way @p claimer says. */
    [[nodiscard]] StorageID claim(SlotLedger& ledger, Claimer claimer, ValuePosition position)
    {
        if (claimer == Claimer::Place)
            return ledger.place(K_PRODUCER, position, K_SHAPE, DType::Float32, span_of({}));

        return match(ledger, K_PRODUCER, position, K_SHAPE, DType::Float32, {{K_READER, 0}});
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

TEST(Small_SlotLedger, TheFirstMatchCreatesASlot)
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

TEST(Small_SlotLedger, AnUnchangedProducerKeepsItsSlot)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    EXPECT_EQ(ledger.keep(K_PRODUCER, ValuePosition{2, 0}, span_of({{K_READER, 0}})), slot);
    EXPECT_TRUE(ledger.created().empty());
    EXPECT_TRUE(ledger.end_step().empty());
}

TEST(Small_SlotLedger, KeepFindsNoSlotAtAnotherPosition)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    (void)first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    EXPECT_EQ(ledger.keep(K_PRODUCER, ValuePosition{3, 0}, span_of({{K_READER, 0}})),
              K_NO_STORAGE);
}

TEST(Small_SlotLedger, AChangedProducerReusesTheSlotItsReaderRead)
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

TEST(Small_SlotLedger, AReaderThatIsNotTrustedNamesNoSlot)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    (void)first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    EXPECT_EQ(ledger.match_readers(K_PRODUCER, ValuePosition{5, 0}, K_SHAPE, DType::Float32,
                                   span_of({}), span_of({{K_READER, 0}})),
              K_NO_STORAGE);
}

class Small_SlotLedgerClaim : public ::testing::TestWithParam<ClaimRow>
{
};

/** @brief Tests that a slot is claimed once: the first claim takes it, the second does not. */
TEST_P(Small_SlotLedgerClaim, ASlotIsClaimedOnce)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});
    const ClaimRow row = GetParam();

    ledger.begin_step();
    const StorageID first = claim(ledger, row.claimer, ValuePosition{2, 0});
    const StorageID second = claim(ledger, row.claimer, row.second_at);

    EXPECT_EQ(first, slot);
    EXPECT_NE(second, slot);
    EXPECT_EQ(ledger.keep(K_PRODUCER, ValuePosition{2, 0}, span_of({})), K_NO_STORAGE);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_SlotLedgerClaim,
                         ::testing::Values(ClaimRow{Claimer::Place, ValuePosition{2, 0}},
                                           ClaimRow{Claimer::Match, ValuePosition{3, 0}}));

/** @brief Holds the buffer description a changed slot is asked for. */
struct ChangeRow
{
    Shape shape;  /**< The extents asked for. */
    DType dtype;  /**< The element type asked for. */
};

class Small_SlotLedgerChange : public ::testing::TestWithParam<ChangeRow>
{
};

/** @brief Tests that a changed shape or element type gets a new slot and releases the old. */
TEST_P(Small_SlotLedgerChange, AChangedBufferGetsANewSlotAndReleasesTheOld)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    const StorageID changed = match(ledger, K_PRODUCER, ValuePosition{2, 0},
                                    GetParam().shape, GetParam().dtype, {{K_READER, 0}});

    EXPECT_NE(changed, slot);
    EXPECT_EQ(ledger.end_step(), std::vector<StorageID>{slot});
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_SlotLedgerChange,
                         ::testing::Values(ChangeRow{Shape{8, 16}, DType::Float32},
                                           ChangeRow{Shape{8, 4}, DType::Float64}));

TEST(Small_SlotLedger, ASlotOfAnotherProducerIsNotTaken)
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

TEST(Small_SlotLedger, AnUnclaimedSlotIsReleasedAtTheEndOfTheStep)
{
    Eager::StorageIDSource ids;
    SlotLedger ledger(ids);
    const StorageID slot = first_step(ledger, ValuePosition{2, 0}, {{K_READER, 0}});

    ledger.begin_step();
    EXPECT_EQ(ledger.end_step(), std::vector<StorageID>{slot});
    EXPECT_EQ(ledger.size(), 0u);
}

TEST(Small_SlotLedger, ForgetReturnsEverySlotOnce)
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
