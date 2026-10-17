/**************************************************************************/
/* slot_ledger.hpp                                                        */
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
 * @file slot_ledger.hpp
 * @brief Declares SlotLedger, which matches a shared buffer with its storage of the last step.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/region_names.hpp>
#include <SushiAI/eager/storage_id.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /** @brief Names a shared buffer as one region reads it. */
        struct ReaderSlot
        {
            RegionID reader = K_NO_REGION_ID; /**< The region that reads the buffer. */
            std::uint32_t ordinal = 0;        /**< Index of the value among its imports. */
        };

        /** @brief Describes storage a step needs and the session does not hold yet. */
        struct SlotRecord
        {
            StorageID id = K_NO_STORAGE;  /**< The id the storage is known by. */
            Shape shape;                  /**< The extents to allocate. */
            DType dtype = DType::Float32; /**< The element type to allocate. */
        };

        /**
         * @brief Gives each shared buffer of a step a storage id, reusing last step's.
         * @see modules/core/eager/README.md
         */
        class SlotLedger final
        {
            public:
                /**
                 * @brief Starts with no slot.
                 * @param ids Hands out the id of each new slot; must outlive the ledger.
                 */
                explicit SlotLedger(StorageIDSource& ids) noexcept;

                SlotLedger(const SlotLedger&) = delete;
                SlotLedger& operator=(const SlotLedger&) = delete;
                SlotLedger(SlotLedger&&) = delete;
                SlotLedger& operator=(SlotLedger&&) = delete;
                ~SlotLedger() = default;

                /** @brief Opens a step; no slot of the last step is claimed yet. */
                void begin_step();

                /**
                 * @brief Claims the slot @p producer wrote at @p position in the last step.
                 * @param readers The regions that read the buffer in this step.
                 * @return The slot's id, or K_NO_STORAGE when none exists or it is claimed.
                 */
                [[nodiscard]] StorageID keep(RegionID producer, ValuePosition position,
                                             SushiRuntime::span<const ReaderSlot> readers);

                /**
                 * @brief Claims the slot of @p producer that one of @p trusted read last step.
                 * @param trusted The readers whose own structure did not change.
                 * @param readers Every region that reads the buffer in this step.
                 * @return The id of a slot of the same extents and element type, or
                 *         K_NO_STORAGE when no trusted reader names one.
                 * @see modules/core/eager/README.md
                 */
                [[nodiscard]] StorageID match_readers(
                    RegionID producer, ValuePosition position, const Shape& shape, DType dtype,
                    SushiRuntime::span<const ReaderSlot> trusted,
                    SushiRuntime::span<const ReaderSlot> readers);

                /**
                 * @brief Claims the slot @p producer wrote at @p position if it fits, or
                 *        creates one.
                 * @param readers Every region that reads the buffer in this step.
                 * @return The id of a slot of the same extents and element type.
                 */
                [[nodiscard]] StorageID place(RegionID producer, ValuePosition position,
                                              const Shape& shape, DType dtype,
                                              SushiRuntime::span<const ReaderSlot> readers);

                /**
                 * @brief Closes the step and forgets the slots it did not claim.
                 * @return The ids of the forgotten slots.
                 */
                [[nodiscard]] std::vector<StorageID> end_step();

                /**
                 * @brief Forgets every slot, claimed or not.
                 * @return The ids of the forgotten slots.
                 */
                [[nodiscard]] std::vector<StorageID> forget();

                /** @brief Returns the slots created since begin_step(). */
                [[nodiscard]] SushiRuntime::span<const SlotRecord> created() const noexcept
                {
                    return SushiRuntime::span<const SlotRecord>(created_.data(), created_.size());
                }

                /** @brief Returns how many slots the last closed step claimed. */
                [[nodiscard]] std::size_t size() const noexcept
                {
                    return slots_.size();
                }

            private:
                /** @brief Holds one slot and the name its producer gave it. */
                struct Slot
                {
                    StorageID id = K_NO_STORAGE;         /**< The slot's storage id. */
                    RegionID producer = K_NO_REGION_ID;  /**< The region that writes it first. */
                    ValuePosition position;              /**< Its place in that region. */
                    Shape shape;                         /**< The extents it was created for. */
                    DType dtype = DType::Float32;        /**< Its element type. */
                    bool claimed = false;                /**< Whether this step took it. */
                };

                /** @brief Holds two words that name a slot from one side. */
                struct Key
                {
                    std::uint64_t owner = 0; /**< The producer or the reader. */
                    std::uint64_t place = 0; /**< The position or the import ordinal. */

                    /** @brief Reports whether both words agree. */
                    bool operator==(const Key& other) const noexcept
                    {
                        return owner == other.owner && place == other.place;
                    }
                };

                /** @brief Hashes a Key. */
                struct KeyHash
                {
                    /** @brief Mixes the two words of @p key. */
                    std::size_t operator()(const Key& key) const noexcept
                    {
                        return static_cast<std::size_t>(
                            (key.owner * 0x9E3779B97F4A7C15ull) ^ (key.place + (key.owner << 7)));
                    }
                };

                /** @brief Maps a name to an index into a slot list. */
                using Index = std::unordered_map<Key, std::size_t, KeyHash>;

                /** @brief Returns the name @p producer gives the slot at @p position. */
                [[nodiscard]] static Key position_key(RegionID producer,
                                                      ValuePosition position) noexcept;

                /** @brief Returns the name @p reader gives the slot it reads. */
                [[nodiscard]] static Key reader_key(const ReaderSlot& reader) noexcept;

                /** @brief Returns the unclaimed slot @p index names under @p key, or null. */
                [[nodiscard]] Slot* find_unclaimed(const Index& index, const Key& key);

                /** @brief Lists @p slot among this step's slots under its new names. */
                void carry(Slot slot, RegionID producer, ValuePosition position,
                           SushiRuntime::span<const ReaderSlot> readers);

                /// @brief Hands out the id of each new slot.
                StorageIDSource* ids_;

                /// @brief The slots the last closed step claimed.
                std::vector<Slot> slots_;

                /// @brief The slots the open step has claimed or created.
                std::vector<Slot> next_;

                /// @brief Names slots_ by producer and position.
                Index by_position_;

                /// @brief Names slots_ by reader and import ordinal.
                Index by_reader_;

                /// @brief Names next_ by producer and position.
                Index next_by_position_;

                /// @brief Names next_ by reader and import ordinal.
                Index next_by_reader_;

                /// @brief The slots created since begin_step().
                std::vector<SlotRecord> created_;

                /// @brief Whether begin_step() was called and end_step() was not.
                bool open_ = false;
        };

    } // namespace Eager
} // namespace SushiAI
