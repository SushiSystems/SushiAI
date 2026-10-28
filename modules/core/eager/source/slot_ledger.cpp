/**************************************************************************/
/* slot_ledger.cpp                                                        */
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
 * @file slot_ledger.cpp
 * @brief Implements SlotLedger, the rule that matches a shared buffer with a slot.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/slot_ledger.hpp>

#include <utility>

namespace SushiAI
{
    namespace Eager
    {
        SlotLedger::SlotLedger(StorageIDSource& ids) noexcept : ids_(&ids)
        {

        }

        SlotLedger::Key SlotLedger::position_key(RegionID producer,
                                                 ValuePosition position) noexcept
        {
            return Key{producer,
                       (static_cast<std::uint64_t>(position.node) << 32) | position.result};
        }

        SlotLedger::Key SlotLedger::reader_key(const ReaderSlot& reader) noexcept
        {
            return Key{reader.reader, reader.ordinal};
        }

        SlotLedger::Slot* SlotLedger::find_unclaimed(const Index& index, const Key& key)
        {
            const auto found = index.find(key);
            if (found == index.end() || slots_[found->second].claimed)
                return nullptr;

            return &slots_[found->second];
        }

        void SlotLedger::carry(Slot slot, RegionID producer, ValuePosition position,
                               SushiRuntime::span<const ReaderSlot> readers)
        {
            slot.producer = producer;
            slot.position = position;
            slot.claimed = false;

            const std::size_t at = next_.size();
            next_.push_back(std::move(slot));
            next_by_position_[position_key(producer, position)] = at;
            for (const ReaderSlot& reader : readers)
                next_by_reader_[reader_key(reader)] = at;
        }

        void SlotLedger::begin_step() noexcept
        {
            next_.clear();
            next_by_position_.clear();
            next_by_reader_.clear();
            created_.clear();
            open_ = true;

            for (Slot& slot : slots_)
                slot.claimed = false;
        }

        StorageID SlotLedger::keep(RegionID producer, ValuePosition position,
                                   SushiRuntime::span<const ReaderSlot> readers) noexcept
        {
            Slot* const slot = find_unclaimed(by_position_, position_key(producer, position));
            if (slot == nullptr)
                return K_NO_STORAGE;

            slot->claimed = true;
            carry(*slot, producer, position, readers);
            return slot->id;
        }

        StorageID SlotLedger::match_readers(RegionID producer, ValuePosition position,
                                            const Shape& shape, DType dtype,
                                            SushiRuntime::span<const ReaderSlot> trusted,
                                            SushiRuntime::span<const ReaderSlot> readers) noexcept
        {
            for (const ReaderSlot& reader : trusted)
            {
                Slot* const slot = find_unclaimed(by_reader_, reader_key(reader));
                if (slot == nullptr || slot->producer != producer || slot->shape != shape
                    || slot->dtype != dtype)
                    continue;

                slot->claimed = true;
                carry(*slot, producer, position, readers);
                return slot->id;
            }

            return K_NO_STORAGE;
        }

        StorageID SlotLedger::place(RegionID producer, ValuePosition position, const Shape& shape,
                                    DType dtype, SushiRuntime::span<const ReaderSlot> readers) noexcept
        {
            Slot* const written = find_unclaimed(by_position_, position_key(producer, position));
            if (written != nullptr && written->shape == shape && written->dtype == dtype)
            {
                written->claimed = true;
                carry(*written, producer, position, readers);
                return written->id;
            }

            Slot fresh;
            fresh.id = ids_->next();
            fresh.shape = shape;
            fresh.dtype = dtype;
            created_.push_back(SlotRecord{fresh.id, shape, dtype});
            carry(fresh, producer, position, readers);
            return fresh.id;
        }

        std::vector<StorageID> SlotLedger::end_step() noexcept
        {
            std::vector<StorageID> released;
            for (const Slot& slot : slots_)
            {
                if (!slot.claimed)
                    released.push_back(slot.id);
            }

            slots_.swap(next_);
            by_position_.swap(next_by_position_);
            by_reader_.swap(next_by_reader_);
            next_.clear();
            next_by_position_.clear();
            next_by_reader_.clear();
            open_ = false;
            return released;
        }

        std::vector<StorageID> SlotLedger::forget() noexcept
        {
            std::vector<StorageID> released;
            for (const Slot& slot : slots_)
                released.push_back(slot.id);

            // Only an open step holds created slots that slots_ does not list yet.
            if (open_)
            {
                for (const SlotRecord& record : created_)
                    released.push_back(record.id);
            }

            slots_.clear();
            next_.clear();
            by_position_.clear();
            by_reader_.clear();
            next_by_position_.clear();
            next_by_reader_.clear();
            created_.clear();
            open_ = false;
            return released;
        }

    } // namespace Eager
} // namespace SushiAI
