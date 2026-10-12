/**************************************************************************/
/* storage_bank.hpp                                                       */
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
 * @file storage_bank.hpp
 * @brief Declares StorageBank, which holds the tensor behind each storage id of a session.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <unordered_map>

#include <SushiBLAS/engine/engine.hpp>

#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/eager/storage_id.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @brief Maps storage ids to tensors that live across steps.
         * @see modules/core/eager/README.md
         */
        class StorageBank final
        {
            public:
                /**
                 * @brief Starts with no tensor.
                 * @param engine The engine whose context allocates; must outlive the bank.
                 */
                explicit StorageBank(SushiBLAS::Engine& engine) noexcept;

                StorageBank(const StorageBank&) = delete;
                StorageBank& operator=(const StorageBank&) = delete;
                StorageBank(StorageBank&&) = delete;
                StorageBank& operator=(StorageBank&&) = delete;
                ~StorageBank() = default;

                /**
                 * @brief Holds @p tensor under @p id; the bank shares its storage.
                 * @pre @p id is not K_NO_STORAGE and names no tensor yet; else the process stops.
                 * @return Success, or OutOfMemory when the map cannot grow.
                 */
                [[nodiscard]] Status adopt(StorageID id, SushiBLAS::Tensor tensor);

                /**
                 * @brief Allocates a tensor of @p shape and @p dtype and holds it under @p id.
                 * @pre @p id is not K_NO_STORAGE and names no tensor yet; else the process stops.
                 * @return Success, or the failure of the allocation.
                 */
                [[nodiscard]] Status allocate(StorageID id, const Shape& shape, DType dtype);

                /** @brief Lets go of the tensor under @p id; an unknown id does nothing. */
                void release(StorageID id) noexcept;

                /** @brief Reports whether @p id names a tensor. */
                [[nodiscard]] bool holds(StorageID id) const noexcept;

                /**
                 * @brief Returns the tensor under @p id.
                 * @pre holds(@p id); an unknown id stops the process.
                 * @return A reference valid until release(@p id).
                 */
                [[nodiscard]] SushiBLAS::Tensor& tensor(StorageID id);

                /** @brief Const overload of tensor(). */
                [[nodiscard]] const SushiBLAS::Tensor& tensor(StorageID id) const;

                /** @brief Returns how many tensors the bank holds. */
                [[nodiscard]] std::size_t size() const noexcept
                {
                    return tensors_.size();
                }

                /** @brief Returns how many tensors allocate() has created since construction. */
                [[nodiscard]] std::size_t allocation_count() const noexcept
                {
                    return allocations_;
                }

                /** @brief Returns the engine the bank allocates from. */
                [[nodiscard]] SushiBLAS::Engine& engine() const noexcept
                {
                    return *engine_;
                }

            private:
                /// @brief The engine whose context allocates.
                SushiBLAS::Engine* engine_;

                /// @brief The tensor of each id.
                std::unordered_map<StorageID, SushiBLAS::Tensor> tensors_;

                /// @brief Counts the tensors allocate() created.
                std::size_t allocations_ = 0;
        };

    } // namespace Eager
} // namespace SushiAI
