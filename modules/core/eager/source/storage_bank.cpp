/**************************************************************************/
/* storage_bank.cpp                                                       */
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
 * @file storage_bank.cpp
 * @brief Implements StorageBank, the map from a storage id to its tensor.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/storage_bank.hpp>

#include <utility>

#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>

namespace SushiAI
{
    namespace Eager
    {
        StorageBank::StorageBank(SushiBLAS::Engine& engine) noexcept : engine_(&engine)
        {

        }

        Status StorageBank::adopt(StorageID id, SushiBLAS::Tensor tensor)
        {
            SA_FATAL_IF(id == K_NO_STORAGE, "Storage id 0 names no storage and holds no tensor");
            SA_FATAL_IF(holds(id), "Storage {} already holds a tensor", id);

            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                tensors_.emplace(id, std::move(tensor));
                return Status();
            });
        }

        Status StorageBank::allocate(StorageID id, const Shape& shape, DType dtype)
        {
            SA_TRY_ASSIGN(SushiBLAS::Tensor tensor, allocate_tensor(*engine_, shape, dtype));
            SA_TRY(adopt(id, std::move(tensor)));
            ++allocations_;
            return Status();
        }

        void StorageBank::release(StorageID id) noexcept
        {
            tensors_.erase(id);
        }

        bool StorageBank::holds(StorageID id) const noexcept
        {
            return tensors_.find(id) != tensors_.end();
        }

        SushiBLAS::Tensor& StorageBank::tensor(StorageID id)
        {
            const auto found = tensors_.find(id);
            SA_FATAL_IF(found == tensors_.end(), "Storage {} holds no tensor", id);
            return found->second;
        }

        const SushiBLAS::Tensor& StorageBank::tensor(StorageID id) const
        {
            const auto found = tensors_.find(id);
            SA_FATAL_IF(found == tensors_.end(), "Storage {} holds no tensor", id);
            return found->second;
        }

    } // namespace Eager
} // namespace SushiAI
