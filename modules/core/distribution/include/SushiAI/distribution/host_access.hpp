/**************************************************************************/
/* host_access.hpp                                                        */
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
 * @file host_access.hpp
 * @brief Declares the checks that a tensor the host copies is host-visible.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstddef>
#include <string_view>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /** @brief Returns whether the host may address @p tensor's memory directly. */
        [[nodiscard]] inline bool is_host_visible(const SushiBLAS::Tensor& tensor) noexcept
        {
            return tensor.storage
                   && tensor.storage->residency() != SushiRuntime::Residency::Device;
        }

        /**
         * @brief Stops the process when the host may not address @p tensor's memory.
         * @param what Names the tensor's role in the message, with @p index.
         */
        inline void require_host_visible(const SushiBLAS::Tensor& tensor, std::string_view what,
                                         std::size_t index)
        {
            SA_FATAL_IF(!is_host_visible(tensor),
                        "{} {} lives in device memory, which a host copy cannot address", what,
                        index);
        }

    } // namespace Distribution
} // namespace SushiAI
