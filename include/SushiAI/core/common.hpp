/**************************************************************************/
/* common.hpp                                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file common.hpp
 * @brief Vocabulary shared by every SushiAI subsystem.
 *
 * Deliberately tiny. Anything that lands here is used by more than one
 * subsystem and has nowhere better to live; everything else belongs behind
 * its own boundary. The element-type vocabulary is SushiBLAS's, aliased
 * rather than redefined, because a second dtype enum that must be kept in
 * step with the first is a bug waiting for a deadline.
 */

#include <string>
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <SushiBLAS/core/common.hpp>

namespace SushiAI
{
    /**
     * @brief The element type of a tensor.
     *
     * An alias, not a parallel enum: SushiAI performs no arithmetic of its
     * own, so the set of types it can express is exactly the set SushiBLAS
     * can compute on.
     */
    using DType = SushiBLAS::Core::DataType;

    /** @brief The memory layout of a tensor, aliased from SushiBLAS for the same reason. */
    using Layout = SushiBLAS::Core::Layout;

    /** @brief Maximum tensor rank, fixed by SushiBLAS's own storage of shape and strides. */
    inline constexpr int32_t MAX_RANK = static_cast<int32_t>(SushiBLAS::Core::MAX_TENSOR_RANK);

    /**
     * @brief Bytes occupied by one element of @p dtype.
     *
     * The single source of truth for this mapping inside SushiAI. SushiBLAS
     * currently repeats the same table in four places; SushiAI states it once
     * so a new dtype is one edit here rather than a hunt.
     *
     * @param dtype The element type to size.
     * @return The element's size in bytes; never zero.
     */
    constexpr std::size_t dtype_size(DType dtype) noexcept
    {
        switch (dtype)
        {
            case DType::HALF:      return 2;
            case DType::FLOAT32:   return 4;
            case DType::FLOAT64:   return 8;
            case DType::COMPLEX32: return 8;
            case DType::COMPLEX64: return 16;
        }
        return 4;
    }

    /**
     * @brief A human-readable name for @p dtype, for diagnostics and error text.
     * @param dtype The element type to name.
     * @return A null-terminated literal owned by the library; never null.
     */
    constexpr const char* dtype_name(DType dtype) noexcept
    {
        switch (dtype)
        {
            case DType::HALF:      return "half";
            case DType::FLOAT32:   return "float32";
            case DType::FLOAT64:   return "float64";
            case DType::COMPLEX32: return "complex32";
            case DType::COMPLEX64: return "complex64";
        }
        return "unknown";
    }

    /**
     * @brief Reports whether @p dtype is a real (non-complex) floating type.
     *
     * The training path is real-valued throughout; this is the predicate every
     * op uses to reject a complex tensor at the boundary rather than deep
     * inside a kernel.
     *
     * @param dtype The element type to test.
     * @return True for HALF, FLOAT32 and FLOAT64; false otherwise.
     */
    constexpr bool is_real_float(DType dtype) noexcept
    {
        return dtype == DType::HALF || dtype == DType::FLOAT32 || dtype == DType::FLOAT64;
    }

    /**
     * @brief The exception every SushiAI failure surfaces as.
     *
     * A distinct type so a caller can tell a SushiAI contract violation from a
     * SushiBLAS or SushiRuntime one without string matching. It derives from
     * std::runtime_error so a caller that does not care still catches it.
     */
    class Error : public std::runtime_error
    {
        public:
            /**
             * @brief Constructs the exception from a fully formatted message.
             * @param message The message, already formatted; copied.
             */
            explicit Error(const std::string& message) : std::runtime_error(message) {}
    };

} // namespace SushiAI

/**
 * @brief Throws SushiAI::Error with a formatted message when @p condition holds.
 *
 * Input validation, not internal-invariant checking: it stays compiled in
 * under NDEBUG on purpose, because the conditions it guards are shape and
 * dtype mismatches supplied by a caller, and a silent wrong answer is worse
 * than a thrown one. Mirrors SushiBLAS's SB_THROW_IF so the two libraries
 * fail the same way.
 */
#define SA_THROW_IF(condition, ...)                                            \
    do {                                                                       \
        if (condition) {                                                       \
            throw ::SushiAI::Error(std::format(__VA_ARGS__));                  \
        }                                                                      \
    } while (0)

/** @brief Short alias for the library's namespace, matching sb:: and sr:: */
namespace sa = SushiAI;
