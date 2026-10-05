/****************************************************************/
/* common.hpp                                                   */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file common.hpp
 * @brief Declares shared types, constants, and error utilities for SushiAI.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/format.hpp>

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
     * @brief Returns the size in bytes occupied by one element of @p dtype.
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
            case DType::INT32:     return 4;
            case DType::INT64:     return 8;
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
            case DType::INT32:     return "int32";
            case DType::INT64:     return "int64";
        }
        return "unknown";
    }

    /**
     * @brief Reports whether @p dtype is a real floating-point type.
     * @param dtype The element type to test.
     * @return True for HALF, FLOAT32, and FLOAT64; false otherwise.
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
 * @param condition Expression to evaluate.
 * @param ... Format string and arguments forwarded to SushiBLAS::Core::format.
 */
#define SA_THROW_IF(condition, ...)                                            \
    do {                                                                       \
        if (condition) {                                                       \
            throw ::SushiAI::Error(::SushiBLAS::Core::format(__VA_ARGS__));    \
        }                                                                      \
    } while (0)

/** @brief Short alias for the library's namespace, matching sb:: and sr:: */
namespace sa = SushiAI;
