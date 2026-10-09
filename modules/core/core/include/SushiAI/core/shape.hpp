/**************************************************************************/
/* shape.hpp                                                              */
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
 * @file shape.hpp
 * @brief Defines the tensor shape value type and hashing utilities.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <string>
#include <cstdint>
#include <initializer_list>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    /**
     * @class Shape
     * @brief Represents tensor extents as a fixed-capacity value type.
     */
    class Shape
    {
        public:
            /** @brief Constructs the rank-0 (scalar) shape. */
            constexpr Shape() noexcept = default;

            /**
             * @brief Constructs from an extent list, e.g. Shape{batch, features}.
             * @param dims The extent of each dimension, outermost first.
             * @throws Error If the rank exceeds MAX_RANK or any extent is not positive.
             */
            Shape(std::initializer_list<int64_t> dims)
            {
                SA_THROW_IF(dims.size() > static_cast<std::size_t>(K_MAX_RANK),
                            "Shape rank {} exceeds the maximum of {}", dims.size(), K_MAX_RANK);

                for (int64_t d : dims)
                {
                    SA_THROW_IF(d <= 0, "Shape extents must be positive; got {}", d);
                    dims_[static_cast<std::size_t>(rank_++)] = d;
                }
            }

            /**
             * @brief Constructs extents from a runtime span.
             * @param dims The extent of each dimension, outermost first.
             * @throws Error If the rank exceeds MAX_RANK or any extent is not positive.
             */
            explicit Shape(SushiRuntime::span<const int64_t> dims)
            {
                SA_THROW_IF(dims.size() > static_cast<std::size_t>(K_MAX_RANK),
                            "Shape rank {} exceeds the maximum of {}", dims.size(), K_MAX_RANK);

                for (std::size_t i = 0; i < dims.size(); ++i)
                {
                    SA_THROW_IF(dims[i] <= 0, "Shape extents must be positive; got {}", dims[i]);
                    dims_[static_cast<std::size_t>(rank_++)] = dims[i];
                }
            }

            /** @brief The number of dimensions; 0 for a scalar. */
            constexpr int32_t rank() const noexcept { return rank_; }

            /**
             * @brief The extent of one dimension.
             * @param index Dimension index; negative counts from the innermost.
             * @return The extent at @p index.
             * @throws Error If @p index does not name a dimension of this shape.
             */
            int64_t dim(int32_t index) const
            {
                const int32_t i = index < 0 ? index + rank_ : index;
                SA_THROW_IF(i < 0 || i >= rank_,
                            "Dimension {} is out of range for a rank-{} shape", index, rank_);
                return dims_[static_cast<std::size_t>(i)];
            }

            /** @brief Unchecked extent access, for hot paths that have already validated. */
            constexpr int64_t operator[](int32_t index) const noexcept
            {
                return dims_[static_cast<std::size_t>(index)];
            }

            /** @brief The total element count; 1 for a scalar, never zero. */
            constexpr int64_t num_elements() const noexcept
            {
                int64_t n = 1;
                for (int32_t i = 0; i < rank_; ++i) n *= dims_[static_cast<std::size_t>(i)];
                return n;
            }

            /** @brief The extents as a contiguous span, for handing to SushiBLAS. */
            constexpr const int64_t* data() const noexcept { return dims_.data(); }

            /** @brief Value equality: same rank and same extents. */
            friend constexpr bool operator==(const Shape& a, const Shape& b) noexcept
            {
                if (a.rank_ != b.rank_) return false;
                for (int32_t i = 0; i < a.rank_; ++i)
                    if (a.dims_[static_cast<std::size_t>(i)] != b.dims_[static_cast<std::size_t>(i)])
                        return false;
                return true;
            }

            /** @brief Value inequality, the negation of operator==. */
            friend constexpr bool operator!=(const Shape& a, const Shape& b) noexcept { return !(a == b); }

            /**
             * @brief A bracketed rendering such as "[64, 784]", for error messages.
             * @return A freshly built string; not cached.
             */
            std::string to_string() const
            {
                std::string out = "[";
                for (int32_t i = 0; i < rank_; ++i)
                {
                    if (i) out += ", ";
                    out += std::to_string(dims_[static_cast<std::size_t>(i)]);
                }
                return out + "]";
            }

            /**
             * @brief Computes an FNV-1a hash over rank and extents.
             * @return The 64-bit hash value.
             */
            constexpr std::uint64_t hash() const noexcept
            {
                std::uint64_t h = 1469598103934665603ull;
                const auto mix = [&h](std::uint64_t v) constexpr
                {
                    for (int b = 0; b < 8; ++b)
                    {
                        h ^= (v >> (b * 8)) & 0xffull;
                        h *= 1099511628211ull;
                    }
                };
                mix(static_cast<std::uint64_t>(rank_));
                for (int32_t i = 0; i < rank_; ++i)
                    mix(static_cast<std::uint64_t>(dims_[static_cast<std::size_t>(i)]));
                return h;
            }

        private:
            std::array<int64_t, static_cast<std::size_t>(K_MAX_RANK)> dims_{};
            int32_t rank_ = 0;
    };

} // namespace SushiAI

namespace std
{
    /** @brief Enables Shape as an unordered-container key. */
    template<>
    struct hash<::SushiAI::Shape>
    {
        std::size_t operator()(const ::SushiAI::Shape& s) const noexcept
        {
            return static_cast<std::size_t>(s.hash());
        }
    };
}
