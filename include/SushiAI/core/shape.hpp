/**************************************************************************/
/* shape.hpp                                                              */
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
 * @file shape.hpp
 * @brief A tensor's extents as a comparable, hashable value type.
 *
 * SushiBLAS's Tensor stores shape as a zero-padded fixed array alongside
 * strides, an offset and a storage handle — the right shape for a memory
 * view, the wrong one for a graph key. SushiAI reasons about extents long
 * before any memory exists (shape inference at trace time, cache keys for a
 * compiled step), so extents get their own type with value semantics.
 */

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
     * @brief The extents of a tensor, independent of any storage.
     *
     * Fixed capacity (MAX_RANK), so it is trivially copyable and can live in a
     * cache key or a hash map without allocating. Dimensions are always
     * positive; a zero or negative extent is rejected at construction rather
     * than producing a tensor nothing can be done with.
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
                SA_THROW_IF(dims.size() > static_cast<std::size_t>(MAX_RANK),
                            "Shape rank {} exceeds the maximum of {}", dims.size(), MAX_RANK);

                for (int64_t d : dims)
                {
                    SA_THROW_IF(d <= 0, "Shape extents must be positive; got {}", d);
                    dims_[static_cast<std::size_t>(rank_++)] = d;
                }
            }

            /**
             * @brief Constructs from extents that are only known at run time.
             *
             * The initializer-list constructor above cannot serve a caller
             * whose rank comes from a file or from another library's tensor:
             * an initializer list is a compile-time construct, so a reader
             * holding @c rank extents in a buffer had no way to build a Shape
             * short of a switch over every rank. Validation is identical - the
             * two constructors reject the same shapes with the same messages.
             *
             * Explicit, because a span silently becoming a Shape would let an
             * extent buffer be passed where a shape was meant.
             *
             * @param dims The extent of each dimension, outermost first.
             * @throws Error If the rank exceeds MAX_RANK or any extent is not positive.
             */
            explicit Shape(SushiRuntime::span<const int64_t> dims)
            {
                SA_THROW_IF(dims.size() > static_cast<std::size_t>(MAX_RANK),
                            "Shape rank {} exceeds the maximum of {}", dims.size(), MAX_RANK);

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
             * @brief A hash suitable for keying a compiled-step cache on shape.
             *
             * FNV-1a over the rank and extents. Not cryptographic; it only has
             * to spread well enough for an unordered_map of a few dozen entries.
             *
             * @return The hash value.
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
            std::array<int64_t, static_cast<std::size_t>(MAX_RANK)> dims_{};
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
