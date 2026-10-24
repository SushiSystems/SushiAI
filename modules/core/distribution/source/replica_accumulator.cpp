/**************************************************************************/
/* replica_accumulator.cpp                                                */
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
 * @file replica_accumulator.cpp
 * @brief Implements the host sum and mean of a replica's micro-batch results.
 * @author Mustafa Garip
 */

#include "replica_accumulator.hpp"

#include <complex>
#include <cstdint>
#include <cstring>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiAI/distribution/exact_sum.hpp>
#include <SushiAI/distribution/host_access.hpp>
#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        namespace
        {
            using SushiBLAS::Core::DataType;

            /** @brief Adds @p count elements of @p source to @p target, element by element. */
            template<typename Element>
            void add_elements(void* target, const void* source, std::size_t count) noexcept
            {
                Element* sum = static_cast<Element*>(target);
                const Element* addend = static_cast<const Element*>(source);
                for (std::size_t index = 0; index < count; ++index)
                    sum[index] += addend[index];
            }

            /** @brief Divides @p count elements of @p values by @p divisor, in their own type. */
            template<typename Element, typename Divisor>
            void divide_elements(void* values, std::size_t count, Divisor divisor) noexcept
            {
                Element* elements = static_cast<Element*>(values);
                for (std::size_t index = 0; index < count; ++index)
                    elements[index] /= divisor;
            }
        } // namespace

        ReplicaAccumulator::ReplicaAccumulator(SushiBLAS::PackLayout layout)
            : layout_(std::move(layout)), sum_(layout_.size_bytes()),
              half_sum_(layout_.segment(DataType::Half).element_count)
        {

        }

        void ReplicaAccumulator::reset() noexcept
        {
            count_ = 0;
        }

        void ReplicaAccumulator::add(SushiRuntime::span<const SushiBLAS::Tensor> results)
        {
            SA_FATAL_IF(results.size() != layout_.entries().size(),
                        "ReplicaAccumulator::add: {} results for a layout of {}", results.size(),
                        layout_.entries().size());

            for (std::size_t index = 0; index < results.size(); ++index)
            {
                const SushiBLAS::Tensor& result = results[index];
                SA_FATAL_IF(!SushiBLAS::PackLayout::matches(layout_.entries()[index], result)
                                || !result.is_contiguous(),
                            "ReplicaAccumulator::add: result {} is not the contiguous tensor its "
                            "layout entry names",
                            index);
                if (layout_.entries()[index].size_bytes == 0)
                    continue;

                require_host_visible(result, "Replica result", index);
                if (count_ == 0)
                    copy_entry(index, result.data());
                else
                    add_entry(index, result.data());
            }

            ++count_;
        }

        Status ReplicaAccumulator::write_mean(const SushiBLAS::BoundPack& target) const
        {
            if (count_ > 1)
            {
                for (const SushiBLAS::PackEntry& entry : layout_.entries())
                    SA_FAIL_IF(sums_exactly(entry.dtype), StatusCode::Unsupported,
                               "A replica of {} micro-batches holds an integer result, whose "
                               "mean is not an integer",
                               count_);
            }

            const SushiBLAS::PackSegment& half = layout_.segment(DataType::Half);
            for (std::size_t index = 0; index < layout_.entries().size(); ++index)
            {
                const SushiBLAS::PackEntry& entry = layout_.entries()[index];
                if (entry.size_bytes == 0)
                    continue;

                // A bound pack is host memory: SushiRuntime borrows every view as Shared.
                void* const values = target.tensor(index).data;
                if (count_ == 0)
                {
                    std::memset(values, 0, entry.size_bytes);
                    continue;
                }

                // One micro-batch is the replica's result as it is, byte for byte.
                std::memcpy(values, sum_.data() + entry.offset_bytes, entry.size_bytes);
                if (count_ == 1)
                    continue;

                switch (entry.dtype)
                {
                    case DataType::Half:
                    {
                        sycl::half* const elements = static_cast<sycl::half*>(values);
                        const std::size_t first = (entry.offset_bytes - half.offset_bytes)
                                                  / sizeof(sycl::half);
                        const float divisor = static_cast<float>(count_);
                        for (std::size_t element = 0; element < entry.element_count; ++element)
                            elements[element] =
                                static_cast<sycl::half>(half_sum_[first + element] / divisor);
                        break;
                    }
                    case DataType::Float32:
                        divide_elements<float>(values, entry.element_count,
                                               static_cast<float>(count_));
                        break;
                    case DataType::Float64:
                        divide_elements<double>(values, entry.element_count,
                                                static_cast<double>(count_));
                        break;
                    case DataType::Complex32:
                        divide_elements<std::complex<float>>(values, entry.element_count,
                                                             static_cast<float>(count_));
                        break;
                    case DataType::Complex64:
                        divide_elements<std::complex<double>>(values, entry.element_count,
                                                              static_cast<double>(count_));
                        break;
                    case DataType::Int32:
                    case DataType::Int64:
                    case DataType::UInt32:
                        break;
                }
            }

            return Status();
        }

        void ReplicaAccumulator::copy_entry(std::size_t index, const void* source)
        {
            const SushiBLAS::PackEntry& entry = layout_.entries()[index];
            std::memcpy(sum_.data() + entry.offset_bytes, source, entry.size_bytes);
            if (entry.dtype != DataType::Half)
                return;

            const std::size_t first = (entry.offset_bytes
                                       - layout_.segment(DataType::Half).offset_bytes)
                                      / sizeof(sycl::half);
            const sycl::half* const elements = static_cast<const sycl::half*>(source);
            for (std::size_t element = 0; element < entry.element_count; ++element)
                half_sum_[first + element] = static_cast<float>(elements[element]);
        }

        void ReplicaAccumulator::add_entry(std::size_t index, const void* source)
        {
            const SushiBLAS::PackEntry& entry = layout_.entries()[index];
            void* const sum = sum_.data() + entry.offset_bytes;
            switch (entry.dtype)
            {
                case DataType::Half:
                {
                    const std::size_t first = (entry.offset_bytes
                                               - layout_.segment(DataType::Half).offset_bytes)
                                              / sizeof(sycl::half);
                    const sycl::half* const elements = static_cast<const sycl::half*>(source);
                    for (std::size_t element = 0; element < entry.element_count; ++element)
                        half_sum_[first + element] += static_cast<float>(elements[element]);
                    break;
                }
                case DataType::Float32:
                    add_elements<float>(sum, source, entry.element_count);
                    break;
                case DataType::Float64:
                    add_elements<double>(sum, source, entry.element_count);
                    break;
                case DataType::Complex32:
                    add_elements<std::complex<float>>(sum, source, entry.element_count);
                    break;
                case DataType::Complex64:
                    add_elements<std::complex<double>>(sum, source, entry.element_count);
                    break;
                case DataType::Int32:
                    // Unsigned addition wraps.
                    add_elements<std::uint32_t>(sum, source, entry.element_count);
                    break;
                case DataType::Int64:
                    add_elements<std::uint64_t>(sum, source, entry.element_count);
                    break;
                case DataType::UInt32:
                    add_elements<std::uint32_t>(sum, source, entry.element_count);
                    break;
            }
        }

    } // namespace Distribution
} // namespace SushiAI
