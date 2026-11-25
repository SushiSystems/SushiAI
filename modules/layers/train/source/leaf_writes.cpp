/**************************************************************************/
/* leaf_writes.cpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file leaf_writes.cpp
 * @brief Implements the check that a feed wrote every leaf its objective declared.
 * @author Mustafa Garip
 */

#include <SushiAI/train/leaf_writes.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Holds the bytes the unwritten pattern repeats: the float32 NaN 0x7FC0DEAD. */
            constexpr std::array<std::uint8_t, 4> K_UNWRITTEN_BYTES = {0xAD, 0xDE, 0xC0, 0x7F};

            /** @brief Returns a zeroed host buffer as large as the elements of @p tensor. */
            [[nodiscard]] Result<std::vector<std::uint8_t>> host_bytes(
                const SushiBLAS::Tensor& tensor) noexcept
            {
                return Detail::guard_value(Detail::SeamPhase::Allocate, [&tensor]()
                {
                    return std::vector<std::uint8_t>(static_cast<std::size_t>(tensor.num_elements)
                                                     * dtype_size(tensor.dtype));
                });
            }

            /** @brief Marks the absence of an unwritten element. */
            constexpr std::size_t K_NONE = static_cast<std::size_t>(-1);

            /**
             * @brief Returns the index of the first element of @p bytes whose every byte holds
             *        the pattern, or K_NONE when each element has a byte that differs.
             * @param bytes The storage of the leaf, @p count bytes long.
             * @param element_size The bytes of one element; at least 1.
             */
            [[nodiscard]] std::size_t first_unwritten_element(const std::uint8_t* bytes,
                                                              std::size_t count,
                                                              std::size_t element_size) noexcept
            {
                for (std::size_t first = 0; first + element_size <= count; first += element_size)
                {
                    bool is_marked = true;
                    for (std::size_t index = first; is_marked && index < first + element_size;
                         ++index)
                        is_marked = bytes[index]
                                    == K_UNWRITTEN_BYTES[index % K_UNWRITTEN_BYTES.size()];

                    if (is_marked)
                        return first / element_size;
                }

                return K_NONE;
            }
        } // namespace

        Status mark_leaves_unwritten(const ObjectiveLeaves& leaves,
                                     Graph::TensorPool& pool) noexcept
        {
            for (const std::pair<std::string, Graph::ValueID>& leaf : leaves.all())
            {
                if (!pool.is_bound(leaf.second))
                    continue;

                const SushiBLAS::Tensor& tensor = pool.tensor(leaf.second);
                if (tensor.num_elements == 0)
                    continue;

                SA_TRY_ASSIGN(std::vector<std::uint8_t> bytes, host_bytes(tensor));
                for (std::size_t index = 0; index < bytes.size(); ++index)
                    bytes[index] = K_UNWRITTEN_BYTES[index % K_UNWRITTEN_BYTES.size()];
                SA_TRY(copy_from_host(tensor, 0, tensor.num_elements, dtype_size(tensor.dtype),
                                      bytes.data()));
            }

            return Status();
        }

        Status require_leaves_written(const ObjectiveLeaves& leaves,
                                      Graph::TensorPool& pool) noexcept
        {
            for (const std::pair<std::string, Graph::ValueID>& leaf : leaves.all())
            {
                if (!pool.is_bound(leaf.second))
                    continue;

                const SushiBLAS::Tensor& tensor = pool.tensor(leaf.second);
                if (tensor.num_elements == 0)
                    continue;

                const std::size_t element_size = dtype_size(tensor.dtype);
                SA_TRY_ASSIGN(std::vector<std::uint8_t> bytes, host_bytes(tensor));
                SA_TRY(copy_to_host(tensor, 0, tensor.num_elements, element_size, bytes.data()));
                const std::size_t unwritten =
                    first_unwritten_element(bytes.data(), bytes.size(), element_size);

                SA_FAIL_IF(unwritten != K_NONE, StatusCode::InvalidDescription,
                           "The objective declared the leaf '{}' but its feed left element {} "
                           "unwritten; a feed writes every element of every leaf its objective "
                           "declares, so the step would have read what an earlier step left there",
                           leaf.first, unwritten);
            }

            return Status();
        }

    } // namespace Train
} // namespace SushiAI
