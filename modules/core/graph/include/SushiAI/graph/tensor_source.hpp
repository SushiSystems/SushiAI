/**************************************************************************/
/* tensor_source.hpp                                                      */
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

/**
 * @file tensor_source.hpp
 * @brief Declares TensorSource, the port a TensorPool draws tensors from, and its adapter.
 * @author Mustafa Garip
 * @see modules/core/graph/README.md
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class TensorSource
         * @brief Makes the standalone tensors a TensorPool binds to graph values, to ones and to
         *        scratch.
         *
         * A pool built over an engine allocates them through allocate_tensor(). A pool built over
         * a source asks it, so the bookkeeping of the pool runs without a device.
         */
        class TensorSource
        {
            public:
                virtual ~TensorSource() = default;

                /**
                 * @brief Makes a tensor of @p shape and @p dtype.
                 * @return The tensor, or the failure of making it.
                 */
                [[nodiscard]] virtual Result<SushiBLAS::Tensor> create(const Shape& shape,
                                                                       DType dtype) noexcept = 0;
        };

        /**
         * @class InMemoryTensorSource
         * @brief Describes each tensor and gives it no Storage; touches no device, file, thread
         *        or clock.
         *
         * A tensor it returns holds its extents and element type and no element. It keeps the
         * requests it received, so a test reads how many tensors a pool asked for.
         */
        class InMemoryTensorSource final : public TensorSource
        {
            public:
                /** @brief Holds the shape and the element type of one request. */
                struct Request
                {
                    Shape shape;                      /**< The extents asked for. */
                    DType dtype = DType::Float32;     /**< The element type asked for. */
                };

                /** @brief Records the request and returns a tensor without Storage. */
                [[nodiscard]] Result<SushiBLAS::Tensor> create(const Shape& shape,
                                                               DType dtype) noexcept override
                {
                    namespace Seam = ::SushiAI::Detail;
                    return Seam::guard_value(Seam::SeamPhase::Allocate, [&]()
                    {
                        requests_.push_back(Request{shape, dtype});
                        return SushiBLAS::Tensor(
                            SushiRuntime::sushi_ptr<SushiBLAS::Storage>(),
                            SushiRuntime::span<const int64_t>(
                                shape.data(), static_cast<std::size_t>(shape.rank())),
                            dtype, 0, SushiBLAS::Core::Layout::RowMajor);
                    });
                }

                /** @brief Returns every request received, first received first. */
                [[nodiscard]] const std::vector<Request>& requests() const noexcept
                {
                    return requests_;
                }

            private:
                /// @brief Holds the requests received, in order.
                std::vector<Request> requests_;
        };

    } // namespace Graph
} // namespace SushiAI
