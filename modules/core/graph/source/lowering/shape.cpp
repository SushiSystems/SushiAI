/**************************************************************************/
/* shape.cpp                                                              */
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
 * @file shape.cpp
 * @brief Implements the lowering rows of the shape operations.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/shape.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/spatial/spatial.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/context.hpp>
#include <SushiAI/graph/permuted_copy.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Holds the strides of a SushiBLAS tensor view, one per axis it may have. */
            using StrideArray = std::array<std::int64_t, SushiBLAS::Core::K_MAX_TENSOR_RANK>;
        } // namespace

        Status lower_leaf(const LoweringContext& context) noexcept
        {
            // A leaf takes any number of operands and results; only the binding is checked.
            const Node& node = context.node();
            for (std::size_t index = 0; index < node.outputs.size(); ++index)
            {
                SA_FATAL_IF(!context.output_is_bound(index),
                            "Leaf value {} ('{}') has no tensor bound. Bind it, or call "
                            "TensorPool::materialise_leaves() before lowering.",
                            node.outputs[index], context.output_value(index).debug_name);
            }

            context.count(0);
            return Status();
        }

        Status lower_reshape(const LoweringContext& context) noexcept
        {
            context.require_arity(1, 1);

            const SushiBLAS::Tensor& source = context.input(0);
            SA_FATAL_IF(!source.is_contiguous(),
                        "RESHAPE (node {}) needs a contiguous operand; the value it reads was "
                        "produced as a strided view, which reshape's reinterpretation would "
                        "read wrong",
                        context.id());

            SA_TRY_ASSIGN(SushiBLAS::Tensor& destination, context.output(0));

            // The plan made the result a view of the operand, so there is nothing to copy.
            if (destination.storage == source.storage
                && destination.storage_offset == source.storage_offset)
            {
                return Status();
            }

            // reshaped reads the operand's storage, which the check above found contiguous.
            const Shape& out_shape = context.output_value(0).shape;
            const SushiBLAS::Tensor reshaped(
                source.storage,
                SushiRuntime::span<const std::int64_t>(
                    out_shape.data(), static_cast<std::size_t>(out_shape.rank())),
                source.dtype, source.storage_offset, source.layout);

            SushiBLAS::BLASOps blas(context.recorder());
            blas.copy(reshaped, destination);

            context.count(1);
            return Status();
        }

        Status lower_transpose(const LoweringContext& context) noexcept
        {
            context.require_arity(1, 1);

            const SushiBLAS::Tensor& source = context.input(0);
            SA_FATAL_IF(!source.is_contiguous(),
                        "TRANSPOSE (node {}) needs a contiguous operand; the plan walks its "
                        "strides as row-major ones",
                        context.id());
            SA_TRY_ASSIGN(SushiBLAS::Tensor& destination, context.output(0));

            const Shape& in_shape = context.input_value(0).shape;
            const std::size_t count = static_cast<std::size_t>(in_shape.rank()) - 1;
            const auto permutation = permutation_of(context.node(), count);

            const PermutedCopy plan = plan_permuted_copy(
                in_shape, SushiRuntime::span<const std::int64_t>(permutation.data(), count));
            SA_FATAL_IF(!plan.fits,
                        "TRANSPOSE (node {}) moves more runs of axes than one copy walks; the "
                        "builder refuses such a node",
                        context.id());

            // Both views carry the result's extents; only the strides differ.
            std::array<std::int64_t, K_COPY_AXES> extents{};
            StrideArray source_steps{};
            StrideArray result_steps{};
            std::int64_t running = 1;
            for (std::size_t axis = K_COPY_AXES; axis-- > 0;)
            {
                extents[axis] = plan.axes[axis].extent;
                source_steps[axis] = plan.axes[axis].stride;
                result_steps[axis] = running;
                running *= extents[axis];
            }

            const SushiRuntime::span<const std::int64_t> dims(extents.data(), K_COPY_AXES);
            const SushiBLAS::Tensor source_view(source.storage, dims, source_steps, source.dtype,
                                                source.storage_offset, source.layout);
            SushiBLAS::Tensor result_view(destination.storage, dims, result_steps,
                                          destination.dtype, destination.storage_offset,
                                          destination.layout);

            // The layout conversion copies by logical position over any two stride sets.
            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.layout_convert(source_view, result_view);

            context.count(1);
            return Status();
        }

        Status lower_layout_convert(const LoweringContext& context) noexcept
        {
            context.require_arity(1, 1);

            const SushiBLAS::Tensor& source = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& destination, context.output(0));

            const TensorLayout source_layout = context.input_value(0).layout;
            const TensorLayout dest_layout = context.output_value(0).layout;

            const SushiBLAS::Tensor source_view = (source_layout == TensorLayout::NCHW)
                ? source.transpose(1, 2).transpose(2, 3) : source;
            SushiBLAS::Tensor dest_view = (dest_layout == TensorLayout::NCHW)
                ? destination.transpose(1, 2).transpose(2, 3) : destination;

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.layout_convert(source_view, dest_view);

            context.count(1);
            return Status();
        }

        Status lower_stop_gradient(const LoweringContext& context) noexcept
        {
            context.require_arity(1, 1);

            const SushiBLAS::Tensor& source = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& destination, context.output(0));

            SushiBLAS::BLASOps blas(context.recorder());
            blas.copy(source, destination);

            context.count(1);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
