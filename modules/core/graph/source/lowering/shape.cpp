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
 * @file shape.cpp
 * @brief Implements the lowering rows of the shape operations.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/shape.hpp>

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/spatial/spatial.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        Status lower_leaf(const LoweringContext& context)
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

        Status lower_reshape(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const SushiBLAS::Tensor& source = context.input(0);
            SA_FATAL_IF(!source.is_contiguous(),
                        "RESHAPE (node {}) needs a contiguous operand; the value it reads was "
                        "produced as a strided view, which reshape's reinterpretation would "
                        "read wrong",
                        context.id());

            SA_TRY_ASSIGN(SushiBLAS::Tensor& destination, context.output(0));

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

        Status lower_layout_convert(const LoweringContext& context)
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

        Status lower_stop_gradient(const LoweringContext& context)
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
