/**************************************************************************/
/* activation.cpp                                                         */
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
 * @file activation.cpp
 * @brief Implements the lowering rows of the activation operations.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/activation.hpp>

#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/nonlinear/nonlinear.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief A NonLinearOps member that rewrites one tensor in place. */
            using InPlaceKernel =
                SushiBLAS::TaskHandle (SushiBLAS::NonLinearOps::*)(SushiBLAS::Tensor&);

            /** @brief A NonLinearOps member that writes dx from dy and one forward tensor. */
            using BackwardKernel = SushiBLAS::TaskHandle (SushiBLAS::NonLinearOps::*)(
                const SushiBLAS::Tensor&, const SushiBLAS::Tensor&, SushiBLAS::Tensor&);

            /**
             * @brief Copies the operand into the result, then records one in-place kernel on it.
             * @param context The node to lower; it has one operand and one result.
             * @param kernel The member of NonLinearOps that records the in-place kernel.
             * @return Ok when both tasks are recorded.
             */
            [[nodiscard]] Status record_copy_then(const LoweringContext& context,
                                                  InPlaceKernel kernel)
            {
                context.require_arity(1, 1);
                const SushiBLAS::Tensor& x = context.input(0);
                SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));

                SushiBLAS::BLASOps blas(context.recorder());
                SushiBLAS::NonLinearOps nonlinear(context.recorder());
                blas.copy(x, y);
                (nonlinear.*kernel)(y);
                context.count(2);
                return Status();
            }

            /**
             * @brief Records one gradient kernel: dx = kernel(dy, forward tensor).
             * @param context The node to lower; it has two operands and one result.
             * @param kernel The member of NonLinearOps that records the gradient kernel.
             * @return Ok when the task is recorded.
             */
            [[nodiscard]] Status record_backward(const LoweringContext& context,
                                                 BackwardKernel kernel)
            {
                context.require_arity(2, 1);
                const SushiBLAS::Tensor& dy = context.input(0);
                const SushiBLAS::Tensor& primal = context.input(1);
                SA_TRY_ASSIGN(SushiBLAS::Tensor& dx, context.output(0));

                SushiBLAS::NonLinearOps nonlinear(context.recorder());
                (nonlinear.*kernel)(dy, primal, dx);
                context.count(1);
                return Status();
            }
        } // namespace

        Status lower_relu(const LoweringContext& context)
        {
            return record_copy_then(context, &SushiBLAS::NonLinearOps::relu);
        }

        Status lower_gelu(const LoweringContext& context)
        {
            return record_copy_then(context, &SushiBLAS::NonLinearOps::gelu);
        }

        Status lower_sigmoid(const LoweringContext& context)
        {
            return record_copy_then(context, &SushiBLAS::NonLinearOps::sigmoid);
        }

        Status lower_tanh(const LoweringContext& context)
        {
            return record_copy_then(context, &SushiBLAS::NonLinearOps::tanh);
        }

        Status lower_silu(const LoweringContext& context)
        {
            return record_copy_then(context, &SushiBLAS::NonLinearOps::silu);
        }

        Status lower_softplus(const LoweringContext& context)
        {
            return record_copy_then(context, &SushiBLAS::NonLinearOps::softplus);
        }

        Status lower_leaky_relu(const LoweringContext& context)
        {
            context.require_arity(1, 1);
            const SushiBLAS::Tensor& x = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));

            SushiBLAS::BLASOps blas(context.recorder());
            SushiBLAS::NonLinearOps nonlinear(context.recorder());
            blas.copy(x, y);
            nonlinear.leaky_relu(y, static_cast<float>(context.node().attributes.get(0)));
            context.count(2);
            return Status();
        }

        Status lower_relu_backward(const LoweringContext& context)
        {
            return record_backward(context, &SushiBLAS::NonLinearOps::relu_backward);
        }

        Status lower_gelu_backward(const LoweringContext& context)
        {
            return record_backward(context, &SushiBLAS::NonLinearOps::gelu_backward);
        }

        Status lower_sigmoid_backward(const LoweringContext& context)
        {
            return record_backward(context, &SushiBLAS::NonLinearOps::sigmoid_backward);
        }

        Status lower_tanh_backward(const LoweringContext& context)
        {
            return record_backward(context, &SushiBLAS::NonLinearOps::tanh_backward);
        }

        Status lower_silu_backward(const LoweringContext& context)
        {
            return record_backward(context, &SushiBLAS::NonLinearOps::silu_backward);
        }

        Status lower_softplus_backward(const LoweringContext& context)
        {
            return record_backward(context, &SushiBLAS::NonLinearOps::softplus_backward);
        }

        Status lower_leaky_relu_backward(const LoweringContext& context)
        {
            context.require_arity(2, 1);
            const SushiBLAS::Tensor& dy = context.input(0);
            const SushiBLAS::Tensor& primal = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& dx, context.output(0));

            SushiBLAS::NonLinearOps nonlinear(context.recorder());
            nonlinear.leaky_relu_backward(dy, primal, dx,
                                          static_cast<float>(context.node().attributes.get(0)));
            context.count(1);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
