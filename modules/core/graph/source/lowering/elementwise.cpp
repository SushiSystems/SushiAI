/**************************************************************************/
/* elementwise.cpp                                                        */
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
 * @file elementwise.cpp
 * @brief Implements the lowering rows of the elementwise operations.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/elementwise.hpp>

#include <string_view>
#include <SushiBLAS/engine/blas.hpp>
#include <SushiBLAS/engine/logic/logic.hpp>
#include <SushiBLAS/engine/math/elementwise.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>
#include <SushiBLAS/engine/math/random.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /**
             * @brief Records one two-operand kernel of a view: result = kernel(first, second).
             * @tparam View The SushiBLAS view that owns the kernel.
             * @param context The node to lower; it has two operands and one result.
             * @param kernel The member of View that records the kernel.
             * @return Ok when the kernel is recorded.
             */
            template<typename View>
            [[nodiscard]] Status record_binary(
                const LoweringContext& context,
                SushiBLAS::TaskHandle (View::*kernel)(const SushiBLAS::Tensor&,
                                                      const SushiBLAS::Tensor&,
                                                      SushiBLAS::Tensor&))
            {
                context.require_arity(2, 1);
                const SushiBLAS::Tensor& first = context.input(0);
                const SushiBLAS::Tensor& second = context.input(1);
                SA_TRY_ASSIGN(SushiBLAS::Tensor& result, context.output(0));

                View view(context.recorder());
                (view.*kernel)(first, second, result);
                context.count(1);
                return Status();
            }

            /**
             * @brief Copies the operand into the result, then records one in-place kernel on it.
             * @tparam View The SushiBLAS view that owns the kernel.
             * @param context The node to lower; it has one operand and one result.
             * @param kernel The member of View that records the in-place kernel.
             * @return Ok when both tasks are recorded.
             */
            template<typename View>
            [[nodiscard]] Status record_copy_then(
                const LoweringContext& context,
                SushiBLAS::TaskHandle (View::*kernel)(SushiBLAS::Tensor&))
            {
                context.require_arity(1, 1);
                const SushiBLAS::Tensor& source = context.input(0);
                SA_TRY_ASSIGN(SushiBLAS::Tensor& result, context.output(0));

                SushiBLAS::BLASOps blas(context.recorder());
                blas.copy(source, result);

                View view(context.recorder());
                (view.*kernel)(result);
                context.count(2);
                return Status();
            }
        } // namespace

        Status lower_add(const LoweringContext& context)
        {
            return record_binary(context, &SushiBLAS::ElementwiseOps::add);
        }

        Status lower_sub(const LoweringContext& context)
        {
            return record_binary(context, &SushiBLAS::ElementwiseOps::sub);
        }

        Status lower_mul(const LoweringContext& context)
        {
            return record_binary(context, &SushiBLAS::ElementwiseOps::mul);
        }

        Status lower_div(const LoweringContext& context)
        {
            return record_binary(context, &SushiBLAS::ElementwiseOps::div);
        }

        Status lower_minimum(const LoweringContext& context)
        {
            return record_binary(context, &SushiBLAS::ElementwiseOps::min);
        }

        Status lower_maximum(const LoweringContext& context)
        {
            return record_binary(context, &SushiBLAS::ElementwiseOps::max);
        }

        Status lower_scale(const LoweringContext& context)
        {
            context.require_arity(1, 1);
            const SushiBLAS::Tensor& x = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));

            SushiBLAS::BLASOps blas(context.recorder());
            blas.copy(x, y);
            blas.scal(static_cast<float>(context.node().attributes.get(0)), y);
            context.count(2);
            return Status();
        }

        Status lower_dynamic_scale(const LoweringContext& context)
        {
            context.require_arity(2, 1);
            const SushiBLAS::Tensor& x = context.input(0);
            const SushiBLAS::Tensor& factor = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));

            SA_FAIL_IF(factor.num_elements != 1, StatusCode::InvalidDescription,
                       "DYNAMIC_SCALE (node {}) was given a {}-element factor; it must be one",
                       context.id(), factor.num_elements);

            SushiBLAS::BLASOps blas(context.recorder());
            blas.copy(x, y);

            // The factor is a declared read: the kernel reads it at execution, not at lowering.
            constexpr const char* K_TASK_NAME = "sushiai.dynamic_scale";
            static_assert(Ops::K_DYNAMIC_SCALE.name == std::string_view(K_TASK_NAME),
                          "the task name must be the string its OpID hashes");

            SushiBLAS::Dispatch::execute_unary_inplace(
                context.recorder(), y, K_TASK_NAME, Ops::K_DYNAMIC_SCALE.id,
                SushiBLAS::Dispatch::bind_reads(
                    [](auto v, auto f)
                    {
                        return v * f[0];
                    },
                    SushiBLAS::Dispatch::read_as_element(factor)));
            context.count(2);
            return Status();
        }

        Status lower_exp(const LoweringContext& context)
        {
            return record_copy_then(context, &SushiBLAS::ElementwiseOps::exp);
        }

        Status lower_abs(const LoweringContext& context)
        {
            return record_copy_then(context, &SushiBLAS::ElementwiseOps::abs);
        }

        Status lower_atan(const LoweringContext& context)
        {
            return record_copy_then(context, &SushiBLAS::ElementwiseOps::atan);
        }

        Status lower_less_equal(const LoweringContext& context)
        {
            return record_binary(context, &SushiBLAS::LogicOps::less_equal);
        }

        Status lower_select(const LoweringContext& context)
        {
            context.require_arity(3, 1);
            const SushiBLAS::Tensor& condition = context.input(0);
            const SushiBLAS::Tensor& when_true = context.input(1);
            const SushiBLAS::Tensor& when_false = context.input(2);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& result, context.output(0));

            SushiBLAS::LogicOps logic(context.recorder());
            logic.where(condition, when_true, when_false, result);
            context.count(1);
            return Status();
        }

        Status lower_zero(const LoweringContext& context)
        {
            context.require_arity(0, 1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& zeros, context.output(0));

            SushiBLAS::RandomOps random(context.recorder(), context.stream());
            random.constant(zeros, 0.0);
            context.count(1);
            return Status();
        }

        Status lower_constant(const LoweringContext& context)
        {
            context.require_arity(0, 1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& filled, context.output(0));

            SushiBLAS::RandomOps random(context.recorder(), context.stream());
            random.constant(filled, context.node().attributes.get(0));
            context.count(1);
            return Status();
        }

        Status lower_accumulate(const LoweringContext& context)
        {
            context.require_arity(2, 1);

            // The result shares the accumulator's storage, so the axpy below writes it in place.
            SA_TRY(context.alias_output(0, 0));
            SushiBLAS::Tensor& accumulator = context.mutable_input(0);
            const SushiBLAS::Tensor& delta = context.input(1);

            SushiBLAS::BLASOps blas(context.recorder());
            blas.axpy(1.0f, delta, accumulator);
            context.count(1);
            return Status();
        }

        Status lower_cast(const LoweringContext& context)
        {
            context.require_arity(1, 1);
            const SushiBLAS::Tensor& source = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& destination, context.output(0));

            SushiBLAS::ElementwiseOps elementwise(context.recorder());
            elementwise.convert(source, destination);
            context.count(1);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
