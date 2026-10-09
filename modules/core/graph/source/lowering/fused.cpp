/**************************************************************************/
/* fused.cpp                                                              */
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
 * @file fused.cpp
 * @brief Implements the lowering rows of the fused operations.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/fused.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/blas/dispatch.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/elementwise/dispatch.hpp>
#include <SushiBLAS/nonlinear/nonlinear.hpp>
#include <SushiBLAS/support/safe_math.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Adds a per-column bias to the GEMM's result. */
            struct BiasEpilogue
            {
                /// @brief The bias elements, one per column.
                const void* bias;

                /** @brief Returns @p value plus the bias of column @p col. */
                template<typename T>
                T operator()(T value, std::int64_t, std::int64_t col) const
                {
                    return value + static_cast<const T*>(bias)[col];
                }
            };

            /** @brief Adds a per-column bias, then SushiBLAS's ReLU. */
            struct BiasReluEpilogue
            {
                /// @brief The bias elements, one per column.
                const void* bias;

                /** @brief Returns the ReLU of @p value plus the bias of column @p col. */
                template<typename T>
                T operator()(T value, std::int64_t, std::int64_t col) const
                {
                    const T y = value + static_cast<const T*>(bias)[col];
                    return SushiBLAS::Dispatch::greater_than_zero(y) ? y : T(0);
                }
            };

            /** @brief Adds a per-column bias, then SushiBLAS's logistic sigmoid. */
            struct BiasSigmoidEpilogue
            {
                /// @brief The bias elements, one per column.
                const void* bias;

                /** @brief Returns the sigmoid of @p value plus the bias of column @p col. */
                template<typename T>
                T operator()(T value, std::int64_t, std::int64_t col) const
                {
                    const T y = value + static_cast<const T*>(bias)[col];
                    return T(1) / (T(1) + SushiBLAS::Dispatch::safe_exp(-y));
                }
            };

            /** @brief Adds a per-column bias, then SushiBLAS's tanh. */
            struct BiasTanhEpilogue
            {
                /// @brief The bias elements, one per column.
                const void* bias;

                /** @brief Returns the tanh of @p value plus the bias of column @p col. */
                template<typename T>
                T operator()(T value, std::int64_t, std::int64_t col) const
                {
                    return SushiBLAS::Dispatch::safe_tanh(
                        value + static_cast<const T*>(bias)[col]);
                }
            };

            /** @brief Adds a per-column bias, stores the sum in a second buffer, then GELU. */
            struct BiasGeluSpillEpilogue
            {
                /// @brief The bias elements, one per column.
                const void* bias;

                /// @brief The buffer that receives the pre-activation.
                void* aux;

                /// @brief Element distance between two rows of the buffer.
                std::int64_t aux_row_stride;

                /// @brief Element distance between two columns of the buffer.
                std::int64_t aux_col_stride;

                /** @brief Stores the biased @p value at (@p row, @p col) and returns its GELU. */
                template<typename T>
                T operator()(T value, std::int64_t row, std::int64_t col) const
                {
                    const T y = value + static_cast<const T*>(bias)[col];

                    static_cast<T*>(aux)[row * aux_row_stride + col * aux_col_stride] = y;

                    // The constants are those of the SushiBLAS GELU kernel.
                    const T y3 = y * y * y;
                    const T inner = T(0.7978845608028654) * (y + T(0.044715) * y3);
                    return T(0.5) * y * (T(1) + SushiBLAS::Dispatch::safe_tanh(inner));
                }
            };

            /** @brief Names the in-place activation a fused GEMM row ends in. */
            using ActivationKernel =
                SushiBLAS::TaskHandle (SushiBLAS::NonLinearOps::*)(SushiBLAS::Tensor&);

            /**
             * @brief Records one GEMM node in the form the fusion selector chooses.
             * @param form The form the node matched; a member of the GEMM_BIAS family.
             * @param identity The operation the whole-match task is recorded under.
             * @param tail The activation after the bias, or null when @p form has none.
             * @param make_epilogue Returns the epilogue of @p form from `(bias, aux)`; aux is
             *        null unless the form spills.
             */
            template<typename MakeEpilogue>
            [[nodiscard]] Status record_fused_gemm(const LoweringContext& context, FusedForm form,
                                                   const Ops::OpName& identity,
                                                   ActivationKernel tail,
                                                   MakeEpilogue&& make_epilogue)
            {
                const bool spills = fused_form_spills(form);
                context.require_arity(3, spills ? 2u : 1u);

                const Value& result = context.output_value(0);
                const Shape& out_shape = result.shape;
                const DType dtype = result.dtype;

                // The epilogue indexes one row and one column, so the result is unbatched.
                SA_FATAL_IF(out_shape.rank() != 2,
                            "{} (node {}) needs a rank-2 result; got {}",
                            context.name(), context.id(), out_shape.to_string());

                if (spills)
                {
                    const Value& spilled = context.output_value(1);
                    SA_FATAL_IF(spilled.shape != out_shape,
                                "{} (node {}) spills its pre-activation into a {} buffer, but its "
                                "result is {}; the two are the same tensor written twice",
                                context.name(), context.id(),
                                spilled.shape.to_string(), out_shape.to_string());
                    SA_FATAL_IF(spilled.dtype != dtype,
                                "{} (node {}) spills its pre-activation as {} but computes in {}; "
                                "the epilogue stores the value it is holding, in the element type "
                                "the GEMM was instantiated for",
                                context.name(), context.id(),
                                dtype_name(spilled.dtype), dtype_name(dtype));
                }

                const SushiBLAS::Tensor& a = context.input(0);
                const SushiBLAS::Tensor& b = context.input(1);
                const SushiBLAS::Tensor& bias = context.input(2);
                SA_TRY_ASSIGN(SushiBLAS::Tensor& c, context.output(0));

                SushiBLAS::Tensor* aux = nullptr;
                if (spills)
                {
                    SA_TRY_ASSIGN(SushiBLAS::Tensor& spill, context.output(1));
                    aux = &spill;
                }

                SushiBLAS::BLASOps blas(context.recorder());
                SushiBLAS::NonLinearOps nonlinear(context.recorder());

                const Node& node = context.node();
                const auto transpose_a = transpose_of(node.attributes.flag(0));
                const auto transpose_b = transpose_of(node.attributes.flag(1));

                FusionQuery query;
                query.matched = form;
                query.dtype = dtype;
                query.rows = out_shape[0];
                query.columns = out_shape[1];
                query.operand_is_contiguous = bias.is_contiguous();

                const FusionChoice choice = context.select(query);
                const std::size_t before = context.emitted();
                std::size_t tasks = 0;

                // context.select() refuses every answer but form, GemmBias and None.
                if (choice.form == form)
                {
                    SushiBLAS::Dispatch::execute_gemm(
                        context.recorder(), a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                        identity.name.data(), identity.id,
                        std::forward<MakeEpilogue>(make_epilogue)(bias, aux));
                    tasks += 1;
                }
                else if (choice.form == FusedForm::GemmBias)
                {
                    SushiBLAS::Dispatch::execute_gemm(
                        context.recorder(), a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                        Ops::K_GEMM_BIAS.name.data(), Ops::K_GEMM_BIAS.id,
                        SushiBLAS::Dispatch::gemm_epilogue(BiasEpilogue{bias.data()}).reads(bias));
                    tasks += 1;
                }
                else
                {
                    SA_TRY_ASSIGN(SushiBLAS::Tensor& rows,
                                  context.ones(Shape{out_shape[0]}, dtype));

                    blas.gemm(a, b, c, transpose_a, transpose_b, 1.0, 0.0);
                    blas.ger(rows, bias, c, 1.0f);
                    tasks += 2;
                }

                // Only the whole match spills and folds the activation inside its epilogue.
                if (spills && choice.form != form)
                {
                    blas.copy(c, *aux);
                    tasks += 1;
                }

                if (tail != nullptr && choice.form != form)
                {
                    (nonlinear.*tail)(c);
                    tasks += 1;
                }

                context.count(tasks);
                context.close_selection(before);
                return Status();
            }
        } // namespace

        Status lower_gemm_bias(const LoweringContext& context)
        {
            return record_fused_gemm(
                context, FusedForm::GemmBias, Ops::K_GEMM_BIAS, nullptr,
                [](const SushiBLAS::Tensor& bias, SushiBLAS::Tensor*)
                {
                    return SushiBLAS::Dispatch::gemm_epilogue(BiasEpilogue{bias.data()})
                        .reads(bias);
                });
        }

        Status lower_gemm_bias_relu(const LoweringContext& context)
        {
            return record_fused_gemm(
                context, FusedForm::GemmBiasRelu, Ops::K_GEMM_BIAS_RELU,
                &SushiBLAS::NonLinearOps::relu,
                [](const SushiBLAS::Tensor& bias, SushiBLAS::Tensor*)
                {
                    return SushiBLAS::Dispatch::gemm_epilogue(BiasReluEpilogue{bias.data()})
                        .reads(bias);
                });
        }

        Status lower_gemm_bias_sigmoid(const LoweringContext& context)
        {
            return record_fused_gemm(
                context, FusedForm::GemmBiasSigmoid, Ops::K_GEMM_BIAS_SIGMOID,
                &SushiBLAS::NonLinearOps::sigmoid,
                [](const SushiBLAS::Tensor& bias, SushiBLAS::Tensor*)
                {
                    return SushiBLAS::Dispatch::gemm_epilogue(BiasSigmoidEpilogue{bias.data()})
                        .reads(bias);
                });
        }

        Status lower_gemm_bias_tanh(const LoweringContext& context)
        {
            return record_fused_gemm(
                context, FusedForm::GemmBiasTanh, Ops::K_GEMM_BIAS_TANH,
                &SushiBLAS::NonLinearOps::tanh,
                [](const SushiBLAS::Tensor& bias, SushiBLAS::Tensor*)
                {
                    return SushiBLAS::Dispatch::gemm_epilogue(BiasTanhEpilogue{bias.data()})
                        .reads(bias);
                });
        }

        Status lower_gemm_bias_gelu(const LoweringContext& context)
        {
            return record_fused_gemm(
                context, FusedForm::GemmBiasGelu, Ops::K_GEMM_BIAS_GELU,
                &SushiBLAS::NonLinearOps::gelu,
                [](const SushiBLAS::Tensor& bias, SushiBLAS::Tensor* aux)
                {
                    return SushiBLAS::Dispatch::gemm_epilogue(
                               BiasGeluSpillEpilogue{bias.data(), aux->data(),
                                                     aux->strides[0], aux->strides[1]})
                        .reads(bias)
                        .writes(*aux);
                });
        }

        Status lower_exp_sub_scale(const LoweringContext& context)
        {
            context.require_arity(2, 1);

            const Value& result = context.output_value(0);
            const Shape& out_shape = result.shape;

            SushiBLAS::Tensor& a = context.mutable_input(0);
            SushiBLAS::Tensor& b = context.mutable_input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& out, context.output(0));

            SushiBLAS::BLASOps blas(context.recorder());
            SushiBLAS::ElementwiseOps elementwise(context.recorder());

            // The fused arm multiplies by the float that the unfused arm hands to scal.
            const float alpha = static_cast<float>(context.node().attributes.get(0));

            FusionQuery query;
            query.matched = FusedForm::ExpSubScale;
            query.dtype = result.dtype;
            query.rows = out_shape.rank() > 0 ? out_shape[0] : 0;
            query.columns = out_shape.rank() > 1 ? out_shape[1] : 1;

            const FusionChoice choice = context.select(query);
            const std::size_t before = context.emitted();
            std::size_t tasks = 0;

            if (choice.form == FusedForm::None)
            {
                blas.copy(a, out);
                elementwise.exp(out);
                elementwise.sub(out, b, out);
                blas.scal(alpha, out);
                tasks = 4;
            }
            else
            {
                const std::array<SushiBLAS::Tensor*, 3> operands{&out, &a, &b};

                SushiBLAS::Dispatch::execute_zip_inplace(
                    context.recorder(), operands,
                    Ops::K_EXP_SUB_SCALE.name.data(), Ops::K_EXP_SUB_SCALE.id,
                    [alpha](auto& value, auto& logit, auto& target)
                    {
                        using T = std::decay_t<decltype(value)>;
                        value = (SushiBLAS::Dispatch::safe_exp(logit) - target)
                              * static_cast<T>(alpha);
                    });
                tasks = 1;
            }

            context.count(tasks);
            context.close_selection(before);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
