/**************************************************************************/
/* expert_projection.hpp                                                  */
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
 * @file expert_projection.hpp
 * @brief Declares ExpertProjection, the second linear layer of every expert in one matrix.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Maps the gated hidden values of every expert back to the width, and sums. */
        struct ExpertProjection
        {
            /** @brief Holds the width of the result. */
            int64_t dim = 0;

            /** @brief Holds the hidden width of one expert. */
            int64_t hidden = 0;

            /** @brief Holds the number of experts. */
            int64_t experts = 0;

            /** @brief Holds [dim, experts * hidden]; expert e owns columns e * hidden onward. */
            NN::Parameter weight{};

            /** @brief Holds [experts, dim]; row e is the bias of expert e. */
            NN::Parameter bias{};

            /** @brief Lists the two parameters, under "weight" and "bias". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::field("weight", &ExpertProjection::weight),
                    NN::field("bias", &ExpertProjection::bias),
                };
            }

            /**
             * @brief Emits gated * weight^T + gate * bias: the sum over the experts of the
             *        projection of each, times its gate.
             * @param gated A [R, experts * hidden] value: each hidden value times its gate.
             * @param gate The [R, experts] gate, of the element type of @p gated.
             * @return The [R, dim] result's id, or InvalidDescription when an operand misfits.
             */
            [[nodiscard]] Result<Graph::ValueID> mix(const Graph::GraphBuilder& builder,
                                                     Graph::ValueID gated, Graph::ValueID gate);
        };

    } // namespace Language::Blocks
} // namespace SushiAI
