/**************************************************************************/
/* dropout.hpp                                                            */
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
 * @file dropout.hpp
 * @brief Declares Dropout, the layer that holds a drop rate and passes its input through.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @brief Holds a drop rate: the training form masks its input anew at each run of the
         *        step, and the evaluation form passes it through.
         */
        struct Dropout
        {
            /** @brief Holds the probability of dropping an element, in [0, 1). */
            double rate = 0.0;

            /** @brief Lists no state: the layer has no parameter and no buffer. */
            static constexpr auto fields()
            {
                return std::tuple<>{};
            }

            /**
             * @brief Emits K_DROPOUT over @p x, seeded by the trace's seed and the order of this
             *        site in the trace; a rate of 0 emits no node.
             * @return The masked value, or @p x at a rate of 0 and after trace.fail() when the
             *         rate is outside [0, 1) or the emitter refuses @p x.
             */
            Graph::ValueID trace_training(const Trace& trace, Graph::ValueID x);

            /**
             * @brief Returns @p x and emits no node, as dropout does outside training.
             * @return @p x, after trace.fail() when rate is outside [0, 1).
             */
            Graph::ValueID trace_evaluation(const Trace& trace, Graph::ValueID x);

            /** @brief Returns the per-sample shape either form produces, which is @p input. */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(ModalModule<Dropout>, "Dropout must satisfy the ModalModule contract");

        /**
         * @brief Builds a Dropout node from the required key "rate", a number in [0, 1).
         * @param inputs One per-sample shape of any rank.
         * @param context What to call the row in a diagnostic, e.g. "layer 0 'dropout'".
         * @return The node, or InvalidDescription when @p inputs or the key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_dropout_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace NN
} // namespace SushiAI
