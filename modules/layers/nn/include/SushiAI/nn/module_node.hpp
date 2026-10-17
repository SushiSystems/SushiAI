/**************************************************************************/
/* module_node.hpp                                                        */
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
 * @file module_node.hpp
 * @brief Declares ModuleNode, which presents a compile-time layer as a one-in one-out node.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @brief Presents a Traceable layer that has output_shape(const Shape&) as a one-in
         *        one-out node.
         * @tparam L The layer type; a Module, a TraceModule or a ModalModule.
         */
        template<typename L>
        class ModuleNode final : public Node
        {
            public:
                static_assert(Traceable<L>,
                              "ModuleNode's argument needs a static fields() and one of three "
                              "forms: forward(const Graph::GraphBuilder&, Graph::ValueID), "
                              "forward(const Trace&, Graph::ValueID), or the pair trace_training "
                              "and trace_evaluation over (const Trace&, Graph::ValueID), each "
                              "returning exactly Graph::ValueID. Check the layer named in the "
                              "instantiation below, not this line.");
                static_assert(Detail::HasOutputShapeOfInput<L>::value,
                              "ModuleNode's argument needs output_shape(const Shape&) returning "
                              "exactly Shape, which maps the per-sample input shape to the shape "
                              "of the node's one port. Check the layer named in the "
                              "instantiation below, not this line.");

                /**
                 * @brief Wraps @p layer and declares the one port it produces over @p input.
                 * @param input The per-sample shape the layer reads, batch axis excluded.
                 * @return The node, or the failure of the layer's output_shape() over @p input.
                 */
                [[nodiscard]] static Result<std::unique_ptr<ModuleNode>> create(L layer,
                                                                                const Shape& input)
                {
                    SA_TRY_ASSIGN(const Shape produced, Detail::bridge_throwing([&layer, &input]()
                    {
                        return layer.output_shape(input);
                    }));

                    return std::unique_ptr<ModuleNode>(
                        new ModuleNode(std::move(layer), input, produced));
                }

                /**
                 * @brief Traces the layer in the form trace_layer() picks and appends its result.
                 * @return InvalidDescription unless @p inputs holds exactly one value the layer
                 *         accepts; @p outputs is unchanged then.
                 */
                [[nodiscard]] Status forward(const Trace& trace, ValueSpan inputs,
                                             std::vector<Graph::ValueID>& outputs) override
                {
                    SA_TRY(Detail::require_input_count(inputs, K_INPUT_COUNT));
                    const Result<Graph::ValueID> traced =
                        Detail::bridge_throwing([this, &trace, &inputs]()
                    {
                        return trace_layer(layer_, trace, inputs[0]);
                    });

                    // A failure the layer recorded is the cause of whatever it threw afterwards.
                    SA_TRY(trace.status());
                    SA_TRY(traced);

                    outputs.push_back(traced.value());
                    return Status();
                }

                /**
                 * @brief Traces the layer's incremental form over the shape create() was given.
                 * @return Unsupported for a layer without forward_incremental(), or
                 *         InvalidDescription unless @p inputs holds one value it accepts.
                 */
                [[nodiscard]] Status forward_incremental(
                    const Trace& trace, ValueSpan inputs,
                    std::vector<Graph::ValueID>& outputs) override
                {
                    if constexpr (IncrementalModule<L>)
                    {
                        SA_TRY(Detail::require_input_count(inputs, K_INPUT_COUNT));
                        const Result<Graph::ValueID> traced =
                            Detail::bridge_throwing([this, &trace, &inputs]()
                        {
                            return trace_layer_incremental(layer_, trace, inputs[0], input_);
                        });

                        // A failure the layer recorded caused whatever it threw afterwards.
                        SA_TRY(trace.status());
                        SA_TRY(traced);

                        outputs.push_back(traced.value());
                        return Status();
                    }
                    else
                    {
                        return Node::forward_incremental(trace, inputs, outputs);
                    }
                }

                /** @brief Returns one, the number of values a layer reads. */
                [[nodiscard]] std::size_t input_count() const noexcept override
                {
                    return K_INPUT_COUNT;
                }

                /** @brief Walks the layer's state in the order NN::walk_state() yields. */
                void walk_state(std::string_view prefix, const StateVisitor& visitor) override
                {
                    NN::walk_state(layer_, prefix, visitor);
                }

                /** @brief Lists the single port, named K_SINGLE_PORT_NAME. */
                [[nodiscard]] SushiRuntime::span<const Port> ports() const noexcept override
                {
                    return SushiRuntime::span<const Port>(&port_, 1);
                }

                /** @brief Returns the wrapped layer. */
                [[nodiscard]] L& layer() noexcept
                {
                    return layer_;
                }

                /** @brief Returns the wrapped layer, read-only. */
                [[nodiscard]] const L& layer() const noexcept
                {
                    return layer_;
                }

            private:
                /** @brief Holds the number of values a layer reads. */
                static constexpr std::size_t K_INPUT_COUNT = 1;

                /** @brief Binds the layer, the shape it reads and the shape it produces. */
                ModuleNode(L layer, const Shape& input, const Shape& produced)
                    : layer_(std::move(layer)),
                      input_(input),
                      port_{std::string(K_SINGLE_PORT_NAME), produced}
                {

                }

                /** @brief Holds the wrapped layer. */
                L layer_;

                /** @brief Holds the per-sample shape the layer reads, batch axis excluded. */
                Shape input_;

                /** @brief Holds the one port forward() appends to. */
                Port port_;
        };

    } // namespace NN
} // namespace SushiAI
