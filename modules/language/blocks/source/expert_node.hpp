/**************************************************************************/
/* expert_node.hpp                                                        */
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
 * @file expert_node.hpp
 * @brief Defines the node that presents a block with experts through three ports.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/expert_trace.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace Detail
        {
            /**
             * @brief Presents a block that traces an ExpertTrace as a one-in node with the
             *        ports "output", K_BALANCE_PORT and K_USAGE_PORT.
             * @tparam Block Has trace_experts(trace, x) and trace_experts_incremental(trace,
             *         x, sample), each returning Result<ExpertTrace>, and a static fields().
             */
            template<typename Block>
            class ExpertNode final : public NN::Node
            {
                public:
                    /**
                     * @brief Wraps @p block and declares its three ports.
                     * @param input The per-sample [T, C] shape the block reads and returns.
                     * @param experts The extent of the usage port; at least 1.
                     * @return The node, or OutOfMemory.
                     */
                    [[nodiscard]] static Result<std::unique_ptr<NN::Node>> create(
                        Block block, const Shape& input, int64_t experts)
                    {
                        return SushiAI::Detail::guard_value(
                            SushiAI::Detail::SeamPhase::Allocate, [&]()
                        {
                            return std::unique_ptr<NN::Node>(
                                new ExpertNode(std::move(block), input, experts));
                        });
                    }

                    /**
                     * @brief Traces the block and appends its result, balance and usage.
                     * @return InvalidDescription unless @p inputs holds one value the block
                     *         accepts; @p outputs is unchanged then.
                     */
                    [[nodiscard]] Status forward(const NN::Trace& trace, NN::ValueSpan inputs,
                                                 std::vector<Graph::ValueID>& outputs) override
                    {
                        SA_TRY(NN::Detail::require_input_count(inputs, K_INPUT_COUNT));

                        return append(trace, block_.trace_experts(trace, inputs[0]), outputs);
                    }

                    /**
                     * @brief Traces the block's incremental form and appends its three values.
                     * @return InvalidDescription unless @p inputs holds one value the block
                     *         accepts; @p outputs is unchanged then.
                     */
                    [[nodiscard]] Status forward_incremental(
                        const NN::Trace& trace, NN::ValueSpan inputs,
                        std::vector<Graph::ValueID>& outputs) override
                    {
                        SA_TRY(NN::Detail::require_input_count(inputs, K_INPUT_COUNT));

                        return append(
                            trace, block_.trace_experts_incremental(trace, inputs[0], input_),
                            outputs);
                    }

                    /** @brief Walks the block's state in the order NN::walk_state() yields. */
                    void walk_state(std::string_view prefix,
                                    const NN::StateVisitor& visitor) override
                    {
                        NN::walk_state(block_, prefix, visitor);
                    }

                    /** @brief Returns one, the number of values the block reads. */
                    [[nodiscard]] std::size_t input_count() const noexcept override
                    {
                        return K_INPUT_COUNT;
                    }

                    /** @brief Lists the three ports: the result, the balance and the usage. */
                    [[nodiscard]] SushiRuntime::span<const NN::Port> ports() const noexcept override
                    {
                        return SushiRuntime::span<const NN::Port>(ports_.data(), ports_.size());
                    }

                    /** @brief Returns the wrapped block, read-only. */
                    [[nodiscard]] const Block& block() const noexcept
                    {
                        return block_;
                    }

                private:
                    /** @brief Holds the number of values the block reads. */
                    static constexpr std::size_t K_INPUT_COUNT = 1;

                    /** @brief Holds the number of ports. */
                    static constexpr std::size_t K_PORT_COUNT = 3;

                    /** @brief Binds the block and the shapes of its ports behind one axis. */
                    ExpertNode(Block block, const Shape& input, int64_t experts)
                        : block_(std::move(block)),
                          input_(input),
                          ports_{NN::Port{std::string(NN::K_SINGLE_PORT_NAME), input},
                                 NN::Port{std::string(K_BALANCE_PORT), Shape{1}},
                                 NN::Port{std::string(K_USAGE_PORT), Shape{experts}}}
                    {

                    }

                    /**
                     * @brief Appends the three values of @p traced in the order of ports().
                     * @return The failure a child recorded on @p trace, else that of @p traced.
                     */
                    [[nodiscard]] static Status append(const NN::Trace& trace,
                                                       const Result<ExpertTrace>& traced,
                                                       std::vector<Graph::ValueID>& outputs)
                    {
                        // A failure a child recorded is the cause of the one returned after it.
                        SA_TRY(trace.status());
                        SA_TRY(traced);

                        const ExpertTrace& values = traced.value();
                        return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate,
                                                      [&]() -> Status
                        {
                            outputs.reserve(outputs.size() + K_PORT_COUNT);
                            outputs.push_back(values.output);
                            outputs.push_back(values.balance);
                            outputs.push_back(values.usage);

                            return Status();
                        });
                    }

                    /** @brief Holds the wrapped block. */
                    Block block_;

                    /** @brief Holds the per-sample shape the block reads. */
                    Shape input_;

                    /** @brief Holds the ports in the order forward() appends to them. */
                    std::array<NN::Port, K_PORT_COUNT> ports_;
            };

        } // namespace Detail
    } // namespace Language::Blocks
} // namespace SushiAI
