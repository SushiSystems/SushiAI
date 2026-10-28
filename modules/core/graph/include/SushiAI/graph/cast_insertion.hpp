/**************************************************************************/
/* cast_insertion.hpp                                                     */
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
 * @file cast_insertion.hpp
 * @brief Declares the pass that inserts type-conversion casts into graphs.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/error.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @struct CastOptions
         * @brief What the pass is allowed to change.
         */
        struct CastOptions
        {
            /**
             * @brief Determines each node's precision, or null for uniform.
             */
            const PrecisionPolicy* policy = nullptr;

            /**
             * @brief Holds values read by id whose element types must not change.
             */
            SushiRuntime::span<const ValueID> observed{};
        };

        /**
         * @struct InsertedCast
         * @brief One conversion the pass added.
         */
        struct InsertedCast
        {
            /** @brief The node whose dtype boundary this cast resolves. */
            NodeID node = K_NO_NODE;

            /** @brief That node's operation, in the graph as it arrived. */
            OpID op{};

            /** @brief The value read. */
            ValueID source = K_NO_VALUE;

            /** @brief The value written. */
            ValueID destination = K_NO_VALUE;

            /** @brief The source's element type. */
            DType from = DType::Float32;

            /** @brief The destination's element type. */
            DType to = DType::Float32;

            /**
             * @brief Whether this cast feeds an operand or widens a result.
             *
             * The two directions fail differently and are worth telling apart
             * in a report: an operand cast that is missing throws, a result
             * cast that is missing writes a low-precision answer into a
             * high-precision buffer and trains on it.
             */
            bool feeds_operand = true;
        };

        /**
         * @class CastReport
         * @brief Summarises the casts and value retitlings inserted by the pass.
         */
        class CastReport
        {
            public:
                /** @brief Every conversion added, in graph order. */
                SushiRuntime::span<const InsertedCast> casts() const noexcept
                {
                    return SushiRuntime::span<const InsertedCast>(casts_.data(), casts_.size());
                }

                /** @brief How many conversions were added. */
                std::size_t count() const noexcept { return casts_.size(); }

                /** @brief How many conversions into @p dtype were added. */
                std::size_t count_into(DType dtype) const noexcept;

                /** @brief How many nodes the graph had before the pass. */
                std::size_t nodes_before() const noexcept { return before_; }

                /** @brief How many it has after. */
                std::size_t nodes_after() const noexcept { return after_; }

                /** @brief How many values changed element type without changing id. */
                std::size_t retyped_values() const noexcept { return retyped_; }

                /** @brief The policy's name, as it named itself. */
                std::string_view policy() const noexcept { return policy_; }

                /** @brief A multi-line human-readable summary; short when nothing happened. */
                std::string to_string() const;

            private:
                friend Result<CastReport> insert_casts(Graph& graph, const CastOptions& options);

                std::vector<InsertedCast> casts_;
                std::size_t before_ = 0;
                std::size_t after_ = 0;
                std::size_t retyped_ = 0;
                std::string_view policy_{};
        };

        /**
         * @brief Rewrites @p graph so every node's operands match its computed dtype.
         * @param graph The graph to rewrite in place.
         * @param options The policy and preserved values.
         * @return The inserted conversions, or InvalidDescription, graph unchanged, for a
         *         non-float conversion or for operands in two layouts.
         * @pre @p graph is ordered, each node writes a value, each observed id is its own.
         */
        [[nodiscard]] Result<CastReport> insert_casts(Graph& graph,
                                                      const CastOptions& options = {});

        /**
         * @struct DTypeMismatch
         * @brief One place a graph disagrees with a precision policy.
         */
        struct DTypeMismatch
        {
            /** @brief The offending node. */
            NodeID node = K_NO_NODE;

            /** @brief Its operation. */
            OpID op{};

            /** @brief The operand or result that disagrees. */
            ValueID value = K_NO_VALUE;

            /** @brief That value's element type. */
            DType found = DType::Float32;

            /** @brief The element type the policy says the node computes in. */
            DType expected = DType::Float32;

            /** @brief Whether @c value is one of the node's results. */
            bool is_result = false;
        };

        /**
         * @brief Finds every node whose operands or results disagree with @p policy.
         * @param graph The graph to check.
         * @param policy The precision policy to conform to.
         * @return Disagreements found, or empty if invariant holds.
         */
        std::vector<DTypeMismatch> find_dtype_mismatches(const Graph& graph,
                                                         const PrecisionPolicy& policy);

        /** @brief A one-line description of @p mismatch, for a test failure or a report. */
        std::string describe(const DTypeMismatch& mismatch);

    } // namespace Graph
} // namespace SushiAI
