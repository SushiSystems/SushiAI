/****************************************************************/
/* cast_insertion.hpp                                           */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
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
            const IPrecisionPolicy* policy = nullptr;

            /**
             * @brief Holds values read by id whose element types must not change.
             */
            SushiRuntime::span<const ValueId> observed{};
        };

        /**
         * @struct InsertedCast
         * @brief One conversion the pass added.
         */
        struct InsertedCast
        {
            /** @brief The node whose dtype boundary this cast resolves. */
            NodeId node = NO_NODE;

            /** @brief That node's operation, in the graph as it arrived. */
            OpID op{};

            /** @brief The value read. */
            ValueId source = NO_VALUE;

            /** @brief The value written. */
            ValueId destination = NO_VALUE;

            /** @brief The source's element type. */
            DType from = DType::FLOAT32;

            /** @brief The destination's element type. */
            DType to = DType::FLOAT32;

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
                friend CastReport insert_casts(Graph& graph, const CastOptions& options);

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
         * @return A summary of inserted conversions.
         * @throws Error If topologically unordered or conversion fails.
         */
        CastReport insert_casts(Graph& graph, const CastOptions& options = {});

        /**
         * @struct DTypeMismatch
         * @brief One place a graph disagrees with a precision policy.
         */
        struct DTypeMismatch
        {
            /** @brief The offending node. */
            NodeId node = NO_NODE;

            /** @brief Its operation. */
            OpID op{};

            /** @brief The operand or result that disagrees. */
            ValueId value = NO_VALUE;

            /** @brief That value's element type. */
            DType found = DType::FLOAT32;

            /** @brief The element type the policy says the node computes in. */
            DType expected = DType::FLOAT32;

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
                                                         const IPrecisionPolicy& policy);

        /** @brief A one-line description of @p mismatch, for a test failure or a report. */
        std::string describe(const DTypeMismatch& mismatch);

    } // namespace Graph
} // namespace SushiAI
