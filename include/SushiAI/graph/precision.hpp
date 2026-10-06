/**************************************************************************/
/* precision.hpp                                                          */
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
 * @file precision.hpp
 * @brief Declares precision policies for controlling operation compute types.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @struct PrecisionQuery
         * @brief One operation, described in the terms a precision policy reasons in.
         */
        struct PrecisionQuery
        {
            /** @brief Which operation this is; see SushiAI::Ops. */
            OpID op{};

            /**
             * @brief The element type the node's result carries today.
             *
             * The graph's own precision for this node, before any narrowing -
             * which is what a policy narrows *from*, so that a model traced in
             * fp64 is not silently dropped to fp32 by a policy whose author was
             * thinking about fp32 models.
             */
            DType declared = DType::FLOAT32;
        };

        /**
         * @struct PrecisionChoice
         * @brief What the operation should compute in, and why.
         */
        struct PrecisionChoice
        {
            /** @brief The element type every operand and result must agree on. */
            DType dtype = DType::FLOAT32;

            /** @brief A sentence for the report; a static string, never owned. */
            std::string_view reason{};
        };

        /**
         * @class IPrecisionPolicy
         * @brief Interface selecting execution element types per operation.
         *
         * Implementations must be idempotent and reproducible.
         */
        class IPrecisionPolicy
        {
            public:
                virtual ~IPrecisionPolicy() = default;

                /**
                 * @brief Chooses the element type @p query's operation computes in.
                 * @param query The operation and its declared precision.
                 * @return The element type and the reason for it.
                 */
                virtual PrecisionChoice choose(const PrecisionQuery& query) const = 0;

                /**
                 * @brief Returns a short identifier for reporting and diagnostics.
                 *
                 * @return Static string identifying the policy name.
                 */
                virtual std::string_view name() const noexcept = 0;
        };

        /**
         * @class UniformPrecisionPolicy
         * @brief Preserves declared precision across all graph operations.
         *
         * Returns declared data types directly, inserting zero conversion casts.
         */
        class UniformPrecisionPolicy final : public IPrecisionPolicy
        {
            public:
                PrecisionChoice choose(const PrecisionQuery& query) const override;

                std::string_view name() const noexcept override { return "fp32"; }
        };

        /**
         * @class MixedPrecisionPolicy
         * @brief Selects lower precision for compute-bound operations.
         *
         * Narrows matrix multiplications while keeping sensitive reductions wide.
         */
        class MixedPrecisionPolicy final : public IPrecisionPolicy
        {
            public:
                /**
                 * @brief Builds a policy narrowing the compute-bound ops to @p low.
                 * @param low The narrow element type; must be a real float.
                 * @throws Error If @p low is complex or integral.
                 */
                explicit MixedPrecisionPolicy(DType low = DType::HALF);

                /** @brief The narrow element type this policy narrows to. */
                DType low() const noexcept { return low_; }

                PrecisionChoice choose(const PrecisionQuery& query) const override;

                std::string_view name() const noexcept override { return "mixed"; }

            private:
                DType low_;
        };

        /**
         * @brief Checks whether an operation is narrowed under mixed precision.
         *
         * @param op Operation identifier to check.
         * @return True if op is narrowed; false if preserved at declared precision.
         */
        bool narrows_under_mixed_precision(OpID op) noexcept;

    } // namespace Graph
} // namespace SushiAI
