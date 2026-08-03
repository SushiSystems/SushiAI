/**************************************************************************/
/* ops.hpp                                                                */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file ops.hpp
 * @brief The identity of every SushiAI operation, as compile-time constants.
 *
 * SushiRuntime identifies an operation by @c OpID, a 32-bit FNV-1a hash of a
 * name, and offers no registry to register into outside the distributed build.
 * So the compiler is the registry: every identity is a @c constexpr constant
 * declared here, and a @c static_assert at the bottom of this file proves the
 * whole set is collision-free. Thirty-two bits is not many — an accidental
 * collision between two ops would silently route one to the other's profiling
 * row, so it is checked rather than hoped for.
 *
 * Every name carries the @c sushiai. prefix so it cannot collide with
 * SushiBLAS's own identities (@c blas.lvl3.gemm and friends) or a future
 * sibling's.
 *
 * This header is part of the host lane: it pulls in no SYCL and can be
 * included by a device-free unit test.
 */

#include <span>
#include <array>
#include <cstddef>
#include <string_view>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiAI
{
    /** @brief The runtime's operation-identity type, aliased for brevity. */
    using OpID = SushiRuntime::Graph::OpID;

    namespace Ops
    {
        using namespace SushiRuntime::Graph::Literals;

        /**
         * @struct OpName
         * @brief One operation's identity: the string and its hash, kept together.
         *
         * Pairing them is what lets the uniqueness check report *which* two
         * names collided, and lets a diagnostic print a name for an id.
         */
        struct OpName
        {
            /** @brief The dotted, prefixed name; also what the profiler shows. */
            std::string_view name;

            /** @brief The hash of @c name, as SushiRuntime identifies it. */
            OpID id;
        };

        /** @brief Builds an OpName, hashing the name at compile time. */
        consteval OpName op(std::string_view n)
        {
            return OpName{n, OpID(OpID::hash_str(n.data(), n.size()))};
        }

        // --- Structural -----------------------------------------------------
        /** @brief A graph leaf: a parameter or an input, produced by nothing. */
        inline constexpr OpName LEAF = op("sushiai.leaf");
        /** @brief Accumulates a gradient contribution into an existing buffer. */
        inline constexpr OpName ACCUMULATE = op("sushiai.accumulate");

        // --- Linear algebra -------------------------------------------------
        /** @brief 2-D matrix product, transposition expressed by flags not views. */
        inline constexpr OpName MATMUL = op("sushiai.matmul");
        /** @brief Adds a row vector to every row of a matrix. */
        inline constexpr OpName BIAS_ADD = op("sushiai.bias_add");

        // --- Elementwise ----------------------------------------------------
        /** @brief Elementwise sum of two equally shaped tensors. */
        inline constexpr OpName ADD = op("sushiai.add");
        /** @brief Elementwise difference of two equally shaped tensors. */
        inline constexpr OpName SUB = op("sushiai.sub");
        /** @brief Elementwise product of two equally shaped tensors. */
        inline constexpr OpName MUL = op("sushiai.mul");
        /** @brief Multiplies every element by a host-side constant. */
        inline constexpr OpName SCALE = op("sushiai.scale");

        /** @brief Elementwise natural exponential, out of place. */
        inline constexpr OpName EXP = op("sushiai.exp");
        /** @brief An all-zero tensor, read by nothing upstream. */
        inline constexpr OpName ZERO = op("sushiai.zero");

        // --- Activations ----------------------------------------------------
        /** @brief Rectified linear unit. */
        inline constexpr OpName RELU = op("sushiai.relu");
        /** @brief Gaussian error linear unit, tanh approximation. */
        inline constexpr OpName GELU = op("sushiai.gelu");
        /** @brief Logistic sigmoid. */
        inline constexpr OpName SIGMOID = op("sushiai.sigmoid");
        /** @brief Hyperbolic tangent. */
        inline constexpr OpName TANH = op("sushiai.tanh");

        // --- Activation gradients -------------------------------------------
        // Each takes [dy, primal] and yields [dx]. Which tensor "primal" is
        // differs by activation and is not a detail a caller may guess:
        // relu and gelu need the forward INPUT, sigmoid and tanh need the
        // forward OUTPUT. That is SushiBLAS's convention, mirrored here so the
        // lowering is a straight one-to-one map with nothing to reinterpret.

        /** @brief Gradient of RELU; second operand is the forward input. */
        inline constexpr OpName RELU_BACKWARD = op("sushiai.relu_backward");
        /** @brief Gradient of GELU; second operand is the forward input. */
        inline constexpr OpName GELU_BACKWARD = op("sushiai.gelu_backward");
        /** @brief Gradient of SIGMOID; second operand is the forward output. */
        inline constexpr OpName SIGMOID_BACKWARD = op("sushiai.sigmoid_backward");
        /** @brief Gradient of TANH; second operand is the forward output. */
        inline constexpr OpName TANH_BACKWARD = op("sushiai.tanh_backward");

        // --- Losses and normalisation ---------------------------------------
        /** @brief Row-wise log-softmax, computed max-shifted for stability. */
        inline constexpr OpName LOG_SOFTMAX = op("sushiai.log_softmax");
        /** @brief Mean negative log-likelihood against integer-valued targets. */
        inline constexpr OpName NLL_LOSS = op("sushiai.nll_loss");

        // --- Reductions -----------------------------------------------------
        /** @brief Sums a matrix along its rows, yielding one value per column. */
        inline constexpr OpName SUM_ROWS = op("sushiai.sum_rows");

        /**
         * @brief Every identity declared above, in one enumerable table.
         *
         * The uniqueness proof reads this, and so does the diagnostic that
         * turns an OpID back into a printable name.
         */
        inline constexpr std::array ALL = {
            LEAF, ACCUMULATE,
            MATMUL, BIAS_ADD,
            ADD, SUB, MUL, SCALE, EXP, ZERO,
            RELU, GELU, SIGMOID, TANH,
            RELU_BACKWARD, GELU_BACKWARD, SIGMOID_BACKWARD, TANH_BACKWARD,
            LOG_SOFTMAX, NLL_LOSS,
            SUM_ROWS,
        };

        /**
         * @brief Reports whether every identity in @p table hashes distinctly.
         *
         * Quadratic, but it runs at compile time over a table of a few dozen
         * entries, so the simple form is the right one.
         *
         * @param table The identities to check.
         * @return True when no two entries share an OpID.
         */
        consteval bool all_ids_unique(std::span<const OpName> table)
        {
            for (std::size_t i = 0; i < table.size(); ++i)
                for (std::size_t j = i + 1; j < table.size(); ++j)
                    if (table[i].id == table[j].id)
                        return false;
            return true;
        }

        static_assert(all_ids_unique(std::span<const OpName>(ALL.data(), ALL.size())),
                      "Two SushiAI operation names hash to the same 32-bit OpID. "
                      "Rename one of them - a collision would silently merge their "
                      "profiling rows and confuse any downstream dispatch on id.");

        /**
         * @brief The declared name behind an OpID, for diagnostics and reports.
         * @param id The identity to look up.
         * @return The matching name, or "sushiai.unknown" when @p id is not ours.
         */
        constexpr std::string_view name_of(OpID id) noexcept
        {
            for (const OpName& entry : ALL)
                if (entry.id == id) return entry.name;
            return "sushiai.unknown";
        }

    } // namespace Ops
} // namespace SushiAI
