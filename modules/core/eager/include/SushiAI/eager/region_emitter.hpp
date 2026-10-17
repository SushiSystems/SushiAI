/**************************************************************************/
/* region_emitter.hpp                                                     */
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
 * @file region_emitter.hpp
 * @brief Declares RegionEmitter, which records one region's program on a runtime recorder.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <vector>

#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/replay_prelude.hpp>
#include <SushiAI/graph/tensor_pool.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @brief Lowers one region's program into the recorder it is handed.
         * @see modules/core/eager/README.md
         */
        class RegionEmitter final
        {
            public:
                /**
                 * @brief Binds the emitter to the program it records and the pool it fills.
                 * @param selector Picks the form of a fused node; null selects by device.
                 * @pre Every argument outlives the emitter.
                 */
                RegionEmitter(SushiBLAS::Engine& engine, const RegionProgram& program,
                              Graph::TensorPool& pool,
                              const Graph::FusionSelector* selector) noexcept;

                RegionEmitter(const RegionEmitter&) = delete;
                RegionEmitter& operator=(const RegionEmitter&) = delete;
                RegionEmitter(RegionEmitter&&) = delete;
                RegionEmitter& operator=(RegionEmitter&&) = delete;
                ~RegionEmitter() = default;

                /**
                 * @brief Records every node of the program on @p recorder, in node order.
                 * @pre The pool holds the tensor of every import and export of the program.
                 * @return Success, or the first refusal of the lowering or of the recorder.
                 */
                [[nodiscard]] Status emit(SushiRuntime::Recorder& recorder);

                /** @brief Returns how many times emit() has been called. */
                [[nodiscard]] std::size_t emit_count() const noexcept
                {
                    return emits_;
                }

                /** @brief Returns how many tasks the last emit() recorded. */
                [[nodiscard]] std::size_t operation_count() const noexcept
                {
                    return operations_;
                }

                /** @brief Returns the host work the last emit() asks for before each run. */
                [[nodiscard]] SushiRuntime::span<const Graph::ReplayPrelude> preludes()
                    const noexcept
                {
                    return SushiRuntime::span<const Graph::ReplayPrelude>(preludes_.data(),
                                                                          preludes_.size());
                }

                /** @brief Returns the selector's choice for each fused node of the last emit(). */
                [[nodiscard]] SushiRuntime::span<const Graph::FusionSelection> selections()
                    const noexcept
                {
                    return SushiRuntime::span<const Graph::FusionSelection>(selections_.data(),
                                                                            selections_.size());
                }

            private:
                /// @brief The engine whose scratch pool, mode and random stream the tasks use.
                SushiBLAS::Engine* engine_;

                /// @brief The program emit() records.
                const RegionProgram* program_;

                /// @brief The pool that holds the tensor of each value of the program.
                Graph::TensorPool* pool_;

                /// @brief The caller's selector, or null to select by device.
                const Graph::FusionSelector* selector_;

                /// @brief The replay preludes of the last emit(), in node order.
                std::vector<Graph::ReplayPrelude> preludes_;

                /// @brief The fusion decisions of the last emit(), in execution order.
                std::vector<Graph::FusionSelection> selections_;

                /// @brief The number of tasks the last emit() recorded.
                std::size_t operations_ = 0;

                /// @brief The number of emit() calls.
                std::size_t emits_ = 0;
        };

    } // namespace Eager
} // namespace SushiAI
