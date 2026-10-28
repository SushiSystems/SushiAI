/**************************************************************************/
/* test_graph.hpp                                                         */
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

/**
 * @file test_graph.hpp
 * @brief Declares TestGraph, a graph a test owns with the recorder that records on it.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @brief Owns one graph on the runtime of an engine and a recorder over it, so a test
         *        records on its own graph and runs it.
         *
         * The graph runs once; a test that records again builds a second TestGraph.
         */
        class TestGraph final
        {
            public:
                /**
                 * @brief Creates the graph and binds a recorder to it.
                 * @param engine Lends the runtime, the scratch pool and the mode; it must outlive
                 *               this object.
                 */
                explicit TestGraph(SushiBLAS::Engine& engine) noexcept
                    : late_counts_(engine.runtime()), graph_(Detail::create_graph(engine).value()),
                      recorder_(graph_.recorder(), engine.scratch_pool(), late_counts_,
                                engine.mode())
                {

                }

                TestGraph(const TestGraph&) = delete;
                TestGraph& operator=(const TestGraph&) = delete;
                TestGraph(TestGraph&&) = delete;
                TestGraph& operator=(TestGraph&&) = delete;
                ~TestGraph() = default;

                /** @brief Returns the recorder every operation of the test records through. */
                [[nodiscard]] SushiBLAS::TaskRecorder& recorder() noexcept
                {
                    return recorder_;
                }

                /**
                 * @brief Returns the runtime recorder of the graph, which recorder() records on.
                 */
                [[nodiscard]] SushiRuntime::Recorder& graph_recorder() noexcept
                {
                    return graph_.recorder();
                }

                /** @brief Runs every task recorded on the graph and returns when they finish. */
                [[nodiscard]] Status run() noexcept
                {
                    return Detail::run_once(graph_);
                }

            private:
                /// @brief Keeps the late-count tasks of graph_; declared first so it dies last.
                SushiBLAS::LateCountHold late_counts_;

                /// @brief The graph the test records on.
                SushiRuntime::Graph graph_;

                /// @brief The recorder over graph_.
                SushiBLAS::TaskRecorder recorder_;
        };
    } // namespace Test
} // namespace SushiAI
