/**************************************************************************/
/* step_recording.hpp                                                     */
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
 * @file step_recording.hpp
 * @brief Declares StepRecording, the two recorders a step recorder records through.
 * @author Mustafa Garip
 */

#pragma once

namespace SushiBLAS
{
    class TaskRecorder;
} // namespace SushiBLAS

namespace SushiRuntime
{
    class Recorder;
} // namespace SushiRuntime

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Hands a step recorder the task recorder of a graph and the graph's own
         *        runtime recorder, which the task recorder records on.
         *
         * Holds both by address and owns neither; both outlive the recording.
         */
        class StepRecording final
        {
            public:
                /** @brief Takes @p tasks and @p graph, the recorder @p tasks was built on. */
                StepRecording(SushiBLAS::TaskRecorder& tasks,
                              SushiRuntime::Recorder& graph) noexcept
                    : tasks_(&tasks), graph_(&graph)
                {

                }

                /** @brief Returns the recorder of SushiBLAS operations and packs. */
                [[nodiscard]] SushiBLAS::TaskRecorder& tasks() const noexcept
                {
                    return *tasks_;
                }

                /** @brief Returns the runtime recorder of the graph, for scopes and remote nodes. */
                [[nodiscard]] SushiRuntime::Recorder& graph() const noexcept
                {
                    return *graph_;
                }

            private:
                /// @brief The task recorder; never null.
                SushiBLAS::TaskRecorder* tasks_;

                /// @brief The runtime recorder tasks_ records on; never null.
                SushiRuntime::Recorder* graph_;
        };

    } // namespace Graph
} // namespace SushiAI
