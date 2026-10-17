/**************************************************************************/
/* step.hpp                                                               */
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
 * @file step.hpp
 * @brief Declares Step, which traces one step of a session and runs it when it is read.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <SushiBLAS/engine/engine.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/scope_trace.hpp>
#include <SushiAI/eager/session.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/step_recorder.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @brief Owns the graph, the scopes and the flush points of one step.
         * @see modules/core/eager/README.md
         */
        class Step final
        {
            public:
                Step(const Step&) = delete;
                Step& operator=(const Step&) = delete;
                Step(Step&&) = delete;
                Step& operator=(Step&&) = delete;

                /** @brief Closes the step; work that has not run is dropped. */
                ~Step();

                /** @brief Returns the builder that emits into this step's graph. */
                [[nodiscard]] const Graph::GraphBuilder& builder() const noexcept
                {
                    return builder_;
                }

                /** @brief Returns the trace a Scope opens its name on. */
                [[nodiscard]] ScopeTrace& scopes() noexcept
                {
                    return scopes_;
                }

                /** @brief Returns the graph traced so far. */
                [[nodiscard]] const Graph::Graph& graph() const noexcept
                {
                    return trace_.graph;
                }

                /**
                 * @brief Returns this step's leaf for @p state, declaring it on first use.
                 * @return The leaf, or InvalidDescription for a state the session has not bound.
                 */
                [[nodiscard]] Result<Graph::ValueID> state(Graph::StateID state);

                /**
                 * @brief Declares an input leaf shaped like @p batch and feeds @p batch into it.
                 * @param batch Copied when the step runs; it must hold its values until then.
                 * @param name A label for diagnostics; must outlive the step.
                 */
                [[nodiscard]] Result<Graph::ValueID> input(const SushiBLAS::Tensor& batch,
                                                           std::string_view name = "input");

                /**
                 * @brief Feeds @p source into @p leaf, a leaf some other call declared.
                 * @param source Copied when the step runs; it must hold its values until then.
                 * @return InvalidDescription for a value that is not a leaf of this step.
                 */
                [[nodiscard]] Status feed(Graph::ValueID leaf, const SushiBLAS::Tensor& source);

                /**
                 * @brief Asks the step to leave @p value readable after it has run.
                 * @return InvalidDescription for an unknown value, Unsupported once the step ran.
                 */
                [[nodiscard]] Status keep(Graph::ValueID value);

                /**
                 * @brief Appends the gradients of @p loss for every trainable state traced.
                 * @param loss A one-element value of this step; kept readable.
                 * @return InvalidDescription when called twice or after the step ran, or the
                 *         refusal of the differentiation.
                 */
                [[nodiscard]] Status backward(Graph::ValueID loss);

                /**
                 * @brief Runs the step with @p update recorded behind every region.
                 * @param update Records work that reads Session::gradient(); it must outlive the
                 *        session's executor and replay unchanged.
                 */
                [[nodiscard]] Status apply(Graph::StepRecorder& update);

                /**
                 * @brief Returns the tensor of @p value, running the step first if it has not run.
                 * @return A tensor valid until the session's next step runs.
                 */
                [[nodiscard]] Result<const SushiBLAS::Tensor*> read(Graph::ValueID value);

                /**
                 * @brief Returns the one element of @p value as a double.
                 * @return InvalidDescription for a value of another size or element type.
                 */
                [[nodiscard]] Result<double> read_scalar(Graph::ValueID value);

                /** @brief Runs the step if it has not run. */
                [[nodiscard]] Status finish();

                /** @brief Reports whether the step has run. */
                [[nodiscard]] bool has_run() const noexcept
                {
                    return stage_ == Stage::Ran;
                }

            private:
                friend class Session;

                /** @brief Says how far the step has come. */
                enum class Stage : std::uint8_t
                {
                    Tracing,        /**< Forward nodes are being traced. */
                    Differentiated, /**< backward() has appended the gradients. */
                    Ran,            /**< The step has run; its kept values are readable. */
                };

                /** @brief Opens a step of @p session with an empty graph. */
                explicit Step(Session& session);

                /** @brief Returns Unsupported when a node was traced after the step was closed. */
                [[nodiscard]] Status require_no_late_node() const;

                /** @brief Seals the scopes if needed and has the session run the trace. */
                [[nodiscard]] Status flush();

                /// @brief The session this step belongs to.
                Session* session_;

                /// @brief What the step traced; the session reads it when the step runs.
                StepTrace trace_;

                /// @brief Emits into trace_.graph.
                Graph::GraphBuilder builder_;

                /// @brief Cuts the forward nodes into segments.
                ScopeTrace scopes_;

                /// @brief How far the step has come.
                Stage stage_ = Stage::Tracing;

                /// @brief The node count when the trace was closed for tracing.
                std::size_t closed_at_ = 0;
        };

    } // namespace Eager
} // namespace SushiAI
