/**************************************************************************/
/* buffer.hpp                                                             */
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
 * @file buffer.hpp
 * @brief Declares Buffer, a tensor of model state that no optimizer moves.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <utility>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @brief Describes a tensor of model state that no optimizer moves.
         *
         * define() draws the identity, so a defined buffer always holds one. A copy is new
         * state with its own identity; a move carries the identity away and leaves the source
         * undefined. A buffer that is defined and never read adds no leaf to a graph.
         */
        class Buffer final
        {
            public:
                /** @brief Starts undefined and without an identity. */
                Buffer() noexcept = default;

                /** @brief Copies the description of @p other and draws a new identity for it. */
                Buffer(const Buffer& other) noexcept
                    : shape_(other.shape_),
                      initial_value_(other.initial_value_),
                      dtype_(other.dtype_),
                      defined_(other.defined_)
                {
                    draw_identity_if_defined();
                }

                /** @brief Replaces this buffer with new state described as @p other is. */
                Buffer& operator=(const Buffer& other) noexcept
                {
                    if (this == &other)
                        return *this;

                    shape_ = other.shape_;
                    identity_ = Graph::StateIdentity();
                    initial_value_ = other.initial_value_;
                    dtype_ = other.dtype_;
                    defined_ = other.defined_;
                    draw_identity_if_defined();
                    return *this;
                }

                /** @brief Takes the description and the identity of @p other, left undefined. */
                Buffer(Buffer&& other) noexcept
                    : shape_(other.shape_),
                      identity_(std::move(other.identity_)),
                      initial_value_(other.initial_value_),
                      dtype_(other.dtype_),
                      defined_(other.defined_)
                {
                    other.defined_ = false;
                }

                /** @brief Takes the description and the identity of @p other, left undefined. */
                Buffer& operator=(Buffer&& other) noexcept
                {
                    if (this == &other)
                        return *this;

                    shape_ = other.shape_;
                    identity_ = std::move(other.identity_);
                    initial_value_ = other.initial_value_;
                    dtype_ = other.dtype_;
                    defined_ = other.defined_;
                    other.defined_ = false;
                    return *this;
                }

                /** @brief Releases nothing; a buffer owns no tensor. */
                ~Buffer() = default;

                /**
                 * @brief Fixes the shape, element type and starting value, and draws the identity.
                 * @param shape The buffer's extents.
                 * @param initial_value The value every element holds before the first update.
                 * @return InvalidDescription when the buffer is already defined and one of the
                 *         three arguments differs; the first description is kept then.
                 */
                [[nodiscard]] Status define(Shape shape, double initial_value,
                                            DType dtype = DType::FLOAT32)
                {
                    if (defined_)
                    {
                        SA_FAIL_IF(shape != shape_ || dtype != dtype_ ||
                                       initial_value != initial_value_,
                                   StatusCode::InvalidDescription,
                                   "Buffer was defined as {} {} filled with {} and is now defined "
                                   "as {} {} filled with {}",
                                   shape_.to_string(), dtype_name(dtype_), initial_value_,
                                   shape.to_string(), dtype_name(dtype), initial_value);
                        return Status();
                    }

                    shape_ = shape;
                    dtype_ = dtype;
                    initial_value_ = initial_value;
                    defined_ = true;
                    draw_identity_if_defined();
                    return Status();
                }

                /**
                 * @brief Declares this buffer's leaf in the builder's graph, once per graph.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @return The leaf of this buffer in builder.graph().
                 * @pre define() has been called; an undefined buffer is fatal.
                 */
                Graph::ValueId read(const Graph::GraphBuilder& builder, std::string_view debug_name)
                {
                    SA_FATAL_IF(!defined_,
                                "Buffer '{}' is read before define() has fixed its shape",
                                debug_name);

                    Graph::Graph& graph = builder.graph();
                    const Graph::ValueId held = graph.state_leaf(identity_.peek());
                    if (held != Graph::NO_VALUE)
                        return held;

                    const Graph::ValueId leaf = builder.buffer(shape_, debug_name, dtype_);
                    graph.bind_state_leaf(identity_.peek(), leaf);
                    return leaf;
                }

                /** @brief Returns the leaf of this buffer in @p graph, or NO_VALUE when none. */
                [[nodiscard]] Graph::ValueId id_in(const Graph::Graph& graph) const noexcept
                {
                    return graph.state_leaf(identity_.peek());
                }

                /** @brief Returns this buffer's identity, or K_NO_STATE before define(). */
                [[nodiscard]] Graph::StateId state() const noexcept
                {
                    return identity_.peek();
                }

                /** @brief Reports whether define() has been called. */
                [[nodiscard]] bool declared() const noexcept
                {
                    return defined_;
                }

                /** @brief Returns the extents define() fixed. */
                [[nodiscard]] const Shape& shape() const noexcept
                {
                    return shape_;
                }

                /** @brief Returns the element type define() fixed. */
                [[nodiscard]] DType dtype() const noexcept
                {
                    return dtype_;
                }

                /** @brief Returns the value every element holds before the first update. */
                [[nodiscard]] double initial_value() const noexcept
                {
                    return initial_value_;
                }

            private:
                /** @brief Draws the identity of a defined buffer that holds none. */
                void draw_identity_if_defined() noexcept
                {
                    if (defined_)
                        static_cast<void>(identity_.get());
                }

                /** @brief Holds the extents define() fixed. */
                Shape shape_{};

                /** @brief Names this buffer across graphs; drawn exactly when defined_ holds. */
                Graph::StateIdentity identity_{};

                /** @brief Holds the value every element starts at. */
                double initial_value_ = 0.0;

                /** @brief Holds the element type define() fixed. */
                DType dtype_ = DType::FLOAT32;

                /** @brief Records that define() has been called. */
                bool defined_ = false;
        };

    } // namespace NN
} // namespace SushiAI
