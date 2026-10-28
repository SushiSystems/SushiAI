/**************************************************************************/
/* model_graph.hpp                                                        */
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
 * @file model_graph.hpp
 * @brief Declares ModelGraph, the container of named nodes, and NodeRef, which names a port.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @brief Names one port of one node of a ModelGraph, or the graph's input.
         *
         * Only a ModelGraph makes one. A ref is an index pair and holds no pointer, so it
         * stays valid when its graph moves.
         */
        class NodeRef final
        {
            public:
                /**
                 * @brief Returns the index of the node in the order it was added.
                 * @pre !is_input(); the input is not a node and has no index.
                 */
                [[nodiscard]] constexpr std::size_t node() const noexcept
                {
                    return node_;
                }

                /** @brief Returns the index of the port in the node's ports(); 0 for the input. */
                [[nodiscard]] constexpr std::size_t port() const noexcept
                {
                    return port_;
                }

                /** @brief Reports whether this ref names the graph's input. */
                [[nodiscard]] constexpr bool is_input() const noexcept
                {
                    return node_ == K_INPUT;
                }

                /** @brief Reports whether @p left and @p right name the same port. */
                [[nodiscard]] friend constexpr bool operator==(NodeRef left,
                                                               NodeRef right) noexcept
                {
                    return left.node_ == right.node_ && left.port_ == right.port_;
                }

                /** @brief Reports whether @p left and @p right name different ports. */
                [[nodiscard]] friend constexpr bool operator!=(NodeRef left,
                                                               NodeRef right) noexcept
                {
                    return !(left == right);
                }

            private:
                friend class ModelGraph;

                /** @brief Marks the ref of the graph's input; no node has this index. */
                static constexpr std::size_t K_INPUT = static_cast<std::size_t>(-1);

                /** @brief Binds the two indices. */
                constexpr NodeRef(std::size_t node, std::size_t port) noexcept
                    : node_(node), port_(port)
                {

                }

                /** @brief Holds the node's index, or K_INPUT. */
                std::size_t node_;

                /** @brief Holds the port's index in the node's ports(). */
                std::size_t port_;
        };

        /**
         * @brief Holds named nodes that read earlier nodes, and the outputs the model publishes.
         *
         * A node reads only what was added before it, so the nodes are in topological order
         * and no cycle can be written. The graph has one input and owns its nodes.
         */
        class ModelGraph final
        {
            public:
                /**
                 * @brief Starts a graph with no node over one input.
                 * @param input_sample The per-sample shape of the input, batch axis excluded.
                 */
                explicit ModelGraph(Shape input_sample) noexcept;

                /** @brief Takes the nodes and outputs of @p other, which is left with none. */
                ModelGraph(ModelGraph&& other) noexcept = default;

                /** @brief Replaces this graph with the nodes and outputs of @p other. */
                ModelGraph& operator=(ModelGraph&& other) noexcept = default;

                /** @brief Refuses a copy; each node has one owner. */
                ModelGraph(const ModelGraph&) = delete;

                /** @brief Refuses a copy assignment; each node has one owner. */
                ModelGraph& operator=(const ModelGraph&) = delete;

                /** @brief Releases the nodes. */
                ~ModelGraph() = default;

                /** @brief Returns the ref of the graph's input. */
                [[nodiscard]] NodeRef input() const noexcept;

                /**
                 * @brief Appends @p node under @p name, reading @p from in that order.
                 * @param name Non-empty, without '.', and not yet in the graph; it prefixes the
                 *             node's state.
                 * @param node The node to own; fatal when null, portless, or with a repeated port.
                 * @return The node's port 0, or InvalidDescription for a refused name, a foreign
                 *         source or a wrong source count; the graph is unchanged then.
                 */
                [[nodiscard]] Result<NodeRef> add(std::string name, std::unique_ptr<Node> node,
                                                  SushiRuntime::span<const NodeRef> from) noexcept;

                /**
                 * @brief Appends a Traceable layer that reads one value, wrapped in a ModuleNode.
                 * @tparam L A layer with output_shape(const Shape&); see ModuleNode.
                 * @return The node's one port, the failure the node form of add() returns, or
                 *         the failure of the layer's output_shape() over the source's shape.
                 */
                template<typename L>
                [[nodiscard]] Result<NodeRef> add(std::string name, L layer, NodeRef from) noexcept;

                /**
                 * @brief Selects the port named @p name of the node @p of belongs to.
                 * @return InvalidDescription listing the node's ports when none has that name,
                 *         or when @p of is the input or is not held by this graph.
                 */
                [[nodiscard]] Result<NodeRef> port(NodeRef of, std::string_view name) const noexcept;

                /**
                 * @brief Publishes the value @p from names as the model output @p name.
                 * @return InvalidDescription for an empty or duplicate name or an unknown source.
                 */
                [[nodiscard]] Status output(std::string name, NodeRef from) noexcept;

                /**
                 * @brief Publishes every port of the node @p of belongs to, under the port's name.
                 * @return InvalidDescription when a name is taken, or @p of is the input or
                 *         unknown; no port is published then.
                 */
                [[nodiscard]] Status output_ports(NodeRef of) noexcept;

                /**
                 * @brief Traces every node in the order added and returns the published outputs.
                 * @param input A value of trace.builder().graph(); any other id is fatal.
                 * @return InvalidDescription when @p input or a traced port differs from its
                 *         declared per-sample shape, a node refuses its inputs, or no output
                 *         is published. Nodes traced before the failure stay in the graph.
                 */
                [[nodiscard]] Result<ModelOutputs> trace(const Trace& trace, Graph::ValueID input) noexcept;

                /**
                 * @brief Traces the graph and returns its single published output.
                 * @return The output's id, or @p input after trace.fail() when trace() fails
                 *         or the graph does not publish exactly one output.
                 */
                Graph::ValueID forward(const Trace& trace, Graph::ValueID input) noexcept;

                /**
                 * @brief Traces the incremental form of every node in the order added; no
                 *        per-sample shape is checked, as a slice has the shape its node gives it.
                 * @param trace A carrying trace.
                 * @param input The newest slice of the model input; another graph's id is fatal.
                 * @return The published outputs, Unsupported naming the first node that has no
                 *         incremental form, or the failure of a node.
                 */
                [[nodiscard]] Result<ModelOutputs> trace_incremental(const Trace& trace,
                                                                     Graph::ValueID input) noexcept;

                /**
                 * @brief Traces the incremental form and returns its single published output.
                 * @param sample Not read; the graph was started over its own input shape.
                 * @return The output's id, or @p input after trace.fail() when
                 *         trace_incremental() fails or one output is not what is published.
                 */
                Graph::ValueID forward_incremental(const Trace& trace, Graph::ValueID input,
                                                   const Shape& sample) noexcept;

                /**
                 * @brief Returns the per-sample shape forward() produces.
                 * @param input The shape of the value read; forward() refuses any but the
                 *              shape the graph was started over.
                 * @return The shape of the single published output, or InvalidDescription
                 *         naming the output count when the graph does not publish exactly one.
                 */
                Result<Shape> output_shape(const Shape& input) const noexcept;

                /**
                 * @brief Hands the state of every node to @p visitor, in the order added.
                 * @param prefix Dotted prefix; a node's state hangs under "<prefix>.<name>".
                 */
                void walk_state(std::string_view prefix, const StateVisitor& visitor) noexcept;

                /** @brief Lists no state of its own; the nodes hold all of it. */
                static constexpr auto fields() noexcept
                {
                    return std::tuple<>{};
                }

                /** @brief Returns how many nodes the graph holds. */
                [[nodiscard]] std::size_t size() const noexcept;

                /**
                 * @brief Returns the per-sample shape of the value @p ref names.
                 * @pre @p ref came from this graph; an unknown ref is fatal.
                 */
                [[nodiscard]] const Shape& shape_of(NodeRef ref) const noexcept;

                /**
                 * @brief Returns the node at @p index.
                 * @pre index < size(); anything else is fatal.
                 */
                [[nodiscard]] Node& node(std::size_t index) const noexcept;

            private:
                /** @brief Holds one node with its name and what it reads. */
                struct Entry
                {
                    std::string name;             /**< The name add() was given. */
                    std::unique_ptr<Node> node;   /**< The node; never null. */
                    std::vector<NodeRef> sources; /**< What the node reads, in order. */
                    std::size_t first_value = 0;  /**< Ports of all earlier nodes. */
                };

                /** @brief Pairs a published output's name with the port it reads. */
                using Output = std::pair<std::string, NodeRef>;

                /** @brief Names the entry of Node that one walk of the nodes calls. */
                using NodeForm = Status (Node::*)(const Trace&, ValueSpan,
                                                  std::vector<Graph::ValueID>&);

                /** @brief States whether a walk compares each port with its declared shape. */
                enum class PortCheck : std::uint8_t
                {
                    PerSample, /**< A port is its declared shape behind one batch axis. */
                    None,      /**< A port has the shape its node gives it. */
                };

                /**
                 * @brief Calls @p form on every node in the order added and publishes.
                 * @pre @p input is a value of trace.builder().graph().
                 * @return The published outputs, or the failure of a node or of a port check.
                 */
                [[nodiscard]] Result<ModelOutputs> trace_nodes(const Trace& trace,
                                                               Graph::ValueID input,
                                                               NodeForm form, PortCheck check);

                /**
                 * @brief Checks that @p ref is the input or a port of a node this graph holds.
                 * @return InvalidDescription stating the index and the count it exceeds.
                 */
                [[nodiscard]] Status require_source(NodeRef ref) const;

                /**
                 * @brief Checks that @p name can name a new node of this graph.
                 * @return InvalidDescription for an empty name, one that contains '.', or one
                 *         a node of this graph has.
                 */
                [[nodiscard]] Status require_free_name(std::string_view name) const;

                /** @brief Reports whether an output named @p name is published. */
                [[nodiscard]] bool has_output(std::string_view name) const noexcept;

                /** @brief Returns the port count of every node together. */
                [[nodiscard]] std::size_t port_count() const noexcept;

                /** @brief Holds the per-sample shape of the input. */
                Shape input_sample_;

                /** @brief Holds the nodes in the order added; each reads earlier entries only. */
                std::vector<Entry> entries_;

                /** @brief Holds the published outputs in the order published; names are unique. */
                std::vector<Output> outputs_;
        };

        static_assert(TraceModule<ModelGraph>,
                      "ModelGraph must satisfy the TraceModule contract so it nests as a layer");

        template<typename L>
        Result<NodeRef> ModelGraph::add(std::string name, L layer, NodeRef from) noexcept
        {
            SA_TRY(require_free_name(name));
            SA_TRY(SA_ANNOTATE(require_source(from), "node '{}'", name));

            Result<std::unique_ptr<ModuleNode<L>>> made =
                ModuleNode<L>::create(std::move(layer), shape_of(from));
            SA_TRY(SA_ANNOTATE(made.status(), "node '{}'", name));

            return add(std::move(name), std::unique_ptr<Node>(std::move(made).value()),
                       SushiRuntime::span<const NodeRef>(&from, 1));
        }

    } // namespace NN
} // namespace SushiAI
