/**************************************************************************/
/* model_graph.cpp                                                        */
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
 * @file model_graph.cpp
 * @brief Implements ModelGraph: adding nodes, publishing outputs, the trace and the state walk.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/model_graph.hpp>
#include <SushiLog/text/text_format.hpp>

#include <algorithm>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/graph/builder.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Joins the names of @p ports with ", ". */
            std::string list_ports(SushiRuntime::span<const Port> ports)
            {
                std::string names;
                for (std::size_t index = 0; index < ports.size(); ++index)
                {
                    if (index != 0)
                        names += ", ";

                    names += ports[index].name;
                }

                return names;
            }

            /** @brief Returns the first port of @p ports whose name an earlier one has. */
            const Port* find_repeated_port(SushiRuntime::span<const Port> ports) noexcept
            {
                for (std::size_t later = 1; later < ports.size(); ++later)
                {
                    for (std::size_t earlier = 0; earlier < later; ++earlier)
                    {
                        if (ports[earlier].name == ports[later].name)
                            return &ports[later];
                    }
                }

                return nullptr;
            }

            /** @brief Reports whether @p traced is @p sample with one batch axis in front. */
            bool is_batch_of(const Shape& traced, const Shape& sample) noexcept
            {
                if (traced.rank() != sample.rank() + 1)
                    return false;

                for (int32_t axis = 0; axis < sample.rank(); ++axis)
                {
                    if (traced.dim(axis + 1) != sample.dim(axis))
                        return false;
                }

                return true;
            }

            /** @brief Traces @p model and returns the value of its single published output. */
            Result<Graph::ValueID> trace_single_output(ModelGraph& model, const Trace& trace,
                                                       Graph::ValueID input)
            {
                SA_TRY_ASSIGN(const ModelOutputs outputs, model.trace(trace, input));

                return outputs.only();
            }

            /** @brief Traces the incremental form and returns its single published output. */
            Result<Graph::ValueID> trace_single_incremental(ModelGraph& model, const Trace& trace,
                                                            Graph::ValueID input)
            {
                SA_TRY_ASSIGN(const ModelOutputs outputs, model.trace_incremental(trace, input));

                return outputs.only();
            }

        } // namespace

        ModelGraph::ModelGraph(Shape input_sample) : input_sample_(input_sample)
        {

        }

        NodeRef ModelGraph::input() const noexcept
        {
            return NodeRef(NodeRef::K_INPUT, 0);
        }

        Result<NodeRef> ModelGraph::add(std::string name, std::unique_ptr<Node> node,
                                        SushiRuntime::span<const NodeRef> from)
        {
            SA_FATAL_IF(node == nullptr, "a null node was added to a model graph as '{}'", name);
            SA_FATAL_IF(node->ports().empty(),
                        "the node added to a model graph as '{}' declares no port", name);

            const Port* const repeated = find_repeated_port(node->ports());
            SA_FATAL_IF(repeated != nullptr,
                        "the node added to a model graph as '{}' declares two ports named '{}'",
                        name, repeated != nullptr ? repeated->name : std::string());

            SA_TRY(require_free_name(name));
            SA_FAIL_IF(from.size() != node->input_count(), StatusCode::InvalidDescription,
                       "node '{}' reads {} values; the number of sources it was given is {}",
                       name, node->input_count(), from.size());

            for (const NodeRef source : from)
                SA_TRY(SA_ANNOTATE(require_source(source), "node '{}'", name));

            const std::size_t index = entries_.size();
            const std::size_t first_value = port_count();
            entries_.push_back(Entry{std::move(name), std::move(node),
                                     std::vector<NodeRef>(from.begin(), from.end()),
                                     first_value});

            return NodeRef(index, 0);
        }

        Result<NodeRef> ModelGraph::port(NodeRef of, std::string_view name) const
        {
            SA_TRY(SA_ANNOTATE(require_source(of), "the lookup of port '{}'", name));
            SA_FAIL_IF(of.is_input(), StatusCode::InvalidDescription,
                       "the model input has no port named '{}'; only a node has named ports",
                       name);

            const SushiRuntime::span<const Port> ports = entries_[of.node()].node->ports();
            for (std::size_t index = 0; index < ports.size(); ++index)
            {
                if (ports[index].name == name)
                    return NodeRef(of.node(), index);
            }

            return Status::from_message(StatusCode::InvalidDescription, [&]()
            {
                return SushiLog::format_text("node '{}' has no port named '{}'; its ports: {}",
                                               entries_[of.node()].name, name,
                                               list_ports(ports));
            });
        }

        Status ModelGraph::output(std::string name, NodeRef from)
        {
            SA_FAIL_IF(name.empty(), StatusCode::InvalidDescription,
                       "a model output needs a name; the model publishes {} outputs",
                       outputs_.size());
            SA_FAIL_IF(has_output(name), StatusCode::InvalidDescription,
                       "the model already publishes an output named '{}'", name);
            SA_TRY(SA_ANNOTATE(require_source(from), "output '{}'", name));

            outputs_.emplace_back(std::move(name), from);
            return Status();
        }

        Status ModelGraph::output_ports(NodeRef of)
        {
            SA_TRY(SA_ANNOTATE(require_source(of), "the publication of a node's ports"));
            SA_FAIL_IF(of.is_input(), StatusCode::InvalidDescription,
                       "the model input has no ports to publish; publish it with output()");

            const Entry& entry = entries_[of.node()];
            const SushiRuntime::span<const Port> ports = entry.node->ports();
            for (std::size_t index = 0; index < ports.size(); ++index)
            {
                SA_FAIL_IF(has_output(ports[index].name), StatusCode::InvalidDescription,
                           "port '{}' of node '{}' cannot be published under its own name; an "
                           "output named '{}' exists",
                           ports[index].name, entry.name, ports[index].name);
            }

            outputs_.reserve(outputs_.size() + ports.size());
            for (std::size_t index = 0; index < ports.size(); ++index)
                outputs_.emplace_back(ports[index].name, NodeRef(of.node(), index));

            return Status();
        }

        Result<ModelOutputs> ModelGraph::trace(const Trace& trace, Graph::ValueID input)
        {
            const Graph::Graph& graph = trace.builder().graph();
            SA_FATAL_IF(input >= graph.value_count(),
                        "a model graph was traced over value {}; the traced graph holds {} values",
                        input, graph.value_count());
            SA_FAIL_IF(!is_batch_of(graph.value(input).shape, input_sample_),
                       StatusCode::InvalidDescription,
                       "the model input is declared as {} per sample; the traced value is {}",
                       input_sample_.to_string(), graph.value(input).shape.to_string());

            return trace_nodes(trace, input, &Node::forward, PortCheck::PerSample);
        }

        Result<ModelOutputs> ModelGraph::trace_incremental(const Trace& trace,
                                                           Graph::ValueID input)
        {
            const Graph::Graph& graph = trace.builder().graph();
            SA_FATAL_IF(input >= graph.value_count(),
                        "a model graph was traced over value {}; the traced graph holds {} values",
                        input, graph.value_count());
            SA_FAIL_IF(!trace.is_carrying(), StatusCode::InvalidDescription,
                       "the incremental form of a model graph was traced in a trace whose step "
                       "keeps nothing between runs; it has {} nodes",
                       entries_.size());

            return trace_nodes(trace, input, &Node::forward_incremental, PortCheck::None);
        }

        Result<ModelOutputs> ModelGraph::trace_nodes(const Trace& trace, Graph::ValueID input,
                                                     NodeForm form, PortCheck check)
        {
            const Graph::Graph& graph = trace.builder().graph();
            SA_FAIL_IF(outputs_.empty(), StatusCode::InvalidDescription,
                       "the model publishes no output; name one with output() or output_ports() "
                       "among its {} nodes",
                       entries_.size());

            // values[entry.first_value + port] is the traced id of that port.
            std::vector<Graph::ValueID> values;
            values.reserve(port_count());
            const auto value_of = [&](NodeRef ref)
            {
                if (ref.is_input())
                    return input;

                return values[entries_[ref.node()].first_value + ref.port()];
            };

            std::vector<Graph::ValueID> read;
            for (const Entry& entry : entries_)
            {
                read.clear();
                for (const NodeRef source : entry.sources)
                    read.push_back(value_of(source));

                const Status traced = ((*entry.node).*form)(
                    trace, ValueSpan(read.data(), read.size()), values);
                SA_TRY(SA_ANNOTATE(traced, "node '{}'", entry.name));

                const SushiRuntime::span<const Port> ports = entry.node->ports();
                SA_FATAL_IF(values.size() != entry.first_value + ports.size(),
                            "node '{}' declares {} ports and appended {} values", entry.name,
                            ports.size(), values.size() - entry.first_value);

                for (std::size_t index = 0; index < ports.size(); ++index)
                {
                    const Graph::ValueID id = values[entry.first_value + index];
                    SA_FATAL_IF(id >= graph.value_count(),
                                "node '{}' returned value {} for port '{}'; the traced graph "
                                "holds {} values",
                                entry.name, id, ports[index].name, graph.value_count());
                    SA_FAIL_IF(check == PortCheck::PerSample
                                   && !is_batch_of(graph.value(id).shape, ports[index].shape),
                               StatusCode::InvalidDescription,
                               "node '{}' declares port '{}' as {} per sample; the traced value "
                               "is {}",
                               entry.name, ports[index].name, ports[index].shape.to_string(),
                               graph.value(id).shape.to_string());
                }
            }

            ModelOutputs published;
            for (const Output& output : outputs_)
                SA_TRY(published.add(output.first, value_of(output.second)));

            return published;
        }

        Graph::ValueID ModelGraph::forward(const Trace& trace, Graph::ValueID input)
        {
            return trace.value_or_fail(trace_single_output(*this, trace, input), input);
        }

        Graph::ValueID ModelGraph::forward_incremental(const Trace& trace, Graph::ValueID input,
                                                       const Shape&)
        {
            return trace.value_or_fail(trace_single_incremental(*this, trace, input), input);
        }

        Shape ModelGraph::output_shape(const Shape&) const
        {
            if (outputs_.size() != 1)
                return Shape{};

            return shape_of(outputs_.front().second);
        }

        void ModelGraph::walk_state(std::string_view prefix, const StateVisitor& visitor)
        {
            for (const Entry& entry : entries_)
                entry.node->walk_state(Detail::join(prefix, entry.name), visitor);
        }

        std::size_t ModelGraph::size() const noexcept
        {
            return entries_.size();
        }

        const Shape& ModelGraph::shape_of(NodeRef ref) const
        {
            const Status known = require_source(ref);
            SA_FATAL_IF(!known.is_ok(), "ModelGraph::shape_of was given an unknown ref: {}",
                        known.message());

            if (ref.is_input())
                return input_sample_;

            return entries_[ref.node()].node->ports()[ref.port()].shape;
        }

        Node& ModelGraph::node(std::size_t index) const
        {
            SA_FATAL_IF(index >= entries_.size(),
                        "ModelGraph::node was asked for node {}; the graph holds {} nodes", index,
                        entries_.size());

            return *entries_[index].node;
        }

        Status ModelGraph::require_source(NodeRef ref) const
        {
            if (ref.is_input())
            {
                SA_FAIL_IF(ref.port() != 0, StatusCode::InvalidDescription,
                           "the model input has one port; port {} was named", ref.port());
                return Status();
            }

            SA_FAIL_IF(ref.node() >= entries_.size(), StatusCode::InvalidDescription,
                       "source node {} is not in the graph, which holds {} nodes; a source is "
                       "the input or a node added earlier",
                       ref.node(), entries_.size());

            const std::size_t ports = entries_[ref.node()].node->ports().size();
            SA_FAIL_IF(ref.port() >= ports, StatusCode::InvalidDescription,
                       "source node '{}' has {} ports; port {} was named",
                       entries_[ref.node()].name, ports, ref.port());

            return Status();
        }

        Status ModelGraph::require_free_name(std::string_view name) const
        {
            SA_FAIL_IF(name.empty(), StatusCode::InvalidDescription,
                       "a node of a model graph needs a name; the graph holds {} nodes",
                       entries_.size());
            SA_FAIL_IF(name.find('.') != std::string_view::npos, StatusCode::InvalidDescription,
                       "node name '{}' contains '.', which separates the segments of a state "
                       "name",
                       name);

            const auto same_name = std::find_if(entries_.begin(), entries_.end(),
                                                [name](const Entry& entry)
            {
                return entry.name == name;
            });
            SA_FAIL_IF(same_name != entries_.end(), StatusCode::InvalidDescription,
                       "the model graph already has a node named '{}', at index {}", name,
                       same_name - entries_.begin());

            return Status();
        }

        bool ModelGraph::has_output(std::string_view name) const noexcept
        {
            return std::any_of(outputs_.begin(), outputs_.end(), [name](const Output& output)
            {
                return output.first == name;
            });
        }

        std::size_t ModelGraph::port_count() const noexcept
        {
            if (entries_.empty())
                return 0;

            return entries_.back().first_value + entries_.back().node->ports().size();
        }

    } // namespace NN
} // namespace SushiAI
