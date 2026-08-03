/**************************************************************************/
/* backward.cpp                                                           */
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

#include <array>
#include <unordered_map>
#include <SushiAI/autograd/backward.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @class Differentiator
         * @brief One differentiation run's bookkeeping.
         *
         * Holds the contributions per primal value, the memoised sums, and the
         * set of nodes a fusion has already absorbed. Lives only for the length
         * of a @c differentiate() call.
         */
        class Differentiator
        {
            public:
                Differentiator(Graph::Graph& graph, Graph::ValueId loss,
                               Graph::ValueId seed_factor)
                    : graph_(graph),
                      loss_(loss),
                      builder_(graph),
                      forward_node_count_(graph.node_count()),
                      skipped_(graph.node_count(), false),
                      tape_(Graph::GraphBuilder(graph), *this, seed_factor)
                {
                }

                /** @brief Runs the reverse sweep over the forward nodes. */
                void run();

                /** @brief The summed gradient of @p primal, materialising the sum on first read. */
                Graph::ValueId resolve(Graph::ValueId primal);

                /** @brief Records one gradient contribution to @p primal. */
                void contribute(Graph::ValueId primal, Graph::ValueId gradient);

                /** @brief Marks @p node as already differentiated by a fusion. */
                void skip(Graph::NodeId node);

                /** @brief Whether @p value is the scalar being differentiated. */
                bool is_root(Graph::ValueId value) const noexcept { return value == loss_; }

                /**
                 * @brief The node the tape currently points at.
                 *
                 * A snapshot, not a reference into the graph. Every rule emits
                 * while it is running, and emitting grows the node vector,
                 * which reallocates and would leave a rule holding a dangling
                 * reference to the very node it is differentiating. Copying one
                 * node per differentiated node is unmeasurable; the alternative
                 * is a use-after-free that only appears once a graph outgrows
                 * its initial capacity.
                 */
                const Graph::Node& current_node() const noexcept { return current_node_; }

                /** @brief The graph being extended. */
                Graph::Graph& graph() noexcept { return graph_; }

            private:
                Graph::Graph& graph_;
                Graph::ValueId loss_;
                Graph::GraphBuilder builder_;
                std::size_t forward_node_count_;
                std::vector<bool> skipped_;
                std::unordered_map<Graph::ValueId, std::vector<Graph::ValueId>> contributions_;
                std::unordered_map<Graph::ValueId, Graph::ValueId> resolved_;
                GradientTape tape_;
                Graph::Node current_node_{};
        };

        // =====================================================================
        // Backward rules
        // =====================================================================

        namespace
        {
            void matmul_backward(const GradientTape& tape)
            {
                const Graph::Node& node = tape.node();
                const Graph::ValueId dc = tape.output_grad(0);
                if (dc == Graph::NO_VALUE) return;

                const Graph::GraphBuilder& builder = tape.builder();
                const Graph::ValueId a = node.inputs.at(0);
                const Graph::ValueId b = node.inputs.at(1);
                const bool ta = node.attributes.flag(0);
                const bool tb = node.attributes.flag(1);

                // Four transposition cases, each derived by treating the product
                // as U*V with U = op(A), V = op(B): dU = dC V^T, dV = U^T dC, and
                // then transposing back through whichever operand was flagged.
                if (tape.input_requires_grad(0))
                {
                    Graph::ValueId da;
                    if (!ta && !tb)     da = builder.matmul(dc, b, false, true);
                    else if (ta && !tb) da = builder.matmul(b, dc, false, true);
                    else if (!ta)       da = builder.matmul(dc, b, false, false);
                    else                da = builder.matmul(b, dc, true, true);
                    tape.contribute_input(0, da);
                }

                if (tape.input_requires_grad(1))
                {
                    Graph::ValueId db;
                    if (!ta && !tb)     db = builder.matmul(a, dc, true, false);
                    else if (ta && !tb) db = builder.matmul(a, dc, false, false);
                    else if (!ta)       db = builder.matmul(dc, a, true, false);
                    else                db = builder.matmul(dc, a, true, true);
                    tape.contribute_input(1, db);
                }
            }

            void bias_add_backward(const GradientTape& tape)
            {
                const Graph::ValueId dy = tape.output_grad(0);
                if (dy == Graph::NO_VALUE) return;

                // Y = X + 1 b^T is the identity in X, so the gradient passes
                // straight through with nothing emitted, and it sums over the
                // broadcast axis in b.
                if (tape.input_requires_grad(0)) tape.contribute_input(0, dy);
                if (tape.input_requires_grad(1)) tape.contribute_input(1, tape.builder().sum_rows(dy));
            }

            void add_backward(const GradientTape& tape)
            {
                const Graph::ValueId dc = tape.output_grad(0);
                if (dc == Graph::NO_VALUE) return;

                if (tape.input_requires_grad(0)) tape.contribute_input(0, dc);
                if (tape.input_requires_grad(1)) tape.contribute_input(1, dc);
            }

            void sub_backward(const GradientTape& tape)
            {
                const Graph::ValueId dc = tape.output_grad(0);
                if (dc == Graph::NO_VALUE) return;

                if (tape.input_requires_grad(0)) tape.contribute_input(0, dc);
                if (tape.input_requires_grad(1)) tape.contribute_input(1, tape.builder().scale(dc, -1.0));
            }

            void mul_backward(const GradientTape& tape)
            {
                const Graph::Node& node = tape.node();
                const Graph::ValueId dc = tape.output_grad(0);
                if (dc == Graph::NO_VALUE) return;

                const Graph::GraphBuilder& builder = tape.builder();
                if (tape.input_requires_grad(0))
                    tape.contribute_input(0, builder.mul(dc, node.inputs.at(1)));
                if (tape.input_requires_grad(1))
                    tape.contribute_input(1, builder.mul(dc, node.inputs.at(0)));
            }

            void scale_backward(const GradientTape& tape)
            {
                const Graph::ValueId dy = tape.output_grad(0);
                if (dy == Graph::NO_VALUE) return;
                if (!tape.input_requires_grad(0)) return;

                tape.contribute_input(0, tape.builder().scale(dy, tape.node().attributes.get(0)));
            }

            void exp_backward(const GradientTape& tape)
            {
                const Graph::ValueId dy = tape.output_grad(0);
                if (dy == Graph::NO_VALUE) return;
                if (!tape.input_requires_grad(0)) return;

                // d/dx exp(x) = exp(x), which the forward pass already produced.
                tape.contribute_input(0, tape.builder().mul(dy, tape.node().outputs.at(0)));
            }

            /**
             * @brief The rule for every activation, read out of Ops::ACTIVATIONS.
             *
             * One function rather than four, because the only thing that
             * differs between them is which gradient operation to emit and
             * which tensor to differentiate at - and both of those are already
             * stated in @c Ops::ACTIVATIONS, which is also what decides whether
             * a fused form may fold the activation away. Restating either here
             * would be a second place for the two to disagree, and a
             * disagreement is silent: the two candidate primals have the same
             * shape, so the graph builds and the model learns the wrong thing.
             *
             * @param tape The rule's context.
             */
            void activation_backward(const GradientTape& tape)
            {
                const Graph::ValueId dy = tape.output_grad(0);
                if (dy == Graph::NO_VALUE) return;
                if (!tape.input_requires_grad(0)) return;

                const Graph::Node& node = tape.node();
                const Ops::Activation* activation = Ops::activation_of(node.op);

                SA_THROW_IF(activation == nullptr,
                            "{} is registered as an activation's backward rule but is not in "
                            "Ops::ACTIVATIONS. Add it there rather than here; that table is what "
                            "the fusion catalog also reads.",
                            Ops::name_of(node.op));

                const Graph::ValueId primal =
                    activation->gradient_source == Ops::GradientSource::OUTPUT
                        ? node.outputs.at(0)
                        : node.inputs.at(0);

                tape.contribute_input(
                    0, tape.builder().activation_backward(activation->backward, dy, primal));
            }

            void sum_rows_backward(const GradientTape& tape)
            {
                const Graph::ValueId ds = tape.output_grad(0);
                if (ds == Graph::NO_VALUE) return;
                if (!tape.input_requires_grad(0)) return;

                const Graph::ValueId x = tape.node().inputs.at(0);
                const Graph::Value& value = tape.graph().value(x);

                // Summing over rows broadcasts back over rows; adding the row
                // vector to a zero matrix is exactly that broadcast, and reuses
                // the bias lowering rather than inventing a second one.
                const Graph::ValueId zeros = tape.builder().zeros(value.shape, value.dtype);
                tape.contribute_input(0, tape.builder().bias_add(zeros, ds));
            }

            void accumulate_backward(const GradientTape& tape)
            {
                const Graph::ValueId dout = tape.output_grad(0);
                if (dout == Graph::NO_VALUE) return;

                if (tape.input_requires_grad(0)) tape.contribute_input(0, dout);
                if (tape.input_requires_grad(1)) tape.contribute_input(1, dout);
            }

            void log_softmax_backward(const GradientTape&)
            {
                // Reachable only when a LOG_SOFTMAX result is consumed by
                // something other than NLL_LOSS. The standalone rule is
                // `dx = dy - exp(L) * rowsum(dy)`, so it needs a row-wise sum of
                // the incoming gradient - and a backward rule writes IR, never
                // SushiBLAS calls, so what it is missing is an IR *node* for that
                // sum, not the kernel underneath it.
                //
                // The kernel exists now: SushiBLAS grew `sum_along` alongside the
                // `max_along` that fixed log-softmax's own stabilising shift. What
                // is still absent here is an op identity, a builder entry point, a
                // shape rule and a lowering for a row-wise sum, plus this rule
                // written against them. Named rather than left implied, because
                // the honest blocker moved and a stale message would send the next
                // reader to the wrong repository.
                SA_THROW_IF(true,
                            "LOG_SOFTMAX can only be differentiated as part of the fused "
                            "LOG_SOFTMAX + NLL_LOSS pair in this release: the standalone rule "
                            "needs a row-wise reduction of the incoming gradient, and SushiAI has "
                            "no IR node for one yet (SushiBLAS::ReductionOps::sum_along is the "
                            "kernel it would lower to). Feed its result to NLL_LOSS (see "
                            "NN::CrossEntropyLoss), or add a row-sum operation first.");
            }

            void nll_loss_backward(const GradientTape& tape)
            {
                const Graph::Node& node = tape.node();

                SA_THROW_IF(!tape.output_is_root(0),
                            "NLL_LOSS is differentiable only as the value passed to "
                            "differentiate(); its gradient is seeded as the implicit scalar one, "
                            "so it cannot sit inside a larger expression yet.");

                const Graph::GraphBuilder& builder = tape.builder();
                const Graph::ValueId log_probs = node.inputs.at(0);
                const Graph::ValueId targets = node.inputs.at(1);

                const Graph::Value& value = tape.graph().value(log_probs);
                const double batch = static_cast<double>(value.shape.dim(0));

                const Graph::NodeId producer = value.producer;
                const bool fused = producer != Graph::NO_NODE
                                && tape.graph().node(producer).op == Ops::LOG_SOFTMAX.id;

                if (!fused)
                {
                    // loss = -sum(L * Y) / N, so dL is just the targets, scaled.
                    if (tape.input_requires_grad(0))
                        tape.contribute_input(0, tape.apply_seed_factor(
                            builder.scale(targets, -1.0 / batch)));
                    return;
                }

                // Fused: differentiate log-softmax and the loss together. The
                // composite gradient with respect to the logits is
                // (softmax(X) - Y) / N, which is both far cheaper than chaining
                // through the seven-op log-softmax recipe and far better
                // conditioned - the p and (1-p) factors that cancel analytically
                // would otherwise be computed and subtracted numerically.
                //
                // Valid because the targets are one-hot: the general form is
                // (p * rowsum(Y) - Y) / N, and rowsum(Y) is 1.
                tape.skip(producer);

                const Graph::ValueId logits = tape.graph().node(producer).inputs.at(0);
                if (!tape.graph().value(logits).requires_grad) return;

                const Graph::ValueId probabilities = builder.exp(log_probs);
                const Graph::ValueId difference = builder.sub(probabilities, targets);
                // The loss scale multiplies *here*, on the seeded gradient and
                // before anything narrows. Two reasons it is this value and not
                // another. It is the last point at which one multiply covers the
                // whole backward pass - every gradient below it is linear in it -
                // so one node buys the scaling of the entire graph. And the
                // cross-entropy tail is kept in fp32 by the precision policy
                // (graph/precision.hpp rule 4), so the factor is applied while
                // there is still exponent range to apply it in; a scale inserted
                // after the first cast to fp16 would overflow the very value it
                // exists to keep out of the subnormals.
                tape.contribute(logits, tape.apply_seed_factor(
                    builder.scale(difference, 1.0 / batch)));
            }

            /** @brief One operation's backward rule, paired with its identity. */
            struct RuleEntry
            {
                OpID op;
                BackwardRule rule;
            };

            /**
             * @brief Every backward rule, as a flat compile-time table.
             *
             * A plain array rather than a self-registering map: see the header
             * for why linker garbage collection makes the latter silently
             * incomplete in a static library.
             */
            constexpr std::array<RuleEntry, 14> RULES = {{
                {Ops::MATMUL.id,           &matmul_backward},
                {Ops::BIAS_ADD.id,         &bias_add_backward},
                {Ops::ADD.id,              &add_backward},
                {Ops::SUB.id,              &sub_backward},
                {Ops::MUL.id,              &mul_backward},
                {Ops::SCALE.id,            &scale_backward},
                {Ops::EXP.id,              &exp_backward},
                {Ops::RELU.id,             &activation_backward},
                {Ops::GELU.id,             &activation_backward},
                {Ops::SIGMOID.id,          &activation_backward},
                {Ops::TANH.id,             &activation_backward},
                {Ops::SUM_ROWS.id,         &sum_rows_backward},
                {Ops::ACCUMULATE.id,       &accumulate_backward},
                {Ops::NLL_LOSS.id,         &nll_loss_backward},
            }};

            /** @brief The extra entry that only exists to produce a good error. */
            constexpr RuleEntry LOG_SOFTMAX_RULE{Ops::LOG_SOFTMAX.id, &log_softmax_backward};

            /** @brief Finds a rule, or null when the operation has none. */
            constexpr BackwardRule find_rule(OpID op) noexcept
            {
                for (const RuleEntry& entry : RULES)
                    if (entry.op == op) return entry.rule;
                if (op == LOG_SOFTMAX_RULE.op) return LOG_SOFTMAX_RULE.rule;
                return nullptr;
            }
        }

        bool has_backward_rule(OpID op) noexcept
        {
            return find_rule(op) != nullptr;
        }

        // =====================================================================
        // GradientTape
        // =====================================================================

        const Graph::Node& GradientTape::node() const
        {
            return owner_->current_node();
        }

        Graph::ValueId GradientTape::apply_seed_factor(Graph::ValueId gradient) const
        {
            if (seed_factor_ == Graph::NO_VALUE) return gradient;
            return builder_.dynamic_scale(gradient, seed_factor_, "loss_scaled_grad");
        }

        Graph::ValueId GradientTape::output_grad(std::size_t index) const
        {
            const Graph::Node& n = node();
            SA_THROW_IF(index >= n.outputs.size(),
                        "{} has no result {}", Ops::name_of(n.op), index);
            return owner_->resolve(n.outputs[index]);
        }

        bool GradientTape::output_is_root(std::size_t index) const
        {
            const Graph::Node& n = node();
            SA_THROW_IF(index >= n.outputs.size(),
                        "{} has no result {}", Ops::name_of(n.op), index);
            return owner_->is_root(n.outputs[index]);
        }

        bool GradientTape::input_requires_grad(std::size_t index) const
        {
            const Graph::Node& n = node();
            SA_THROW_IF(index >= n.inputs.size(),
                        "{} has no operand {}", Ops::name_of(n.op), index);
            return owner_->graph().value(n.inputs[index]).requires_grad;
        }

        void GradientTape::contribute(Graph::ValueId primal, Graph::ValueId gradient) const
        {
            owner_->contribute(primal, gradient);
        }

        void GradientTape::contribute_input(std::size_t index, Graph::ValueId gradient) const
        {
            const Graph::Node& n = node();
            SA_THROW_IF(index >= n.inputs.size(),
                        "{} has no operand {}", Ops::name_of(n.op), index);
            owner_->contribute(n.inputs[index], gradient);
        }

        void GradientTape::skip(Graph::NodeId node_to_skip) const
        {
            owner_->skip(node_to_skip);
        }

        // =====================================================================
        // Differentiator
        // =====================================================================

        void Differentiator::contribute(Graph::ValueId primal, Graph::ValueId gradient)
        {
            SA_THROW_IF(resolved_.find(primal) != resolved_.end(),
                        "A gradient contribution to value {} arrived after its gradient was "
                        "already read. The reverse walk visits every consumer before the "
                        "producer, so this is an ordering bug in a backward rule, not a "
                        "caller error.",
                        primal);

            const Graph::Value& a = graph_.value(primal);
            const Graph::Value& b = graph_.value(gradient);
            SA_THROW_IF(a.shape != b.shape,
                        "A gradient of shape {} was contributed to value {} of shape {}",
                        b.shape.to_string(), primal, a.shape.to_string());

            contributions_[primal].push_back(gradient);
        }

        void Differentiator::skip(Graph::NodeId node)
        {
            SA_THROW_IF(node >= skipped_.size(),
                        "Cannot skip node {}; the forward graph has {} nodes",
                        node, skipped_.size());
            skipped_[node] = true;
        }

        Graph::ValueId Differentiator::resolve(Graph::ValueId primal)
        {
            // The differentiation root's own gradient is the implicit scalar
            // one; there is no tensor for it and the rule that seeds the sweep
            // knows not to ask for one.
            if (primal == loss_) return Graph::NO_VALUE;

            const auto cached = resolved_.find(primal);
            if (cached != resolved_.end()) return cached->second;

            Graph::ValueId result = Graph::NO_VALUE;
            const auto found = contributions_.find(primal);

            if (found != contributions_.end() && !found->second.empty())
            {
                const std::vector<Graph::ValueId>& parts = found->second;
                result = parts.front();
                for (std::size_t i = 1; i < parts.size(); ++i)
                    result = builder_.accumulate(result, parts[i]);
            }

            resolved_.emplace(primal, result);
            return result;
        }

        void Differentiator::run()
        {
            graph_.propagate_requires_grad();

            SA_THROW_IF(graph_.value(loss_).shape.num_elements() != 1,
                        "differentiate() needs a scalar to differentiate; value {} has shape {}",
                        loss_, graph_.value(loss_).shape.to_string());

            // Being one element is necessary and not sufficient. The root's own
            // gradient is the implicit scalar one, so there is no tensor for
            // resolve() to hand back, and every rule but nll_loss_backward
            // reads that NO_VALUE as "nothing flows here" and returns. A [1, 1]
            // MATMUL therefore differentiates to a complete set of empty
            // gradients and no error - the caller only finds out one layer
            // later, from bind_parameters blaming the model for a loss-side
            // mistake. Reject it here, where the actual cause is still known.
            const Graph::NodeId root_producer = graph_.value(loss_).producer;
            SA_THROW_IF(root_producer == Graph::NO_NODE,
                        "differentiate() cannot seed a gradient for value {}: nothing produces "
                        "it. The loss must be the result of NLL_LOSS - see NN::CrossEntropyLoss.",
                        loss_);
            SA_THROW_IF(graph_.node(root_producer).op != Ops::NLL_LOSS.id,
                        "differentiate() seeds the loss gradient as the implicit scalar one, and "
                        "only NLL_LOSS knows to start from it. Value {} is produced by {}, whose "
                        "rule declines an absent output gradient and returns, so every requested "
                        "gradient would come back empty with nothing reported. Produce the loss "
                        "with NLL_LOSS - see NN::CrossEntropyLoss.",
                        loss_, Ops::name_of(graph_.node(root_producer).op));

            for (std::size_t offset = forward_node_count_; offset > 0; --offset)
            {
                const Graph::NodeId id = static_cast<Graph::NodeId>(offset - 1);
                if (skipped_[id]) continue;

                // Snapshot before anything below can grow the graph.
                current_node_ = graph_.node(id);

                bool any_input_needs_grad = false;
                for (Graph::ValueId in : current_node_.inputs)
                    if (graph_.value(in).requires_grad) { any_input_needs_grad = true; break; }

                // A leaf has no operands, so this also skips every leaf, and a
                // frozen subgraph costs nothing at all.
                if (!any_input_needs_grad) continue;

                bool reachable = false;
                for (Graph::ValueId out : current_node_.outputs)
                    if (is_root(out) || resolve(out) != Graph::NO_VALUE) { reachable = true; break; }

                if (!reachable) continue;

                const BackwardRule rule = find_rule(current_node_.op);
                SA_THROW_IF(rule == nullptr,
                            "No backward rule for {} (node {}), but its operands require a "
                            "gradient. Add one to the table in backward.cpp, or mark the "
                            "subgraph as not requiring gradients.",
                            Ops::name_of(current_node_.op), id);

                tape_.retarget(id);
                rule(tape_);
            }
        }

        std::vector<Graph::ValueId> differentiate(Graph::Graph& graph,
                                                  Graph::ValueId loss,
                                                  SushiRuntime::span<const Graph::ValueId> wrt,
                                                  Graph::ValueId seed_factor)
        {
            Differentiator differentiator(graph, loss, seed_factor);
            differentiator.run();

            std::vector<Graph::ValueId> gradients;
            gradients.reserve(wrt.size());

            for (Graph::ValueId value : wrt)
            {
                const Graph::ValueId gradient = differentiator.resolve(value);
                gradients.push_back(gradient);

                // A parameter's gradient outlives the step exactly as the
                // parameter does, so it belongs in the gradient arena rather
                // than being packed against the activations.
                if (gradient != Graph::NO_VALUE
                    && graph.value(value).lifetime == Graph::Lifetime::PARAMETER)
                    graph.value(gradient).lifetime = Graph::Lifetime::GRADIENT;
            }

            return gradients;
        }

    } // namespace Autograd
} // namespace SushiAI
