/**************************************************************************/
/* session.hpp                                                            */
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
 * @file session.hpp
 * @brief Declares Session, which owns what the eager front end keeps across steps.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <SushiBLAS/engine/engine.hpp>

#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/region_partitioner.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/eager/region_table.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/eager/step_executor.hpp>
#include <SushiAI/eager/storage_bank.hpp>
#include <SushiAI/eager/storage_id.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/step_recorder.hpp>

namespace SushiAI
{
    namespace Eager
    {
        class Step;

        /** @brief States how a session compiles its regions. */
        struct SessionOptions
        {
            ProgramOptions program; /**< What the passes do to every region. */
        };

        /** @brief Names a leaf of a step and the tensor copied into it before the run. */
        struct Feed
        {
            Graph::ValueID leaf = Graph::K_NO_VALUE; /**< The leaf the copy fills. */
            SushiBLAS::Tensor source;                /**< The tensor copied; shares its storage. */
        };

        /** @brief Pairs a trainable state with the value that holds its gradient in a step. */
        struct StateGradient
        {
            Graph::StateID state = Graph::K_NO_STATE;    /**< The parameter. */
            Graph::ValueID value = Graph::K_NO_VALUE;    /**< Its gradient, or none. */
        };

        /**
         * @brief Holds what one step traced, as the session reads it.
         * @see modules/core/eager/README.md
         */
        struct StepTrace
        {
            Graph::Graph graph;                   /**< The forward nodes, then the backward ones. */
            std::vector<Segment> segments;        /**< The segments of the forward nodes. */
            Autograd::Provenance provenance;      /**< The origin of each backward node. */
            std::vector<Feed> feeds;              /**< The leaves filled from the host. */
            std::vector<Graph::ValueID> observed; /**< The values the caller reads. */
            std::vector<StateGradient> gradients; /**< The gradient of each traced parameter. */
            Graph::StepRecorder* trailing = nullptr; /**< The update behind the step, or null. */
        };

        /** @brief Holds what the host spent on the last step, in nanoseconds, and what it did. */
        struct StepCost
        {
            std::uint64_t partition_ns = 0; /**< Assigning each node a region key. */
            std::uint64_t plan_ns = 0;      /**< Finding boundaries and comparing signatures. */
            std::uint64_t compile_ns = 0;   /**< Running the passes over the built regions. */
            std::uint64_t execute_ns = 0;   /**< Feeding, recording and running the step. */
            std::uint64_t total_ns = 0;     /**< The whole flush. */
            std::size_t kept = 0;           /**< Regions that ran as recorded. */
            std::size_t recorded = 0;       /**< Regions recorded again under another key. */
            std::size_t built = 0;          /**< Regions compiled and recorded. */
            std::size_t dropped = 0;        /**< Regions of the last step that left. */
        };

        /**
         * @brief Owns the state storage, the region table and the boundary storage of a loop.
         * @see modules/core/eager/README.md
         */
        class Session final
        {
            public:
                /**
                 * @brief Starts a session with no state and no region.
                 * @param engine Allocates the session's storage and copies its feeds.
                 * @param executor Runs each step.
                 * @param partitioner Cuts each step into regions.
                 * @pre The three references outlive the session.
                 */
                Session(SushiBLAS::Engine& engine, StepExecutor& executor,
                        const RegionPartitioner& partitioner, SessionOptions options = {});

                Session(const Session&) = delete;
                Session& operator=(const Session&) = delete;
                Session(Session&&) = delete;
                Session& operator=(Session&&) = delete;
                ~Session() = default;

                /**
                 * @brief Binds @p state to @p tensor for every step of the session.
                 * @param tensor Shares its storage with the session; the caller keeps its own.
                 * @param trainable Whether the session keeps a gradient tensor for the state.
                 * @return InvalidDescription for a state bound twice, without identity or
                 *         without storage; or the failure of an allocation or of the zero fill.
                 */
                [[nodiscard]] Status bind_state(Graph::StateID state,
                                                const SushiBLAS::Tensor& tensor, bool trainable);

                /**
                 * @brief Returns the tensor every step writes the gradient of @p state into.
                 * @return A tensor sharing the session's storage, or InvalidDescription for a
                 *         state that is not bound as trainable.
                 */
                [[nodiscard]] Result<SushiBLAS::Tensor> gradient(Graph::StateID state) const;

                /**
                 * @brief Opens a step.
                 * @pre No other step of this session is alive; a second one stops the process.
                 */
                [[nodiscard]] Step begin_step();

                /** @brief Returns how many steps have run. */
                [[nodiscard]] std::uint64_t step_count() const noexcept
                {
                    return steps_;
                }

                /** @brief Returns what the host spent on the last step that ran. */
                [[nodiscard]] const StepCost& last_cost() const noexcept
                {
                    return cost_;
                }

                /** @brief Returns the decisions of the last step that ran. */
                [[nodiscard]] const StepPlan& last_plan() const noexcept
                {
                    return plan_;
                }

                /** @brief Returns the names of the regions the session has seen. */
                [[nodiscard]] const RegionNames& region_names() const noexcept
                {
                    return table_.names();
                }

                /** @brief Returns the storage the session keeps across steps. */
                [[nodiscard]] const StorageBank& storage() const noexcept
                {
                    return bank_;
                }

            private:
                friend class Step;

                /** @brief Holds what the session keeps for one bound state. */
                struct BoundState
                {
                    StorageID storage = K_NO_STORAGE;  /**< The state's own tensor. */
                    StorageID gradient = K_NO_STORAGE; /**< Its gradient tensor, or none. */
                    Shape shape;                       /**< The state's extents. */
                    DType dtype = DType::Float32;      /**< The state's element type. */
                    bool gradient_is_zero = false;     /**< Whether the gradient holds zeros. */
                    std::uint64_t written_in = 0;      /**< One past the step that wrote it. */
                };

                /** @brief Holds one buffer a feed is copied into. */
                struct InputBuffer
                {
                    StorageID storage = K_NO_STORAGE; /**< The buffer's tensor. */
                    Shape shape;                      /**< The extents it was allocated for. */
                    DType dtype = DType::Float32;     /**< Its element type. */
                };

                /** @brief Returns the entry of @p state, or null when it is not bound. */
                [[nodiscard]] const BoundState* find_state(Graph::StateID state) const;

                /**
                 * @brief Partitions, plans, compiles and runs @p trace.
                 * @return Success, or the first failure; after one the next step builds
                 *         every region.
                 */
                [[nodiscard]] Status run(const StepTrace& trace);

                /**
                 * @brief Returns the tensor the last run left @p value in.
                 * @return InvalidDescription for a value the step did not keep.
                 */
                [[nodiscard]] Result<const SushiBLAS::Tensor*> tensor_of(
                    Graph::ValueID value) const;

                /** @brief Marks the session as having no open step. */
                void close_step() noexcept
                {
                    open_ = false;
                }

                /**
                 * @brief Writes the storage of each state leaf, feed and gradient of @p trace.
                 * @param preset Receives one storage id per value id of the trace's graph.
                 */
                [[nodiscard]] Status place_leaves(const StepTrace& trace,
                                                  std::vector<StorageID>& preset);

                /** @brief Returns the buffer of feed @p ordinal, allocating one that fits. */
                [[nodiscard]] Result<StorageID> input_buffer(std::size_t ordinal,
                                                             const Graph::Value& leaf);

                /** @brief Copies every feed and zeroes the gradients no region writes. */
                [[nodiscard]] Status prepare_storage(const StepTrace& trace,
                                                     const std::vector<StorageID>& preset);

                /** @brief Forgets every region after @p failure and returns it. */
                [[nodiscard]] Status recover(const Status& failure);

                /// @brief Allocates storage and runs the copies and fills.
                SushiBLAS::Engine* engine_;

                /// @brief Runs each step.
                StepExecutor* executor_;

                /// @brief Cuts each step into regions.
                const RegionPartitioner* partitioner_;

                /// @brief What the passes do to every region.
                SessionOptions options_;

                /// @brief Hands out the id of every tensor the session holds.
                StorageIDSource ids_;

                /// @brief The tensors of the states, the feeds and the boundary values.
                StorageBank bank_;

                /// @brief The regions of the last step, by name.
                RegionTable table_;

                /// @brief The bound states, by identity.
                std::unordered_map<Graph::StateID, BoundState> states_;

                /// @brief The feed buffers, by the feed's position in its step.
                std::vector<InputBuffer> inputs_;

                /// @brief The decisions of the last step that ran.
                StepPlan plan_;

                /// @brief What the host spent on the last step that ran.
                StepCost cost_;

                /// @brief The number of steps that have run.
                std::uint64_t steps_ = 0;

                /// @brief Whether a Step of this session is alive.
                bool open_ = false;
        };

    } // namespace Eager
} // namespace SushiAI
