/**************************************************************************/
/* module.hpp                                                             */
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
 * @file module.hpp
 * @brief Defines Parameter, the field kinds a layer lists, and the walk over model state.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/nn/buffer.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @enum ParameterRole
         * @brief What a parameter is for, which is what decides how it is initialised.
         */
        enum class ParameterRole : std::uint8_t
        {
            /** @brief A weight matrix; gets a fan-scaled random initialisation. */
            Weight,
            /** @brief A bias vector; always starts at zero. */
            Bias,
            /** @brief A per-channel scale, such as a normalisation's gamma; starts at one. */
            Scale,
        };

        /** @brief Describes how one parameter is filled before training. */
        struct ParameterInit
        {
            /** @brief Names the fill a parameter starts from. */
            enum class Kind : std::uint8_t
            {
                FanScaled, /**< @brief Fills at random, scaled by the parameter's fans. */
                Constant,  /**< @brief Sets every element to one stated value. */
            };

            /** @brief Selects the fill. */
            Kind kind = Kind::FanScaled;

            /** @brief Holds the value of a Constant fill; unread for FanScaled. */
            double constant = 0.0;

            /** @brief Returns the fan-scaled random fill. */
            [[nodiscard]] static constexpr ParameterInit fan_scaled() noexcept
            {
                return ParameterInit{Kind::FanScaled, 0.0};
            }

            /** @brief Returns the fill that sets every element to @p value. */
            [[nodiscard]] static constexpr ParameterInit constant_value(double value) noexcept
            {
                return ParameterInit{Kind::Constant, value};
            }
        };

        namespace Detail
        {
            /** @brief Returns the fill a parameter of @p role starts from. */
            [[nodiscard]] constexpr ParameterInit init_of(ParameterRole role) noexcept
            {
                if (role == ParameterRole::Bias)
                    return ParameterInit::constant_value(0.0);

                if (role == ParameterRole::Scale)
                    return ParameterInit::constant_value(1.0);

                return ParameterInit::fan_scaled();
            }

            /** @brief Erases the type of @p callable and keeps its address. */
            template<typename Fn>
            [[nodiscard]] void* erase_callable(Fn& callable) noexcept
            {
                // The invoker casts back to the deduced Fn, which restores constness.
                return const_cast<void*>(static_cast<const void*>(std::addressof(callable)));
            }

        } // namespace Detail

        /**
         * @brief Holds a learnable tensor's description and its identity across graphs.
         *
         * A copy is new state and draws its own identity; a move carries the identity away.
         */
        class Parameter
        {
            public:
                Parameter() = default;

                /**
                 * @brief Declares into builder.graph() unless that graph holds this parameter.
                 * @param role Selects the fill: fan-scaled for a weight, zero for a bias, one
                 *             for a scale.
                 * @param debug_name Name that must outlive the graph.
                 * @return This parameter's leaf in builder.graph().
                 * @pre A parameter declared before is given the same shape and dtype.
                 */
                Graph::ValueId declare(const Graph::GraphBuilder& builder,
                                       Shape shape,
                                       ParameterRole role,
                                       int64_t fan_in,
                                       int64_t fan_out,
                                       std::string_view debug_name,
                                       DType dtype = DType::Float32)
                {
                    return declare_described(builder, shape, Detail::init_of(role), role,
                                             fan_in, fan_out, debug_name, dtype);
                }

                /**
                 * @brief Declares as the role form does, with a stated fill in place of the role's.
                 * @param role What the parameter is for; role() reports it whatever @p init is.
                 * @param init The fill init() reports.
                 * @param debug_name Name that must outlive the graph.
                 * @return This parameter's leaf in builder.graph().
                 * @pre A parameter declared before is given the same shape and dtype.
                 */
                Graph::ValueId declare(const Graph::GraphBuilder& builder,
                                       Shape shape,
                                       ParameterRole role,
                                       ParameterInit init,
                                       int64_t fan_in,
                                       int64_t fan_out,
                                       std::string_view debug_name,
                                       DType dtype = DType::Float32)
                {
                    return declare_described(builder, shape, init, role, fan_in, fan_out,
                                             debug_name, dtype);
                }

                /** @brief Reports whether a declare has fixed this parameter's shape. */
                bool declared() const noexcept { return described_; }

                /** @brief Returns this parameter's leaf in @p graph, or NO_VALUE when none. */
                [[nodiscard]] Graph::ValueId id_in(const Graph::Graph& graph) const noexcept
                {
                    return graph.state_leaf(identity_.peek());
                }

                /** @brief Returns this parameter's identity, or K_NO_STATE when none was drawn. */
                [[nodiscard]] Graph::StateId state() const noexcept { return identity_.peek(); }

                /**
                 * @brief Returns the leaf of the most recent declare, or NO_VALUE before one.
                 * @return An id of the graph that declare ran in; id_in() serves any graph.
                 */
                Graph::ValueId id() const noexcept { return id_; }

                /** @brief The parameter's extents. */
                const Shape& shape() const noexcept { return shape_; }

                /** @brief The parameter's element type. */
                DType dtype() const noexcept { return dtype_; }

                /** @brief Returns what the parameter is for: a weight, a bias or a scale. */
                ParameterRole role() const noexcept { return role_; }

                /** @brief Returns the fill this parameter starts from. */
                [[nodiscard]] ParameterInit init() const noexcept { return init_; }

                /** @brief Inputs feeding one output unit. */
                int64_t fan_in() const noexcept { return fan_in_; }

                /** @brief Outputs one input unit feeds. */
                int64_t fan_out() const noexcept { return fan_out_; }

            private:
                /** @brief Keeps the first description and declares one leaf per graph. */
                Graph::ValueId declare_described(const Graph::GraphBuilder& builder,
                                                 Shape shape,
                                                 ParameterInit init,
                                                 ParameterRole role,
                                                 int64_t fan_in,
                                                 int64_t fan_out,
                                                 std::string_view debug_name,
                                                 DType dtype)
                {
                    if (described_)
                    {
                        SA_THROW_IF(shape != shape_ || dtype != dtype_,
                                    "Parameter '{}' was declared as {} {} and is now traced as "
                                    "{} {}",
                                    debug_name, shape_.to_string(), dtype_name(dtype_),
                                    shape.to_string(), dtype_name(dtype));
                    }
                    else
                    {
                        shape_ = shape;
                        init_ = init;
                        role_ = role;
                        fan_in_ = fan_in;
                        fan_out_ = fan_out;
                        dtype_ = dtype;
                        described_ = true;
                    }

                    Graph::Graph& graph = builder.graph();
                    const Graph::ValueId held = graph.state_leaf(identity_.peek());
                    if (held != Graph::K_NO_VALUE)
                    {
                        id_ = held;
                        return held;
                    }

                    id_ = builder.parameter(shape_, debug_name, dtype_);
                    graph.bind_state_leaf(identity_.get(), id_);
                    return id_;
                }

                Shape shape_{};

                /** @brief Names this parameter across the graphs it is declared in. */
                Graph::StateIdentity identity_{};

                /** @brief Holds the fill the first declare stated. */
                ParameterInit init_{};

                Graph::ValueId id_ = Graph::K_NO_VALUE;
                DType dtype_ = DType::Float32;
                ParameterRole role_ = ParameterRole::Weight;
                int64_t fan_in_ = 0;
                int64_t fan_out_ = 0;

                /** @brief Records that a declare has fixed the description. */
                bool described_ = false;
        };

        /**
         * @struct ParameterField
         * @brief One entry of a layer's fields() list: a name and a member.
         * @tparam M The layer type owning the member.
         */
        template<typename M>
        struct ParameterField
        {
            /** @brief The name this parameter carries in the parameter tree. */
            std::string_view name;

            /** @brief Which member of @c M it is. */
            Parameter M::* member;
        };

        /**
         * @brief Names one Buffer member in a layer's fields() list.
         * @tparam M The layer type owning the member.
         */
        template<typename M>
        struct BufferField
        {
            /** @brief The name this buffer carries in the state tree. */
            std::string_view name;

            /** @brief Which member of @c M it is. */
            Buffer M::* member;
        };

        /**
         * @brief Names one child member in a layer's fields() list.
         * @tparam M The layer type owning the member.
         * @tparam C The child's type; anything walk_state() can enumerate.
         */
        template<typename M, typename C>
        struct ChildField
        {
            /** @brief The name the child's state hangs under in the state tree. */
            std::string_view name;

            /** @brief Which member of @c M it is. */
            C M::* member;
        };

        /**
         * @brief Builds one entry of a layer's fields() list.
         * @param name The parameter's name, e.g. "weight".
         * @param member A pointer to the layer's member.
         * @return The field descriptor.
         */
        template<typename M>
        constexpr ParameterField<M> field(std::string_view name, Parameter M::* member) noexcept
        {
            return ParameterField<M>{name, member};
        }

        /**
         * @brief Builds the fields() entry of a buffer.
         * @param name The buffer's name, e.g. "running_mean".
         * @param member A pointer to the layer's member.
         */
        template<typename M>
        constexpr BufferField<M> buffer(std::string_view name, Buffer M::* member) noexcept
        {
            return BufferField<M>{name, member};
        }

        /**
         * @brief Builds the fields() entry of a child layer or child container.
         * @param name The child's name, e.g. "cv1"; its state is named "cv1.<leaf>".
         * @param member A pointer to the layer's member.
         */
        template<typename M, typename C>
        constexpr ChildField<M, C> child(std::string_view name, C M::* member) noexcept
        {
            return ChildField<M, C>{name, member};
        }

        /**
         * @class ParameterVisitor
         * @brief Non-owning type-erased parameter callback wrapper.
         */
        class ParameterVisitor
        {
            public:
                /**
                 * @brief Wraps a callable reference that must outlive this visitor.
                 * @tparam Fn Callable type invocable as (const std::string&, Parameter&).
                 * @param callable Target callback stored by address.
                 */
                template<typename Fn,
                         typename = std::enable_if_t<
                             !std::is_same<std::decay_t<Fn>, ParameterVisitor>::value>>
                ParameterVisitor(Fn& callable) noexcept
                    : target_(Detail::erase_callable(callable)),
                      invoke_([](void* target, const std::string& name, Parameter& parameter)
                              {
                                  (*static_cast<Fn*>(target))(name, parameter);
                              })
                {
                }

                /**
                 * @brief Applies the wrapped callback.
                 * @param name The parameter's dotted name.
                 * @param parameter The parameter itself.
                 */
                void operator()(const std::string& name, Parameter& parameter) const
                {
                    invoke_(target_, name, parameter);
                }

            private:
                void* target_ = nullptr;
                void (*invoke_)(void*, const std::string&, Parameter&) = nullptr;
        };

        namespace Detail
        {
            /**
             * @struct ChildProbe
             * @brief Stand-in visitor detecting whether a type accepts child traversal.
             */
            struct ChildProbe
            {
                /**
                 * @brief Accepts any (index, child) pair, and does nothing with it.
                 * @tparam T The child layer's type, deduced.
                 */
                template<typename T>
                void operator()(std::size_t, T&) const noexcept {}
            };

            /**
             * @struct IsModule
             * @brief Detects @ref SushiAI::NN::Module; see it for the contract.
             * @tparam T The candidate layer type.
             */
            template<typename T, typename = void>
            struct IsModule : std::false_type {};

            /** @brief The satisfied case: T has fields() and a conforming forward(). */
            template<typename T>
            struct IsModule<T, std::void_t<
                decltype(T::fields()),
                std::enable_if_t<std::is_same<
                    decltype(std::declval<T&>().forward(
                        std::declval<const Graph::GraphBuilder&>(),
                        std::declval<Graph::ValueId>())),
                    Graph::ValueId>::value>>>
                : std::true_type {};

            /**
             * @struct IsModuleContainer
             * @brief Detects @ref SushiAI::NN::ModuleContainer; see it for the contract.
             * @tparam T The candidate container type.
             */
            template<typename T, typename = void>
            struct IsModuleContainer : std::false_type {};

            /** @brief The satisfied case: T accepts a child visitor. */
            template<typename T>
            struct IsModuleContainer<T, std::void_t<
                decltype(std::declval<T&>().visit_children(ChildProbe{}))>>
                : std::true_type {};

            /**
             * @struct HasParameterWalk
             * @brief Detects whether a type provides a walk_parameters method.
             * @tparam T Candidate type inspected for parameter walk support.
             */
            template<typename T, typename = void>
            struct HasParameterWalk : std::false_type {};

            /** @brief The satisfied case: T offers walk_parameters(prefix, visitor). */
            template<typename T>
            struct HasParameterWalk<T, std::void_t<
                decltype(std::declval<T&>().walk_parameters(
                    std::declval<std::string_view>(),
                    std::declval<const ParameterVisitor&>()))>>
                : std::true_type {};

            /** @brief Joins a parameter-path prefix to a leaf name, PyTorch style. */
            inline std::string join(std::string_view prefix, std::string_view leaf)
            {
                if (prefix.empty()) return std::string(leaf);

                std::string joined;
                joined.reserve(prefix.size() + leaf.size() + 1);
                joined.append(prefix).push_back('.');
                joined.append(leaf);
                return joined;
            }

            /** @brief Collects the name of every entry of @p fields, in order. */
            template<typename... Fs, std::size_t... Indices>
            [[nodiscard]] constexpr std::array<std::string_view, sizeof...(Fs)>
            field_names(const std::tuple<Fs...>& fields, std::index_sequence<Indices...>) noexcept
            {
                return {{std::get<Indices>(fields).name...}};
            }

            /** @brief Reports whether no two entries of one fields() list share a name. */
            template<typename... Fs>
            [[nodiscard]] constexpr bool has_unique_names(const std::tuple<Fs...>& fields) noexcept
            {
                const std::array<std::string_view, sizeof...(Fs)> names =
                    field_names(fields, std::index_sequence_for<Fs...>{});

                for (std::size_t first = 0; first < sizeof...(Fs); ++first)
                {
                    for (std::size_t second = first + 1; second < sizeof...(Fs); ++second)
                    {
                        if (names[first] == names[second])
                            return false;
                    }
                }

                return true;
            }

            /** @brief Accepts any piece of state and does nothing with it. */
            struct IgnoredState
            {
                /** @brief Discards one named piece of state. */
                template<typename S>
                void operator()(const std::string&, S&) const noexcept
                {

                }
            };

        } // namespace Detail

        /** @brief Receives every piece of state a walk reaches; non-owning. */
        class StateVisitor final
        {
            public:
                /**
                 * @brief Wraps two callables that must outlive this visitor.
                 * @param on_parameter Called as on_parameter(const std::string&, Parameter&).
                 * @param on_buffer Called as on_buffer(const std::string&, Buffer&).
                 */
                template<typename OnParameter, typename OnBuffer>
                StateVisitor(OnParameter& on_parameter, OnBuffer& on_buffer) noexcept
                    : parameter_target_(Detail::erase_callable(on_parameter)),
                      buffer_target_(Detail::erase_callable(on_buffer)),
                      invoke_parameter_([](void* target, const std::string& name, Parameter& state)
                                        {
                                            (*static_cast<OnParameter*>(target))(name, state);
                                        }),
                      invoke_buffer_([](void* target, const std::string& name, Buffer& state)
                                     {
                                         (*static_cast<OnBuffer*>(target))(name, state);
                                     })
                {

                }

                /** @brief Hands one named parameter to the parameter callable. */
                void parameter(const std::string& name, Parameter& parameter) const
                {
                    invoke_parameter_(parameter_target_, name, parameter);
                }

                /** @brief Hands one named buffer to the buffer callable. */
                void buffer(const std::string& name, Buffer& buffer) const
                {
                    invoke_buffer_(buffer_target_, name, buffer);
                }

            private:
                /** @brief Points at the parameter callable. */
                void* parameter_target_ = nullptr;

                /** @brief Points at the buffer callable. */
                void* buffer_target_ = nullptr;

                /** @brief Calls the parameter callable through its erased address. */
                void (*invoke_parameter_)(void*, const std::string&, Parameter&) = nullptr;

                /** @brief Calls the buffer callable through its erased address. */
                void (*invoke_buffer_)(void*, const std::string&, Buffer&) = nullptr;
        };

        /**
         * @brief Visits every declared parameter and defined buffer of @p module, in one order.
         *
         * Tries, in order: the type's own walk_state(prefix, visitor); walk_parameters(prefix,
         * visitor), which yields parameters only; visit_children, named by index; fields(),
         * entry by entry in declaration order.
         * @param prefix Dotted prefix the names of this module's state hang under.
         */
        template<typename M>
        void walk_state(M& module, std::string_view prefix, const StateVisitor& visitor);

        namespace Detail
        {
            /** @brief Detects whether a type walks its own state through walk_state(). */
            template<typename T, typename = void>
            struct HasStateWalk : std::false_type
            {

            };

            /** @brief Holds for a T that offers walk_state(prefix, visitor). */
            template<typename T>
            struct HasStateWalk<T, std::void_t<
                decltype(std::declval<T&>().walk_state(
                    std::declval<std::string_view>(),
                    std::declval<const StateVisitor&>()))>>
                : std::true_type
            {

            };

            /** @brief Hands one parameter entry of @p module to @p visitor if it is declared. */
            template<typename M>
            void visit_field(M& module, ParameterField<M> descriptor,
                             std::string_view prefix, const StateVisitor& visitor)
            {
                Parameter& parameter = module.*(descriptor.member);
                if (!parameter.declared())
                    return;

                visitor.parameter(join(prefix, descriptor.name), parameter);
            }

            /** @brief Hands one buffer entry of @p module to @p visitor if it is defined. */
            template<typename M>
            void visit_field(M& module, BufferField<M> descriptor,
                             std::string_view prefix, const StateVisitor& visitor)
            {
                Buffer& buffer = module.*(descriptor.member);
                if (!buffer.declared())
                    return;

                visitor.buffer(join(prefix, descriptor.name), buffer);
            }

            /** @brief Walks one child entry of @p module under the child's own name. */
            template<typename M, typename C>
            void visit_field(M& module, ChildField<M, C> descriptor,
                             std::string_view prefix, const StateVisitor& visitor)
            {
                walk_state(module.*(descriptor.member), join(prefix, descriptor.name), visitor);
            }

        } // namespace Detail

        /**
         * @var Module
         * @brief Concept variable checking whether T defines fields() and forward().
         * @tparam T Candidate type checked against the module contract.
         */
        template<typename T>
        inline constexpr bool Module = Detail::IsModule<T>::value;

        /**
         * @var ModuleContainer
         * @brief Concept variable checking whether T supports visit_children.
         * @tparam T Candidate container type.
         */
        template<typename T>
        inline constexpr bool ModuleContainer = Detail::IsModuleContainer<T>::value;

        template<typename M>
        void walk_state(M& module, std::string_view prefix, const StateVisitor& visitor)
        {
            if constexpr (Detail::HasStateWalk<M>::value)
            {
                module.walk_state(prefix, visitor);
            }
            else if constexpr (Detail::HasParameterWalk<M>::value)
            {
                const auto forward_parameter = [&visitor](const std::string& name,
                                                          Parameter& parameter)
                {
                    visitor.parameter(name, parameter);
                };
                const ParameterVisitor erased(forward_parameter);
                module.walk_parameters(prefix, erased);
            }
            else if constexpr (ModuleContainer<M>)
            {
                module.visit_children([&](std::size_t index, auto& child)
                {
                    walk_state(child, Detail::join(prefix, std::to_string(index)), visitor);
                });
            }
            else
            {
                static_assert(Detail::has_unique_names(M::fields()),
                              "Two entries of one fields() list share a name. Every parameter, "
                              "buffer and child of a layer needs its own name, because a "
                              "checkpoint stores each one under it. Check the layer named in "
                              "the instantiation below, not this line.");

                std::apply([&](auto... descriptors)
                {
                    (Detail::visit_field(module, descriptors, prefix, visitor), ...);
                }, M::fields());
            }
        }

        /**
         * @brief Visits the declared parameters of @p module in the order walk_state() yields.
         * @tparam M Module or container type.
         * @param prefix Dotted prefix representing parameter hierarchy.
         * @param visitor Called as visitor(const std::string& name, Parameter&).
         */
        template<typename M, typename Fn>
        void for_each_parameter(M& module, std::string_view prefix, Fn&& visitor)
        {
            const Detail::IgnoredState ignored;
            const StateVisitor parameters_only(visitor, ignored);
            walk_state(module, prefix, parameters_only);
        }

        /**
         * @brief Visits every declared parameter of @p module from the root.
         * @param module The model to walk.
         * @param visitor Called as visitor(const std::string& name, Parameter&).
         */
        template<typename M, typename Fn>
        void for_each_parameter(M& module, Fn&& visitor)
        {
            for_each_parameter(module, std::string_view{}, std::forward<Fn>(visitor));
        }

        /**
         * @brief Visits the defined buffers of @p module in the order walk_state() yields.
         * @tparam M Module or container type.
         * @param prefix Dotted prefix representing buffer hierarchy.
         * @param visitor Called as visitor(const std::string& name, Buffer&).
         */
        template<typename M, typename Fn>
        void for_each_buffer(M& module, std::string_view prefix, Fn&& visitor)
        {
            const Detail::IgnoredState ignored;
            const StateVisitor buffers_only(ignored, visitor);
            walk_state(module, prefix, buffers_only);
        }

        /**
         * @brief Visits every defined buffer of @p module from the root.
         * @param module The model to walk.
         * @param visitor Called as visitor(const std::string& name, Buffer&).
         */
        template<typename M, typename Fn>
        void for_each_buffer(M& module, Fn&& visitor)
        {
            for_each_buffer(module, std::string_view{}, std::forward<Fn>(visitor));
        }

        /**
         * @brief Counts @p module's declared parameters.
         * @param module The model to walk.
         * @return How many parameters the walk yields.
         */
        template<typename M>
        std::size_t parameter_count(M& module)
        {
            std::size_t count = 0;
            for_each_parameter(module, [&count](const std::string&, Parameter&) { ++count; });
            return count;
        }

        /**
         * @brief Counts @p module's defined buffers.
         * @param module The model to walk.
         * @return How many buffers the walk yields.
         */
        template<typename M>
        std::size_t buffer_count(M& module)
        {
            std::size_t count = 0;
            for_each_buffer(module, [&count](const std::string&, Buffer&) { ++count; });
            return count;
        }

        /**
         * @brief Sums the element counts of @p module's declared parameters.
         * @param module The model to walk.
         * @return How many learnable scalars the model holds.
         */
        template<typename M>
        int64_t parameter_elements(M& module)
        {
            int64_t total = 0;
            for_each_parameter(module, [&total](const std::string&, Parameter& parameter)
            {
                total += parameter.shape().num_elements();
            });
            return total;
        }

    } // namespace NN
} // namespace SushiAI
