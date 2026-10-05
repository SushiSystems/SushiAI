/****************************************************************/
/* module.hpp                                                   */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file module.hpp
 * @brief Defines module concepts, parameters, and parameter traversal utilities.
 * @author Mustafa Garip
 */

#pragma once

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
            WEIGHT,
            /** @brief A bias vector; always starts at zero. */
            BIAS,
        };

        /**
         * @class Parameter
         * @brief Holds a learnable tensor's metadata and graph value identifier.
         */
        class Parameter
        {
            public:
                Parameter() = default;

                /**
                 * @brief Declares this parameter into the graph idempotently.
                 * @param builder Graph builder where the parameter node is registered.
                 * @param shape Extents of the parameter tensor.
                 * @param role Role indicating whether parameter is weight or bias.
                 * @param debug_name Name that must outlive the graph builder.
                 * @return ValueId assigned to this parameter.
                 */
                Graph::ValueId declare(const Graph::GraphBuilder& builder,
                                       Shape shape,
                                       ParameterRole role,
                                       int64_t fan_in,
                                       int64_t fan_out,
                                       std::string_view debug_name,
                                       DType dtype = DType::FLOAT32)
                {
                    if (id_ != Graph::NO_VALUE) return id_;

                    shape_ = shape;
                    role_ = role;
                    fan_in_ = fan_in;
                    fan_out_ = fan_out;
                    dtype_ = dtype;
                    id_ = builder.parameter(shape, debug_name, dtype);
                    return id_;
                }

                /** @brief Whether forward() has declared this parameter yet. */
                bool declared() const noexcept { return id_ != Graph::NO_VALUE; }

                /** @brief The parameter's value id, or Graph::NO_VALUE before declaration. */
                Graph::ValueId id() const noexcept { return id_; }

                /** @brief The parameter's extents. */
                const Shape& shape() const noexcept { return shape_; }

                /** @brief The parameter's element type. */
                DType dtype() const noexcept { return dtype_; }

                /** @brief Whether this is a weight or a bias. */
                ParameterRole role() const noexcept { return role_; }

                /** @brief Inputs feeding one output unit. */
                int64_t fan_in() const noexcept { return fan_in_; }

                /** @brief Outputs one input unit feeds. */
                int64_t fan_out() const noexcept { return fan_out_; }

            private:
                Shape shape_{};
                Graph::ValueId id_ = Graph::NO_VALUE;
                DType dtype_ = DType::FLOAT32;
                ParameterRole role_ = ParameterRole::WEIGHT;
                int64_t fan_in_ = 0;
                int64_t fan_out_ = 0;
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
                    // Invariant: Target pointer preserves constness through deduced Fn invocation.
                    : target_(const_cast<void*>(static_cast<const void*>(std::addressof(callable)))),
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

            /** @brief Applies @p visitor to one declared field of @p module. */
            template<typename M, typename Fn>
            void visit_field(M& module, ParameterField<M> descriptor,
                             std::string_view prefix, Fn& visitor)
            {
                Parameter& parameter = module.*(descriptor.member);
                if (!parameter.declared()) return;
                visitor(join(prefix, descriptor.name), parameter);
            }
        }

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

        /**
         * @brief Traverses declared parameters of a module in deterministic order.
         * @tparam M Module or container type.
         * @param module Root module or layer instance to inspect.
         * @param prefix Dotted prefix representing parameter hierarchy.
         * @param visitor Callback receiving parameter name and reference.
         */
        template<typename M, typename Fn>
        void for_each_parameter(M& module, std::string_view prefix, Fn&& visitor)
        {
            if constexpr (Detail::HasParameterWalk<M>::value)
            {
                // Invariant: Erased visitor references local lvalue without heap allocation.
                const ParameterVisitor erased(visitor);
                module.walk_parameters(prefix, erased);
            }
            else if constexpr (ModuleContainer<M>)
            {
                module.visit_children([&](std::size_t index, auto& child)
                {
                    for_each_parameter(child, Detail::join(prefix, std::to_string(index)), visitor);
                });
            }
            else
            {
                std::apply([&](auto... descriptors)
                {
                    (Detail::visit_field(module, descriptors, prefix, visitor), ...);
                }, M::fields());
            }
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
