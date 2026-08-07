/**************************************************************************/
/* module.hpp                                                             */
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

#pragma once

/**
 * @file module.hpp
 * @brief What a layer is, and how its parameters are found.
 *
 * @section registration No base class, no register_module
 *
 * The usual design has every layer inherit a Module base and call
 * @c register_module (or assign through a hooked setter) for each parameter it
 * owns. Its failure mode is the reason this library does not use it: forget
 * the call and everything still compiles, the layer still runs, and the
 * forward pass is still correct - the optimizer simply never sees that
 * parameter, so one layer silently never learns. What you observe is a model
 * that converges to a slightly worse loss than it should, which is
 * indistinguishable from a hyperparameter you have not tuned yet. There is no
 * error to catch and no assertion that would have fired.
 *
 * Here a layer is an aggregate struct that publishes its parameters through a
 * @c static @c constexpr @c fields() list. Forgetting an entry is not a
 * runtime accident; it is a visible omission in a two-line function that sits
 * directly beneath the members it lists. The @c Module trait checks the
 * contract where the layer is declared, not where it is first used, so a
 * mistake names the layer rather than some template instantiation four frames
 * deep.
 *
 * One generic walk - @c for_each_parameter - derives the whole named-parameter
 * tree from those lists. The optimizer, the initializer and any future
 * serializer consume the same walk, so they cannot disagree about which
 * tensors are parameters or what they are called. Names follow PyTorch's
 * convention ("0.weight", "0.bias", "2.weight"), which makes a checkpoint
 * comparison against a reference implementation a diff rather than a
 * translation exercise.
 *
 * @section runtime A layer list that only exists at run time
 *
 * A model built from a JSON file cannot publish a @c fields() list, because
 * @c fields() is @c static and returns member pointers of one concrete type,
 * and it cannot be walked as a container either, because its children are all
 * the same @c ILayer& and a @c fields() walk over that finds nothing while
 * reporting no error. So @c for_each_parameter has a third case: a type that
 * offers @c walk_parameters is asked to enumerate itself, through the
 * type-erased @ref ParameterVisitor.
 *
 * The arm is strictly additive - no type that compiled before it existed takes
 * it, and layers.hpp asserts exactly that - and it is what lets one walk serve
 * both the compile-time @c Sequential and the runtime @c LayerStack. Because
 * everything downstream is a template over this walk, nothing else in the
 * library learned that a second kind of model exists.
 *
 * Host lane: no SushiBLAS, no SYCL.
 */

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
         * @brief A learnable tensor's description and its identity in the graph.
         *
         * Default-constructible and empty, so a layer holding one stays an
         * aggregate and @c Linear{784, 128} keeps working. It gains a shape and
         * an id the first time the layer's forward() declares it; until then
         * declared() is false and the walks skip it, which is also how an
         * optional bias disappears cleanly rather than needing a special case
         * in every consumer.
         */
        class Parameter
        {
            public:
                Parameter() = default;

                /**
                 * @brief Declares this parameter into the graph, once.
                 *
                 * Idempotent: a second call with the same layer returns the
                 * existing id rather than declaring a duplicate, so a model may
                 * be traced more than once without accumulating parameters.
                 *
                 * @param builder Where to declare it.
                 * @param shape The parameter's extents.
                 * @param role Whether it is a weight or a bias.
                 * @param fan_in Inputs feeding one output unit, for initialisation.
                 * @param fan_out Outputs one input unit feeds, for initialisation.
                 * @param debug_name A label; must be a literal or otherwise outlive the graph.
                 * @param dtype The element type.
                 * @return This parameter's value id.
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
         * @brief A non-owning reference to a parameter callback, for a virtual walk.
         *
         * @ref for_each_parameter takes its visitor as a template parameter,
         * which a virtual function cannot be. A layer list decided at run time
         * therefore needs the callback type-erased before it can cross the
         * @c ILayer boundary - and erased without allocating, because the walk
         * runs on the trace path and inside checkpoint I/O, neither of which
         * should heap-allocate to describe a callable the caller already holds.
         *
         * It *refers* to the callable rather than owning it, exactly as a
         * @c string_view refers to characters: the referent must outlive the
         * visitor. Every construction site in this library passes a named
         * local, so there is no temporary to dangle.
         */
        class ParameterVisitor
        {
            public:
                /**
                 * @brief Wraps @p callable, which must outlive this visitor.
                 *
                 * Excluded for @c ParameterVisitor itself so that wrapping one
                 * a second time copies it rather than building a trampoline
                 * that jumps through another trampoline.
                 *
                 * @tparam Fn Anything callable as (const std::string&, Parameter&).
                 * @param callable The callback; stored by address.
                 */
                template<typename Fn,
                         typename = std::enable_if_t<
                             !std::is_same<std::decay_t<Fn>, ParameterVisitor>::value>>
                ParameterVisitor(Fn& callable) noexcept
                    // The const_cast is what lets a const callable - which is
                    // what `Fn` deduces to whenever the walk is already inside
                    // one erasure - be referred to through one `void*` member.
                    // It is never cast back to a non-const pointer: `Fn` below
                    // carries the constness, so the call is const-correct.
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
             * @brief A stand-in child visitor, used only to ask whether a type takes one.
             *
             * @ref IsModuleContainer needs to name the argument @c visit_children
             * would be called with, and a lambda cannot be named: writing one inside
             * a @c decltype - which is what the C++20 concept this replaces did - is
             * only legal from C++20 on. So the visitor gets a name.
             *
             * It is generic in its second parameter and fixed in its first for the
             * same reason the lambda was: a container hands each child's index and
             * the child itself, and the children have different types. A probe that
             * only accepted one child type would still compile here and would quietly
             * answer "not a container" for every real container, which
             * @c for_each_parameter's @c if @c constexpr would then read as "this is a
             * leaf" - walking none of the nested parameters and reporting no error.
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
             * @brief Detects a layer that enumerates its own parameters at run time.
             *
             * Neither of the other two cases can serve a @c LayerStack, which
             * holds @c std::unique_ptr<ILayer>: @c fields() is static and
             * returns member pointers of one concrete type, and
             * @c visit_children would hand every child out as the same
             * @c ILayer& - so a @c fields() walk over it would find nothing and
             * report no error, which is the silent kind of wrong this file
             * exists to refuse. Such a type publishes @c walk_parameters
             * instead and this detector routes the walk to it.
             *
             * @tparam T The candidate layer type.
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
         * @brief The contract a layer satisfies: it lists its parameters and it traces.
         *
         * A layer must offer a static @c fields() and a
         * @c forward(const @c GraphBuilder&, @c ValueId) whose return type is
         * **exactly** @c Graph::ValueId - @c std::is_same, not convertible-to, so a
         * @c forward() returning something that merely converts fails the contract it
         * should fail rather than passing it by accident.
         *
         * Checked where a layer is declared (see the static_asserts in layers.hpp),
         * so a missing fields() or a forward() with the wrong signature is reported
         * against the layer rather than against whatever template first tried to use
         * it.
         *
         * A variable template rather than a C++20 concept, because the library is
         * built at C++17 (see docs/ARCHITECTURE.md §1). Every use site is spelled the
         * same either way - @c Module<Linear> - so this is the same contract asked the
         * same way; what it cannot do is constrain a template parameter directly,
         * which is why Sequential carries a static_assert instead.
         *
         * @tparam T The candidate layer type.
         */
        template<typename T>
        inline constexpr bool Module = Detail::IsModule<T>::value;

        /**
         * @var ModuleContainer
         * @brief A module built out of other modules, such as Sequential.
         *
         * Recognised by offering a child visit rather than by inheriting
         * anything, so a user's own container joins the parameter walk without
         * touching this file.
         *
         * @tparam T The candidate container type.
         */
        template<typename T>
        inline constexpr bool ModuleContainer = Detail::IsModuleContainer<T>::value;

        /**
         * @brief Visits every declared parameter of @p module, deepest name first.
         *
         * The single walk the optimizer, the initializer and any serializer all
         * go through. Three cases, tried in this order:
         *
         * 1. A layer that enumerates its own parameters at run time - a
         *    @c LayerStack, whose list came from a config file - is asked to do
         *    so through @c walk_parameters. First, so a type that offers both a
         *    walk and a @c fields() list is enumerated by the walk it wrote
         *    rather than by the empty list it inherited.
         * 2. A container recurses with its child index appended to the prefix.
         * 3. A leaf layer applies its @c fields() list.
         *
         * Everything downstream is @c template<typename @c M> over this one
         * walk, so all three cases reach @c NN::initialize,
         * @c Train::build_graph, @c NN::parameter_elements and the checkpoint
         * I/O without any of them knowing there is more than one.
         *
         * @param module The model or layer to walk.
         * @param prefix The name prefix; empty at the root.
         * @param visitor Called as visitor(const std::string& name, Parameter&).
         */
        template<typename M, typename Fn>
        void for_each_parameter(M& module, std::string_view prefix, Fn&& visitor)
        {
            if constexpr (Detail::HasParameterWalk<M>::value)
            {
                // ParameterVisitor refers to its callable rather than owning
                // it, so the erased visitor is a named local pointing at
                // `visitor` - which is itself a named lvalue here, whatever
                // Fn deduced to.
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
