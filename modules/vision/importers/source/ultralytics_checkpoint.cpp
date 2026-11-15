/**************************************************************************/
/* ultralytics_checkpoint.cpp                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file ultralytics_checkpoint.cpp
 * @brief Defines the walk of the module tree of an Ultralytics checkpoint.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/ultralytics_checkpoint.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/pickle_data.hpp>
#include <SushiAI/io/pickle_reader.hpp>
#include <SushiAI/io/torch_archive.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /**
             * @brief Lists the module prefixes whose classes a checkpoint pickles: Ultralytics',
             *        the YOLOv5 repository's and PyTorch's layers.
             */
            constexpr std::string_view K_MODULE_GLOBALS[] = {"ultralytics.", "models.common.",
                                                             "models.yolo.", "torch.nn.modules."};

            /** @brief Holds the deepest module nesting the walk follows. */
            constexpr std::size_t K_MAX_DEPTH = 32;

            /** @brief Names the buffer of a normalisation that counts its steps. */
            constexpr std::string_view K_STEP_COUNTER = "num_batches_tracked";

            /** @brief Returns the pairs of a dict or of a call that rebuilt an ordered dict. */
            const std::vector<std::pair<IO::PickleRef, IO::PickleRef>>* pairs_of(
                const IO::PickleValue* value)
            {
                if (value == nullptr)
                    return nullptr;
                if (value->kind == IO::PickleKind::Dict || value->kind == IO::PickleKind::Call)
                    return &value->entries;
                return nullptr;
            }

            /** @brief Returns whether the set @p names, as pickled, holds @p name. */
            bool set_holds(const IO::PickleValue* names, std::string_view name)
            {
                if (names == nullptr)
                    return false;
                const std::vector<IO::PickleRef>* items = &names->items;
                if (names->kind == IO::PickleKind::Call && !names->items.empty())
                    items = &names->items.front()->items;
                for (const IO::PickleRef& item : *items)
                {
                    if (item->kind == IO::PickleKind::String && item->text == name)
                        return true;
                }
                return false;
            }

            /** @brief Collects the tensors of the modules of one checkpoint. */
            class ModuleWalk final
            {
                public:
                    /** @brief Holds the archive the tensors are read from and where they go. */
                    ModuleWalk(const IO::TorchArchive& archive, UltralyticsCheckpoint& out) noexcept
                        : archive_(archive), out_(out)
                    {

                    }

                    ModuleWalk(const ModuleWalk&) = delete;
                    ModuleWalk& operator=(const ModuleWalk&) = delete;
                    ModuleWalk(ModuleWalk&&) = delete;
                    ModuleWalk& operator=(ModuleWalk&&) = delete;
                    ~ModuleWalk() = default;

                    /** @brief Collects the tensors of @p module and its children under @p path. */
                    Status walk(const IO::PickleValue& module, const std::string& path,
                                std::size_t depth)
                    {
                        SA_FAIL_IF(depth > K_MAX_DEPTH, StatusCode::InvalidFile,
                                   "the module '{}' nests deeper than {} levels", path,
                                   K_MAX_DEPTH);
                        SA_FAIL_IF(module.kind != IO::PickleKind::Call || module.state == nullptr,
                                   StatusCode::InvalidFile,
                                   "the module '{}' is not a pickled module object", path);
                        const IO::PickleValue& state = *module.state;
                        if (!path.empty())
                            out_.modules.push_back(UltralyticsModuleClass{path, class_name(module)});
                        const IO::PickleValue* groups = IO::find_entry(state, "groups");
                        const std::int64_t group_count =
                            groups != nullptr && groups->kind == IO::PickleKind::Int ? groups->integer
                                                                                     : 1;
                        const IO::PickleValue* hidden =
                            IO::find_entry(state, "_non_persistent_buffers_set");
                        SA_TRY(collect(IO::find_entry(state, "_parameters"), path, group_count,
                                       nullptr));
                        SA_TRY(collect(IO::find_entry(state, "_buffers"), path, group_count,
                                       hidden));

                        const auto* children = pairs_of(IO::find_entry(state, "_modules"));
                        if (children == nullptr)
                            return Status();
                        for (const auto& child : *children)
                        {
                            if (child.second->kind == IO::PickleKind::None ||
                                child.first->kind != IO::PickleKind::String)
                                continue;
                            SA_TRY(walk(*child.second, join(path, child.first->text), depth + 1));
                        }
                        return Status();
                    }

                private:
                    /** @brief Returns the class name of @p module, the last word of its Global. */
                    static std::string class_name(const IO::PickleValue& module)
                    {
                        if (module.callable == nullptr)
                            return std::string();
                        const std::string& text = module.callable->text;
                        const std::size_t space = text.rfind(' ');
                        return space == std::string::npos ? text : text.substr(space + 1);
                    }

                    /** @brief Returns @p path and @p name joined by a dot. */
                    static std::string join(const std::string& path, const std::string& name)
                    {
                        return path.empty() ? name : path + "." + name;
                    }

                    /** @brief Collects the tensors of the dict @p tensors of one module. */
                    Status collect(const IO::PickleValue* tensors, const std::string& path,
                                   std::int64_t groups, const IO::PickleValue* hidden)
                    {
                        const auto* entries = pairs_of(tensors);
                        if (entries == nullptr)
                            return Status();
                        for (const auto& entry : *entries)
                        {
                            const IO::PickleValue& value = *entry.second;
                            if (value.kind == IO::PickleKind::None ||
                                entry.first->kind != IO::PickleKind::String ||
                                set_holds(hidden, entry.first->text))
                                continue;
                            const std::string name = join(path, entry.first->text);
                            SA_FAIL_IF(!IO::TorchArchive::holds_tensor(value),
                                       StatusCode::InvalidFile,
                                       "the entry '{}' is not a tensor", name);
                            SA_TRY_ASSIGN(const IO::TorchTensor tensor,
                                          archive_.tensor(value, name));
                            if (!is_real_float(tensor.dtype) && entry.first->text == K_STEP_COUNTER)
                            {
                                out_.skipped.push_back(name);
                                continue;
                            }
                            SA_TRY_ASSIGN(std::vector<float> values,
                                          archive_.read_float32(tensor, name));
                            UltralyticsTensor read;
                            read.name = name;
                            read.groups = groups;
                            read.shape = tensor.shape;
                            read.values = std::move(values);
                            out_.tensors.push_back(std::move(read));
                        }
                        return Status();
                    }

                    /** @brief Holds the archive the tensors are read from. */
                    const IO::TorchArchive& archive_;

                    /** @brief Holds the checkpoint being filled. */
                    UltralyticsCheckpoint& out_;
            };

            /** @brief Returns the model object the checkpoint @p root holds: EMA first. */
            Result<const IO::PickleValue*> find_model(const IO::PickleValue& root,
                                                      std::string_view origin)
            {
                SA_FAIL_IF(root.kind != IO::PickleKind::Dict, StatusCode::InvalidFile,
                           "'{}' holds no checkpoint dictionary", origin);
                const IO::PickleValue* model = IO::find_entry(root, "ema");
                if (model == nullptr || model->kind == IO::PickleKind::None)
                    model = IO::find_entry(root, "model");
                SA_FAIL_IF(model == nullptr || model->kind != IO::PickleKind::Call ||
                               model->state == nullptr,
                           StatusCode::InvalidFile,
                           "'{}' holds neither an 'ema' nor a 'model' object", origin);
                return model;
            }
        } // namespace

        Result<UltralyticsCheckpoint> read_ultralytics_checkpoint(
            SushiRuntime::span<const std::byte> bytes, std::string_view origin) noexcept
        {
            SA_TRY_ASSIGN(const IO::TorchArchive archive,
                          IO::TorchArchive::read(bytes, K_MODULE_GLOBALS, origin));
            SA_TRY_ASSIGN(const IO::PickleValue* const model, find_model(archive.root(), origin));

            UltralyticsCheckpoint out;
            const IO::PickleValue* const yaml = IO::find_entry(*model->state, "yaml");
            SA_FAIL_IF(yaml == nullptr, StatusCode::InvalidFile,
                       "'{}': the model object holds no 'yaml' description", origin);
            SA_TRY_ASSIGN(out.architecture, IO::pickle_to_value(*yaml, origin));
            const IO::PickleValue* const arguments = IO::find_entry(archive.root(), "train_args");
            if (arguments != nullptr && arguments->kind == IO::PickleKind::Dict)
            {
                SA_TRY_ASSIGN(out.training_arguments, IO::pickle_to_value(*arguments, origin));
            }

            const auto* children = pairs_of(IO::find_entry(*model->state, "_modules"));
            const IO::PickleValue* layers = nullptr;
            if (children != nullptr)
            {
                for (const auto& child : *children)
                {
                    if (child.first->kind == IO::PickleKind::String && child.first->text == "model")
                        layers = child.second.get();
                }
            }
            SA_FAIL_IF(layers == nullptr, StatusCode::InvalidFile,
                       "'{}': the model object holds no 'model' layer list", origin);
            ModuleWalk walk(archive, out);
            SA_TRY(SA_ANNOTATE(walk.walk(*layers, std::string(), 0), "'{}'", origin));
            return out;
        }

        Result<UltralyticsCheckpoint> load_ultralytics_checkpoint(
            const std::filesystem::path& path) noexcept
        {
            std::ifstream in(path, std::ios::binary);
            SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                       path.string());
            const std::vector<char> text((std::istreambuf_iterator<char>(in)),
                                         std::istreambuf_iterator<char>());
            SA_FAIL_IF(in.bad(), StatusCode::InvalidFile, "failed reading '{}'", path.string());
            const SushiRuntime::span<const std::byte> bytes(
                reinterpret_cast<const std::byte*>(text.data()), text.size());
            return read_ultralytics_checkpoint(bytes, path.string());
        }
    } // namespace Vision::Importers
} // namespace SushiAI
