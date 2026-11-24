/**************************************************************************/
/* torch_writer.cpp                                                       */
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
 * @file torch_writer.cpp
 * @brief Implements the writer of a PyTorch state dictionary file.
 * @author Mustafa Garip
 */

#include <SushiAI/io/torch_writer.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/pickle_reader.hpp>
#include <SushiAI/io/pickle_writer.hpp>
#include <SushiAI/io/zip_writer.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the alignment of each storage entry, as `torch.save` aligns them. */
            constexpr std::size_t K_STORAGE_ALIGNMENT = 64;

            /** @brief Holds the serialisation version `torch.save` writes. */
            constexpr std::string_view K_TORCH_VERSION = "3\n";

            /** @brief Holds the byte order the file states. */
            constexpr std::string_view K_BYTE_ORDER = "little";

            /** @brief Returns the storage global of @p dtype, or empty for none. */
            [[nodiscard]] std::string_view storage_global(DType dtype) noexcept
            {
                switch (dtype)
                {
                    case DType::Half:    return "torch HalfStorage";
                    case DType::Float32: return "torch FloatStorage";
                    case DType::Float64: return "torch DoubleStorage";
                    default:             return {};
                }
            }

            /** @brief Returns a node of @p kind. */
            [[nodiscard]] PickleRef node(PickleKind kind)
            {
                PickleRef value = std::make_shared<PickleValue>();
                value->kind = kind;
                return value;
            }

            /** @brief Returns a node of @p kind holding @p text. */
            [[nodiscard]] PickleRef text(PickleKind kind, std::string_view content)
            {
                PickleRef value = node(kind);
                value->text = std::string(content);
                return value;
            }

            /** @brief Returns an Int node of @p value. */
            [[nodiscard]] PickleRef whole(std::int64_t value)
            {
                PickleRef out = node(PickleKind::Int);
                out->integer = value;
                return out;
            }

            /** @brief Returns a call of @p global over @p arguments. */
            [[nodiscard]] PickleRef call(std::string_view global, std::vector<PickleRef> arguments)
            {
                PickleRef out = node(PickleKind::Call);
                out->callable = text(PickleKind::Global, global);
                out->items = std::move(arguments);
                return out;
            }

            /** @brief Returns the `_rebuild_tensor_v2` call of @p record, stored as @p key. */
            [[nodiscard]] PickleRef rebuild(const TensorRecord& record, std::string_view global,
                                            const std::string& key)
            {
                PickleRef id = node(PickleKind::Tuple);
                id->items = {text(PickleKind::String, "storage"), text(PickleKind::Global, global),
                             text(PickleKind::String, key), text(PickleKind::String, "cpu"),
                             whole(record.shape.num_elements())};
                PickleRef persistent = node(PickleKind::PersistentID);
                persistent->items = {id};

                PickleRef size = node(PickleKind::Tuple);
                PickleRef stride = node(PickleKind::Tuple);
                std::int64_t step = 1;
                std::vector<PickleRef> steps(record.shape.rank());
                for (std::size_t axis = record.shape.rank(); axis-- > 0;)
                {
                    steps[axis] = whole(step);
                    step *= record.shape[axis];
                }
                for (std::int32_t axis = 0; axis < record.shape.rank(); ++axis)
                    size->items.push_back(whole(record.shape[axis]));
                stride->items = std::move(steps);

                return call("torch._utils _rebuild_tensor_v2",
                            {persistent, whole(0), size, stride, node(PickleKind::Bool),
                             call("collections OrderedDict", {})});
            }

            /** @brief Returns InvalidDescription or Unsupported for a tensor it cannot write. */
            [[nodiscard]] Status check_tensor(const ImportedTensor& tensor)
            {
                const TensorRecord& record = tensor.record;
                SA_FAIL_IF(storage_global(record.dtype).empty(), StatusCode::Unsupported,
                           "the tensor '{}' is {}, and a PyTorch file is written for float32, "
                           "float64 and half",
                           record.name, dtype_name(record.dtype));
                SA_FAIL_IF(tensor.payload.size() != record.payload_bytes(),
                           StatusCode::InvalidDescription,
                           "the tensor '{}' holds {} bytes, but its shape {} needs {}",
                           record.name, tensor.payload.size(), record.shape.to_string(),
                           record.payload_bytes());
                return Status();
            }
        } // namespace

        Result<std::vector<std::byte>> write_torch_state_dict(
            SushiRuntime::span<const ImportedTensor> tensors, std::string_view folder,
            std::string_view state_key) noexcept
        {
            SA_FAIL_IF(folder.empty() || folder.find('/') != std::string_view::npos,
                       StatusCode::InvalidDescription,
                       "the folder '{}' of a PyTorch file must be one non-empty name", folder);
            for (const ImportedTensor& tensor : tensors)
            {
                SA_TRY(check_tensor(tensor));
            }

            SA_TRY_ASSIGN(const PickleRef root,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&]()
            {
                PickleRef dictionary = call("collections OrderedDict", {});
                for (std::size_t k = 0; k < tensors.size(); ++k)
                {
                    const TensorRecord& record = tensors[k].record;
                    dictionary->entries.emplace_back(
                        text(PickleKind::String, record.name),
                        rebuild(record, storage_global(record.dtype), std::to_string(k)));
                }
                if (state_key.empty())
                    return dictionary;

                PickleRef outer = node(PickleKind::Dict);
                outer->entries.emplace_back(text(PickleKind::String, state_key), dictionary);
                return outer;
            }));
            SA_TRY_ASSIGN(const std::vector<std::byte> pickle, write_pickle(*root));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                const std::string prefix = std::string(folder) + "/";
                const auto view = [](std::string_view content)
                {
                    return SushiRuntime::span<const std::byte>(
                        reinterpret_cast<const std::byte*>(content.data()), content.size());
                };
                ZipWriter zip;
                zip.add(prefix + "data.pkl",
                        SushiRuntime::span<const std::byte>(pickle.data(), pickle.size()));
                zip.add(prefix + "byteorder", view(K_BYTE_ORDER));
                for (std::size_t k = 0; k < tensors.size(); ++k)
                    zip.add(prefix + "data/" + std::to_string(k),
                            SushiRuntime::span<const std::byte>(tensors[k].payload.data(),
                                                                tensors[k].payload.size()),
                            K_STORAGE_ALIGNMENT);
                zip.add(prefix + "version", view(K_TORCH_VERSION));
                return zip.finish();
            });
        }
    } // namespace IO
} // namespace SushiAI
