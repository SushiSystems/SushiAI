/**************************************************************************/
/* torch_archive.cpp                                                      */
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
 * @file torch_archive.cpp
 * @brief Defines the reader of a PyTorch zip file and the widening of its tensors to float32.
 * @author Mustafa Garip
 */

#include <SushiAI/io/torch_archive.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/pickle_reader.hpp>
#include <SushiAI/io/zip_archive.hpp>
#include <SushiAI/shape/shape.hpp>
#include "float_widening.hpp"

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Names the rebuilder of a tensor. */
            constexpr std::string_view K_REBUILD_TENSOR = "torch._utils _rebuild_tensor_v2";

            /** @brief Names the rebuilder of a parameter, which wraps a tensor. */
            constexpr std::string_view K_REBUILD_PARAMETER = "torch._utils _rebuild_parameter";

            /** @brief Lists the globals of PyTorch's own every file may name. */
            constexpr std::string_view K_TORCH_GLOBALS[] = {
                "collections OrderedDict",     "torch._utils _rebuild_tensor_v2",
                "torch._utils _rebuild_parameter", "torch Size",
                "__builtin__ set",             "builtins set",
                "torch HalfStorage",           "torch FloatStorage",
                "torch DoubleStorage",         "torch LongStorage",
                "torch IntStorage",            "torch BoolStorage",
                "torch ByteStorage",
            };

            /** @brief Pairs a storage global with the element type it holds. */
            struct StorageType
            {
                std::string_view global; /**< The "module name" of the storage class. */
                DType dtype;             /**< Its element type. */
                std::size_t bytes;       /**< The bytes of one element. */
            };

            /** @brief Lists the storage types whose elements the reader reads. */
            constexpr StorageType K_STORAGE_TYPES[] = {
                {"torch HalfStorage", DType::Half, 2},
                {"torch FloatStorage", DType::Float32, 4},
                {"torch DoubleStorage", DType::Float64, 8},
                {"torch LongStorage", DType::Int64, 8},
                {"torch IntStorage", DType::Int32, 4},
            };

            /** @brief Holds the suffix of the pickle entry of a PyTorch file. */
            constexpr std::string_view K_PICKLE_SUFFIX = "data.pkl";

            /** @brief Returns the bytes of one element of @p dtype as it is stored. */
            std::size_t stored_bytes(DType dtype) noexcept
            {
                for (const StorageType& type : K_STORAGE_TYPES)
                {
                    if (type.dtype == dtype)
                        return type.bytes;
                }
                return 0;
            }

            /** @brief Returns the whole numbers of the tuple @p value, or a failure. */
            Result<std::vector<std::int64_t>> read_integers(const PickleValue& value,
                                                            std::string_view what,
                                                            std::string_view context)
            {
                const PickleValue* tuple = &value;
                if (is_call_of(tuple, "torch Size") && !tuple->items.empty())
                    tuple = tuple->items.front().get();
                SA_FAIL_IF(tuple->kind != PickleKind::Tuple, StatusCode::InvalidFile,
                           "{}: the {} of the tensor is not a tuple", context, what);
                std::vector<std::int64_t> out;
                out.reserve(tuple->items.size());
                for (const PickleRef& item : tuple->items)
                {
                    SA_FAIL_IF(item->kind != PickleKind::Int, StatusCode::InvalidFile,
                               "{}: the {} of the tensor holds a value that is no whole number",
                               context, what);
                    out.push_back(item->integer);
                }
                return out;
            }
        } // namespace

        TorchArchive::TorchArchive(ZipArchive archive, std::string prefix, PickleRef root,
                                   std::string origin) noexcept
            : archive_(std::move(archive)), prefix_(std::move(prefix)), root_(std::move(root)),
              origin_(std::move(origin))
        {

        }

        Result<TorchArchive> TorchArchive::read(SushiRuntime::span<const std::byte> bytes,
                                                SushiRuntime::span<const std::string_view> accepted,
                                                std::string_view origin) noexcept
        {
            SA_TRY_ASSIGN(ZipArchive archive, ZipArchive::read(bytes, origin));
            const ZipEntry* pickle = nullptr;
            for (const ZipEntry& entry : archive.entries())
            {
                const std::string_view name = entry.name;
                const std::size_t slash = name.find('/');
                if (slash != std::string_view::npos && name.substr(slash + 1) == K_PICKLE_SUFFIX)
                    pickle = &entry;
            }
            SA_FAIL_IF(pickle == nullptr, StatusCode::InvalidFile,
                       "'{}' is a zip archive but holds no <folder>/data.pkl, so it is no "
                       "PyTorch file",
                       origin);
            std::string prefix = pickle->name.substr(0, pickle->name.size() - K_PICKLE_SUFFIX.size());

            const ZipEntry* order = archive.find(prefix + "byteorder");
            if (order != nullptr)
            {
                const SushiRuntime::span<const std::byte> text = archive.contents(*order);
                const std::string_view value(reinterpret_cast<const char*>(text.data()),
                                             text.size());
                SA_FAIL_IF(value != "little", StatusCode::Unsupported,
                           "'{}' stores its tensors in '{}' byte order, and only little-endian "
                           "files are read",
                           origin, value);
            }

            std::vector<std::string_view> globals(std::begin(K_TORCH_GLOBALS),
                                                  std::end(K_TORCH_GLOBALS));
            globals.insert(globals.end(), accepted.begin(), accepted.end());
            SA_TRY_ASSIGN(PickleRef root,
                          read_pickle(archive.contents(*pickle),
                                      SushiRuntime::span<const std::string_view>(globals.data(),
                                                                                 globals.size()),
                                      origin));
            return TorchArchive(std::move(archive), std::move(prefix), std::move(root),
                                std::string(origin));
        }

        const PickleValue& TorchArchive::root() const noexcept
        {
            return *root_;
        }

        bool TorchArchive::holds_tensor(const PickleValue& value) noexcept
        {
            return is_call_of(&value, K_REBUILD_TENSOR) || is_call_of(&value, K_REBUILD_PARAMETER);
        }

        Result<TorchTensor> TorchArchive::tensor(const PickleValue& value,
                                                 std::string_view context) const noexcept
        {
            const PickleValue* rebuild = &value;
            if (is_call_of(rebuild, K_REBUILD_PARAMETER))
            {
                SA_FAIL_IF(rebuild->items.empty(), StatusCode::InvalidFile,
                           "{}: the parameter holds no tensor", context);
                rebuild = rebuild->items.front().get();
            }
            SA_FAIL_IF(!is_call_of(rebuild, K_REBUILD_TENSOR) || rebuild->items.size() < 4,
                       StatusCode::InvalidFile,
                       "{}: the value is not a tensor of _rebuild_tensor_v2 with its storage, "
                       "offset, size and stride",
                       context);

            const PickleValue& persistent = *rebuild->items[0];
            SA_FAIL_IF(persistent.kind != PickleKind::PersistentID || persistent.items.empty() ||
                           persistent.items[0]->kind != PickleKind::Tuple ||
                           persistent.items[0]->items.size() < 3,
                       StatusCode::InvalidFile,
                       "{}: the storage of the tensor is not a persistent id of a storage",
                       context);
            const std::vector<PickleRef>& id = persistent.items[0]->items;
            SA_FAIL_IF(id[1]->kind != PickleKind::Global || id[2]->kind != PickleKind::String,
                       StatusCode::InvalidFile,
                       "{}: the persistent id of the tensor names no storage type and key",
                       context);

            TorchTensor out;
            bool known = false;
            for (const StorageType& type : K_STORAGE_TYPES)
            {
                if (type.global == id[1]->text)
                {
                    out.dtype = type.dtype;
                    known = true;
                }
            }
            SA_FAIL_IF(!known, StatusCode::Unsupported,
                       "{}: the tensor is stored as '{}', which the reader does not read", context,
                       id[1]->text);
            out.storage = archive_.find(prefix_ + "data/" + id[2]->text);
            SA_FAIL_IF(out.storage == nullptr, StatusCode::InvalidFile,
                       "{}: the storage '{}' of the tensor is not in '{}'", context, id[2]->text,
                       origin_);

            SA_FAIL_IF(rebuild->items[1]->kind != PickleKind::Int ||
                           rebuild->items[1]->integer < 0,
                       StatusCode::InvalidFile, "{}: the storage offset of the tensor is no "
                       "whole number", context);
            out.storage_offset = rebuild->items[1]->integer;
            SA_TRY_ASSIGN(const std::vector<std::int64_t> extents,
                          read_integers(*rebuild->items[2], "size", context));
            SA_TRY_ASSIGN(out.strides, read_integers(*rebuild->items[3], "stride", context));
            SA_FAIL_IF(extents.size() != out.strides.size(), StatusCode::InvalidFile,
                       "{}: the tensor states {} extents and {} strides", context, extents.size(),
                       out.strides.size());
            SA_TRY_ASSIGN(out.shape,
                          Shape::make(SushiRuntime::span<const std::int64_t>(extents.data(),
                                                                             extents.size())));
            return out;
        }

        Result<std::vector<float>> TorchArchive::read_float32(const TorchTensor& tensor,
                                                              std::string_view context) const noexcept
        {
            SA_FAIL_IF(tensor.dtype != DType::Half && tensor.dtype != DType::Float32 &&
                           tensor.dtype != DType::Float64,
                       StatusCode::Unsupported,
                       "{}: the tensor holds {} elements, and only half, float32 and float64 "
                       "are read as float32",
                       context, dtype_name(tensor.dtype));
            const SushiRuntime::span<const std::byte> storage = archive_.contents(*tensor.storage);
            const std::size_t width = stored_bytes(tensor.dtype);
            const std::size_t capacity = storage.size() / width;
            const std::int32_t rank = tensor.shape.rank();
            const std::size_t count = static_cast<std::size_t>(tensor.shape.num_elements());

            std::vector<float> out(count);
            std::vector<std::int64_t> index(static_cast<std::size_t>(rank), 0);
            for (std::size_t n = 0; n < count; ++n)
            {
                std::int64_t element = tensor.storage_offset;
                for (std::int32_t axis = 0; axis < rank; ++axis)
                    element += index[static_cast<std::size_t>(axis)] *
                               tensor.strides[static_cast<std::size_t>(axis)];
                SA_FAIL_IF(element < 0 || static_cast<std::size_t>(element) >= capacity,
                           StatusCode::InvalidFile,
                           "{}: the tensor reads element {} of a storage of {}", context, element,
                           capacity);
                const std::byte* const raw = storage.data() + static_cast<std::size_t>(element) * width;
                if (tensor.dtype == DType::Half)
                {
                    const std::uint16_t bits = static_cast<std::uint16_t>(
                        static_cast<unsigned>(raw[0]) | (static_cast<unsigned>(raw[1]) << 8u));
                    out[n] = Detail::widen_half(bits);
                }
                else if (tensor.dtype == DType::Float32)
                {
                    std::memcpy(&out[n], raw, sizeof(float));
                }
                else
                {
                    double wide = 0.0;
                    std::memcpy(&wide, raw, sizeof(double));
                    out[n] = static_cast<float>(wide);
                }
                for (std::int32_t axis = rank; axis-- > 0;)
                {
                    const std::size_t a = static_cast<std::size_t>(axis);
                    if (++index[a] < tensor.shape[axis])
                        break;
                    index[a] = 0;
                }
            }
            return out;
        }
    } // namespace IO
} // namespace SushiAI
