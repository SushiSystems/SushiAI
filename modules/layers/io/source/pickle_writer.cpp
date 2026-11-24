/**************************************************************************/
/* pickle_writer.cpp                                                      */
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
 * @file pickle_writer.cpp
 * @brief Implements the writer of a pickle stream of protocol 2.
 * @author Mustafa Garip
 */

#include <SushiAI/io/pickle_writer.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/pickle_reader.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the opcodes the writer emits. */
            namespace Op
            {
                constexpr std::uint8_t K_PROTO = 0x80;           /**< The protocol. */
                constexpr std::uint8_t K_STOP = '.';             /**< The end of the stream. */
                constexpr std::uint8_t K_NONE = 'N';             /**< None. */
                constexpr std::uint8_t K_NEWTRUE = 0x88;         /**< True. */
                constexpr std::uint8_t K_NEWFALSE = 0x89;        /**< False. */
                constexpr std::uint8_t K_BININT1 = 'K';          /**< An unsigned byte. */
                constexpr std::uint8_t K_BININT = 'J';           /**< A signed 32-bit integer. */
                constexpr std::uint8_t K_LONG1 = 0x8a;           /**< A sized integer. */
                constexpr std::uint8_t K_BINFLOAT = 'G';         /**< A big-endian double. */
                constexpr std::uint8_t K_BINUNICODE = 'X';       /**< A UTF-8 text. */
                constexpr std::uint8_t K_SHORT_BINSTRING = 'U';  /**< Bytes below 256. */
                constexpr std::uint8_t K_BINSTRING = 'T';        /**< Bytes. */
                constexpr std::uint8_t K_EMPTY_LIST = ']';       /**< An empty list. */
                constexpr std::uint8_t K_EMPTY_TUPLE = ')';      /**< An empty tuple. */
                constexpr std::uint8_t K_EMPTY_DICT = '}';       /**< An empty dict. */
                constexpr std::uint8_t K_MARK = '(';             /**< A mark. */
                constexpr std::uint8_t K_TUPLE = 't';            /**< A tuple to the mark. */
                constexpr std::uint8_t K_APPENDS = 'e';          /**< Appends to the mark. */
                constexpr std::uint8_t K_SETITEMS = 'u';         /**< Pairs to the mark. */
                constexpr std::uint8_t K_GLOBAL = 'c';           /**< A module and a name. */
                constexpr std::uint8_t K_REDUCE = 'R';           /**< A call. */
                constexpr std::uint8_t K_BUILD = 'b';            /**< A state set onto a call. */
                constexpr std::uint8_t K_BINPERSID = 'Q';        /**< A persistent id. */
            } // namespace Op

            /** @brief Holds the protocol the writer states. */
            constexpr std::uint8_t K_PROTOCOL = 2;

            /** @brief Appends the opcodes of a pickle tree to a stream. */
            class Emitter final
            {
                public:
                    /** @brief Appends @p value, named @p path in a message. */
                    [[nodiscard]] Status write(const PickleValue& value, const std::string& path);

                    /** @brief Holds the stream written so far. */
                    std::vector<std::byte> out;

                private:
                    /** @brief Appends one byte. */
                    void op(std::uint8_t byte) { out.push_back(static_cast<std::byte>(byte)); }

                    /** @brief Appends the low @p count bytes of @p value, low byte first. */
                    void little(std::uint64_t value, std::size_t count)
                    {
                        for (std::size_t i = 0; i < count; ++i)
                            op(static_cast<std::uint8_t>((value >> (8u * i)) & 0xFFu));
                    }

                    /** @brief Appends the bytes of @p text. */
                    void raw(std::string_view text)
                    {
                        for (const char c : text)
                            op(static_cast<std::uint8_t>(c));
                    }

                    /** @brief Appends a whole number in the shortest opcode that holds it. */
                    void integer(std::int64_t value);

                    /** @brief Appends every node of @p items under a mark. */
                    [[nodiscard]] Status items(const std::vector<PickleRef>& items,
                                               const std::string& path);

                    /** @brief Appends every pair of @p entries under a mark, then SETITEMS. */
                    [[nodiscard]] Status pairs(
                        const std::vector<std::pair<PickleRef, PickleRef>>& entries,
                        const std::string& path);

                    /** @brief Appends a call and what is built onto it. */
                    [[nodiscard]] Status call(const PickleValue& value, const std::string& path);
            };

            void Emitter::integer(std::int64_t value)
            {
                if (value >= 0 && value <= 0xFF)
                {
                    op(Op::K_BININT1);
                    little(static_cast<std::uint64_t>(value), 1);
                }
                else if (value >= std::numeric_limits<std::int32_t>::min() &&
                         value <= std::numeric_limits<std::int32_t>::max())
                {
                    op(Op::K_BININT);
                    little(static_cast<std::uint32_t>(static_cast<std::int32_t>(value)), 4);
                }
                else
                {
                    op(Op::K_LONG1);
                    op(8);
                    little(static_cast<std::uint64_t>(value), 8);
                }
            }

            Status Emitter::items(const std::vector<PickleRef>& values, const std::string& path)
            {
                op(Op::K_MARK);
                for (std::size_t i = 0; i < values.size(); ++i)
                {
                    SA_FAIL_IF(values[i] == nullptr, StatusCode::InvalidDescription,
                               "the pickle node '{}[{}]' is null", path, i);
                    SA_TRY(write(*values[i], path + "[" + std::to_string(i) + "]"));
                }
                return Status();
            }

            Status Emitter::pairs(const std::vector<std::pair<PickleRef, PickleRef>>& entries,
                                  const std::string& path)
            {
                op(Op::K_MARK);
                for (std::size_t i = 0; i < entries.size(); ++i)
                {
                    SA_FAIL_IF(entries[i].first == nullptr || entries[i].second == nullptr,
                               StatusCode::InvalidDescription,
                               "the pair {} of the pickle node '{}' is null", i, path);
                    const std::string at = path + "{" + std::to_string(i) + "}";
                    SA_TRY(write(*entries[i].first, at));
                    SA_TRY(write(*entries[i].second, at));
                }
                op(Op::K_SETITEMS);
                return Status();
            }

            Status Emitter::call(const PickleValue& value, const std::string& path)
            {
                SA_FAIL_IF(value.callable == nullptr ||
                               value.callable->kind != PickleKind::Global,
                           StatusCode::InvalidDescription,
                           "the pickle call '{}' names no global to call", path);
                SA_TRY(write(*value.callable, path));
                if (value.items.empty())
                {
                    op(Op::K_EMPTY_TUPLE);
                }
                else
                {
                    SA_TRY(items(value.items, path));
                    op(Op::K_TUPLE);
                }
                op(Op::K_REDUCE);

                if (!value.entries.empty())
                {
                    SA_TRY(pairs(value.entries, path));
                }
                if (!value.appended.empty())
                {
                    SA_TRY(items(value.appended, path));
                    op(Op::K_APPENDS);
                }
                if (value.state != nullptr)
                {
                    SA_TRY(write(*value.state, path + ".state"));
                    op(Op::K_BUILD);
                }
                return Status();
            }

            Status Emitter::write(const PickleValue& value, const std::string& path)
            {
                switch (value.kind)
                {
                    case PickleKind::None:
                        op(Op::K_NONE);
                        return Status();
                    case PickleKind::Bool:
                        op(value.boolean ? Op::K_NEWTRUE : Op::K_NEWFALSE);
                        return Status();
                    case PickleKind::Int:
                        integer(value.integer);
                        return Status();
                    case PickleKind::Float:
                    {
                        std::uint64_t bits = 0;
                        std::memcpy(&bits, &value.real, sizeof(bits));
                        op(Op::K_BINFLOAT);
                        for (std::size_t i = 0; i < 8; ++i)
                            op(static_cast<std::uint8_t>((bits >> (8u * (7u - i))) & 0xFFu));
                        return Status();
                    }
                    case PickleKind::String:
                        op(Op::K_BINUNICODE);
                        little(value.text.size(), 4);
                        raw(value.text);
                        return Status();
                    case PickleKind::Bytes:
                        if (value.text.size() < 256)
                        {
                            op(Op::K_SHORT_BINSTRING);
                            little(value.text.size(), 1);
                        }
                        else
                        {
                            op(Op::K_BINSTRING);
                            little(value.text.size(), 4);
                        }
                        raw(value.text);
                        return Status();
                    case PickleKind::List:
                        op(Op::K_EMPTY_LIST);
                        if (!value.items.empty())
                        {
                            SA_TRY(items(value.items, path));
                            op(Op::K_APPENDS);
                        }
                        return Status();
                    case PickleKind::Tuple:
                        if (value.items.empty())
                        {
                            op(Op::K_EMPTY_TUPLE);
                            return Status();
                        }
                        SA_TRY(items(value.items, path));
                        op(Op::K_TUPLE);
                        return Status();
                    case PickleKind::Dict:
                        op(Op::K_EMPTY_DICT);
                        if (!value.entries.empty())
                        {
                            SA_TRY(pairs(value.entries, path));
                        }
                        return Status();
                    case PickleKind::Global:
                    {
                        const std::size_t space = value.text.find(' ');
                        SA_FAIL_IF(space == std::string::npos || space == 0 ||
                                       space + 1 == value.text.size() ||
                                       value.text.find_first_of(" \n", space + 1) !=
                                           std::string::npos,
                                   StatusCode::InvalidDescription,
                                   "the pickle global '{}' at '{}' is not \"module name\"",
                                   value.text, path);
                        op(Op::K_GLOBAL);
                        raw(std::string_view(value.text).substr(0, space));
                        op('\n');
                        raw(std::string_view(value.text).substr(space + 1));
                        op('\n');
                        return Status();
                    }
                    case PickleKind::Call:
                        return call(value, path);
                    case PickleKind::PersistentID:
                        SA_FAIL_IF(value.items.size() != 1 || value.items[0] == nullptr,
                                   StatusCode::InvalidDescription,
                                   "the persistent id at '{}' holds {} ids, and one is needed",
                                   path, value.items.size());
                        SA_TRY(write(*value.items[0], path));
                        op(Op::K_BINPERSID);
                        return Status();
                    case PickleKind::Set:
                        break;
                }

                SA_FAIL_IF(true, StatusCode::Unsupported,
                           "the pickle node '{}' is a set, which protocol 2 cannot state", path);
                return Status();
            }
        } // namespace

        Result<std::vector<std::byte>> write_pickle(const PickleValue& root) noexcept
        {
            Emitter emitter;
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                emitter.out.push_back(static_cast<std::byte>(Op::K_PROTO));
                emitter.out.push_back(static_cast<std::byte>(K_PROTOCOL));
                SA_TRY(emitter.write(root, "root"));
                emitter.out.push_back(static_cast<std::byte>(Op::K_STOP));
                return Status();
            }));
            return std::move(emitter.out);
        }
    } // namespace IO
} // namespace SushiAI
