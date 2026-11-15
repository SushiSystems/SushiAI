/**************************************************************************/
/* pickle_reader.cpp                                                      */
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
 * @file pickle_reader.cpp
 * @brief Defines the pickle machine that builds a value tree and calls nothing it names.
 * @author Mustafa Garip
 */

#include <SushiAI/io/pickle_reader.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the opcodes the machine runs, by the byte that spells each. */
            namespace Op
            {
                constexpr std::uint8_t K_PROTO = 0x80;            /**< Protocol number. */
                constexpr std::uint8_t K_FRAME = 0x95;            /**< Frame length. */
                constexpr std::uint8_t K_STOP = '.';              /**< End of stream. */
                constexpr std::uint8_t K_MARK = '(';              /**< Pushes a mark. */
                constexpr std::uint8_t K_EMPTY_DICT = '}';        /**< Pushes {}. */
                constexpr std::uint8_t K_EMPTY_LIST = ']';        /**< Pushes []. */
                constexpr std::uint8_t K_EMPTY_TUPLE = ')';       /**< Pushes (). */
                constexpr std::uint8_t K_EMPTY_SET = 0x8f;        /**< Pushes set(). */
                constexpr std::uint8_t K_TUPLE1 = 0x85;           /**< One-element tuple. */
                constexpr std::uint8_t K_TUPLE2 = 0x86;           /**< Two-element tuple. */
                constexpr std::uint8_t K_TUPLE3 = 0x87;           /**< Three-element tuple. */
                constexpr std::uint8_t K_TUPLE = 't';             /**< Tuple to the mark. */
                constexpr std::uint8_t K_LIST = 'l';              /**< List to the mark. */
                constexpr std::uint8_t K_DICT = 'd';              /**< Dict to the mark. */
                constexpr std::uint8_t K_FROZENSET = 0x91;        /**< Frozen set to the mark. */
                constexpr std::uint8_t K_APPEND = 'a';            /**< Appends one element. */
                constexpr std::uint8_t K_APPENDS = 'e';           /**< Appends to the mark. */
                constexpr std::uint8_t K_ADDITEMS = 0x90;         /**< Adds to a set. */
                constexpr std::uint8_t K_SETITEM = 's';           /**< Sets one pair. */
                constexpr std::uint8_t K_SETITEMS = 'u';          /**< Sets pairs to the mark. */
                constexpr std::uint8_t K_BININT = 'J';            /**< Signed 4-byte int. */
                constexpr std::uint8_t K_BININT1 = 'K';           /**< Unsigned 1-byte int. */
                constexpr std::uint8_t K_BININT2 = 'M';           /**< Unsigned 2-byte int. */
                constexpr std::uint8_t K_LONG1 = 0x8a;            /**< Int of up to 8 bytes. */
                constexpr std::uint8_t K_BINFLOAT = 'G';          /**< Big-endian double. */
                constexpr std::uint8_t K_SHORT_BINUNICODE = 0x8c; /**< Text, 1-byte length. */
                constexpr std::uint8_t K_BINUNICODE = 'X';        /**< Text, 4-byte length. */
                constexpr std::uint8_t K_BINUNICODE8 = 0x8d;      /**< Text, 8-byte length. */
                constexpr std::uint8_t K_SHORT_BINSTRING = 'U';   /**< Bytes as str, 1-byte. */
                constexpr std::uint8_t K_BINSTRING = 'T';         /**< Bytes as str, 4-byte. */
                constexpr std::uint8_t K_SHORT_BINBYTES = 'C';    /**< Bytes, 1-byte length. */
                constexpr std::uint8_t K_BINBYTES = 'B';          /**< Bytes, 4-byte length. */
                constexpr std::uint8_t K_BINBYTES8 = 0x8e;        /**< Bytes, 8-byte length. */
                constexpr std::uint8_t K_NONE = 'N';              /**< Pushes None. */
                constexpr std::uint8_t K_NEWTRUE = 0x88;          /**< Pushes True. */
                constexpr std::uint8_t K_NEWFALSE = 0x89;         /**< Pushes False. */
                constexpr std::uint8_t K_BINPUT = 'q';            /**< Memo store, 1 byte. */
                constexpr std::uint8_t K_LONG_BINPUT = 'r';       /**< Memo store, 4 bytes. */
                constexpr std::uint8_t K_MEMOIZE = 0x94;          /**< Memo store, next slot. */
                constexpr std::uint8_t K_BINGET = 'h';            /**< Memo fetch, 1 byte. */
                constexpr std::uint8_t K_LONG_BINGET = 'j';       /**< Memo fetch, 4 bytes. */
                constexpr std::uint8_t K_GLOBAL = 'c';            /**< Global, two lines. */
                constexpr std::uint8_t K_STACK_GLOBAL = 0x93;     /**< Global from the stack. */
                constexpr std::uint8_t K_REDUCE = 'R';            /**< Call with a tuple. */
                constexpr std::uint8_t K_NEWOBJ = 0x81;           /**< cls.__new__(*args). */
                constexpr std::uint8_t K_NEWOBJ_EX = 0x92;        /**< NEWOBJ with kwargs. */
                constexpr std::uint8_t K_BUILD = 'b';             /**< Sets a Call's state. */
                constexpr std::uint8_t K_BINPERSID = 'Q';         /**< Persistent id. */
            } // namespace Op

            /** @brief Holds the newest protocol the machine reads. */
            constexpr std::uint8_t K_NEWEST_PROTOCOL = 5;

            /** @brief Returns a new node of @p kind. */
            PickleRef make(PickleKind kind)
            {
                PickleRef value = std::make_shared<PickleValue>();
                value->kind = kind;
                return value;
            }

            /** @brief Returns whether the global @p module @p name is one @p accepted admits. */
            bool admits(SushiRuntime::span<const std::string_view> accepted,
                        std::string_view module, std::string_view name)
            {
                for (const std::string_view rule : accepted)
                {
                    const bool prefix = !rule.empty() && rule.back() == '.';
                    if (prefix && (module.substr(0, rule.size()) == rule ||
                                   module == rule.substr(0, rule.size() - 1)))
                        return true;
                    const std::size_t space = rule.find(' ');
                    if (!prefix && space != std::string_view::npos &&
                        rule.substr(0, space) == module && rule.substr(space + 1) == name)
                        return true;
                }
                return false;
            }

            /**
             * @class Machine
             * @brief Runs the opcodes of one stream over a stack, a mark list and a memo.
             */
            class Machine final
            {
                public:
                    /** @brief Holds the stream, the accepted globals and the stream's name. */
                    Machine(SushiRuntime::span<const std::byte> bytes,
                            SushiRuntime::span<const std::string_view> accepted,
                            std::string_view origin) noexcept
                        : bytes_(bytes), accepted_(accepted), origin_(origin)
                    {

                    }

                    /** @brief Runs the stream to its STOP and returns the value left on top. */
                    Result<PickleRef> run()
                    {
                        while (true)
                        {
                            SA_FAIL_IF(at_ >= bytes_.size(), StatusCode::InvalidFile,
                                       "'{}' ends at byte {} before its STOP opcode", origin_,
                                       at_);
                            opcode_at_ = at_;
                            const std::uint8_t opcode = static_cast<std::uint8_t>(bytes_[at_++]);
                            if (opcode == Op::K_STOP)
                                break;
                            SA_TRY(step(opcode));
                        }
                        SA_FAIL_IF(stack_.size() != 1 || !marks_.empty(), StatusCode::InvalidFile,
                                   "'{}' stops with {} values and {} marks on its stack, not one "
                                   "value",
                                   origin_, stack_.size(), marks_.size());
                        return stack_.back();
                    }

                private:
                    /** @brief Returns InvalidFile naming the opcode being run and @p what. */
                    Status malformed(std::string_view what) const
                    {
                        SA_FAIL_IF(true, StatusCode::InvalidFile,
                                   "'{}': the opcode at byte {} {}", origin_, opcode_at_, what);
                        return Status();
                    }

                    /** @brief Reads the next @p count bytes as a view. */
                    Result<SushiRuntime::span<const std::byte>> take(std::uint64_t count)
                    {
                        SA_FAIL_IF(count > bytes_.size() - at_, StatusCode::InvalidFile,
                                   "'{}': the opcode at byte {} needs {} bytes past the end",
                                   origin_, opcode_at_, count);
                        const SushiRuntime::span<const std::byte> out =
                            bytes_.subspan(at_, static_cast<std::size_t>(count));
                        at_ += static_cast<std::size_t>(count);
                        return out;
                    }

                    /** @brief Reads the next @p width bytes as a little-endian unsigned value. */
                    Result<std::uint64_t> take_le(std::size_t width)
                    {
                        SA_TRY_ASSIGN(const SushiRuntime::span<const std::byte> raw, take(width));
                        std::uint64_t value = 0;
                        for (std::size_t i = 0; i < width; ++i)
                            value |= static_cast<std::uint64_t>(raw[i]) << (8u * i);
                        return value;
                    }

                    /** @brief Reads text up to the next newline, the newline left out. */
                    Result<std::string> take_line()
                    {
                        std::size_t end = at_;
                        while (end < bytes_.size() && static_cast<char>(bytes_[end]) != '\n')
                            ++end;
                        SA_FAIL_IF(end == bytes_.size(), StatusCode::InvalidFile,
                                   "'{}': the opcode at byte {} has no end of line", origin_,
                                   opcode_at_);
                        std::string line(reinterpret_cast<const char*>(bytes_.data() + at_),
                                         end - at_);
                        at_ = end + 1;
                        return line;
                    }

                    /** @brief Pushes a node of @p kind holding @p text, read as @p width bytes. */
                    Status push_text(PickleKind kind, std::size_t width)
                    {
                        SA_TRY_ASSIGN(const std::uint64_t length, take_le(width));
                        SA_TRY_ASSIGN(const SushiRuntime::span<const std::byte> raw, take(length));
                        PickleRef value = make(kind);
                        value->text.assign(reinterpret_cast<const char*>(raw.data()), raw.size());
                        stack_.push_back(std::move(value));
                        return Status();
                    }

                    /** @brief Pushes an Int of @p value. */
                    void push_int(std::int64_t value)
                    {
                        PickleRef node = make(PickleKind::Int);
                        node->integer = value;
                        stack_.push_back(std::move(node));
                    }

                    /** @brief Removes and returns the top of the stack. */
                    Result<PickleRef> pop()
                    {
                        SA_FAIL_IF(stack_.size() <= floor(), StatusCode::InvalidFile,
                                   "'{}': the opcode at byte {} pops an empty stack", origin_,
                                   opcode_at_);
                        PickleRef top = std::move(stack_.back());
                        stack_.pop_back();
                        return top;
                    }

                    /** @brief Returns the top of the stack without removing it. */
                    Result<PickleRef> top() const
                    {
                        SA_FAIL_IF(stack_.size() <= floor(), StatusCode::InvalidFile,
                                   "'{}': the opcode at byte {} reads an empty stack", origin_,
                                   opcode_at_);
                        return stack_.back();
                    }

                    /** @brief Returns the stack depth below which the latest mark keeps values. */
                    std::size_t floor() const noexcept
                    {
                        return marks_.empty() ? 0 : marks_.back();
                    }

                    /** @brief Removes the values above the latest mark and the mark itself. */
                    Result<std::vector<PickleRef>> pop_mark()
                    {
                        SA_FAIL_IF(marks_.empty(), StatusCode::InvalidFile,
                                   "'{}': the opcode at byte {} needs a mark and none is set",
                                   origin_, opcode_at_);
                        const std::size_t mark = marks_.back();
                        marks_.pop_back();
                        std::vector<PickleRef> items(
                            std::make_move_iterator(stack_.begin() +
                                                    static_cast<std::ptrdiff_t>(mark)),
                            std::make_move_iterator(stack_.end()));
                        stack_.resize(mark);
                        return items;
                    }

                    /** @brief Pushes a sequence of @p kind holding the @p count top values. */
                    Status push_sequence(PickleKind kind, std::size_t count)
                    {
                        SA_FAIL_IF(stack_.size() < floor() + count, StatusCode::InvalidFile,
                                   "'{}': the opcode at byte {} takes {} values from a shorter "
                                   "stack",
                                   origin_, opcode_at_, count);
                        PickleRef value = make(kind);
                        value->items.assign(
                            std::make_move_iterator(stack_.end() -
                                                    static_cast<std::ptrdiff_t>(count)),
                            std::make_move_iterator(stack_.end()));
                        stack_.resize(stack_.size() - count);
                        stack_.push_back(std::move(value));
                        return Status();
                    }

                    /** @brief Adds @p pairs, flattened key then value, onto @p target. */
                    Status set_pairs(PickleValue& target, std::vector<PickleRef> pairs)
                    {
                        if (target.kind != PickleKind::Dict && target.kind != PickleKind::Call)
                            return malformed("sets items on a value that is no dict");
                        if (pairs.size() % 2 != 0)
                            return malformed("sets an odd count of keys and values");
                        for (std::size_t i = 0; i < pairs.size(); i += 2)
                            target.entries.emplace_back(std::move(pairs[i]),
                                                        std::move(pairs[i + 1]));
                        return Status();
                    }

                    /** @brief Appends @p items onto @p target, a list, a set or a Call. */
                    Status append_items(PickleValue& target, std::vector<PickleRef> items)
                    {
                        std::vector<PickleRef>* sink = nullptr;
                        if (target.kind == PickleKind::List || target.kind == PickleKind::Set)
                            sink = &target.items;
                        else if (target.kind == PickleKind::Call)
                            sink = &target.appended;
                        else
                            return malformed("appends to a value that is no list");
                        for (PickleRef& item : items)
                            sink->push_back(std::move(item));
                        return Status();
                    }

                    /** @brief Pushes a Global after checking @p module and @p name are accepted. */
                    Status push_global(std::string_view module, std::string_view name)
                    {
                        SA_FAIL_IF(!admits(accepted_, module, name), StatusCode::Unsupported,
                                   "'{}' names the global '{} {}' at byte {}, which the reader "
                                   "does not accept",
                                   origin_, module, name, opcode_at_);
                        PickleRef value = make(PickleKind::Global);
                        value->text.reserve(module.size() + 1 + name.size());
                        value->text.append(module).append(" ").append(name);
                        stack_.push_back(std::move(value));
                        return Status();
                    }

                    /** @brief Pushes a Call of @p callable with the elements of @p arguments. */
                    Status push_call(PickleRef callable, const PickleRef& arguments)
                    {
                        if (callable->kind != PickleKind::Global)
                            return malformed("calls a value that is no global");
                        if (arguments->kind != PickleKind::Tuple)
                            return malformed("calls with arguments that are no tuple");
                        PickleRef value = make(PickleKind::Call);
                        value->callable = std::move(callable);
                        value->items = arguments->items;
                        stack_.push_back(std::move(value));
                        return Status();
                    }

                    /** @brief Stores the top of the stack in memo slot @p slot. */
                    Status memoize(std::uint64_t slot)
                    {
                        SA_TRY_ASSIGN(PickleRef value, top());
                        memo_[slot] = std::move(value);
                        return Status();
                    }

                    /** @brief Pushes the value of memo slot @p slot. */
                    Status fetch(std::uint64_t slot)
                    {
                        const auto found = memo_.find(slot);
                        SA_FAIL_IF(found == memo_.end(), StatusCode::InvalidFile,
                                   "'{}': the opcode at byte {} fetches memo slot {}, which "
                                   "holds nothing",
                                   origin_, opcode_at_, slot);
                        stack_.push_back(found->second);
                        return Status();
                    }

                    /** @brief Runs the opcodes that build containers. */
                    Status step_container(std::uint8_t opcode, bool& ran);

                    /** @brief Runs the opcodes that push numbers, texts and constants. */
                    Status step_scalar(std::uint8_t opcode, bool& ran);

                    /** @brief Runs the opcodes of the memo, globals, calls and persistent ids. */
                    Status step_object(std::uint8_t opcode, bool& ran);

                    /** @brief Runs one opcode. */
                    Status step(std::uint8_t opcode)
                    {
                        bool ran = false;
                        SA_TRY(step_container(opcode, ran));
                        if (!ran)
                            SA_TRY(step_scalar(opcode, ran));
                        if (!ran)
                            SA_TRY(step_object(opcode, ran));
                        SA_FAIL_IF(!ran, StatusCode::Unsupported,
                                   "'{}' holds the opcode {} at byte {}, which the reader "
                                   "does not run",
                                   origin_, static_cast<unsigned>(opcode), opcode_at_);
                        return Status();
                    }

                    /** @brief Holds the stream. */
                    SushiRuntime::span<const std::byte> bytes_;

                    /** @brief Holds the globals the stream may name. */
                    SushiRuntime::span<const std::string_view> accepted_;

                    /** @brief Names the stream in a message. */
                    std::string_view origin_;

                    /** @brief Holds the offset of the next byte to read. */
                    std::size_t at_ = 0;

                    /** @brief Holds the offset of the opcode being run. */
                    std::size_t opcode_at_ = 0;

                    /** @brief Holds the values the opcodes push and pop. */
                    std::vector<PickleRef> stack_;

                    /** @brief Holds the stack depth of each mark, latest last. */
                    std::vector<std::size_t> marks_;

                    /** @brief Holds the values stored for later fetches, by slot. */
                    std::unordered_map<std::uint64_t, PickleRef> memo_;
            };

            Status Machine::step_container(std::uint8_t opcode, bool& ran)
            {
                ran = true;
                switch (opcode)
                {
                    case Op::K_MARK:
                        marks_.push_back(stack_.size());
                        return Status();
                    case Op::K_EMPTY_DICT:
                        stack_.push_back(make(PickleKind::Dict));
                        return Status();
                    case Op::K_EMPTY_LIST:
                        stack_.push_back(make(PickleKind::List));
                        return Status();
                    case Op::K_EMPTY_TUPLE:
                        stack_.push_back(make(PickleKind::Tuple));
                        return Status();
                    case Op::K_EMPTY_SET:
                        stack_.push_back(make(PickleKind::Set));
                        return Status();
                    case Op::K_TUPLE1:
                        return push_sequence(PickleKind::Tuple, 1);
                    case Op::K_TUPLE2:
                        return push_sequence(PickleKind::Tuple, 2);
                    case Op::K_TUPLE3:
                        return push_sequence(PickleKind::Tuple, 3);
                    case Op::K_TUPLE:
                    case Op::K_LIST:
                    case Op::K_FROZENSET:
                    {
                        SA_TRY_ASSIGN(std::vector<PickleRef> items, pop_mark());
                        PickleRef value = make(opcode == Op::K_TUPLE  ? PickleKind::Tuple
                                               : opcode == Op::K_LIST ? PickleKind::List
                                                                      : PickleKind::Set);
                        value->items = std::move(items);
                        stack_.push_back(std::move(value));
                        return Status();
                    }
                    case Op::K_DICT:
                    {
                        SA_TRY_ASSIGN(std::vector<PickleRef> pairs, pop_mark());
                        PickleRef value = make(PickleKind::Dict);
                        SA_TRY(set_pairs(*value, std::move(pairs)));
                        stack_.push_back(std::move(value));
                        return Status();
                    }
                    case Op::K_APPEND:
                    {
                        SA_TRY_ASSIGN(PickleRef item, pop());
                        SA_TRY_ASSIGN(const PickleRef target, top());
                        std::vector<PickleRef> items;
                        items.push_back(std::move(item));
                        return append_items(*target, std::move(items));
                    }
                    case Op::K_APPENDS:
                    case Op::K_ADDITEMS:
                    {
                        SA_TRY_ASSIGN(std::vector<PickleRef> items, pop_mark());
                        SA_TRY_ASSIGN(const PickleRef target, top());
                        return append_items(*target, std::move(items));
                    }
                    case Op::K_SETITEM:
                    {
                        SA_TRY_ASSIGN(PickleRef value, pop());
                        SA_TRY_ASSIGN(PickleRef key, pop());
                        SA_TRY_ASSIGN(const PickleRef target, top());
                        std::vector<PickleRef> pairs;
                        pairs.push_back(std::move(key));
                        pairs.push_back(std::move(value));
                        return set_pairs(*target, std::move(pairs));
                    }
                    case Op::K_SETITEMS:
                    {
                        SA_TRY_ASSIGN(std::vector<PickleRef> pairs, pop_mark());
                        SA_TRY_ASSIGN(const PickleRef target, top());
                        return set_pairs(*target, std::move(pairs));
                    }
                    default:
                        ran = false;
                        return Status();
                }
            }

            Status Machine::step_scalar(std::uint8_t opcode, bool& ran)
            {
                ran = true;
                switch (opcode)
                {
                    case Op::K_PROTO:
                    {
                        SA_TRY_ASSIGN(const std::uint64_t protocol, take_le(1));
                        SA_FAIL_IF(protocol > K_NEWEST_PROTOCOL, StatusCode::Unsupported,
                                   "'{}' is pickle protocol {}, and the reader reads up to {}",
                                   origin_, protocol, K_NEWEST_PROTOCOL);
                        return Status();
                    }
                    case Op::K_FRAME:
                    {
                        SA_TRY_ASSIGN(const std::uint64_t ignored, take_le(8));
                        static_cast<void>(ignored);
                        return Status();
                    }
                    case Op::K_BININT:
                    {
                        SA_TRY_ASSIGN(const std::uint64_t raw, take_le(4));
                        push_int(static_cast<std::int32_t>(static_cast<std::uint32_t>(raw)));
                        return Status();
                    }
                    case Op::K_BININT1:
                    case Op::K_BININT2:
                    {
                        SA_TRY_ASSIGN(const std::uint64_t raw,
                                      take_le(opcode == Op::K_BININT1 ? 1 : 2));
                        push_int(static_cast<std::int64_t>(raw));
                        return Status();
                    }
                    case Op::K_LONG1:
                    {
                        SA_TRY_ASSIGN(const std::uint64_t width, take_le(1));
                        if (width > 8)
                            return malformed("holds an integer wider than 64 bits");
                        SA_TRY_ASSIGN(std::uint64_t raw, take_le(static_cast<std::size_t>(width)));
                        if (width > 0 && width < 8 && (raw >> (8 * width - 1)) != 0)
                            raw |= ~std::uint64_t(0) << (8 * width);
                        push_int(static_cast<std::int64_t>(raw));
                        return Status();
                    }
                    case Op::K_BINFLOAT:
                    {
                        SA_TRY_ASSIGN(const SushiRuntime::span<const std::byte> raw, take(8));
                        std::uint64_t bits = 0;
                        for (std::size_t i = 0; i < 8; ++i)
                            bits = (bits << 8u) | static_cast<std::uint64_t>(raw[i]);
                        PickleRef value = make(PickleKind::Float);
                        std::memcpy(&value->real, &bits, sizeof(bits));
                        stack_.push_back(std::move(value));
                        return Status();
                    }
                    case Op::K_SHORT_BINUNICODE:
                        return push_text(PickleKind::String, 1);
                    case Op::K_BINUNICODE:
                        return push_text(PickleKind::String, 4);
                    case Op::K_BINUNICODE8:
                        return push_text(PickleKind::String, 8);
                    case Op::K_SHORT_BINSTRING:
                    case Op::K_SHORT_BINBYTES:
                        return push_text(PickleKind::Bytes, 1);
                    case Op::K_BINSTRING:
                    case Op::K_BINBYTES:
                        return push_text(PickleKind::Bytes, 4);
                    case Op::K_BINBYTES8:
                        return push_text(PickleKind::Bytes, 8);
                    case Op::K_NONE:
                        stack_.push_back(make(PickleKind::None));
                        return Status();
                    case Op::K_NEWTRUE:
                    case Op::K_NEWFALSE:
                    {
                        PickleRef value = make(PickleKind::Bool);
                        value->boolean = opcode == Op::K_NEWTRUE;
                        stack_.push_back(std::move(value));
                        return Status();
                    }
                    default:
                        ran = false;
                        return Status();
                }
            }

            Status Machine::step_object(std::uint8_t opcode, bool& ran)
            {
                ran = true;
                switch (opcode)
                {
                    case Op::K_BINPUT:
                    case Op::K_LONG_BINPUT:
                    {
                        SA_TRY_ASSIGN(const std::uint64_t slot,
                                      take_le(opcode == Op::K_BINPUT ? 1 : 4));
                        return memoize(slot);
                    }
                    case Op::K_MEMOIZE:
                        return memoize(memo_.size());
                    case Op::K_BINGET:
                    case Op::K_LONG_BINGET:
                    {
                        SA_TRY_ASSIGN(const std::uint64_t slot,
                                      take_le(opcode == Op::K_BINGET ? 1 : 4));
                        return fetch(slot);
                    }
                    case Op::K_GLOBAL:
                    {
                        SA_TRY_ASSIGN(const std::string module, take_line());
                        SA_TRY_ASSIGN(const std::string name, take_line());
                        return push_global(module, name);
                    }
                    case Op::K_STACK_GLOBAL:
                    {
                        SA_TRY_ASSIGN(const PickleRef name, pop());
                        SA_TRY_ASSIGN(const PickleRef module, pop());
                        if (name->kind != PickleKind::String || module->kind != PickleKind::String)
                            return malformed("names a global with values that are no text");
                        return push_global(module->text, name->text);
                    }
                    case Op::K_REDUCE:
                    case Op::K_NEWOBJ:
                    {
                        SA_TRY_ASSIGN(const PickleRef arguments, pop());
                        SA_TRY_ASSIGN(PickleRef callable, pop());
                        return push_call(std::move(callable), arguments);
                    }
                    case Op::K_NEWOBJ_EX:
                    {
                        SA_TRY_ASSIGN(const PickleRef keywords, pop());
                        SA_TRY_ASSIGN(const PickleRef arguments, pop());
                        SA_TRY_ASSIGN(PickleRef callable, pop());
                        SA_TRY(push_call(std::move(callable), arguments));
                        stack_.back()->entries = keywords->entries;
                        return Status();
                    }
                    case Op::K_BUILD:
                    {
                        SA_TRY_ASSIGN(PickleRef state, pop());
                        SA_TRY_ASSIGN(const PickleRef target, top());
                        if (target->kind != PickleKind::Call)
                            return malformed("builds a value that is no call");
                        target->state = std::move(state);
                        return Status();
                    }
                    case Op::K_BINPERSID:
                    {
                        SA_TRY_ASSIGN(PickleRef id, pop());
                        PickleRef value = make(PickleKind::PersistentID);
                        value->items.push_back(std::move(id));
                        stack_.push_back(std::move(value));
                        return Status();
                    }
                    default:
                        ran = false;
                        return Status();
                }
            }
        } // namespace

        bool is_call_of(const PickleValue* value, std::string_view name) noexcept
        {
            return value != nullptr && value->kind == PickleKind::Call &&
                   value->callable != nullptr && value->callable->text == name;
        }

        const PickleValue* find_entry(const PickleValue& value, std::string_view key) noexcept
        {
            for (const std::pair<PickleRef, PickleRef>& entry : value.entries)
            {
                if (entry.first->kind == PickleKind::String && entry.first->text == key)
                    return entry.second.get();
            }
            return nullptr;
        }

        Result<PickleRef> read_pickle(SushiRuntime::span<const std::byte> bytes,
                                      SushiRuntime::span<const std::string_view> accepted,
                                      std::string_view origin) noexcept
        {
            Machine machine(bytes, accepted, origin);
            return machine.run();
        }
    } // namespace IO
} // namespace SushiAI
