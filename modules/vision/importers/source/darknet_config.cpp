/**************************************************************************/
/* darknet_config.cpp                                                     */
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
 * @file darknet_config.cpp
 * @brief Defines the line grammar of Darknet's .cfg text and the assembly of its sections.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/darknet_config.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /**
             * @brief Returns whether @p character is removed from every line before it is read.
             * @return True for a space, a tab, a carriage return and a line feed.
             */
            bool is_removed(char character) noexcept
            {
                return character == ' ' || character == '\t' || character == '\r' ||
                       character == '\n';
            }

            /**
             * @brief Returns the line @p raw without the characters that is_removed names.
             * @return The line with every such character removed, inner ones included.
             */
            std::string strip_whitespace(std::string_view raw)
            {
                std::string stripped;
                stripped.reserve(raw.size());
                for (char character : raw)
                {
                    if (!is_removed(character))
                        stripped.push_back(character);
                }
                return stripped;
            }

            /**
             * @brief Returns whether a stripped line carries nothing: it is empty or a comment.
             * @return True when the line is empty or starts with '#' or ';'.
             */
            bool is_skipped(std::string_view stripped) noexcept
            {
                return stripped.empty() || stripped.front() == '#' || stripped.front() == ';';
            }

            /**
             * @brief Reads the name between the brackets of a section line.
             * @param stripped A stripped line that starts with '['.
             * @param origin The name the diagnostic quotes.
             * @param number The 1-based line the diagnostic quotes.
             * @return The name, or InvalidDescription when the line does not end with ']', or the
             *         name is empty or holds a bracket.
             */
            Result<std::string> section_name(std::string_view stripped, std::string_view origin,
                                             std::int64_t number)
            {
                SA_FAIL_IF(stripped.back() != ']', StatusCode::InvalidDescription,
                           "{}:{}: a section line must end with ']'", origin, number);

                const std::string_view name = stripped.substr(1, stripped.size() - 2);
                SA_FAIL_IF(name.empty() || name.find_first_of("[]") != std::string_view::npos,
                           StatusCode::InvalidDescription,
                           "{}:{}: a section needs one name between '[' and ']'", origin, number);

                return std::string(name);
            }

            /**
             * @brief Splits a stripped key=value line at its first '='.
             * @param stripped A stripped line that is neither a comment nor a section.
             * @param origin The name the diagnostic quotes.
             * @param number The 1-based line the diagnostic quotes.
             * @return The option, or InvalidDescription when the line has no '=' or no key.
             */
            Result<DarknetOption> option_of(std::string_view stripped, std::string_view origin,
                                            std::int64_t number)
            {
                const std::size_t equals = stripped.find('=');
                SA_FAIL_IF(equals == std::string_view::npos, StatusCode::InvalidDescription,
                           "{}:{}: a line is neither a [section] nor key=value", origin, number);
                SA_FAIL_IF(equals == 0, StatusCode::InvalidDescription,
                           "{}:{}: a key is empty", origin, number);

                DarknetOption option;
                option.key = std::string(stripped.substr(0, equals));
                option.value = std::string(stripped.substr(equals + 1));
                option.line = number;
                return option;
            }

            /** @brief Collects the sections of one file and checks the rules of their order. */
            class ConfigAssembler final
            {
                public:
                    /**
                     * @brief Starts an assembly with no section.
                     * @param origin The name the diagnostics quote; copied.
                     */
                    explicit ConfigAssembler(std::string_view origin) : origin_(origin)
                    {

                    }

                    /**
                     * @brief Opens a section named @p type that starts at line @p number.
                     * @param type The name between the brackets.
                     * @param number The 1-based line of the section line.
                     * @return Success, or InvalidDescription when the first section is neither
                     *         [net] nor [network].
                     */
                    [[nodiscard]] Status open_section(std::string type, std::int64_t number);

                    /**
                     * @brief Adds @p option to the section that is open.
                     * @param option The key and value, with the line that spells them.
                     * @return Success, or InvalidDescription when no section is open or the
                     *         section already holds the key.
                     */
                    [[nodiscard]] Status add_option(DarknetOption option);

                    /**
                     * @brief Returns the sections as a configuration, the first one as net.
                     * @return The configuration, or InvalidDescription when no section was opened.
                     */
                    [[nodiscard]] Result<DarknetConfig> finish();

                private:
                    /** @brief Holds the name the diagnostics quote. */
                    std::string origin_;

                    /** @brief Holds the sections in file order; the first becomes net. */
                    std::vector<DarknetSection> sections_;
            };

            Status ConfigAssembler::open_section(std::string type, std::int64_t number)
            {
                SA_FAIL_IF(sections_.empty() && type != "net" && type != "network",
                           StatusCode::InvalidDescription,
                           "{}:{}: the first section is [{}]; a .cfg opens with [net] or [network]",
                           origin_, number, type);

                DarknetSection section;
                section.type = std::move(type);
                section.line = number;
                sections_.push_back(std::move(section));
                return Status();
            }

            Status ConfigAssembler::add_option(DarknetOption option)
            {
                SA_FAIL_IF(sections_.empty(), StatusCode::InvalidDescription,
                           "{}:{}: key '{}' appears before any section",
                           origin_, option.line, option.key);

                std::vector<DarknetOption>& options = sections_.back().options;
                for (const DarknetOption& earlier : options)
                {
                    SA_FAIL_IF(earlier.key == option.key, StatusCode::InvalidDescription,
                               "{}:{}: key '{}' repeats the one on line {}",
                               origin_, option.line, option.key, earlier.line);
                }

                options.push_back(std::move(option));
                return Status();
            }

            Result<DarknetConfig> ConfigAssembler::finish()
            {
                SA_FAIL_IF(sections_.empty(), StatusCode::InvalidDescription,
                           "{}: the file has no [net] or [network] section", origin_);

                DarknetConfig config;
                config.net = std::move(sections_.front());
                sections_.erase(sections_.begin());
                config.layers = std::move(sections_);
                return std::move(config);
            }

            /**
             * @brief Applies one physical line of the file to @p assembler.
             * @param assembler The assembly the line extends.
             * @param raw The line as it was read, without its line feed.
             * @param origin The name the diagnostics quote.
             * @param number The 1-based line of @p raw.
             * @return Success, or the InvalidDescription of the line.
             */
            Status accept_line(ConfigAssembler& assembler, std::string_view raw,
                               std::string_view origin, std::int64_t number)
            {
                const std::string stripped = strip_whitespace(raw);
                if (is_skipped(stripped))
                    return Status();

                if (stripped.front() == '[')
                {
                    SA_TRY_ASSIGN(const std::string type, section_name(stripped, origin, number));
                    return assembler.open_section(type, number);
                }

                SA_TRY_ASSIGN(const DarknetOption option, option_of(stripped, origin, number));
                return assembler.add_option(option);
            }
        } // namespace

        Result<DarknetConfig> parse_darknet_config(std::string_view text, std::string_view origin) noexcept
        {
            ConfigAssembler assembler(origin);
            std::size_t begin = 0;
            std::int64_t number = 0;
            while (begin <= text.size())
            {
                std::size_t end = text.find('\n', begin);
                if (end == std::string_view::npos)
                    end = text.size();

                ++number;
                SA_TRY(accept_line(assembler, text.substr(begin, end - begin), origin, number));
                begin = end + 1;
            }
            return assembler.finish();
        }

        Result<DarknetConfig> load_darknet_config(const std::filesystem::path& path) noexcept
        {
            std::ifstream in(path, std::ios::binary);
            SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                       path.string());

            const std::string text((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
            SA_FAIL_IF(in.bad(), StatusCode::InvalidFile, "could not read '{}'", path.string());

            return parse_darknet_config(text, path.string());
        }
    } // namespace Vision::Importers
} // namespace SushiAI
