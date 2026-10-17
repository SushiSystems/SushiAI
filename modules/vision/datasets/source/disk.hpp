/**************************************************************************/
/* disk.hpp                                                               */
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
 * @file disk.hpp
 * @brief Declares the file system calls the format readers share, each returning a Status.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <exception>
#include <filesystem>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

/** @brief Holds the units of the datasets module that no caller names. */
namespace SushiAI::Vision::Datasets::Detail
{
    /** @brief Names what list_entries keeps of a directory. */
    enum class EntryKind : std::uint8_t
    {
        Directory, /**< Entries that are directories. */
        File,      /**< Entries that are regular files. */
    };

    /** @brief Names how far below a directory list_entries looks. */
    enum class ListDepth : std::uint8_t
    {
        Children, /**< The entries of the directory itself. */
        Tree,     /**< The entries of the directory and of every directory below it. */
    };

    /** @brief Returns the text of a path with forward slashes, for a message. */
    [[nodiscard]] std::string display_of(const std::filesystem::path& path);

    /**
     * @brief Returns the entries of one kind in or below a directory, sorted by path.
     * @param reader Names the calling reader in a message.
     * @return The paths, each opening with @p directory; FileNotFound when it is not a
     *         directory; InvalidFile when it cannot be listed.
     */
    [[nodiscard]] Result<std::vector<std::filesystem::path>> list_entries(
        const std::filesystem::path& directory, EntryKind kind, ListDepth depth,
        std::string_view reader);

    /**
     * @brief Returns the byte count of a regular file.
     * @param reader Names the calling reader in a message.
     * @return The size; FileNotFound when the path names nothing; InvalidFile when it names
     *         something else or the size cannot be read.
     */
    [[nodiscard]] Result<std::uint64_t> size_of_file(const std::filesystem::path& file,
                                                     std::string_view reader);

    /**
     * @brief Reads every byte of a file as text, changing none.
     * @param reader Names the calling reader in a message.
     * @return The bytes; what size_of_file returns; InvalidFile when the read stops early.
     */
    [[nodiscard]] Result<std::string> read_text_file(const std::filesystem::path& file,
                                                     std::string_view reader);

    /** @brief Holds what a message says of an allocation that failed. */
    inline constexpr std::string_view K_ALLOCATION_FAILED = "an allocation failed";

    /** @brief Holds what a message says of an exception that is no std::exception. */
    inline constexpr std::string_view K_UNKNOWN_EXCEPTION = "an unknown exception";

    /** @brief Returns a failure whose message is "<reader>: <what>"; never a success. */
    [[nodiscard]] Status boundary_failure(StatusCode code, std::string_view reader,
                                          std::string_view what) noexcept;

    /** @brief Returns a failure whose message is "'<path>': <reader>: <what>". */
    [[nodiscard]] Status boundary_failure(StatusCode code, std::string_view reader,
                                          const std::filesystem::path& subject,
                                          std::string_view what) noexcept;

    /**
     * @brief Runs a call and returns its Status or Result, or the Status of what it threw.
     * @param call Called once with no argument; returns a Status or a Result.
     * @param subject Nothing, or the one path a failure message opens with.
     * @return What the call returns; OutOfMemory when it throws std::bad_alloc; InvalidFile
     *         with the exception's text when it throws anything else.
     */
    template <typename Call, typename... Subject>
    [[nodiscard]] auto at_boundary(std::string_view reader, Call&& call,
                                   const Subject&... subject) noexcept
        -> decltype(std::forward<Call>(call)())
    {
        try
        {
            return std::forward<Call>(call)();
        }
        catch (const std::bad_alloc&)
        {
            return boundary_failure(StatusCode::OutOfMemory, reader, subject...,
                                    K_ALLOCATION_FAILED);
        }
        catch (const std::exception& error)
        {
            return boundary_failure(StatusCode::InvalidFile, reader, subject..., error.what());
        }
        catch (...)
        {
            return boundary_failure(StatusCode::InvalidFile, reader, subject...,
                                    K_UNKNOWN_EXCEPTION);
        }
    }
} // namespace SushiAI::Vision::Datasets::Detail
