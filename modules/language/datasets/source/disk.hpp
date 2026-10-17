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
 * @brief Declares the file system calls of the language datasets module, each returning a Status.
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

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

/** @brief Holds the units of the language datasets module that no caller names. */
namespace SushiAI::Language::Datasets::Detail
{
    /** @brief Returns the text of a path with forward slashes, for a message. */
    [[nodiscard]] std::string display_of(const std::filesystem::path& path);

    /**
     * @brief Returns the byte count of a regular file.
     * @param reader Names the calling unit in a message.
     * @return The size; FileNotFound when the path names nothing; InvalidFile when it names
     *         something else or the size cannot be read.
     */
    [[nodiscard]] Result<std::uint64_t> size_of_file(const std::filesystem::path& file,
                                                     std::string_view reader);

    /** @brief Holds what a message says of an allocation that failed. */
    inline constexpr std::string_view K_ALLOCATION_FAILED = "an allocation failed";

    /** @brief Holds what a message says of an exception that is no std::exception. */
    inline constexpr std::string_view K_UNKNOWN_EXCEPTION = "an unknown exception";

    /** @brief Returns a failure whose message is "<unit>: <what>"; never a success. */
    [[nodiscard]] Status boundary_failure(StatusCode code, std::string_view unit,
                                          std::string_view what) noexcept;

    /** @brief Returns a failure whose message is "'<path>': <unit>: <what>". */
    [[nodiscard]] Status boundary_failure(StatusCode code, std::string_view unit,
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
    [[nodiscard]] auto at_boundary(std::string_view unit, Call&& call,
                                   const Subject&... subject) noexcept
        -> decltype(std::forward<Call>(call)())
    {
        try
        {
            return std::forward<Call>(call)();
        }
        catch (const std::bad_alloc&)
        {
            return boundary_failure(StatusCode::OutOfMemory, unit, subject...,
                                    K_ALLOCATION_FAILED);
        }
        catch (const std::exception& error)
        {
            return boundary_failure(StatusCode::InvalidFile, unit, subject..., error.what());
        }
        catch (...)
        {
            return boundary_failure(StatusCode::InvalidFile, unit, subject...,
                                    K_UNKNOWN_EXCEPTION);
        }
    }
} // namespace SushiAI::Language::Datasets::Detail
