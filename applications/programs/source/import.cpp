/**************************************************************************/
/* import.cpp                                                             */
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
 * @file import.cpp
 * @brief Runs `sushiai import`: turns the files of another framework into SushiAI's own.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "import.cpp".
 */

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/SushiAI.h>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/io/importer_registry.hpp>
#include <SushiBLAS/core/format.hpp>
#include "registries.hpp"

namespace
{
    /** @brief Holds the text `--help` prints. */
    constexpr std::string_view K_USAGE =
        "sushiai import - write a model of another framework as SushiAI's files.\n"
        "\n"
        "Usage: sushiai import <format> <files...> --output <directory> [--force]\n"
        "\n"
        "  format      the importer to read with; `darknet` takes a .cfg and\n"
        "              optionally a .weights\n"
        "  files       the files of the other framework, in the order the format states\n"
        "  --output    the directory that receives model.json, weights.sushi and\n"
        "              train.json; it is created when absent\n"
        "  --force     replace a file that is already there\n";

    /** @brief Holds the usage line a refused command line quotes. */
    constexpr std::string_view K_USAGE_LINE =
        "usage: sushiai import <format> <files...> --output <directory> [--force]";

    /** @brief Holds the flag that names the output directory, and its short form. */
    constexpr std::string_view K_OUTPUT_FLAGS[] = {"--output", "-o"};

    /** @brief Holds the flag that lets the import replace existing files. */
    constexpr std::string_view K_FORCE_FLAG = "--force";

    /** @brief Holds what the command line of one import states. */
    struct ImportRequest
    {
        /** @brief The format name, the first argument. */
        std::string format;

        /** @brief The foreign files, in the order given. */
        std::vector<std::filesystem::path> files;

        /** @brief The directory the three files are written into. */
        std::filesystem::path output;

        /** @brief Whether a file that exists is refused or replaced. */
        sa::IO::ExistingFiles existing = sa::IO::ExistingFiles::Refuse;
    };

    /** @brief Returns whether @p argument is a spelling of the output flag. */
    [[nodiscard]] bool is_output_flag(std::string_view argument) noexcept
    {
        for (const std::string_view flag : K_OUTPUT_FLAGS)
        {
            if (flag == argument)
                return true;
        }

        return false;
    }

    /**
     * @brief Reads the command line into a request.
     * @return The request; InvalidDescription quoting the usage line for no format, no file,
     *         no output directory, an output flag without a value or a directory given twice.
     */
    [[nodiscard]] sa::Result<ImportRequest> read_request(int argc, char** argv)
    {
        SA_FAIL_IF(argc < 2, sa::StatusCode::InvalidDescription, "{}", K_USAGE_LINE);

        ImportRequest request;
        request.format = argv[1];
        for (int at = 2; at < argc; ++at)
        {
            const std::string_view argument = argv[at];
            if (argument == K_FORCE_FLAG)
            {
                request.existing = sa::IO::ExistingFiles::Replace;
                continue;
            }

            if (!is_output_flag(argument))
            {
                request.files.emplace_back(argument);
                continue;
            }

            SA_FAIL_IF(at + 1 >= argc, sa::StatusCode::InvalidDescription,
                       "'{}' names no directory; {}", argument, K_USAGE_LINE);
            SA_FAIL_IF(!request.output.empty(), sa::StatusCode::InvalidDescription,
                       "the output directory is given twice; {}", K_USAGE_LINE);
            request.output = argv[++at];
        }

        SA_FAIL_IF(request.files.empty(), sa::StatusCode::InvalidDescription,
                   "no file to import was given; {}", K_USAGE_LINE);
        SA_FAIL_IF(request.output.empty(), sa::StatusCode::InvalidDescription,
                   "no output directory was given; {}", K_USAGE_LINE);
        return request;
    }

    /** @brief Prints what was read, the files written and every note of the import. */
    void print_report(const ImportRequest& request, const sa::IO::ImportedModel& imported,
                      const sa::IO::WrittenImport& written)
    {
        std::cout << "sushiai import\n"
                  << "----------------------------------------------------------------\n"
                  << SushiBLAS::Core::format("Format     : {}\n", request.format);
        for (const std::filesystem::path& file : request.files)
            std::cout << SushiBLAS::Core::format("Input      : {}\n", file.string());

        const std::string task =
            imported.model.task.is_null()
                ? std::string("no task")
                : "task " + imported.model.task.at("type", "the task section")
                                .as_string("the task type");
        std::cout << SushiBLAS::Core::format("Model      : {} rows, {} outputs, {}\n",
                                             imported.model.layers.size(),
                                             imported.model.outputs.size(), task)
                  << SushiBLAS::Core::format("Weights    : {} tensors\n\n",
                                             imported.weights.size());

        for (const std::filesystem::path& file : written.files)
            std::cout << SushiBLAS::Core::format("wrote {}\n", file.string());

        std::cout << "\nnotes:\n";
        for (const std::string& note : imported.notes)
            std::cout << "  - " << note << "\n";

        // The data block is the one part of a run no foreign file states.
        std::cout << SushiBLAS::Core::format(
            "  - {} states \"data\": null, and `sushiai train` refuses it until the block "
            "names a dataset\n",
            written.files.back().string());
        if (imported.weights.empty())
            std::cout << "  - no weights were given, so a run initialises the model\n";
    }

    /**
     * @brief Imports the files the arguments name and writes the three files.
     * @return Success, or the first failure of the command line, the importer or the writer.
     */
    [[nodiscard]] sa::Status run(int argc, char** argv)
    {
        if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
        {
            std::cout << K_USAGE;
            return sa::Status();
        }

        SA_TRY_ASSIGN(const ImportRequest request, read_request(argc, argv));
        SA_TRY_ASSIGN(const sa::Apps::Registries registries, sa::Apps::compose_registries());

        // The importer is picked by string, not by type; see applications/programs/README.md.
        SA_TRY_ASSIGN(const sa::IO::ImportedModel imported,
                      sa::IO::import_model(registries.importers, request.format,
                                           request.files));
        SA_TRY_ASSIGN(const sa::IO::WrittenImport written,
                      sa::IO::write_imported_model(imported, request.output,
                                                   request.existing));

        print_report(request, imported, written);
        return sa::Status();
    }
} // namespace

/** @brief Runs the import driver and returns its exit code. */
int main(int argc, char** argv)
{
    try
    {
        const sa::Status status = run(argc, argv);
        if (!status.is_ok())
        {
            std::cerr << "sushiai import failed: " << status.message() << "\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sushiai import failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
