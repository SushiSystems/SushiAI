/**************************************************************************/
/* export.cpp                                                             */
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
 * @file export.cpp
 * @brief Runs `sushiai export`: writes SushiAI's files as the files of another framework.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "export.cpp".
 */

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/io/exporter_registry.hpp>
#include <SushiLog/text/text_format.hpp>
#include "model_export.hpp"
#include "registries.hpp"
#include "weights_export.hpp"

namespace
{
    /** @brief Holds the text `--help` prints. */
    constexpr std::string_view K_USAGE =
        "sushiai export - write a SushiAI model as the files of another framework.\n"
        "\n"
        "Usage: sushiai export <format> <model.json> [<weights.sushi>] --output <directory>\n"
        "                      [--training <train.json>] [--name <stem>] [--force]\n"
        "\n"
        "  format      the exporter to write with: onnx, caffe or darknet\n"
        "  model.json  the model file\n"
        "  weights     the .sushi file whose model-state records become the weights\n"
        "  --output    the directory that receives the files; it is created when absent\n"
        "  --training  the training file, for a format that states a training recipe\n"
        "  --name      the stem of every written file; model when absent\n"
        "  --force     replace a file that is already there\n"
        "\n"
        "Usage: sushiai export weights <safetensors|torch> <model.json> <weights.sushi>\n"
        "                              --output <file> [--map <map.json>] [--force]\n"
        "\n"
        "  model.json     the model file; it says which records are convolution kernels\n"
        "  weights.sushi  the .sushi file whose model-state records are written\n"
        "  --map          the name map `import weights` takes; its renames are applied\n"
        "                 backwards and its state_dict names the key of the dictionary\n"
        "  --output       the weight file to write\n";

    /** @brief Holds the usage line a refused command line quotes. */
    constexpr std::string_view K_USAGE_LINE =
        "usage: sushiai export <format> <model.json> [<weights.sushi>] --output <directory> "
        "[--training <train.json>] [--name <stem>] [--force]";

    /** @brief Holds the usage line a refused weights command line quotes. */
    constexpr std::string_view K_WEIGHTS_USAGE_LINE =
        "usage: sushiai export weights <safetensors|torch> <model.json> <weights.sushi> "
        "--output <file> [--map <map.json>] [--force]";

    /** @brief Holds the flag that names the name map. */
    constexpr std::string_view K_MAP_FLAG = "--map";

    /** @brief Holds the flag that names the training file. */
    constexpr std::string_view K_TRAINING_FLAG = "--training";

    /** @brief Holds the flag that names the stem of the written files. */
    constexpr std::string_view K_NAME_FLAG = "--name";

    /** @brief Holds the flag that names the output, and its short form. */
    constexpr std::string_view K_OUTPUT_FLAGS[] = {"--output", "-o"};

    /** @brief Holds the flag that lets the export replace existing files. */
    constexpr std::string_view K_FORCE_FLAG = "--force";

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
     * @brief Returns the value field of @p request that the flag @p argument sets.
     * @return Null when @p argument is no flag that takes a value.
     */
    [[nodiscard]] std::filesystem::path* model_flag_target(std::string_view argument,
                                                           sa::Apps::ModelExportRequest& request)
    {
        if (is_output_flag(argument))
            return &request.output;
        if (argument == K_TRAINING_FLAG)
            return &request.training;
        return nullptr;
    }

    /**
     * @brief Reads the command line of a model export into a request.
     * @return The request; InvalidDescription quoting the usage line for no format, a model
     *         file count other than one or two, no output directory, a flag without a value or
     *         a flag given twice.
     */
    [[nodiscard]] sa::Result<sa::Apps::ModelExportRequest> read_request(int argc, char** argv)
    {
        SA_FAIL_IF(argc < 2, sa::StatusCode::InvalidDescription, "{}", K_USAGE_LINE);

        sa::Apps::ModelExportRequest request;
        request.format = argv[1];
        std::vector<std::filesystem::path> files;
        bool named = false;
        for (int at = 2; at < argc; ++at)
        {
            const std::string_view argument = argv[at];
            if (argument == K_FORCE_FLAG)
            {
                request.existing = sa::IO::ExistingFiles::Replace;
                continue;
            }

            if (argument == K_NAME_FLAG)
            {
                SA_FAIL_IF(at + 1 >= argc, sa::StatusCode::InvalidDescription,
                           "'{}' names no stem; {}", argument, K_USAGE_LINE);
                SA_FAIL_IF(named, sa::StatusCode::InvalidDescription, "'{}' is given twice; {}",
                           argument, K_USAGE_LINE);
                request.name = argv[++at];
                named = true;
                continue;
            }

            std::filesystem::path* const target = model_flag_target(argument, request);
            if (target == nullptr)
            {
                files.emplace_back(argument);
                continue;
            }

            SA_FAIL_IF(at + 1 >= argc, sa::StatusCode::InvalidDescription,
                       "'{}' names no path; {}", argument, K_USAGE_LINE);
            SA_FAIL_IF(!target->empty(), sa::StatusCode::InvalidDescription,
                       "'{}' is given twice; {}", argument, K_USAGE_LINE);
            *target = argv[++at];
        }

        SA_FAIL_IF(files.empty() || files.size() > 2, sa::StatusCode::InvalidDescription,
                   "expected a model file and optionally a weight file, got {} files; {}",
                   files.size(), K_USAGE_LINE);
        SA_FAIL_IF(request.output.empty(), sa::StatusCode::InvalidDescription,
                   "no output directory was given; {}", K_USAGE_LINE);
        request.model = files[0];
        if (files.size() == 2)
            request.weights = files[1];
        return request;
    }

    /**
     * @brief Reads the command line of `export weights` into a request.
     * @return The request; InvalidDescription quoting the usage line for a file count other than
     *         a format, a model file and a weight file, no output, a flag without a value or a
     *         flag given twice.
     */
    [[nodiscard]] sa::Result<sa::Apps::WeightsExportRequest> read_weights_request(int argc,
                                                                                  char** argv)
    {
        sa::Apps::WeightsExportRequest request;
        std::vector<std::string> words;
        for (int at = 2; at < argc; ++at)
        {
            const std::string_view argument = argv[at];
            if (argument == K_FORCE_FLAG)
            {
                request.existing = sa::IO::ExistingFiles::Replace;
                continue;
            }

            const bool is_map = argument == K_MAP_FLAG;
            if (!is_map && !is_output_flag(argument))
            {
                words.emplace_back(argument);
                continue;
            }

            std::filesystem::path& target = is_map ? request.map : request.output;
            SA_FAIL_IF(at + 1 >= argc, sa::StatusCode::InvalidDescription, "'{}' names no file; {}",
                       argument, K_WEIGHTS_USAGE_LINE);
            SA_FAIL_IF(!target.empty(), sa::StatusCode::InvalidDescription,
                       "'{}' is given twice; {}", argument, K_WEIGHTS_USAGE_LINE);
            target = argv[++at];
        }

        SA_FAIL_IF(words.size() != 3, sa::StatusCode::InvalidDescription,
                   "expected a format, a model file and a weight file, got {} arguments; {}",
                   words.size(), K_WEIGHTS_USAGE_LINE);
        SA_FAIL_IF(request.output.empty(), sa::StatusCode::InvalidDescription,
                   "no output file was given; {}", K_WEIGHTS_USAGE_LINE);
        request.format = words[0];
        request.model = words[1];
        request.weights = words[2];
        return request;
    }

    /** @brief Prints every note under a heading, or that there is none. */
    void print_notes(const std::vector<std::string>& notes)
    {
        std::cout << "\nnotes:\n";
        for (const std::string& note : notes)
            std::cout << "  - " << note << "\n";
        if (notes.empty())
            std::cout << "  - none\n";
    }

    /** @brief Exports one `.sushi` file as a foreign weight file and prints what was written. */
    [[nodiscard]] sa::Status run_weights(int argc, char** argv)
    {
        SA_TRY_ASSIGN(const sa::Apps::WeightsExportRequest request,
                      read_weights_request(argc, argv));
        SA_TRY_ASSIGN(const sa::Apps::Registries registries, sa::Apps::compose_registries());
        SA_TRY_ASSIGN(const sa::Apps::WeightsExportReport report,
                      sa::Apps::export_weights_file(request, registries));

        std::cout << "sushiai export weights\n"
                  << "----------------------------------------------------------------\n"
                  << SushiLog::format_text("Format     : {}\n", request.format)
                  << SushiLog::format_text("Model      : {}\n", request.model.string())
                  << SushiLog::format_text("Weights    : {}\n", request.weights.string())
                  << SushiLog::format_text("Map        : {}\n",
                                           request.map.empty() ? std::string("none")
                                                               : request.map.string())
                  << SushiLog::format_text("Tensors    : {} written\n\n", report.written)
                  << SushiLog::format_text("wrote {}\n", request.output.string());
        print_notes(report.notes);
        return sa::Status();
    }

    /**
     * @brief Exports the files the arguments name and writes the files of the format.
     * @return Success, or the first failure of the command line, the exporter or the writer.
     */
    [[nodiscard]] sa::Status run(int argc, char** argv)
    {
        if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
        {
            std::cout << K_USAGE;
            return sa::Status();
        }

        if (argc >= 2 && sa::IO::K_WEIGHT_EXPORT_WORD == argv[1])
            return run_weights(argc, argv);

        SA_TRY_ASSIGN(const sa::Apps::ModelExportRequest request, read_request(argc, argv));
        SA_TRY_ASSIGN(const sa::Apps::Registries registries, sa::Apps::compose_registries());
        SA_TRY_ASSIGN(const sa::Apps::ModelExportReport report,
                      sa::Apps::export_model_files(request, registries));

        std::cout << "sushiai export\n"
                  << "----------------------------------------------------------------\n"
                  << SushiLog::format_text("Format     : {}\n", request.format)
                  << SushiLog::format_text("Model      : {}\n", request.model.string());
        if (!request.weights.empty())
            std::cout << SushiLog::format_text("Weights    : {}\n", request.weights.string());
        if (!request.training.empty())
            std::cout << SushiLog::format_text("Training   : {}\n", request.training.string());
        std::cout << "\n";
        for (const std::filesystem::path& file : report.written)
            std::cout << SushiLog::format_text("wrote {}\n", file.string());
        print_notes(report.notes);
        return sa::Status();
    }
} // namespace

/** @brief Runs the export driver and returns its exit code. */
int main(int argc, char** argv)
{
    const sa::Status status = run(argc, argv);
    if (status.is_ok())
        return EXIT_SUCCESS;

    std::cerr << "sushiai export failed: " << status.message() << "\n"
              << "  kind: " << sa::status_code_name(status.code()) << "\n";
    return EXIT_FAILURE;
}
