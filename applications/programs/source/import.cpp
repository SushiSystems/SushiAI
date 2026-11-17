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
 * @file import.cpp
 * @brief Runs `sushiai import`: turns the files of another framework into SushiAI's own.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "import.cpp".
 */

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/io/importer_registry.hpp>
#include <SushiLog/text/text_format.hpp>
#include "registries.hpp"
#include "weights_import.hpp"

namespace
{
    /** @brief Holds the text `--help` prints. */
    constexpr std::string_view K_USAGE =
        "sushiai import - write a model of another framework as SushiAI's files.\n"
        "\n"
        "Usage: sushiai import <format> <files...> --output <directory> [--task <task.json>]\n"
        "                      [--force]\n"
        "\n"
        "  format      the importer to read with; `darknet` takes a .cfg and\n"
        "              optionally a .weights; `ultralytics` takes one .pt; `onnx` takes\n"
        "              one .onnx; `caffe` takes a .prototxt and optionally a .caffemodel\n"
        "  files       the files of the other framework, in the order the format states\n"
        "  --output    the directory that receives model.json, weights.sushi and\n"
        "              train.json; it is created when absent\n"
        "  --task      a JSON object with a string \"type\", which becomes the task section\n"
        "              of model.json; refused when the importer wrote one\n"
        "  --force     replace a file that is already there\n"
        "\n"
        "Usage: sushiai import weights <model.json> <weights-file> --map <map.json>\n"
        "                              --output <file.sushi> [--force]\n"
        "\n"
        "  model.json     the model file whose state records the weights fill\n"
        "  weights-file   a PyTorch file (.pth, .pt, .pth.tar) or a .safetensors file; the\n"
        "                 container is read from its first bytes\n"
        "  --map          the name map: renames, tensors to drop and the key of the state\n"
        "                 dictionary\n"
        "  --output       the .sushi file to write\n";

    /** @brief Holds the usage line a refused command line quotes. */
    constexpr std::string_view K_USAGE_LINE =
        "usage: sushiai import <format> <files...> --output <directory> [--task <task.json>] "
        "[--force]";

    /** @brief Holds the usage line a refused weights command line quotes. */
    constexpr std::string_view K_WEIGHTS_USAGE_LINE =
        "usage: sushiai import weights <model.json> <weights-file> --map <map.json> "
        "--output <file.sushi> [--force]";

    /** @brief Holds the first argument that selects the weights import. */
    constexpr std::string_view K_WEIGHTS_COMMAND = "weights";

    /** @brief Holds the flag that names the name map. */
    constexpr std::string_view K_MAP_FLAG = "--map";

    /** @brief Holds the flag that names the output directory, and its short form. */
    constexpr std::string_view K_OUTPUT_FLAGS[] = {"--output", "-o"};

    /** @brief Holds the flag that lets the import replace existing files. */
    constexpr std::string_view K_FORCE_FLAG = "--force";

    /** @brief Holds the flag that names the task file. */
    constexpr std::string_view K_TASK_FLAG = "--task";

    /** @brief Holds what the command line of one import states. */
    struct ImportRequest
    {
        /** @brief The format name, the first argument. */
        std::string format;

        /** @brief The foreign files, in the order given. */
        std::vector<std::filesystem::path> files;

        /** @brief The directory the three files are written into. */
        std::filesystem::path output;

        /** @brief The task file to attach to the model; empty states none. */
        std::filesystem::path task;

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
     *         no output directory, a flag without a value or a directory or task file given
     *         twice.
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

            if (argument == K_TASK_FLAG)
            {
                SA_FAIL_IF(at + 1 >= argc, sa::StatusCode::InvalidDescription,
                           "'{}' names no file; {}", argument, K_USAGE_LINE);
                SA_FAIL_IF(!request.task.empty(), sa::StatusCode::InvalidDescription,
                           "the task file is given twice; {}", K_USAGE_LINE);
                request.task = argv[++at];
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

    /**
     * @brief Sets the task section of @p imported to the object in the file @p path.
     * @return InvalidDescription when the importer already wrote a task section or the file
     *         holds no object with a string "type"; what parse_json_file returns otherwise.
     */
    [[nodiscard]] sa::Status attach_task(const std::filesystem::path& path,
                                         std::string_view format, sa::IO::ImportedModel& imported)
    {
        SA_FAIL_IF(!imported.model.task.is_null(), sa::StatusCode::InvalidDescription,
                   "the '{}' importer wrote a task section, and '{}' would replace it", format,
                   path.string());
        SA_TRY_ASSIGN(sa::Config::Value task, sa::Config::parse_json_file(path));
        SA_FAIL_IF(task.kind() != sa::Config::Kind::Object, sa::StatusCode::InvalidDescription,
                   "the task file '{}' is not a JSON object", path.string());
        SA_TRY_ASSIGN(const sa::Config::Value& type, task.at("type", path.string()));
        SA_TRY(type.as_string("the task type"));
        imported.model.task = std::move(task);
        return sa::Status();
    }

    /**
     * @brief Prints what was read, the files written and every note of the import.
     * @return InvalidDescription when the model's task section states no string type.
     */
    [[nodiscard]] sa::Status print_report(const ImportRequest& request,
                                          const sa::IO::ImportedModel& imported,
                                          const sa::IO::WrittenImport& written)
    {
        std::cout << "sushiai import\n"
                  << "----------------------------------------------------------------\n"
                  << SushiLog::format_text("Format     : {}\n", request.format);
        for (const std::filesystem::path& file : request.files)
            std::cout << SushiLog::format_text("Input      : {}\n", file.string());

        std::string task = "no task";
        if (!imported.model.task.is_null())
        {
            SA_TRY_ASSIGN(const sa::Config::Value& stated,
                          imported.model.task.at("type", "the task section"));
            SA_TRY_ASSIGN(const std::string& type, stated.as_string("the task type"));
            task = "task " + type;
        }
        std::cout << SushiLog::format_text("Model      : {} rows, {} outputs, {}\n",
                                             imported.model.layers.size(),
                                             imported.model.outputs.size(), task)
                  << SushiLog::format_text("Weights    : {} tensors\n\n",
                                             imported.weights.size());

        for (const std::filesystem::path& file : written.files)
            std::cout << SushiLog::format_text("wrote {}\n", file.string());

        std::cout << "\nnotes:\n";
        for (const std::string& note : imported.notes)
            std::cout << "  - " << note << "\n";

        // The data block is the one part of a run no foreign file states.
        std::cout << SushiLog::format_text(
            "  - {} states \"data\": null, and `sushiai train` refuses it until the block "
            "names a dataset\n",
            written.files.back().string());
        if (imported.weights.empty())
            std::cout << "  - no weights were given, so a run initialises the model\n";

        return sa::Status();
    }

    /**
     * @brief Reads the command line of `import weights` into a request.
     * @return The request; InvalidDescription quoting the usage line for a missing or repeated
     *         model file, weight file, map or output, or a flag without a value.
     */
    [[nodiscard]] sa::Result<sa::Apps::WeightsImportRequest> read_weights_request(int argc,
                                                                                  char** argv)
    {
        sa::Apps::WeightsImportRequest request;
        std::vector<std::filesystem::path> files;
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
                files.emplace_back(argument);
                continue;
            }

            std::filesystem::path& target = is_map ? request.map : request.output;
            SA_FAIL_IF(at + 1 >= argc, sa::StatusCode::InvalidDescription, "'{}' names no file; {}",
                       argument, K_WEIGHTS_USAGE_LINE);
            SA_FAIL_IF(!target.empty(), sa::StatusCode::InvalidDescription,
                       "'{}' is given twice; {}", argument, K_WEIGHTS_USAGE_LINE);
            target = argv[++at];
        }

        SA_FAIL_IF(files.size() != 2, sa::StatusCode::InvalidDescription,
                   "expected a model file and a weight file, got {} files; {}", files.size(),
                   K_WEIGHTS_USAGE_LINE);
        SA_FAIL_IF(request.map.empty(), sa::StatusCode::InvalidDescription,
                   "no name map was given; {}", K_WEIGHTS_USAGE_LINE);
        SA_FAIL_IF(request.output.empty(), sa::StatusCode::InvalidDescription,
                   "no output file was given; {}", K_WEIGHTS_USAGE_LINE);
        request.model = files[0];
        request.weights = files[1];
        return request;
    }

    /** @brief Imports one weight file into a `.sushi` file and prints what was written. */
    [[nodiscard]] sa::Status run_weights(int argc, char** argv)
    {
        SA_TRY_ASSIGN(const sa::Apps::WeightsImportRequest request,
                      read_weights_request(argc, argv));
        SA_TRY_ASSIGN(const sa::Apps::Registries registries, sa::Apps::compose_registries());
        SA_TRY_ASSIGN(const sa::Apps::WeightsImportReport report,
                      sa::Apps::import_weights_file(request, registries));

        std::cout << "sushiai import weights\n"
                  << "----------------------------------------------------------------\n"
                  << SushiLog::format_text("Model      : {}\n", request.model.string())
                  << SushiLog::format_text("Weights    : {}\n", request.weights.string())
                  << SushiLog::format_text("Map        : {}\n", request.map.string())
                  << SushiLog::format_text("Tensors    : {} written, {} dropped\n\n",
                                             report.written, report.dropped)
                  << SushiLog::format_text("wrote {}\n", request.output.string());
        return sa::Status();
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

        if (argc >= 2 && K_WEIGHTS_COMMAND == argv[1])
            return run_weights(argc, argv);

        SA_TRY_ASSIGN(const ImportRequest request, read_request(argc, argv));
        SA_TRY_ASSIGN(const sa::Apps::Registries registries, sa::Apps::compose_registries());

        // The importer is picked by string, not by type; see applications/programs/README.md.
        SA_TRY_ASSIGN(sa::IO::ImportedModel imported,
                      sa::IO::import_model(registries.importers, request.format,
                                           request.files));
        if (!request.task.empty())
            SA_TRY(attach_task(request.task, request.format, imported));
        SA_TRY_ASSIGN(const sa::IO::WrittenImport written,
                      sa::IO::write_imported_model(imported, request.output,
                                                   request.existing));

        return print_report(request, imported, written);
    }
} // namespace

/** @brief Runs the import driver and returns its exit code. */
int main(int argc, char** argv)
{
    const sa::Status status = run(argc, argv);
    if (status.is_ok())
        return EXIT_SUCCESS;

    std::cerr << "sushiai import failed: " << status.message() << "\n"
              << "  kind: " << sa::status_code_name(status.code()) << "\n";
    return EXIT_FAILURE;
}
