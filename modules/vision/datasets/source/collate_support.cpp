/**************************************************************************/
/* collate_support.cpp                                                    */
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
 * @file collate_support.cpp
 * @brief Implements the lookups the collators share.
 * @author Mustafa Garip
 */

#include "collate_support.hpp"

#include <SushiLog/text/text_format.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/target.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets::Detail
{
    std::string file_of(const SampleList& list, const FittedSample& sample)
    {
        if (sample.file < list.files.size())
            return display_of(list.files[sample.file]);

        return SushiLog::format_text("file {}", sample.file);
    }

    Result<SushiBLAS::Tensor*> role_tensor(Data::Batch& batch, std::string_view role,
                                           DType dtype, std::int64_t elements,
                                           std::string_view collator)
    {
        SushiBLAS::Tensor* found = nullptr;

        for (Data::BatchTarget& target : batch.targets)
        {
            if (target.role == role)
                found = &target.values;
        }

        SA_FAIL_IF(found == nullptr, StatusCode::Incompatible,
                   "{}: the batch holds no role '{}'", collator, role);
        SA_FAIL_IF(found->dtype != dtype || found->num_elements != elements,
                   StatusCode::Incompatible,
                   "{}: the role '{}' holds {} elements; the collator writes {} of its own "
                   "element type",
                   collator, role, found->num_elements, elements);

        return found;
    }
} // namespace SushiAI::Vision::Datasets::Detail
