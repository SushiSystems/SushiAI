# Report

Nothing below has been built or run. Each task passed `clang++ -fsyntax-only` on the units it
names and the layering, contract, source-comment and documentation checkers.

## Tasks

| Task | Design phase | State |
| --- | --- | --- |
| `write_json`, `write_model` | Model import 1 | Written, wired into `core/config` |
| The `task` section and `AnchorTask` | Model import 2, data half | Written, wired into `vision/detect` |
| `Batch` with roles, `fill_one_hot`, `match_roles` | Datasets 1 | Written, wired; 85 units checked |
| `vision/codecs` | Datasets 2, decoders | Written, wired |
| Readers `image_folder`, `yolo_txt`, `idx`, `cifar` | Datasets 3 | Written, wired |
| Collators, `load_sample_image`, `ReaderDataset` | Datasets 3 and 4 | Written, wired |
| `Dataset::read`, the fallible seam | Datasets 3 | Written; 379 units checked |
| Gather, scatter-add, the integer leaf, `Embedding` | Language 1 | Written, wired; lowered to SushiBLAS `spmm` |
| `truth_from_roles`, the `anchor` objective section, the `schedule` section | Model import 3 | Written, wired |
| Layer normalisation, the causal mask, the transformer block | Language 2 | Written, wired with the `language` tier |
| The tokenizer, the vocabulary file, the token file, `TokenDataset` | Language 3 | Written, wired; 38 files checked |
| `schedule` in the training file, the rate per update, `ObjectiveRegistry`, `run_anchor_training` | Model import 3 | Written, wired; `train` not rewired |
| The input's element type on `Dataset`, `match_input` | Datasets 3 | Written, wired; 404 units checked |
| `sushiai train` over the run table, the `reader` data block, `weights` | Model import 3, Datasets 4 | Written, wired; 356 units checked |
| `sushiai import darknet`, the importer registry, the inference report by task | Model import 4 and 2 | Written, wired; 347 units checked |
| The next-token objective, the normal initialiser, the clip, parameter groups, the cosine schedule, weight tying | Language 4 | Written, wired; 347 units checked |
| Readers `coco_json`, `voc_xml`, `mask_png`, `csv`; four collators; `TableDataset` | Datasets 3, 5 and 6 | Written, wired; 58 units checked |
| `Objective::feed` and `measure`, three dense objectives, `next_token` through `Trainer` | Datasets 5, Language 4 | Written, wired; 403 units checked |
| `vision/segment`, `pixel_cross_entropy`, `dice`, `SegmentHead` | Datasets 7 | Written, wired; 70 files checked |
| A decoder from files, `sushiai tokenize`, samplers, the window generator | Language 5 | Written, wired; 592 units checked; the cache is a design |
| The scatter-add rule, Dice by index, the `sum` objective, `SectionReader`, this session's deprecations removed | Datasets 7, Model import 3 | Written, wired; 609 units checked; the step count needs a file format change |
| The key and value cache on three general seams of the step | Language 5 | Written, wired; 615 units checked |
| The sample cache, epoch order, pipelines, `ThreadedLoader`, masks from polygons | Datasets 2, 3 and 7 | Written, wired; 27 files checked |
| Mixture of experts: the stacked experts, the router, the balance term; the clip identity moved | Language 7 | Written, wired |
| Version 2 of the weight file and `resume`, the streaming JSON reader, the program test binary | Model import 3, Datasets 6 | Written, wired; 77 files checked, includers of three changed headers not |
| The unfused log-softmax rule | Language 2 | Written, wired |
| Rotary positions, host tables of a step, the ring cache and its position feeds | Language 5 | Written, wired; 31 files checked; a ring of two or more blocks differs from the window form after a slide |
| The `seed` key, `RunProgress` and its four records, a resume at the saved run count, epoch and batch | Model import 3 | Written, wired; 19 files checked |
| Replay preludes, `K_DROPOUT`, the host-drawn mask, `NN::Dropout` | Language 2 | Written, wired; 14 units checked; the seed is not in the training file |

## Decided by the orchestrator

- `parse_task` checks that `type` is a string and no more. An empty or unknown `type` is
  refused by the domain that reads the section; the check in `config` was a new `SA_THROW_IF`,
  which the contract checker refuses.
- `mnist.cpp` stays on its own IDX parsing: moving it onto `idx_file` changes its messages, and
  its tests cannot be run yet.

## Ruled by the owner on 2026-10-11

- A rotary model of two or more blocks generates through the ring by default.
- `applications/programs/source/domains_none.cpp` is deleted.
- The training file gains a `seed` key, and the run state gains the step's run count and
  the epoch index, in version 2 of the weight file, which no release has carried.
- B9 and B10 are relayed to the SushiRuntime session.
- No build until he orders one; commits follow the build, in logical parts.
- The decision model, phase 6 of `LANGUAGE.md`, is the first work after the build passes.
- The Unreleased section of the changelog takes the capital spelling of acronyms; work
  folders and the archive keep what they said.

## What the owner's build must confirm

- `test_learning_curve`, `test_determinism` and `test_compile_once`: the synthetic data is now
  copied on the host where it was a `gemm` with the stored one-hot matrix.
- The exact texts the JSON writer tests and the reader tests expect.
- `sushiai_vision_tests` finding `jpeg62.dll` and `libpng16.dll` on the path the CLI sets.
- `longjmp` through libjpeg's frames on Windows, and the four `Unit_Codecs*` suites.

## Dispatches

The model of the first two is not recorded; they ran before this folder was opened.

- dispatch: `write_json` and `write_model` — model and effort not recorded, passed on syntax check
- dispatch: the `task` section and `AnchorTask` — model and effort not recorded, passed on syntax check
- dispatch: `Batch` with roles — opus low, 104 tool uses, passed on syntax check, rung 0
- dispatch: `vision/codecs` — opus low, 60 tool uses, passed on syntax check, rung 0
- dispatch: format readers — opus low, 64 tool uses, passed on syntax check, rung 0
- dispatch: language phase 1 — opus low, 114 tool uses, passed on syntax check of 350 units, rung 0
- dispatch: language phase 2 — opus low, 132 tool uses, passed on syntax check of 350 units, rung 0
- dispatch: the fallible dataset seam — opus low, 62 tool uses, passed on syntax check of 379 units, rung 0
- dispatch: the unfused log-softmax rule and `K_DROPOUT` — opus low, 83 tool uses, rule passed on syntax check of 349 units, dropout stopped as instructed, rung 0
- dispatch: language phase 3, tokenizer and windowed reader — opus low, passed on syntax check of 38 files, rung 0
- dispatch: the training file up to `AnchorTrainingStep` — opus low, 95 tool uses, 364 of 365 units passed; the one failure needed a build edge, added since, rung 0
- dispatch: the `train` program over the anchor path — opus low, 117 tool uses, passed on syntax check of 356 units, rung 0
- dispatch: the input element type on `Dataset` — opus low, 93 tool uses, passed on syntax check of 404 units apart from two of the parallel task's files, rung 0
- dispatch: `sushiai import darknet` and inference from the task section — opus low, 129 tool uses, passed on syntax check of 347 units, rung 0
- dispatch: language phase 4, the objective and the optimizer pieces — opus low, 224 tool uses, passed on syntax check of 347 units, rung 0
- dispatch: the objective-owned feed and the objectives of datasets phase 5 — opus low, 158 tool uses, passed on syntax check of 403 units, rung 0
- dispatch: readers `coco_json`, `voc_xml`, `mask_png`, `csv` and four collators — opus low, passed on syntax check of 58 units, rung 0
- dispatch: language from files, the tokenizer command and generation — opus low, 141 tool uses, passed on syntax check of 592 units, rung 0
- dispatch: `vision/segment`, the per-pixel objective and the head — opus low, passed on syntax check of 70 files, rung 0
- dispatch: the plan of capability changes below the layers — opus medium, by the owner's word, 72 tool uses, 32 items written to `GRAPH_CAPABILITIES.md` and relayed to the SushiRuntime session, rung 0
- dispatch: the key and value cache and its three seams — opus low, 143 tool uses, passed on syntax check of 615 units, rung 0
- dispatch: the raw cache, epoch order, augmentation, the loader on threads, masks from polygons — opus low, 156 tool uses, passed on syntax check of 27 files, rung 0
- dispatch: the scatter-add rule, the composite objective, the deprecations removed, one section reader — opus low, 7 of 9 items done, passed on syntax check of 609 units, rung 0
- dispatch: language phase 7, mixture of experts, and the clip identity — opus low, stopped once for the load of its syntax pass and resumed, passed on its own files, rung 0
- dispatch: version 2 of the weight file, the streaming JSON reader, the last leftovers — opus low, stopped once for the load of its syntax pass and resumed, passed on 77 of its own files, rung 0
- dispatch: rotary positions and a cache that survives a slide — opus low, stopped by the owner for the load of the syntax passes, not resumable
- dispatch: the same, finished from the files left on disk — opus low, 102 tool uses, passed on 31 files one at a time, 15 of them only with three macros that bridge a rename under way in SushiRuntime, rung 0
- dispatch: dropout with a mask drawn at each step — opus low, 148 tool uses, stopped once for the load of its syntax pass and resumed, passed on 14 of its own units, rung 0
- dispatch: the missing sections of the integration test README — opus low, 55 tool uses in two passes, 43 sections and 64 older cases written, checked by script, rung 0
- dispatch: the missing rows and sections of the unit test README, five headers in two module READMEs — opus low, 68 tool uses, 41 rows and 41 sections, 119 sections left, rung 0
- dispatch: unit test sections of core and layers — opus low, 80 tool uses, 45 sections, rung 0
- dispatch: unit test sections of language — opus low, 40 tool uses, 32 sections, rung 0
- dispatch: unit test sections of vision — opus low, 87 tool uses and a closing pass, 44 sections and 230 cases of older sections written, 24 stale headings removed, rung 0
- dispatch: the seed key, the run count and the epoch position across a resume — opus low, passed on 19 files one at a time, rung 0
- dispatch: truth from roles, the anchor objective section, the step schedule section — opus low, 60 tool uses, passed on syntax check, rung 0
- dispatch: collators and `ReaderDataset` — opus low, 61 tool uses, passed on syntax check, rung 0
