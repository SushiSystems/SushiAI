# Language

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-81 | P2 | open | State the frame budget and training data of phase 6 | the language design states the frame budget and the training data of phase 6, and names the first task after the build passes |  | `docs/design/LANGUAGE/README.md` | 2026-10-25 |
| RW-82 | P2 | open | Show per-term losses in a step's view | perplexity inside a sum is that of the fit alone |  |  | 2026-10-19 |
| RW-83 | P2 | open | Add a port that is one value per step | a balance needs no axis of 1 |  |  | 2026-10-19 |
| RW-84 | P2 | open | Allow usage metrics without a balance term | a sum accepts a weight of 0 |  |  | 2026-10-19 |
| RW-85 | P2 | open | Finish rotary positions | a position past 2^28 is refused, rotary runs under mixed precision, `tiny_decoder` uses rotary and the window form shares its tables |  |  | 2026-10-19 |
| RW-86 | P2 | open | Write a row in place in the graph | an in-place row write replaces the scatter-add into zeros |  |  | 2026-10-19 |
| RW-87 | P2 | open | Append to a token file | `sushiai tokenize encode` holds one piece at a time |  |  | 2026-10-19 |
| RW-88 | P2 | open | Give every node one factory signature | `language_head`, the one linked row, takes the signature of the others and `LinkedNodeFactory` goes |  |  | 2026-10-19 |
| RW-89 | P2 | open | Call `match_input` from the anchor loop | the loop matches inputs as well as roles |  |  | 2026-10-19 |
| RW-90 | P2 | open | Redraw a token dataset's windows per epoch | `TokenDataset` implements `Data::EpochAware` and redraws its window starts at each epoch |  |  | 2026-10-25 |
| RW-91 | P2 | open | Give `at_boundary` and the little-endian readers one home | the three `at_boundary` copies (`vision/datasets`, `language/text`, `language/datasets`) and the two `little_endian.hpp` copies are one each |  |  | 2026-10-19 |
| RW-92 | P2 | unmeasured | Measure `train_bpe` on a corpus of tens of megabytes | `sushiai tokenize train` over a corpus of at least 20 MB is timed, and its time and peak memory are in `modules/language/text/README.md` |  |  | 2026-10-19 |
| RW-93 | P2 | open | Settle the masked attention score | the masked score is a value that is minus infinity in HALF, or the language design records -1e9 as a decision |  | `docs/design/LANGUAGE/README.md` | 2026-10-25 |
| RW-94 | P2 | open | Fill the dropout mask on the device | the mask is drawn on the device (B1 and G1 of the graph capabilities plan) and a test covers dropout in HALF and FLOAT64 on an engine |  |  | 2026-10-19 |
| RW-95 | P2 | open | Make the tests' bodies match their names | `AMissingFileFails` checks a `FileNotFound` code and the path, `ARefusedDiceFillChangesNothing` checks all five leaves, and… |  |  | 2026-10-19 |
| RW-96 | P2 | open | Check the older test pages sentence by sentence | the test pages under `tests/` match their sources and none of their lines passes 100 columns (227 do) |  |  | 2026-10-19 |
| RW-97 | P2 | open | Describe the shared check helpers on the importer test page | `tests/vision/importers/README.md` links the test files that exist and describes the helpers of the three `darknet_*_cases` modules from their bodies |  |  | 2026-10-25 |
| RW-98 | P2 | open | Check a token id against the embedding before a step | an input id outside `Embedding::num_embeddings` is refused with a named error before the step |  |  | 2026-10-25 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
