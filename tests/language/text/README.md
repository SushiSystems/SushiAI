# tests/language/text

The tests of the module `language/text`, one file per header of `modules/language/text/include/`. The suite prefix of a case gives its ctest label. The cases of each file are described in the sidecar beside it.

| File | Sidecar | Suites |
| --- | --- | --- |
| `test_bpe_trainer.cpp` | `test_bpe_trainer.md` | `Small_TextBpeTrainer`, `Small_TextBpeTrainerSizeRefusal` |
| `test_codec.cpp` | `test_codec.md` | `Small_TextCodec`, `Small_TextCodecEdgeText`, `Small_TextCodecRefusal` |
| `test_stream_encoder.cpp` | `test_stream_encoder.md` | `Small_TextStreamEncoder`, `Small_TextStreamEncoderPieces` |
| `test_vocabulary.cpp` | `test_vocabulary.md` | `Small_TextVocabulary`, `Small_TextVocabularyMergeRefusal`, `Small_TextVocabularyNameRefusal`, `Medium_TextVocabularyDeath` |
| `test_vocabulary_file.cpp` | `test_vocabulary_file.md` | `Small_TextVocabularyFile*`, `Medium_TextVocabularyFile`, `Medium_TextVocabularyFileRefusal` |
| `compile_token.cpp` | `compile_token.md` | none; a compile-only unit for `token.hpp` |

`text_test_support.hpp` holds what the files share: the scratch directory for the two cases that
touch a file, the seeded byte generator, the merge comparisons, the builder of doubling merges and
the failure check that takes a list of texts.

Only the cases that read or write a file or stop the process are `Medium_`. The layout of the
vocabulary file is checked small on the bytes `serialise_vocabulary_file` returns.
