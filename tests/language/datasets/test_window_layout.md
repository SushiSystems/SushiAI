# tests/language/datasets/test_window_layout.cpp


Suites: `Small_LanguageDatasetsWindowLayout`, `Small_LanguageDatasetsWindowLayoutRefusal`.

Tests `count_windows` and `first_token_of` over a `WindowLayout` in
`modules/language/datasets/include/SushiAI/language/datasets/window_layout.hpp`. A layout is
written here as {window, stride}. No case needs a device or a file.

## WindowsAreCountedAtTheStride

`count_windows(10, {4, 3})` is 3.

## WindowsBeginOneStrideApart

`first_token_of` is 0, 3 and 6 for the windows 0, 1 and 2 of {4, 3}.

## OnlyWholeWindowsAreCounted

`count_windows` is 2 for 8 ids over {4, 4}, 2 for 11 ids over {4, 4}, 1 for 4 ids over {4, 1}
and 0 for 3 ids over {4, 1}.

## ALayoutOutOfBoundsIsRefused

One promise over three rows: `count_windows` fails with `InvalidDescription` for a window of 0
("count_windows", "window of 0 tokens"), a stride of 0 ("stride of 0 tokens") and a window of
`K_MAX_WINDOW_TOKENS + 1` ("window of").
