# tests/vision/datasets/test_csv.cpp


Suites: `Unit_DatasetsCsv`.

Tests `read_csv` in `modules/vision/datasets/include/SushiAI/vision/datasets/csv.hpp` over
files the cases write under the system's temporary directory. Unless a case names another
file, the reader takes `train.csv` with the default columns.

## TheLastColumnIsTheTargetByDefault

A file with the header `a,b,y` and two records gives one file, two samples and a class count
of 0. The table has two features and one target, the columns a, b, y and the values 1, 2, 3,
-0.5, 4, 10; `features(1)[1]` is 4 and `targets(1)[0]` is 10.

## NamedColumnsAreTakenInTheOrderNamed

Over `data/set.csv` with the header `id,x,y,z`, the features z, x and the targets y, z fail
with `InvalidDescription`, "set.csv", "read_csv" and
"the column 'z' is named as a feature and as a target". With the target y alone the columns
are z, x, y and the values 3, 1, 2, 6, 4, 5.

## QuotedFieldsAndCarriageReturnsAreRead

The file opens with a byte order mark, ends its lines with CRLF and has an empty line. Its
header holds a quoted name with a doubled quote and a comma, a quoted name with a line break,
and ` y ` between blanks. With the first column as the feature, the columns are
`the "x", one` and `y`, and the values are 1.5, 2, 3, 4.

## ANonNumericCellNamesItsLineAndColumn

The cell `five` fails with `InvalidFile`, "train.csv", "line 3", "column 2 ('b')", "read_csv"
and "the cell 'five'". An empty target cell fails with "line 2", "column 3 ('y')" and
"the cell ''". After a record whose quoted field spans lines 2 and 3, the cell `inf` fails
with "line 4", "column 3 ('y')" and "the cell 'inf'".

## ARecordOfAnotherFieldCountIsRefused

A record of two fields on line 4, after an empty line, fails with `InvalidFile`, "train.csv",
"line 4", "read_csv" and "the record has 2 fields; the header has 3".

## MalformedQuotesAreRefused

Each of these fails with `InvalidFile` and "line 2": a quote left open ("column 2",
"the quoted field is not closed"), a letter after a closing quote ("column 1",
"text follows the closing quote") and a quote at the end of an unquoted field ("column 2",
"a quote stands inside a field").

## AColumnTheHeaderLacksIsRefused

Over the header `a,b,y`, the target z fails with `InvalidDescription`, "train.csv",
"no target column 'z'" and "'a', 'b', 'y'". The feature c fails with `InvalidDescription` and
"no feature column 'c'".

## AHeaderMustNameItsColumnsOnceAndLeaveAFeature

The header `a, b ,a` on line 2 fails with `InvalidFile`, "line 2", "column 3" and
"the name 'a' is also the name of column 1". A file of the single column y fails with
`InvalidDescription` and "no column is left as a feature".

## AFileWithoutRecordsIsRefused

With the split "val" and no file, the read fails with `FileNotFound`, "val.csv" and
"read_csv". A `val.csv` of two empty lines fails with `InvalidFile` and
"holds no header row", and one with a header alone with "holds no record below its header".
