# modules/vision/datasets/include/SushiAI/vision/datasets/value_table.hpp


A `ValueTable` holds `feature_count + target_count` floats per row, the features first, one row
after another in `values`, and the column names in the same order in `columns`. `features(row)`
and `targets(row)` return views of one row; they check no index. A list of images has a table of
no column and no row.
