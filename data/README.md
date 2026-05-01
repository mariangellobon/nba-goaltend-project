# Data layout

Place **labeled segmented** folders here (each directory name must contain `Segmented`), plus:

- `close_calls_labels.csv` — table with `filename` and `ground_truth` (`legal`, `goaltend`, or `cant_tell`)
- `Close Calls/` — CSV files referenced by that table

Override the data location with the environment variable `GOALTEND_DATA_DIR` (absolute path to this folder).
