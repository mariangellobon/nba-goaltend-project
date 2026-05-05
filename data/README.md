# Data layout

Place **labeled segmented** folders here (each directory name must contain `Segmented`), plus:

- `close_calls_labels.csv` — table with `filename`; **`ground_truth`** (`legal` or **`block`** for non-goaltend, `goaltend`, `cant_tell`) and/or **`eyeballed_contact`**. Rows without a usable binary label are skipped for training and CV.
- `Close Calls/` — CSV files referenced by that table

Override the data location with the environment variable `GOALTEND_DATA_DIR` (absolute path to this folder).
