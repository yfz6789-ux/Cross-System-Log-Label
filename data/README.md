# Input data

Datasets are not included in this repository. Place prepared UTF-8 CSV files in `data/improved/`. Use `--data` to select another prepared-data directory.

| File | Required columns |
|---|---|
| `system_a.csv` | `sample_id,system,timestamp,group_id,text` |
| `system_a_labels.csv` | `sample_id,label` |
| `system_b.csv` | `sample_id,system,timestamp,group_id,text` |
| `system_b_labels.csv` | `sample_id,label` |

Request IDs must be unique. `text` contains the HTTP method, URI, and protocol separated by spaces. Timestamps must be parseable by pandas. AIT groups identify client-session chunks; Biblio groups identify consecutive 50-row blocks. Keep all context rows. Optional `method`, `uri`, and `protocol` columns may be included.

Labels are `normal` or `anomaly`, with one label for every matching request ID. Source data must contain both classes. Target labels are loaded after inference.

`SR-BH 2020.csv` uses the original SR-BH schema: `timestamp,request_http_method,request_http_request,request_http_protocol`, exactly one column containing `Normal`, and one or more attack columns containing ` - `. Label flags are integers: each row must be normal or have at least one attack flag, never both or neither. Timestamps use `DD/Mon/YYYY:HH:MM:SS +ZZZZ` with English month names. Invalid timestamp rows are dropped.

SR-BH is used only as a labelled source, with 50,000 valid rows sampled using seed 42 by default. Its default path is `data/improved/SR-BH 2020.csv`; override it with `--srbh` or the batch runner's `SRBH` environment variable.
