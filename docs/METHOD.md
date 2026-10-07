# Method

The pipeline assigns `normal` or `anomaly` labels to HTTP requests using a local large language model and labelled examples from another system. Its three directions are `b_to_a`, `srbh_to_b`, and `srbh_to_a`. Prompts contain the HTTP method, URI, protocol, and unlabelled request context. Response status, response size, attack categories, and target labels are excluded. Source labels select examples; AIT and Biblio target labels are loaded only after inference for evaluation.

## Source examples and sampling

Source requests are deduplicated by text and label. A fixed 4,096-dimensional character and word hashing representation supports greedy k-center selection of up to six anomalous and six normal examples. Up to three normal examples closest to the anomaly centroid and three additional request templates are included. Duplicate example texts are removed, giving at most 18 examples. The target's 20 most frequent request texts supply an unlabelled traffic profile.

Biblio-US17 uses its complete labelled source dataset. SR-BH is a labelled source sampled without replacement to 50,000 valid rows with seed 42. Examples are selected only from those rows. `--srbh-rows` changes the sample size; zero uses all valid rows. Invalid timestamps are excluded before ordering and sampling. SR-BH request texts are truncated to 400 characters by the reader. AIT and Biblio targets retain their full datasets.

## Direction-specific context

For `b_to_a` and `srbh_to_a`, the AIT candidate receives up to eight neighbours closest in row position within its prepared same-client session group, listed in time order. Group facts include request count, distinct paths, short single-segment alphanumeric paths, and duration. Counts include the candidate. The session prompt distinguishes ordinary navigation and asset loading from enumeration of unrelated paths.

For `srbh_to_b`, up to four other requests from the Biblio target group provide context. Repeated `25` pairs following a percent sign are replaced by explicit repetition counts. Facts report original request length, run count, and maximum consecutive pair count. If the compacted request exceeds 400 characters, its first 250 and final 120 characters are retained with a middle-omission marker. The replaced runs are preserved by the notation; omitted request content is not.

## Inference and outputs

Each eligible distinct request text uses its first eligible occurrence to construct a payload. Its prediction propagates to evaluated rows sharing that text. `--limit` optionally samples distinct texts; zero selects all eligible texts. Candidates generally use a 400-character limit and AIT neighbours a 220-character limit. Context and profile can include unselected target records.

Defaults are Ollama with `deepseek-r1:32b`, thinking disabled, temperature 0, seed 42, a 4,096-token context, and at most 512 output tokens. Responses contain a decision, confidence, reason code, evidence quotes, and a short reason. Confidence is self-reported and uncalibrated. Settings and payload hashes identify resumable response caches. Outputs record configuration, responses, row/text predictions, and coverage.

Precision, recall, and F1 treat anomaly as positive. Metrics cover successfully predicted evaluated rows or distinct texts. A text's reference label is anomaly if any corresponding evaluated row is anomalous. Failed or unselected texts are reported through coverage and excluded from metrics. The offline stub checks the pipeline and does not measure model performance.
