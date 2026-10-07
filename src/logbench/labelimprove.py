"""Cross-system log labelling with per-row predictions.

Supported directions:
  b_to_a: Biblio-US17 -> AIT with nearby client-session context.
  srbh_to_b: SR-BH 2020 -> Biblio-US17 with compact nested encoding.
  srbh_to_a: SR-BH 2020 -> AIT with nearby client-session context.

Target labels are loaded only after model inference.
"""
import argparse
import hashlib
import json
import re
import threading
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

from . import crossllm
from .crossfeatures import Representation
from .crossexternal import load_srbh

SESSION_GUARD = (
    "\nFor session analysis, distinguish ordinary navigation and asset loading from a burst of many "
    "unrelated dictionary paths. Repeated requests for unrelated short path words in the same client "
    "session are evidence of enumeration even if each word is innocuous alone. Evaluate the candidate's "
    "role in that pattern. Diversity alone is not sufficient: legitimate application routes and static "
    "asset loading must remain normal. Session counts are unlabelled observations, not ground truth."
)
PROMPT_TEXT_CHARS = 400
CHARS_PER_TOKEN = 2.5
DIRECTIONS = {
    "b_to_a": {"source": "system_b", "target": "system_a", "method": "nearby_session"},
    "srbh_to_b": {"source": "srbh", "target": "system_b", "method": "compact_encoding"},
    "srbh_to_a": {"source": "srbh", "target": "system_a", "method": "nearby_session"},
}


def system_prompt_for(direction):
    additions = {"b_to_a": SESSION_GUARD, "srbh_to_b": "", "srbh_to_a": SESSION_GUARD}
    return crossllm.SYSTEM + additions[direction]


def text_id(text):
    return "txt-" + hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:16]


def clip(text, limit=PROMPT_TEXT_CHARS):
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " ...[cut]"


def load_prepared(data, system, labelled=False):
    """Load consolidated requests, optionally joining separately stored labels."""
    data = Path(data)
    frame = pd.read_csv(data / f"{system}.csv", dtype=str, keep_default_na=False)
    required = {"sample_id", "system", "timestamp", "group_id", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{system}.csv is missing columns: {sorted(missing)}")
    if not frame.sample_id.is_unique:
        raise ValueError(f"{system}.csv contains duplicate sample_id values")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="raise")
    frame = frame.sort_values(["timestamp", "sample_id"], kind="stable").reset_index(drop=True)
    if labelled:
        labels = pd.read_csv(data / f"{system}_labels.csv", dtype=str, keep_default_na=False)
        if not labels.sample_id.is_unique or not labels.label.isin(["normal", "anomaly"]).all():
            raise ValueError(f"{system}_labels.csv must contain one binary label per sample_id")
        frame["label"] = frame.sample_id.map(labels.set_index("sample_id").label)
        if frame.label.isna().any():
            raise ValueError(f"{system}_labels.csv does not cover every request")
    return frame


def encoding_view(text):
    """Compact repeated %25 runs while preserving depth and the request tail."""
    text = str(text)
    runs = list(re.finditer(r"%(?:25){2,}", text, re.I))
    compact = re.sub(r"%(?:25){2,}",
                     lambda match: "%%[25 repeated %d times]" % ((len(match[0]) - 1) // 2),
                     text, flags=re.I)
    if len(compact) > PROMPT_TEXT_CHARS:
        compact = compact[:250] + " ...[middle omitted]... " + compact[-120:]
    depths = [(len(match[0]) - 1) // 2 for match in runs]
    facts = (
        f"Original request length: {len(text)} characters. "
        f"Nested percent-encoding runs: {len(runs)}. "
        f"Maximum consecutive 25 pairs following percent: {max(depths, default=0)}. "
        "The bracketed repetition notation is a lossless abbreviation of that run, not literal request content."
    )
    return compact, facts


def nearby_session_payload(row, siblings):
    """Eight closest same-client requests plus unlabelled session facts."""
    positions = list(siblings.index)
    index = positions.index(row.name)
    nearby = sorted((i for i in range(len(siblings)) if i != index),
                    key=lambda i: (abs(i - index), i))[:8]
    paths = [str(text).split(" ", 2)[1].split("?")[0] for text in siblings.text]
    short = sum(bool(re.fullmatch(r"/[A-Za-z0-9_-]{1,30}/?", path)) for path in paths)
    duration = (siblings.timestamp.max() - siblings.timestamp.min()).total_seconds()
    return {
        "sample_id": text_id(row.text), "system": row.system, "candidate": clip(row.text),
        "context": [clip(siblings.iloc[i].text, 220) for i in sorted(nearby)],
        "request_facts": (
            f"Same-client session chunk: {len(siblings)} requests, {len(set(paths))} distinct paths, "
            f"{short} single-segment alphanumeric paths, duration {duration:g} seconds. "
            "Context lists the closest requests in this session chunk, in time order. Counts include the candidate."
        ),
    }


def plain_block_payload(row, siblings):
    context = [clip(text) for sample_id, text in zip(siblings.sample_id, siblings.text)
               if sample_id != row.sample_id][:4]
    return {"sample_id": text_id(row.text), "system": row.system,
            "candidate": clip(row.text), "context": context}


def encoding_block_payload(row, siblings):
    payload = plain_block_payload(row, siblings)
    candidate, facts = encoding_view(row.text)
    payload.update(candidate=candidate, request_facts=facts)
    return payload


def build_payloads(target, direction, limit=0, seed=42):
    eligible = target.in_sample.astype(bool) if "in_sample" in target else pd.Series(True, index=target.index)
    first = target.loc[eligible & ~target.text.where(eligible).duplicated()].copy()
    if limit and limit < len(first):
        first = first.sample(limit, random_state=seed).sort_index()
    groups = {name: part for name, part in target.groupby("group_id", sort=False)}
    payloads = []
    for row in first.itertuples():
        source_row = target.loc[row.Index]
        siblings = groups[row.group_id]
        if direction in ("b_to_a", "srbh_to_a"):
            payload = nearby_session_payload(source_row, siblings)
        elif direction == "srbh_to_b":
            payload = encoding_block_payload(source_row, siblings)
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        payloads.append(payload)
    return payloads


class MultiHostProvider:
    """Round-robin requests over equivalent Ollama servers."""
    name = "ollama"

    def __init__(self, model, hosts, seed=42, **kwargs):
        self.providers = [crossllm.OllamaProvider(model, host=host, seed=seed, **kwargs) for host in hosts]
        self._next = 0
        self._lock = threading.Lock()

    def describe(self):
        return self.providers[0].describe()

    def chat(self, messages, options=None):
        with self._lock:
            start = self._next
            self._next = (self._next + 1) % len(self.providers)
        last = None
        for shift in range(len(self.providers)):
            try:
                return self.providers[(start + shift) % len(self.providers)].chat(messages, options)
            except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
                last = error
        raise RuntimeError(f"No Ollama server answered: {last}")


def select_source_shots(source, shots_per_class=6, seed=42):
    distinct = source.drop_duplicates(["text", "label"]).reset_index(drop=True)
    representation = Representation("hashing", 4096, seed).fit(distinct.text.tolist())
    shots = crossllm.select_shots(distinct, representation.transform(distinct.text), shots_per_class, seed)
    return [{**shot, "text": clip(shot["text"])} for shot in shots]


def metric_counts(truth, prediction):
    truth = np.asarray(truth) == "anomaly"
    prediction = np.asarray(prediction) == "anomaly"
    tp, fp = int((truth & prediction).sum()), int((~truth & prediction).sum())
    fn, tn = int((truth & ~prediction).sum()), int((~truth & ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def write_outputs(target, data, system, records, output, target_labels=None):
    answers = {}
    attempted = {record["sample_id"] for record in records}
    for record in records:
        if record.get("status") == "ok":
            answer = record["answer"]
            answers[record["sample_id"]] = {
                "predicted_label": answer["decision"], "confidence": float(answer["confidence"]),
                "reason_code": answer["reason_code"], "short_reason": answer["short_reason"],
            }
    rows = target.copy()
    rows["text_key"] = rows.text.map(text_id)
    for column in ["predicted_label", "confidence", "reason_code", "short_reason"]:
        rows[column] = rows.text_key.map({key: value[column] for key, value in answers.items()})

    # Target truth is deliberately loaded only after annotation has finished.
    labels = (pd.read_csv(Path(data) / f"{system}_labels.csv", dtype=str, keep_default_na=False)
              if target_labels is None else target_labels)
    if not labels.sample_id.is_unique or not labels.label.isin(["normal", "anomaly"]).all():
        raise ValueError("Target labels must contain one binary label per sample_id")
    rows["true_label"] = rows.sample_id.map(labels.set_index("sample_id").label)
    if rows.true_label.isna().any():
        raise ValueError("Target labels do not cover every evaluated request")
    rows["correct"] = rows.predicted_label.eq(rows.true_label).where(rows.predicted_label.notna())
    rows.to_csv(output / "row_predictions.csv", index=False)

    labelled = rows[rows.predicted_label.notna()].copy()
    per_text = labelled.groupby("text_key", sort=True).agg(
        text=("text", "first"), rows=("sample_id", "size"),
        anomaly_rows=("true_label", lambda values: int(values.eq("anomaly").sum())),
        predicted_label=("predicted_label", "first"), confidence=("confidence", "first"),
        reason_code=("reason_code", "first"), short_reason=("short_reason", "first"),
    ).reset_index()
    per_text.to_csv(output / "text_predictions.csv", index=False)
    target_texts = int(rows.text_key.nunique())
    return {
        "coverage": {"target_rows": len(rows), "labelled_rows": len(labelled),
                     "target_texts": target_texts, "attempted_texts": len(attempted),
                     "labelled_texts": len(per_text),
                     "failed_texts": int(len(attempted) - len(per_text)),
                     "unselected_texts": int(target_texts - len(attempted))},
        "row_level": metric_counts(labelled.true_label, labelled.predicted_label),
        "text_level": metric_counts(np.where(per_text.anomaly_rows.gt(0), "anomaly", "normal"),
                                    per_text.predicted_label),
    }


def run(args):
    if args.srbh_rows < 0:
        raise ValueError("--srbh-rows must be non-negative")
    config = DIRECTIONS[args.direction]
    output = Path(args.output) / args.direction
    output.mkdir(parents=True, exist_ok=True)
    source = (load_srbh(args.srbh, sample=args.srbh_rows, seed=args.seed) if config["source"] == "srbh"
              else load_prepared(args.data, config["source"], labelled=True))
    target = load_prepared(args.data, config["target"], labelled=False)
    if config["source"] == "srbh":
        source = source.loc[source.in_sample].copy()
    shots = select_source_shots(source, args.shots_per_class, args.seed)
    profile = [clip(text) for text in target.text.value_counts().head(args.profile_size).index]
    payloads = build_payloads(target, args.direction, args.limit, args.seed)
    system_prompt = system_prompt_for(args.direction)

    if args.provider == "stub":
        provider, workers = crossllm.StubProvider(seed=args.seed), args.workers
    else:
        hosts = [host.strip() for host in args.hosts.split(",") if host.strip()]
        if not hosts:
            raise ValueError("--hosts must contain at least one Ollama URL")
        provider = MultiHostProvider(args.model, hosts, args.seed, think=False,
            options={"temperature": 0, "num_ctx": args.num_ctx, "num_predict": args.num_predict},
            timeout=args.timeout)
        workers = args.workers * len(hosts)
    annotator = crossllm.Annotator(provider, shots, output / "responses.jsonl",
        args.max_attempts, workers, profile, strict_evidence=False, system_prompt=system_prompt)
    sizes = [sum(len(message["content"]) for message in
                 crossllm.render(shots, payload, 0, profile, system_prompt)) for payload in payloads]
    budget = (args.num_ctx - args.num_predict) * CHARS_PER_TOKEN
    if sizes and max(sizes) > budget:
        raise ValueError(f"Prompt may exceed context budget: {max(sizes):.0f} > {budget:.0f} characters")
    selection = {"seed": args.seed, "limit_texts": args.limit,
                 "shots_per_class": args.shots_per_class, "profile_size": args.profile_size,
                 "srbh_source_rows": args.srbh_rows if config["source"] == "srbh" else None,
                 "source_rows": len(source),
                 "evaluated_target_rows": int(target.in_sample.sum()) if "in_sample" in target else len(target)}
    settings = {"direction": args.direction, "source": config["source"], "target": config["target"],
                "method": config["method"],
                "settings_hash": annotator.settings_hash,
                "settings": annotator.settings, "selection": selection, "payloads": len(payloads),
                "target_labels_used_before_inference": False}
    (output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    records = annotator.annotate(payloads, (0,), progress=args.progress)
    evaluated = target[target.in_sample].copy() if "in_sample" in target else target
    summary = write_outputs(evaluated, args.data, config["target"], records, output)
    summary.update(run=annotator.usage(), direction=args.direction, method=config["method"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=sorted(DIRECTIONS), required=True)
    parser.add_argument("--data", type=Path, default=Path("data/improved"))
    parser.add_argument("--srbh", type=Path,
                        default=Path("data/improved/SR-BH 2020.csv"))
    parser.add_argument("--srbh-rows", type=int, default=50000,
                        help="random SR-BH source rows; 0 uses all rows")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--provider", choices=["ollama", "stub"], default="ollama")
    parser.add_argument("--hosts", default="http://127.0.0.1:11500")
    parser.add_argument("--model", default="deepseek-r1:32b")
    parser.add_argument("--workers", type=int, default=6, help="workers per Ollama host")
    parser.add_argument("--shots-per-class", type=int, default=6)
    parser.add_argument("--profile-size", type=int, default=20)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="random unique texts; 0 labels all texts")
    parser.add_argument("--progress", type=int, default=100)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
