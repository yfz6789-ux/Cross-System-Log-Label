"""SR-BH source-data reader for cross-system log labelling.

Every reader returns one frame in time order with the columns the labelling code needs:
sample_id, system, timestamp, group_id, text (method + URI + protocol, the same request line the
other systems use; response status, bytes and bodies are never part of it), an optional binary label
for source-example selection, and in_sample (True for rows in the source sample).
This dataset has no sessions, so groups are blocks of `block` consecutive requests, as for Biblio-US17.
"""
import numpy as np
import pandas as pd

MONTHS = {m: '%02d' % i for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}
MAX_TEXT = 400        # characters; longer request lines are cut so that a prompt always fits the context


def blocks(frame, block):
    frame['group_id'] = [f'{frame.system.iloc[0]}-{i//block:06d}' for i in range(len(frame))]
    return frame


def choose(frame, sample, seed, mode='random', start=0):
    """'random': `sample` rows drawn without replacement (an unbiased estimate of the whole dataset;
    `start` is unused). 'prefix': the time-ordered rows [start, start+sample) (or [start, end) when
    sample is 0, i.e. "the rest from here on") -- a cost/coverage window, not a random sample; the
    reported metrics describe only that window, not the dataset as a whole."""
    if mode == 'prefix':
        if not 0 <= start <= len(frame):
            raise ValueError(f'start ({start}) is outside the dataset ({len(frame)} rows)')
        end = start+sample if sample else len(frame)
        keep = np.zeros(len(frame), bool)
        keep[start:min(end, len(frame))] = True
        frame['in_sample'] = keep
        return frame
    if mode != 'random':
        raise ValueError("mode must be 'random' or 'prefix'")
    frame['in_sample'] = True
    if sample and sample < len(frame):
        keep = np.zeros(len(frame), bool)
        keep[np.random.default_rng(seed).choice(len(frame), size=sample, replace=False)] = True
        frame['in_sample'] = keep
    return frame


def finish(frame, sample, seed, block, mode='random', start=0):
    frame = frame.sort_values(['timestamp', 'sample_id'], kind='stable').reset_index(drop=True)
    if frame.timestamp.isna().any():
        raise ValueError('unparsable timestamps')
    frame['text'] = frame.text.str.slice(0, MAX_TEXT)
    return choose(blocks(frame, block), sample, seed, mode, start)


def apache_time(values):
    """Convert Apache timestamps to UTC without relying on the host locale."""
    parts = values.str.extract(r'^(\d{2})/([A-Za-z]{3})/(\d{4}):(\d{2}:\d{2}:\d{2}) ([+-]\d{4})$')
    iso = parts[2]+'-'+parts[1].map(MONTHS)+'-'+parts[0]+'T'+parts[3]+parts[4]
    return pd.to_datetime(iso, utc=True, errors='coerce')


def load_srbh(path, sample=0, seed=42, block=50, mode='random', start=0, labelled=True):
    """Read SR-BH source requests with optional binary labels.

    ``labelled=False`` reads only the request columns. Both forms produce
    identical sample IDs, ordering, groups, and ``in_sample`` masks.
    """
    request_columns = ['timestamp', 'request_http_method', 'request_http_request',
                       'request_http_protocol']
    frame = pd.read_csv(path, dtype=str, keep_default_na=False,
                        usecols=None if labelled else request_columns)
    if labelled:
        attacks = [c for c in frame.columns if ' - ' in c and 'Normal' not in c]
        normal = [c for c in frame.columns if 'Normal' in c]
        if len(normal) != 1 or not attacks:
            raise ValueError('SR-BH file: expected one "... - Normal" column and attack columns')
    stamps = apache_time(frame.timestamp)
    if stamps.isna().any():        # rows without a timestamp (2 in the published file) cannot be ordered or grouped
        print(f'  SR-BH: dropping {int(stamps.isna().sum())} rows without a valid timestamp', flush=True)
        frame, stamps = frame[stamps.notna()].reset_index(drop=True), stamps[stamps.notna()].reset_index(drop=True)
    out_data = {
        'sample_id': ['srbh-%d' % i for i in range(len(frame))], 'system': 'sr_bh_2020',
        'timestamp': stamps,
        'text': frame.request_http_method+' '+frame.request_http_request+' '+frame.request_http_protocol}
    if labelled:
        flags = frame[attacks].astype(int).sum(axis=1) > 0
        if (flags == frame[normal[0]].astype(int).eq(1)).any():
            raise ValueError('SR-BH file: a row is both or neither normal and attack')
        out_data['label'] = np.where(flags, 'anomaly', 'normal')
    out = pd.DataFrame(out_data)
    return finish(out, sample, seed, block, mode, start)

