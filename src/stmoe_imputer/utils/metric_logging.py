"""Bounded epoch logs; full routing distributions belong to final evaluation only."""
from __future__ import annotations
import gzip
import json
import math
import re
from pathlib import Path

_PATH = re.compile(r'^(.*(?:pair_path|scale_execution_path|scale_path|path))_(.+)_fraction$')
CORE = ('loss', 'mae', 'rmse', 'lr')


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def path_groups(metrics):
    groups = {}
    for key, value in metrics.items():
        match = _PATH.match(key)
        if match and match[2] not in ('max', 'other', 'top_mass', 'logged_top_mass'):
            groups.setdefault(match[1], []).append((key, value))
    return groups


def path_summary(prefix, entries):
    values = [float(v) for _, v in entries if isinstance(v, (float, int)) and math.isfinite(v) and v > 0]
    return {prefix+'_unique_count': float(len(values)),
            prefix+'_entropy': -sum(v*math.log(v) for v in values),
            prefix+'_max_fraction': max(values, default=0.)}


def compact_metrics(metrics):
    """Preserve scalar learning/quality/cost diagnostics, omit conditional/detail tables."""
    if metrics is None:
        return None
    groups = path_groups(metrics)
    detailed = {k for entries in groups.values() for k, _ in entries}
    result = {k: v for k, v in metrics.items()
              if k not in detailed and not k.startswith('coe_condition_')}
    for prefix, entries in groups.items():
        if not prefix.startswith('coe_condition_'):
            result.update(path_summary(prefix, entries))
    return clean_json(result)


def bounded_diagnostics(metrics, top_k=10):
    if metrics is None:
        return None
    groups = path_groups(metrics)
    detailed = {k for entries in groups.values() for k, _ in entries}
    result = {k: v for k, v in metrics.items() if k not in detailed and k not in CORE}
    for prefix, entries in groups.items():
        valid = [(k, float(v)) for k, v in entries if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
        chosen = sorted(valid, key=lambda pair: (-pair[1], pair[0]))[:top_k]
        result.update(chosen)
        result.update(path_summary(prefix, entries))
        result[prefix+'_other_fraction'] = max(0., sum(v for _, v in valid)-sum(v for _, v in chosen))
        invalid = sum(not isinstance(v, (int, float)) or not math.isfinite(v) for _, v in entries)
        if invalid:
            result[prefix+'_nonfinite_entry_count'] = invalid
    return clean_json(result)


def compact_epoch(row):
    return {**row, 'logging_version': 2, 'train': compact_metrics(row.get('train')), 'val': compact_metrics(row.get('val'))}


def write_compressed(path, payload):
    """Atomic, deterministic gzip payload; no random state is consumed."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    try:
        with temporary.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as out:
            out.write(json.dumps(clean_json(payload), ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8'))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class MetricLogPolicy:
    def __init__(self, cfg=None, total_epochs=None):
        cfg = cfg or {}
        self.every = cfg.get('diagnostic_every', 5)
        self.top_k = cfg.get('path_top_k', 10)
        if type(self.every) is not int or self.every < 1 or type(self.top_k) is not int or self.top_k < 1:
            raise ValueError('logging diagnostic_every and path_top_k must be positive integers')
        self.total_epochs = total_epochs

    def describe(self):
        return {'version': 2, 'diagnostic_every': self.every, 'path_top_k': self.top_k,
                'full_paths': 'final_evaluation_only'}

    def due(self, epoch):
        return epoch == 1 or epoch % self.every == 0 or epoch == self.total_epochs

    def diagnostic_path(self, log_dir, epoch):
        return Path(log_dir).parent/'diagnostics'/f'epoch_{epoch:05d}.json.gz'

    def write_diagnostics(self, log_dir, row):
        if self.due(row['epoch']):
            write_compressed(self.diagnostic_path(log_dir, row['epoch']),
                {'epoch': row['epoch'], 'policy': self.describe(),
                 'detail_source': 'checkpoint_summary_only' if row.get('logging_version') == 2 else 'epoch_metrics',
                 **{s: bounded_diagnostics(row.get(s), self.top_k) for s in ('train', 'val')}})

    def append(self, log_dir, row):
        log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir/'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(compact_epoch(row), ensure_ascii=False, allow_nan=False)+'\n')
        for split in ('train', 'val'):
            metrics = row.get(split)
            if metrics is not None:
                payload = {'epoch': row['epoch'], 'metrics': {k: metrics[k] for k in CORE if k in metrics},
                           'perf': row.get('perf', {}), 'is_best': row.get('is_best', False)}
                with (log_dir/f'{split}.log').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(clean_json(payload), ensure_ascii=False, allow_nan=False)+'\n')

    def restore(self, log_dir, history):
        """Use checkpoint history as commit record; preserve already committed diagnostics."""
        log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
        last_epoch = max((r['epoch'] for r in history), default=0)
        for p in (log_dir.parent/'diagnostics').glob('epoch_*.json.gz'):
            if int(p.name.split('_')[1].split('.')[0]) > last_epoch:
                p.unlink()
        for name in ('metrics.jsonl', 'train.log', 'val.log'):
            (log_dir/name).write_text('', encoding='utf-8')
        result = []
        for row in history:
            path = self.diagnostic_path(log_dir, row['epoch'])
            if self.due(row['epoch']) and not path.exists():
                # Older checkpoints include full metrics; new ones retain summaries only.
                self.write_diagnostics(log_dir, row)
            compact = compact_epoch(row); self.append(log_dir, compact); result.append(compact)
        return result


def read_epoch_history(path, include_diagnostics=False):
    """Read either old full JSONL or new summaries, optionally overlay sampled diagnostics."""
    path = Path(path)
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if include_diagnostics and 'epoch' in row:
            diag = path.parent.parent/'diagnostics'/f"epoch_{row['epoch']:05d}.json.gz"
            if diag.exists():
                with gzip.open(diag, 'rt', encoding='utf-8') as f:
                    details = json.load(f)
                for split in ('train', 'val'):
                    if row.get(split) is not None:
                        row[split].update(details.get(split) or {})
        if include_diagnostics and row.get('stage') == 'test':
            final = path.parent.parent/'diagnostics'/'test.json.gz'
            if final.exists():
                with gzip.open(final, 'rt', encoding='utf-8') as f:
                    row['metrics'] = json.load(f)['metrics']
        yield row
