"""Read-only evidence collection for results@9f8fe9e1. Writes only develop evidence."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT.parent / 'llm-arch-tracer-results' / 'models'
FIELDS = ('module', 'from', 'to', 'expect', 'spread', 'axis', 'rank', 'shape',
          'field', 'shape_index', 'op_type', 'nth', 'layer_types')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sample(r, ordinal=None):
    keys = ('op_id', 'module_path', 'op_type', 'input_shape', 'output_shape',
            'weight_shape', 'weight_pos', 'params', 'layers', 'repeat')
    return {**{k: r[k] for k in keys if k in r}, 'ordinal': ordinal}


def csv_value(value):
    if value is None:
        return ''
    if isinstance(value, list):
        return '[' + ', '.join(csv_value(v) for v in value) + ']'
    return str(value)


def layer_numbers(expr):
    result = []
    for part in expr.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        elif part:
            result.append(int(part))
    return result


def collect():
    out = {}
    for directory in sorted(PUB.iterdir()):
        name = directory.name
        full = ROOT / 'models' / name / 'full'
        s = yaml.safe_load((directory / 'structure.yaml').read_text(encoding='utf8'))
        result = {'hashes': {}, 'scope': s['scope'], 'symbols': s['symbols'], 'phases': {},
                  'dead_confirmations': [], 'dead_overrides': []}
        provenance = json.loads((full / 'provenance.json').read_text(encoding='utf8'))
        result['revision'] = provenance['revision_resolved']
        cfg = provenance['config'].get('text_config', provenance['config'])
        raw = {}
        for phase in ('prefill', 'decode'):
            for ext in ('csv', 'jsonl'):
                path = directory / f'{phase}.{ext}'
                result['hashes'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = read_rows(directory / f'{phase}.jsonl')
            with (directory / f'{phase}.csv').open(encoding='utf-8-sig', newline='') as f:
                csvrows = list(csv.DictReader(f))
            groups = list(dict.fromkeys((r['block_type'], r['repeat'], r['layers']) for r in rows))
            diffs = []
            for row, csvrow in zip(rows, csvrows):
                for key, val in row.items():
                    if csv_value(val) != csvrow.get(key):
                        diffs.append([row['op_id'], key, val, csvrow.get(key)])
            expanded = [i for _, _, expr in groups for i in layer_numbers(expr)]
            schedule_mismatches = []
            for kind, count, expr in groups:
                nums = layer_numbers(expr)
                if not nums:
                    continue
                if count != len(nums):
                    schedule_mismatches.append(['repeat', kind, count, nums])
                for idx in nums:
                    if 'Kimi' in name:
                        mla = idx + 1 in cfg['linear_attn_config']['full_attn_layers']
                        expected = ('MLA' if mla else 'attn') + ('+FFN' if idx == 0 else '+MoE')
                    elif 'Llama' in name:
                        expected = 'attn+MoE' if idx in cfg['moe_layers'] else 'attn+FFN'
                    else:
                        expected = 'MLA+MoE' if 'Deep' in name else 'attn+MoE'
                    if kind != expected:
                        schedule_mismatches.append([idx, kind, expected])
                # All members must use the representative's attention schedule.
                schedule = cfg.get('layer_types')
                if schedule and len({schedule[i] for i in nums}) != 1:
                    schedule_mismatches.append(['mixed attention', nums])
            undefined = collections.Counter()
            undefined_rows = []
            negative_weight = []
            leading = collections.Counter()
            for row in rows:
                missing = set()
                for field in ('input_shape', 'output_shape', 'weight_shape'):
                    for token in re.findall(r'\b[A-Za-z_]\w*\b', str(row[field])):
                        if token not in set(s['symbols']) | {'B', 'T', 'None'}:
                            undefined[token] += 1
                            missing.add(token)
                if missing:
                    undefined_rows.append([row['op_id'], sorted(missing)])
                for field in ('input_shape', 'output_shape'):
                    for shape in row[field]:
                        if shape and shape[0] in ('B*n_h', 'n_h'):
                            leading[shape[0]] += 1
                if row['weight_pos'] == -1 and row['op_type'] not in ('rmsnorm', 'layernorm'):
                    negative_weight.append(sample(row))
            result['phases'][phase] = {'rows': len(rows), 'columns': len(csvrows[0]),
                'csv_rows': len(csvrows), 'groups': groups,
                'csv_jsonl_differences': diffs,
                'layer_coverage_ok': sorted(expanded) == list(range(s['symbols']['L'])),
                'schedule_mismatches': schedule_mismatches,
                'undefined_symbol_occurrences': dict(undefined),
                'undefined_symbol_rows': undefined_rows,
                'non_norm_weight_pos_minus_one': negative_weight,
                'leading_attention_axes': dict(leading),
                'same_as_main_models': all((directory / f'{phase}.{ext}').read_bytes() ==
                    (full.parent / f'{phase}.{ext}').read_bytes() for ext in ('csv', 'jsonl')),
                'caveats': sum(bool(r.get('caveat')) for r in rows),
                'operations': dict(collections.Counter(r['op_type'] for r in rows))}
            # K3 has no dead anchors; avoid reading its large raw trace here.
            if 'Kimi' not in name:
                seen = collections.Counter()
                raw[phase] = []
                for row in read_rows(full / f'{phase}.trace.raw.jsonl'):
                    key = (row.get('module_path'), row['op_type'])
                    nth = seen[key]
                    seen[key] += 1
                    # Keep representatives of each distinct layer group.
                    if row.get('layer_idx') in (None, 0, 1, 2, 3, 4):
                        raw[phase].append((row, nth))
        for fn, count, dest in [('label_confirmed.json', 'matched', 'dead_confirmations'),
                                ('label_overrides.json', 'applied', 'dead_overrides')]:
            path = full / fn
            if not path.exists():
                continue
            for record in json.loads(path.read_text(encoding='utf8')):
                if record.get(count):
                    continue
                spec = dict(zip(FIELDS, json.loads(record['id'])))
                entry = {'spec': spec, 'source': record.get('source'), 'structural_hits': {}}
                for phase, pairs in raw.items():
                    hits = []
                    for row, nth in pairs:
                        mod = re.sub(r'\.\d+(?=\.|$)', '.*', row.get('module_path') or '')
                        if not re.search(spec['module'], mod):
                            continue
                        if spec['op_type'] and row['op_type'] != spec['op_type']:
                            continue
                        if spec['nth'] is not None and nth != spec['nth']:
                            continue
                        hits.append(sample(row, nth))
                    entry['structural_hits'][phase] = hits
                result[dest].append(entry)
        out[name] = result
    (ROOT / 'develop/fleet5_artifacts_evidence.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return out


if __name__ == '__main__':
    data = collect()
    for model, info in data.items():
        print(model, {ph: (v['rows'], v['columns']) for ph, v in info['phases'].items()},
              'dead confirmations', len(info['dead_confirmations']),
              'dead overrides', len(info['dead_overrides']))
