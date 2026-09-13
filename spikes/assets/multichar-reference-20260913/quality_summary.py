"""Descriptive ordinal-score counts, preserving missingness and arm identity."""
METRICS = ['exact_count', 'appearance_preserved', 'reference_binding', 'action_obedience', 'hands_and_contacts']
MODELS = ['codex', 'qwen', 'h3']

def summarize_repeats(rows):
    """Pair only identical cohort/count/action; exclude unavailable metric scores."""
    pairs = {}
    for row in rows:
        key = (row['model'], row['cohort'], row['count'], row['action'])
        bucket = pairs.setdefault(key, {})
        repeat = row['repeat']
        if repeat in bucket:
            raise ValueError(f'Duplicate repeat arm: {key}, repeat={repeat}')
        bucket[repeat] = row
    groups = []
    for model in MODELS:
        selected = [(key, pair) for key, pair in pairs.items() if key[0] == model]
        for metric in METRICS:
            observations = []
            for key, pair in selected:
                a, b = pair.get(1), pair.get(2)
                if not all(r and r['state'] == 'succeeded' and r['reviewed']
                           and r.get(metric) in (0, 1, 2) for r in (a, b)):
                    continue
                observations.append(dict(cohort=key[1], count=key[2], action=key[3],
                                         case_ids=[a['case_id'], b['case_id']], scores=[a[metric], b[metric]]))
            groups.append(dict(model=model, metric=metric, planned_pairs=len(selected),
                               evaluated_pairs=len(observations), missing_pairs=len(selected)-len(observations),
                               same_score=sum(o['scores'][0] == o['scores'][1] for o in observations),
                               changed_score=sum(o['scores'][0] != o['scores'][1] for o in observations),
                               both_clear_pass=sum(o['scores'] == [2, 2] for o in observations),
                               pairs=observations))
    return dict(groups=groups, limitation='Two repeats only; descriptive score agreement, not reliability estimation or statistical significance. Codex seed unknown; local seeds differ. Failed, unsupported and unreviewed arms excluded from evaluated pairs but retained as missing pairs. No controls.')

def summarize(rows):
    matched = sorted({r['case_id'] for r in rows if all(
        any(x['case_id'] == r['case_id'] and x['model'] == model
            and x['state'] == 'succeeded' and x['reviewed']
            and all(x.get(metric) in (0, 1, 2) for metric in METRICS)
            for x in rows) for model in MODELS)})
    groups = []
    for scope in ['all_available_baseline', 'three_model_matched_complete']:
        selected = rows if scope == 'all_available_baseline' else [r for r in rows if r['case_id'] in matched]
        for dimension in ['count', 'action', 'repeat']:
            for value in sorted({r[dimension] for r in rows}):
                for model in MODELS:
                    subset = [r for r in selected if r['model'] == model and r[dimension] == value]
                    metrics = {}
                    for metric in METRICS:
                        scores = [r[metric] for r in subset if r['state'] == 'succeeded' and r['reviewed'] and r.get(metric) in (0, 1, 2)]
                        metrics[metric] = {'reviewed': len(scores), 'clear_pass': scores.count(2),
                                           'partial_or_uncertain': scores.count(1), 'clear_failure': scores.count(0),
                                           'missing_or_inapplicable': len(subset) - len(scores)}
                    groups.append({'scope': scope, 'dimension': dimension, 'value': value, 'model': model,
                                   'arms': len(subset), 'succeeded': sum(r['state'] == 'succeeded' for r in subset),
                                   'unsupported': sum(r['state'] == 'unsupported' for r in subset),
                                   'metrics': metrics})
    return {'matched_case_ids': matched, 'groups': groups,
            'limitation': 'Descriptive ordinal counts, no overall ranking or statistical inference. Complete-case matching conditions on successful reviewed output and excludes failures/unsupported arms; consult full matrix for availability. Controls excluded. Ratings are one assistant visual review, not blinded.'}
