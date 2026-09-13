"""Descriptive ordinal-score counts, preserving missingness and arm identity."""
METRICS = ['exact_count', 'appearance_preserved', 'reference_binding', 'action_obedience', 'hands_and_contacts']
MODELS = ['codex', 'qwen', 'h3']

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
