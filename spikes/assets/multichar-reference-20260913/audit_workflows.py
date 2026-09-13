"""Compare accepted provider graphs to case intent and uploaded-file receipts."""
from pathlib import Path
import json
import time

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
checks = []

def check(label, actual, expected):
    checks.append(dict(check=label, passed=actual == expected, actual=actual, expected=expected))

for result_path in sorted((ROOT / 'runs').glob('*/*/result.json')):
    result = read(result_path)
    directory = result_path.parent
    model, cid = directory.parent.name, directory.name
    if model == 'codex' or result['state'] != 'succeeded':
        continue
    case = read(ROOT / 'cases' / cid / 'case.json')
    events = read(directory / 'events.json')
    prepared = next(e['data'] for e in events if e['event'] == 'prepared')
    graph = prepared['body']['prompt']
    history = read(directory / 'history.json')
    run = read(directory / 'run.json')
    journal = read(directory / 'journal' / (run['version_id'] + '.json'))
    check(f'{model}/{cid} provider prompt id', history['prompt'][1], result['prompt_id'])
    # Store only the comparison boolean rather than duplicating large prompts.
    check(f'{model}/{cid} exact accepted graph matches prepared', history['prompt'][2] == graph, True)
    nodes = lambda kind: [n['inputs'] for n in graph.values() if n['class_type'] == kind]
    check(f'{model}/{cid} LoadImage count', len(nodes('LoadImage')), case['count'])
    inputs = {f['path']: f for f in journal['files'] if f['kind'] == 'input'}
    if model == 'h3':
        encoder = nodes('MiniMaxH3ReferenceToVideo')[0]
        check(f'{model}/{cid} prompt exact', encoder['prompt'] == case['h3_prompt'], True)
        check(f'{model}/{cid} dimensions/frames', [encoder[k] for k in ['width', 'height', 'length']], [768, 512, 5])
        check(f'{model}/{cid} seed', nodes('RandomNoise')[0]['noise_seed'], case['seed'])
        scheduler = nodes('BasicScheduler')[0]
        check(f'{model}/{cid} schedule', [scheduler[k] for k in ['steps', 'scheduler', 'denoise']], [20, 'linear_quadratic', 1])
        check(f'{model}/{cid} sampler', nodes('KSamplerSelect')[0]['sampler_name'], 'euler')
        check(f'{model}/{cid} connected refs', len([k for k in encoder if k.startswith('ref_images.')]), case['count'])
        for i, digest in enumerate(case['reference_hashes']):
            node_id = encoder[f'ref_images.ref_image_{i}'][0]
            filename = graph[node_id]['inputs']['image']
            check(f'{model}/{cid} reference {i+1} bytes/order', inputs[filename]['sha256'], digest)
    else:
        sampler = nodes('KSampler')[0]
        check(f'{model}/{cid} sampling parameters', [sampler[k] for k in ['steps', 'cfg', 'seed', 'sampler_name', 'scheduler', 'denoise']], [40, 4, case['seed'], 'euler', 'simple', 1])
        check(f'{model}/{cid} shift', nodes('ModelSamplingAuraFlow')[0]['shift'], 3.1)
        check(f'{model}/{cid} CFGNorm', nodes('CFGNorm')[0]['strength'], 1)
        check(f'{model}/{cid} output dimensions', [graph['24']['inputs'][k] for k in ['width', 'height']], [768, 512])
        check(f'{model}/{cid} positive prompt', graph['7']['inputs']['prompt'] == case['prompt'], True)
        check(f'{model}/{cid} negative prompt', graph['8']['inputs']['prompt'], '')
        for ref in prepared['reference_images']:
            i = ref['order']
            check(f'{model}/{cid} source {i+1} hash/order', ref['sha256'], case['reference_hashes'][i])
            check(f'{model}/{cid} upload {i+1} hash', inputs[ref['uploaded_name']]['sha256'], ref['uploaded_sha256'])
            for encoder_id in ['7', '8']:
                link = graph[encoder_id]['inputs'][f'image{i+1}'][0]
                expected_link = '24' if i == 0 and model == 'qwen' else str(21+i)
                check(f'{model}/{cid} encoder {encoder_id} image{i+1} link', link, expected_link)
                raw_link = graph[link]['inputs']['image'][0] if link == '24' else link
                check(f'{model}/{cid} encoder {encoder_id} image{i+1} file', graph[raw_link]['inputs']['image'], ref['uploaded_name'])

report = dict(checked_epoch=time.time(), scope='Terminal successful GPU arms only; accepted history graph vs prepared graph, key numerical settings, conditioning links and ordered upload hashes. No GPU reruns.', checks=checks, failed=[c for c in checks if not c['passed']])
(ROOT / 'workflow-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(checks=len(checks), failed=report['failed']), ensure_ascii=False))
raise SystemExit(bool(report['failed']))
