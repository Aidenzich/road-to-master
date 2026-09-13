"""Read-only evidence checks; writes a snapshot, never changes receipts or jobs."""
from pathlib import Path
import hashlib
import json
import re
import time
from PIL import Image

ROOT = Path(__file__).resolve().parent
checks = []
files = []

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(label, condition):
    checks.append({'check': label, 'passed': bool(condition)})

def image_info(path):
    with Image.open(path) as im:
        im.load()
        return list(im.size)

for ref in read(ROOT / 'reference-receipts.json'):
    path = ROOT / ref['file']
    check(f"reference hash {ref['id']}", sha(path) == ref['sha256'])
    check(f"reference decode {ref['id']}", image_info(path) == [ref['width'], ref['height']])

case_ids = read(ROOT / 'case-index.json')
for cid in case_ids:
    case = read(ROOT / 'cases' / cid / 'case.json')
    for key, filename in [('prompt', 'prompt.txt'), ('h3_prompt', 'h3-prompt.txt')]:
        check(f'{cid} {filename}', (ROOT / 'cases' / cid / filename).read_text().strip() == case[key].strip())
    if case['repeat'] == 2:
        first = read(ROOT / 'cases' / cid.replace('-r2', '-r1') / 'case.json')
        for key in ['prompt', 'h3_prompt', 'references', 'reference_hashes']:
            check(f'{cid} repeat invariant {key}', case[key] == first[key])

for item in read(ROOT / 'position-control-index.json')['controls']:
    baseline_path = ROOT / 'cases' / item['baseline_case_id'] / 'case.json'
    check(f"baseline frozen {item['case_id']}", sha(baseline_path) == item['baseline_sha256'])
    baseline = read(baseline_path)
    control = read(ROOT / 'cases' / item['case_id'] / 'case.json')
    for key in ['references', 'reference_hashes', 'seed', 'cast', 'camera_ledger', 'action']:
        check(f"control invariant {item['case_id']} {key}", baseline[key] == control[key])
    for key in ['prompt', 'h3_prompt']:
        a, b = baseline[key], control[key]
        pattern = r'The left-to-right order is [^.]+\.'
        check(f"only order changed {item['case_id']} {key}", a != b and re.sub(pattern, '', a) == re.sub(pattern, '', b))

for result_path in sorted((ROOT / 'runs').glob('*/*/result.json')):
    result = read(result_path)
    if result['state'] != 'succeeded':
        continue
    directory = result_path.parent
    model, cid = directory.parent.name, directory.name
    case = read(ROOT / 'cases' / cid / 'case.json')
    run = read(directory / 'run.json')
    check(f'{model}/{cid} reference hashes', run.get('reference_hashes') == case['reference_hashes'])
    for filename in result['outputs']:
        path = directory / filename
        digest = sha(path)
        entry = {'file': str(path.relative_to(ROOT)), 'sha256': digest, 'bytes': path.stat().st_size}
        if path.suffix == '.png':
            entry['dimensions'] = image_info(path)
        files.append(entry)
    if model == 'codex':
        check(f'{model}/{cid} prompt', run['prompt'] == case['prompt'] + '\nPlease produce one landscape image with a 3:2 aspect ratio.')
        if result.get('sha256'):
            check(f'{model}/{cid} original hash', sha(directory / 'output.png') == result['sha256'])
        check(f'{model}/{cid} dimensions', image_info(directory / 'output.png') == [result['width'], result['height']])
    else:
        expected = case['h3_prompt'] if model == 'h3' else case['prompt']
        check(f'{model}/{cid} prompt', run['inputs']['prompt'] == expected)
        check(f'{model}/{cid} seed', run['inputs']['seed'] == case['seed'])
        cleanup = read(directory / 'cleanup.json')
        check(f'{model}/{cid} cleanup prompt id', cleanup['prompt_id'] == result['prompt_id'])
        check(f'{model}/{cid} cleanup receipts', bool(cleanup['files']) and all(f['state'] == 'deleted' for f in cleanup['files']))
        if model == 'h3':
            streams = read(directory / 'ffprobe.json')['streams']
            videos = [s for s in streams if s['codec_type'] == 'video']
            check(f'{model}/{cid} decoded five frames', len(videos) == 1 and videos[0]['nb_read_frames'] == '5')
            check(f'{model}/{cid} first frame fixed', sha(directory / 'output.png') == sha(directory / 'frame-01.png'))

report = {'checked_epoch': time.time(), 'scope': 'Snapshot of terminal successful local receipts, decoded PNGs, stored ffprobe, reference and prompt invariants. Not a live remote deletion audit, workflow-graph semantic audit, independent ffprobe rerun, or completion assertion.', 'checks': checks, 'files': files}
report['failed'] = [c for c in checks if not c['passed']]
(ROOT / 'evidence-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'checks': len(checks), 'files': len(files), 'failed': report['failed']}, ensure_ascii=False))
raise SystemExit(bool(report['failed']))
