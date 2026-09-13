"""Register spatial-order controls without modifying baseline cases or GPU queue."""
from pathlib import Path
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parent


def prepare():
    plan = json.loads((ROOT / 'plan.json').read_text())
    index_path = ROOT / 'position-control-index.json'
    if index_path.exists():
        raise RuntimeError('Controls already registered; inspect rather than overwrite')
    assert time.time() < plan['stop_new_submissions_epoch']
    controls = []
    for action in ('book', 'contact'):
        for count in (2, 3, 5):
            for cohort in ('anime', 'live'):
                baseline_id = f'{cohort}-{count:02d}-{action}-r1'
                source = ROOT / 'cases' / baseline_id / 'case.json'
                raw = source.read_bytes()
                case = json.loads(raw)
                cid = baseline_id + '-reverse-position'
                original = ', '.join(f'Person {i}' for i in range(1, count + 1))
                reverse = ', '.join(f'Person {i}' for i in range(count, 0, -1))
                old = f'The left-to-right order is {original}.'
                new = f'The left-to-right order is {reverse}.'
                assert case['prompt'].count(old) == 1
                case['prompt'] = case['prompt'].replace(old, new)
                old_h3 = 'The left-to-right order is ' + ', '.join(
                    f'<Subject {i}>' for i in range(1, count + 1)) + '.'
                new_h3 = 'The left-to-right order is ' + ', '.join(
                    f'<Subject {i}>' for i in range(count, 0, -1)) + '.'
                assert case['h3_prompt'].count(old_h3) == 1
                case['h3_prompt'] = case['h3_prompt'].replace(old_h3, new_h3)
                case['id'] = cid
                case['baseline_case_id'] = baseline_id
                case['control_type'] = 'reverse_spatial_order_only'
                case['expected_left_to_right_cast'] = list(reversed(case['cast']))
                directory = ROOT / 'cases' / cid
                directory.mkdir(exist_ok=False)
                (directory / 'case.json').write_text(json.dumps(case, indent=2) + '\n')
                (directory / 'prompt.txt').write_text(case['prompt'] + '\n')
                (directory / 'h3-prompt.txt').write_text(case['h3_prompt'])
                controls.append(dict(case_id=cid, baseline_case_id=baseline_id,
                    baseline_sha256=hashlib.sha256(raw).hexdigest(),
                    changed='Only the left-to-right order sentence in each model prompt',
                    unchanged=['reference bytes/order', 'identity labels', 'action roles',
                        'seed where supported', 'settings', 'scene', 'camera'],
                    qwen_supported=case['qwen_supported']))
    record = dict(registered_epoch=time.time(), control_type='reverse_spatial_order_only',
        note='Additional arms; do not pool with the fixed-order baseline matrix. '
             'No active worker or production configuration modified. '
             'Codex has no controllable seed. Unsubmitted model arms remain unexecuted.',
        controls=controls)
    index_path.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Registered {len(controls)} immutable spatial-order controls')


if __name__ == '__main__':
    prepare()
