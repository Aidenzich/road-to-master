"""Close unstarted baseline arms without interrupting the accepted GPU job."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
now = time.time()
closed = []
active = []
for case_id in json.loads((ROOT / 'case-index.json').read_text()):
    case = json.loads((ROOT / 'cases' / case_id / 'case.json').read_text())
    for model in ('codex', 'qwen', 'h3'):
        directory = ROOT / 'runs' / model / case_id
        if (directory / 'result.json').exists():
            continue
        if (directory / 'run.json').exists():
            active.append([model, case_id])
            continue
        unsupported = model == 'qwen' and not case['qwen_supported']
        record = dict(case_id=case_id, model=model,
            state='unsupported' if unsupported else 'not_executed', submitted=False,
            cleanup_complete=True, cleanup_not_applicable=True, review=None,
            notes='Native adapter accepts at most 3 separate references' if unsupported else
                  'User requested wrap-up; no provider submission was made.',
            admission_closed_epoch=now)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'result.json').open('x') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
            f.write('\n')
        closed.append([model, case_id])
(ROOT / 'closure.json').write_text(json.dumps(dict(
    status='draining', reason='user_requested_wrap_up', admission_closed_epoch=now,
    closed_unstarted_arms=closed, already_submitted_arms=active), indent=2) + '\n')
print(json.dumps(dict(closed=len(closed), already_submitted=active)))
