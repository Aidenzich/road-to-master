"""Audit recorded admissions; never infer liveness from a missing result."""
from pathlib import Path
from datetime import datetime, timezone
import json
import time

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / 'plan.json').read_text())
start, stop = plan['start_epoch'], plan['stop_new_submissions_epoch']
rows = []
for file in sorted((ROOT / 'runs').glob('*/*/run.json')):
    run = json.loads(file.read_text())
    result_file = file.parent / 'result.json'
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    admitted = run.get('admitted_epoch', run.get('started_epoch'))
    if admitted is None and run.get('started_at'):
        admitted = datetime.strptime(run['started_at'], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc).timestamp()
    history_file = file.parent / 'history.json'
    history = json.loads(history_file.read_text()) if history_file.exists() else {}
    created_ms = history.get('prompt', [None, None, None, {}])[3].get('create_time')
    provider_received = created_ms / 1000 if created_ms is not None else None
    rows.append(dict(model=file.parent.parent.name, case_id=file.parent.name,
                     admitted_epoch=admitted, provider_received_epoch=provider_received,
                     recorded_admission_in_window=None if admitted is None else start <= admitted < stop,
                     provider_received_in_window=None if provider_received is None else start <= provider_received < stop,
                     terminal_result=result.get('state'), submitted=result.get('submitted'),
                     prompt_id=result.get('prompt_id'), version_id=run.get('version_id')))
violations = [r for r in rows if r['recorded_admission_in_window'] is False or r['provider_received_in_window'] is False]
report = dict(checked_epoch=time.time(), start_epoch=start, stop_new_submissions_epoch=stop,
              scope='Local recorded admission timestamps and exact-job provider receipt timestamps where available. Missing provider timestamps remain unknown. Missing terminal results require live-handle/queue verification, not automatic retry. Snapshot does not prove six-hour completion.',
              runs=rows, outside_window=violations,
              missing_admission=[r for r in rows if r['admitted_epoch'] is None],
              without_terminal_result=[r for r in rows if r['terminal_result'] is None])
(ROOT / 'admission-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(runs=len(rows), outside_window=len(violations),
                      missing_admission=len(report['missing_admission']),
                      without_terminal_result=report['without_terminal_result']), ensure_ascii=False))
raise SystemExit(bool(violations))
