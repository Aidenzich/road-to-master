"""Derived metrics from exact per-job receipts; unavailable intervals stay null."""
from pathlib import Path
from datetime import datetime
import csv,json
ROOT=Path(__file__).resolve().parent
def diff(a,b):
    if a is None or b is None or b<a:return None
    return round(b-a,6)
rows=[]
for f in sorted((ROOT/'runs').glob('*/*/result.json')):
    r=json.loads(f.read_text()); d=f.parent
    if not r.get('submitted'):continue
    events=json.loads((d/'events.json').read_text()) if (d/'events.json').exists() else []
    history=json.loads((d/'history.json').read_text()) if (d/'history.json').exists() else {}
    stamps={}
    for e in events:
        value=e['data'].get('occurred_at')
        if value:stamps[e['event']]=datetime.fromisoformat(value).timestamp()
    messages={name:data for name,data in history.get('status',{}).get('messages',[])}
    for data in messages.values():
        assert data.get('prompt_id')==r.get('prompt_id'), 'Wrong history identity'
    start=messages.get('execution_start',{}).get('timestamp')
    end=messages.get('execution_success',{}).get('timestamp')
    extra=history.get('prompt',[None,None,None,{}])[3]
    created=extra.get('create_time')
    hardware=next((e['data'].get('hardware',{}) for e in events if e['event']=='prepared'),{})
    devices=hardware.get('devices') or [{}]
    row=dict(model=d.parent.name,case_id=d.name,state=r['state'],
             runner_seconds=r.get('elapsed_seconds'),builtin_tool_wall_seconds=r.get('tool_elapsed_seconds'),
             provider_execution_seconds=diff(start/1000 if start else None,end/1000 if end else None),
             provider_queue_seconds=diff(created/1000 if created else None,start/1000 if start else None),
             sampling_node_seconds=diff(stamps.get('sampling_started'),stamps.get('sampling_completed')),
             first_to_last_step_seconds=diff(stamps.get('first_step_observed'),stamps.get('sampling_completed')),
             collection_to_saved_seconds=diff(stamps.get('collection_started'),stamps.get('assets_saved')),
             gpu=devices[0].get('name'),vram_total_bytes=devices[0].get('vram_total'),
             vram_free_before_submit_bytes=devices[0].get('vram_free'),
             prompt_id=r.get('prompt_id'),cleanup_complete=r.get('cleanup_complete'))
    rows.append(row)
(ROOT/'timings.json').write_text(json.dumps(rows,indent=2)+'\n')
with (ROOT/'timings.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
print(f'Wrote {len(rows)} exact-job timing rows; missing stages remain null')
