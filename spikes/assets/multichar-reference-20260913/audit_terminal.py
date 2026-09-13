"""Read-only terminal checks for this experiment's exact requests and schemas."""
import fcntl
import json
import time
import urllib.request
from pathlib import Path
from dotenv import dotenv_values
import psycopg

ROOT = Path(__file__).resolve().parent
runs = [(p, json.loads(p.read_text())) for p in (ROOT / 'runs').glob('*/*/run.json')]
missing = [str(p.relative_to(ROOT)) for p, r in runs if not (p.parent / 'result.json').exists()]
schemas = sorted({r['schema'] for p, r in runs if r.get('schema')})
clients = {'msai-studio-' + r['version_id'] for p, r in runs if r.get('version_id')}
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=15) as response:
    queue = json.load(response)
owned_queue = [dict(prompt_id=row[1], client_id=row[3].get('client_id'))
               for row in queue['queue_running'] + queue['queue_pending']
               if row[3].get('client_id') in clients]
base = dotenv_values('/Users/aiden/Projects/isuper/.repos/veritas/.runtime/task1-studio-preview/preview.env')['VERITAS_PLATFORM_DB']
with psycopg.connect(base) as connection:
    remaining_schemas = [row[0] for row in connection.execute(
        'SELECT nspname FROM pg_namespace WHERE nspname = ANY(%s)', (schemas,))]
lock_free = False
with (ROOT / 'gpu-worker.lock').open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_free = True
    except BlockingIOError:
        pass
results = [json.loads(p.read_text()) for p in (ROOT / 'runs').glob('*/*/result.json')]
unreviewed = [dict(model=r['model'], case_id=r['case_id']) for r in results
              if r.get('state') == 'succeeded' and not r.get('review')]
pending_cleanup = [dict(model=r['model'], case_id=r['case_id']) for r in results
                   if r.get('submitted') and r['model'] != 'codex' and not r.get('cleanup_complete')]
report = dict(checked_epoch=time.time(), checked_schema_count=len(schemas),
    missing_terminal_results=missing, owned_queue=owned_queue,
    remaining_owned_schemas=remaining_schemas, worker_lock_free=lock_free,
    successful_unreviewed=unreviewed, pending_owned_cleanup=pending_cleanup)
report['passed'] = not (missing or owned_queue or remaining_schemas or unreviewed or pending_cleanup) and lock_free
(ROOT / 'terminal-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(not report['passed'])
