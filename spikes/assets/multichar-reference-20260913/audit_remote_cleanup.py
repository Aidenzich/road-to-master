"""Verify archived bytes and remote absence for terminal owned receipts; no deletion."""
from pathlib import Path, PurePosixPath
import base64
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parent
records = []
for result_path in sorted((ROOT / 'runs').glob('*/*/result.json')):
    result = json.loads(result_path.read_text())
    directory = result_path.parent
    if directory.parent.name == 'codex' or not result.get('cleanup_complete') or not result.get('submitted'):
        continue
    run = json.loads((directory / 'run.json').read_text())
    journal = json.loads((directory / 'journal' / f'{run["version_id"]}.json').read_text())
    receipt = json.loads((directory / 'cleanup.json').read_text())
    assert journal['prompt_id'] == result['prompt_id'] == receipt['prompt_id']
    deleted = {(f['kind'], f['path']) for f in receipt['files'] if f['state'] == 'deleted'}
    for f in journal['files']:
        assert f['kind'] in ('input', 'output')
        rel = PurePosixPath(f['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        assert (f['kind'], f['path']) in deleted
        archive = directory / 'journal/objects' / f['sha256']
        raw = archive.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == f['sha256'] and len(raw) == f['size']
        if f['kind'] == 'output':
            exported = directory / ('output' + rel.suffix)
            assert hashlib.sha256(exported.read_bytes()).hexdigest() == f['sha256']
        records.append(dict(case_id=directory.name, model=directory.parent.name,
                            prompt_id=journal['prompt_id'], kind=f['kind'], path=f['path'],
                            sha256=f['sha256'], bytes=f['size'], archive_verified=True,
                            remote_path='/data/comfyui-minimax-h3/' + f['kind'] + '/' + f['path']))

# Send an explicit list; no glob, walk, deletion, or GPU interaction.
payload = base64.b64encode(json.dumps([r['remote_path'] for r in records]).encode()).decode()
script = f'''import base64,json,os
paths=json.loads(base64.b64decode({payload!r}))
result=[]
for path in paths:
    try:
        st=os.lstat(path)
        result.append({{"path":path,"state":"present","bytes":st.st_size}})
    except FileNotFoundError:
        result.append({{"path":path,"state":"absent"}})
print(json.dumps(result))
'''
response = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                           '5090', 'sudo -n python3 -'], input=script, text=True,
                          capture_output=True, check=True, timeout=45)
remote = {r['path']: r for r in json.loads(response.stdout)}
for record in records:
    record['remote_state'] = remote[record['remote_path']]['state']
report = dict(checked_epoch=time.time(), scope='Only terminal submitted owned task files, explicit lstat paths; no deletion or global inventory. Active jobs excluded.',
              files=records, present=[r for r in records if r['remote_state'] != 'absent'])
(ROOT / 'remote-cleanup-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(archived_and_checked=len(records), present=len(report['present']))))
raise SystemExit(bool(report['present']))
