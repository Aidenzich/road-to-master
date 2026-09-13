"""Receipts only: builtin generation must be invoked separately, never via API."""
import argparse, hashlib, json, shutil, time
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('action',choices=['prepare','complete'])
p.add_argument('case_id')
p.add_argument('--source')
p.add_argument('--seconds',type=float)
a=p.parse_args()
c=json.loads((ROOT/'cases'/a.case_id/'case.json').read_text())
out=ROOT/'runs/codex'/a.case_id
if a.action=='prepare':
    assert time.time()<json.loads((ROOT/'plan.json').read_text())['stop_new_submissions_epoch']
    assert not (out/'run.json').exists(), 'Existing submission: inspect before retry'
    for ref,expected in zip(c['references'],c['reference_hashes']):
        assert hashlib.sha256((ROOT/ref).read_bytes()).hexdigest()==expected
    out.mkdir(parents=True,exist_ok=True)
    record=dict(case_id=a.case_id,model='codex-built-in',state='submitting',started_epoch=time.time(),
                prompt=c['prompt']+'\nPlease produce one landscape image with a 3:2 aspect ratio.',
                references=c['references'],reference_hashes=c['reference_hashes'],seed=None,steps=None,cfg=None,
                tool='image_gen',requested_aspect_ratio='3:2')
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(prompt=record['prompt'],referenced_image_paths=[str(ROOT/f) for f in c['references']])))
else:
    assert (out/'run.json').exists() and not (out/'result.json').exists()
    assert a.source and a.seconds is not None
    source=Path(a.source).resolve()
    assert 'generated_images' in source.parts
    with Image.open(source) as im:
        width,height=im.size
        im.verify()
    target=out/'output.png'
    assert not target.exists()
    shutil.copyfile(source,target)
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest()==sha
    result=dict(case_id=a.case_id,model='codex',state='succeeded',submitted=True,
                tool_elapsed_seconds=a.seconds,timing_source='tool wall time, not pure inference',
                width=width,height=height,sha256=sha,outputs=['output.png'],finished_epoch=time.time(),
                seed=None,steps=None,cfg=None,model_version=None,review=None,cleanup_complete=True,
                cleanup_scope='Verified local copy; no 5090 files created. Built-in provider retention unknown.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
