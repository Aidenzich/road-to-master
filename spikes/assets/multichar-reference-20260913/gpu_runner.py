"""Owned Qwen/H3 experiments with durable receipts; no product edits or retries."""
from pathlib import Path
from dotenv import dotenv_values
from dataclasses import replace
import argparse, fcntl, hashlib, json, os, shutil, subprocess, time, uuid
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from PIL import Image
from src.integrations.studio.comfy_cleanup import OutputCleaner
from src.integrations.studio.runtime import StudioCatalog, LocalStudioStore
from src.music_center.adapters.generation.comfyui import ComfyUIGenerator
from src.music_center.adapters.generation.comfyui_qwen_image import ComfyQwenImageGenerator
from src.music_center.domain import comfy_contract as contract
from src.music_center.app import studio_jobs as jobs

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'plan.json').read_text())
STOP = PLAN['stop_new_submissions_epoch']
BASE_ENV = '/Users/aiden/Projects/isuper/.repos/veritas/.runtime/task1-studio-preview/preview.env'
seconds = 5/24
original_build = contract.build_graph
def minimum_graph(inputs):
    return original_build(replace(inputs,frames=5))
contract.build_graph = minimum_graph
class H3Minimum(ComfyUIGenerator):
    def submit(self, request, *, on_submission=None):
        return super().submit(replace(request,duration_seconds=seconds),on_submission=on_submission)

class QwenFullReference(ComfyQwenImageGenerator):
    def graph(self,*args,**kwargs):
        graph=super().graph(*args,**kwargs)
        # Single-variable control: keep full first reference for positive/negative
        # conditioning. Output latent sizing remains exactly the baseline graph.
        graph['7']['inputs']['image1']=['21',0]
        graph['8']['inputs']['image1']=['21',0]
        return graph

def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')
    temporary.replace(path)

def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)]))

def run_one(model, case_id):
    assert model in ('qwen','qwen-fullref','h3')
    is_qwen=model.startswith('qwen')
    case = json.loads((ROOT/'cases'/case_id/'case.json').read_text())
    out = ROOT/'runs'/model/case_id
    if (out/'result.json').exists():
        result = json.loads((out/'result.json').read_text())
        if result.get('cleanup_complete') or result.get('state') == 'unsupported':
            return result
        raise RuntimeError(f'Unresolved existing result: {out}; recover, do not resubmit')
    if (out/'run.json').exists():
        raise RuntimeError(f'Existing submission: {out}; recover, do not resubmit')
    out.mkdir(parents=True,exist_ok=True)
    if is_qwen and not case['qwen_supported']:
        result = dict(case_id=case_id,model=model,state='unsupported',reason='Native adapter accepts at most 3 separate references',submitted=False)
        write(out/'result.json',result)
        return result
    base = dotenv_values(BASE_ENV)['VERITAS_PLATFORM_DB']
    schema = 'qa_multi_'+uuid.uuid4().hex
    dsn = make_conninfo(base, options='-c search_path='+schema)
    os.environ['VERITAS_PLATFORM_DB'] = dsn
    os.environ['VERITAS_BLOB_STORAGE_PATH'] = str(out/'blobs')
    os.environ['VERITAS_STUDIO_BLOB_ROOT'] = str(out/'blobs/studio')
    store = LocalStudioStore(out/'blobs/studio')
    cleaner = OutputCleaner('5090','/data/comfyui-minimax-h3','http://127.0.0.1:8188',out/'journal')
    qwen_class=QwenFullReference if model=='qwen-fullref' else ComfyQwenImageGenerator
    gen = (qwen_class('http://127.0.0.1:8188',output_cleaner=cleaner) if is_qwen else
           H3Minimum('http://127.0.0.1:8188',workflow_id=contract.WORKFLOW_REFERENCE,output_cleaner=cleaner))
    waiting_start = time.time()
    while True:
        if time.time() >= STOP:
            return {'state':'deadline','submitted':False}
        q = gen._call('queue')
        if not q['queue_running'] and not q['queue_pending']:
            break
        write(ROOT/'worker-status.json',dict(state='waiting_for_shared_queue',model=model,case_id=case_id,
             observed_epoch=time.time(),running=len(q['queue_running']),pending=len(q['queue_pending'])))
        print('Waiting shared queue',model,case_id,flush=True)
        time.sleep(15)
    with psycopg.connect(base,autocommit=True) as c:
        c.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    with psycopg.connect(dsn,autocommit=True) as c:
        c.execute(Path('src/migrations/studio/schema.sql').read_text())
    catalog = StudioCatalog(dsn)
    version = None
    clean = False
    try:
        project = catalog.create_studio_project(owner_account_id=1,title='Owned multi-character benchmark')
        pid = str(project['id'])
        episode = catalog.create_studio_episode(project_id=pid,title='QA')
        segment = catalog.create_studio_segment(project_id=pid,episode_id=str(episode['id']),title=case_id,duration_seconds=1)
        assets = []
        for relative in case['references']:
            f = ROOT/relative
            with f.open('rb') as stream:
                asset = jobs.ingest.ingest_media(catalog=catalog,store_factory=lambda:store,project_id=pid,
                    display_name=f.name,kind='image',upload_stream=stream,
                    content_type='image/png' if f.suffix=='.png' else 'image/jpeg')
            assets.append(str(asset.asset['id']))
        prompt = case['prompt'] if is_qwen else case['h3_prompt']
        inputs = dict(prompt=prompt,effective_prompt=prompt,width=PLAN['canvas']['width'],height=PLAN['canvas']['height'],
            steps=PLAN['qwen' if is_qwen else 'h3']['steps'],seed=case['seed'],candidate_count=1,reference_asset_ids=assets,
            workflow_id='qwen-edit-2511' if is_qwen else contract.WORKFLOW_REFERENCE,
            model_id='qwen-image-edit-2511' if is_qwen else 'minimax-h3-standard')
        if is_qwen:
            inputs['cfg_scale'] = PLAN['qwen']['cfg']
        else:
            inputs['duration_seconds'] = seconds
        # Last admission check: no new provider submission after the six-hour window.
        if time.time()>=STOP:
            return {'state':'deadline','submitted':False}
        version = catalog.claim_generation_version(project_id=pid,segment_id=str(segment['id']),
            kind='first_frame' if is_qwen else 'video',idempotency_key=str(uuid.uuid4()),inputs=inputs)['version']
        rid = str(version['id'])
        started = time.time()
        write(out/'run.json',dict(case_id=case_id,model=model,schema=schema,version_id=rid,inputs=inputs,
              admitted_epoch=started,prequeue_wait_seconds=started-waiting_start,reference_hashes=case['reference_hashes']))
        write(ROOT/'worker-status.json',dict(state='running',model=model,case_id=case_id,version_id=rid,
              pid=os.getpid(),started_epoch=started))
        print('START',model,case_id,rid,flush=True)
        result = jobs.run_generation(catalog,gen,lambda:store,version)
        elapsed = time.time()-started
        events = catalog.list_generation_submission_events(rid)
        write(out/'events.json',events)
        write(out/'catalog-result.json',result)
        for _ in range(60):
            receipt = cleaner.reconcile_one(rid,catalog,store)
            if receipt:
                clean=True
                break
            time.sleep(1)
        if not clean:
            raise RuntimeError(f'Owned cleanup pending; retained schema {schema}')
        write(out/'cleanup.json',receipt)
        journal = json.loads((out/'journal'/f'{rid}.json').read_text())
        hist_path = out/'journal/history'/f'{rid}.json'
        history = json.loads(hist_path.read_text()) if hist_path.exists() else {}
        write(out/'history.json',history)
        provider_success = history.get('status',{}).get('status_str') == 'success'
        exported = []
        for f in journal.get('files',[]):
            if f['kind'] != 'output':
                continue
            source = out/'journal/objects'/f['sha256']
            assert hashlib.sha256(source.read_bytes()).hexdigest()==f['sha256']
            dest = out/('output'+Path(f['path']).suffix)
            assert not dest.exists()
            shutil.copyfile(source,dest)
            if model=='h3':
                metadata = probe(dest)
                write(out/'ffprobe.json',metadata)
                video = next(s for s in metadata['streams'] if s['codec_type']=='video')
                assert int(video['nb_read_frames']) == 5
                subprocess.run(['ffmpeg','-v','error','-n','-i',str(dest),'-map','0:v:0','-fps_mode','passthrough',str(out/'frame-%02d.png')],check=True)
                shutil.copyfile(out/'frame-01.png',out/'output.png')
                exported.extend(['output.mp4','output.png']+[f'frame-{i:02d}.png' for i in range(1,6)])
            else:
                with Image.open(dest) as im:
                    im.verify()
                exported.append(dest.name)
        timestamps = {e['event']:e['data'].get('occurred_at') for e in events}
        record = dict(case_id=case_id,model=model,count=case['count'],action=case['action'],cohort=case['cohort'],
            repeat=case['repeat'],seed=case['seed'],state='succeeded' if provider_success and exported else 'failed',
            provider_success=provider_success,catalog_state=result['state'],catalog_error=result.get('error_cause'),
            elapsed_seconds=elapsed,prequeue_wait_seconds=started-waiting_start,submitted=True,
            prompt_id=journal.get('prompt_id'),stage_timestamps=timestamps,outputs=exported,
            cleanup_complete=True,finished_epoch=time.time(),review=None)
        write(out/'result.json',record)
        print('END',model,case_id,record['state'],round(elapsed,2),flush=True)
        return record
    finally:
        if clean or version is None:
            catalog.close()
            with psycopg.connect(base,autocommit=True) as c:
                c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
        else:
            print('RECOVERY REQUIRED',schema,out,flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--one',nargs=2,metavar=('MODEL','CASE'))
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    with (ROOT/'gpu-worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.one:
            run_one(*args.one)
            return
        assert args.run
        index=json.loads((ROOT/'case-index.json').read_text())
        # Small model blocks reduce thrashing, but retain time/order for analysis.
        for offset in range(0,len(index),6):
            models=('qwen','h3') if (offset//6)%2==0 else ('h3','qwen')
            for model in models:
                for case_id in index[offset:offset+6]:
                    if time.time()>=STOP:
                        write(ROOT/'worker-status.json',dict(state='deadline',observed_epoch=time.time()))
                        return
                    run_one(model,case_id)
        write(ROOT/'worker-status.json',dict(state='matrix_finished',observed_epoch=time.time()))

if __name__=='__main__':
    main()
