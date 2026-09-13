"""Record a human/assistant visual review only after inspecting actual output."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser()
p.add_argument('model');p.add_argument('case')
p.add_argument('--scores',nargs=5,type=int,required=True)
p.add_argument('--notes',required=True)
args=p.parse_args()
assert all(0<=x<=2 for x in args.scores)
path=Path(__file__).resolve().parent/'runs'/args.model/args.case/'result.json'
record=json.loads(path.read_text())
assert record['state']=='succeeded'
keys=['exact_count','appearance_preserved','reference_binding','action_obedience','hands_and_contacts']
record['review']={**dict(zip(keys,args.scores)),'reviewer':'assistant visual review','notes':args.notes}
temp=path.with_suffix('.tmp')
temp.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n');temp.replace(path)
