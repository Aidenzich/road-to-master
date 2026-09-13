from pathlib import Path
from urllib.request import Request, urlopen
import hashlib, json, time
from PIL import Image

p = Path(__file__).resolve().parent
refs = json.loads((p/'references.json').read_text())
directory = p/'references'
directory.mkdir(exist_ok=True)
receipts = []
for ref in refs:
    ext = '.png' if '.png' in ref['url'] else '.jpg'
    dest = directory/(ref['id']+ext)
    if not dest.exists():
        request = Request(ref['url'], headers={'User-Agent':'Mozilla/5.0', 'Referer':ref['page']})
        try:
            with urlopen(request, timeout=30) as response:
                data = response.read()
            with dest.open('xb') as f:
                f.write(data)
        except Exception as e:
            print(ref['id'], 'DOWNLOAD FAILED', str(e), flush=True)
            receipts.append({**ref, 'state':'download_failed', 'error':str(e)})
            continue
    with Image.open(dest) as im:
        im.verify()
    with Image.open(dest) as im:
        size = im.size
    receipts.append({**ref,'file':str(dest.relative_to(p)), 'width':size[0], 'height':size[1],
                     'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(), 'state':'downloaded', 'retrieved_epoch':time.time()})
    print(ref['id'], size, 'downloaded', flush=True)
(p/'reference-receipts.json').write_text(json.dumps(receipts,indent=2,ensure_ascii=False)+'\n')
