from concurrent.futures import ThreadPoolExecutor
import http.client
from google.colab import files

BUNDLE_SHA = 'c470ac5f7ddbc96be3f426edfcfca9b318af516d4eab2e85e8de5dc581d45d52'
BUNDLE = DRIVE/'artifacts/three_actions_v1'/('source_'+BUNDLE_SHA[:12]+'.zip')
if not BUNDLE.is_file():
    uploaded = files.upload()
    payload = next((value for value in uploaded.values() if hashlib.sha256(value).hexdigest() == BUNDLE_SHA), None)
    if payload is None:
        raise ValueError('Upload the verified three_actions_bundle.zip, not video or checkpoint files')
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE.write_bytes(payload)
if hashlib.sha256(BUNDLE.read_bytes()).hexdigest() != BUNDLE_SHA:
    raise ValueError('Source bundle checksum mismatch')
with zipfile.ZipFile(BUNDLE) as package:
    assert set(package.namelist()) == {'reviewed_manifest.json','visual_review.json','train_three_actions.py','action_engine.py','person_recovery.py'}
    overlays = {name: package.read(name).decode('utf-8') for name in ['train_three_actions.py','action_engine.py','person_recovery.py']}
    reviewed = json.loads(package.read('reviewed_manifest.json'))
for name, source in overlays.items():
    (DEMO/name).write_text(source,encoding='utf-8')
MANIFEST = DATA/'three_actions_v1/reviewed_20261110_v1.json'
MANIFEST.parent.mkdir(parents=True,exist_ok=True)
encoded = json.dumps(reviewed,ensure_ascii=False,indent=2)
if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != reviewed:
    raise ValueError('Existing review manifest changed; use a new review version')
MANIFEST.write_text(encoded,encoding='utf-8')

def public_request(url):
    if not url.startswith(('https://raw.githubusercontent.com/','https://api.github.com/repos/')):
        raise ValueError('Unexpected public download host')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url,headers={'User-Agent':'three-action-reviewed-colab'})
            with urllib.request.urlopen(request,timeout=90) as response:
                return response.read(26*1024*1024)
        except (OSError,http.client.HTTPException):
            if attempt==2:
                raise
            time.sleep(attempt+1)

def prepare_source(item):
    target=(DATA/item['video']).resolve()
    assert target.is_relative_to(DATA.resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']:
        return
    try:
        content=public_request(item['download_url'])
    except (OSError,http.client.HTTPException):
        content=base64.b64decode(json.loads(public_request(item['blob_url']))['content'])
    assert len(content)==item['size'] and hashlib.sha256(content).hexdigest()==item['sha256'],item['video']
    temporary=target.with_suffix('.partial')
    temporary.write_bytes(content)
    temporary.replace(target)

with ThreadPoolExecutor(max_workers=3) as pool:
    for number,_ in enumerate(pool.map(prepare_source,reviewed['sources']),1):
        if number%5==0 or number==len(reviewed['sources']):
            print('PUBLIC DATA VERIFIED',number,'/',len(reviewed['sources']),flush=True)
subprocess.run([sys.executable,str(DEMO/'train_three_actions.py'),'--manifest',str(MANIFEST),
    '--root',str(DATA),'--output',str(RUN),'--stage','audit'],check=True)
audit=json.loads((RUN/'label_audit.json').read_text())
assert audit['status']=='ready',audit['issues']
print('AUDIT READY:',len(reviewed['sources']),'public clips,',audit['windows'],'reviewed windows; no private iPhone data')
