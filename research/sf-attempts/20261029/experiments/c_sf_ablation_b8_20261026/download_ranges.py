"""Resume pinned Hub bytes; reuse a CDN URL and never forward the Hub token."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urljoin, urlsplit

import requests

ROOT = Path(__file__).resolve().parent
record = json.loads((ROOT/'download.json').read_text())
url = 'https://huggingface.co/'+record['repo']+'/resolve/'+record['revision']+'/'+record['filename']
token = Path('/home/ls/.cache/huggingface/token').read_text().strip()
directory = ROOT/'download.parts'
directory.mkdir(exist_ok=True)
chunk, size = 16*1024*1024, record['bytes']
lock = threading.Lock()
location_cache = ROOT/'download_cdn.private.json'
location = None
if location_cache.exists():
    cached = json.loads(location_cache.read_text())
    if time.time()-cached['saved_at'] < 1800:
        location = cached['url']


def log(**fields):
    print(json.dumps(fields), flush=True)


def get_location():
    global location
    with lock:
        if location:
            return location
        attempt = 0
        while True:
            attempt += 1
            try:
                # Only the official Hub receives authentication. HEAD has no
                # byte range, so the signed CDN location covers the full file.
                with requests.head(url, headers={'Authorization':'Bearer '+token},
                                   allow_redirects=False, timeout=(15,30)) as response:
                    response.raise_for_status()
                    if response.status_code not in (301,302,303,307,308):
                        raise RuntimeError('expected the official Hub to redirect to its CDN')
                    candidate = urljoin(url,response.headers['Location'])
                    if urlsplit(candidate).scheme != 'https' or urlsplit(candidate).hostname == 'huggingface.co':
                        raise RuntimeError('expected an HTTPS CDN download URL')
                location = candidate
                fd = os.open(location_cache,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
                with os.fdopen(fd,'w') as target:
                    json.dump({'url':location,'saved_at':time.time()},target)
                log(cdn_ready=True,host=urlsplit(location).hostname)
                return location
            except requests.RequestException as error:
                log(metadata_attempt=attempt,retry=type(error).__name__)
                time.sleep(min(15,attempt*2))


def receive(index):
    global location
    start,end=index*chunk,min(size,(index+1)*chunk)-1
    path=directory/f'{index:05d}.part'
    if path.exists() and path.stat().st_size==end-start+1:
        return index
    temporary=path.with_suffix('.partial')
    attempt=0
    # Persistent sessions reuse CDN connections; independent sessions per worker.
    session=requests.Session()
    session.trust_env=False  # Official CDN direct path passed a real 1MiB Range probe.
    while True:
        offset=temporary.stat().st_size if temporary.exists() else 0
        if offset==end-start+1:
            temporary.replace(path)
            return index
        attempt+=1
        target_url=get_location()
        try:
            with session.get(target_url,headers={'Range':f'bytes={start+offset}-{end}'},
                             stream=True,timeout=(15,90)) as response:
                if response.status_code in (401,403):
                    with lock:
                        if location==target_url:
                            location=None
                    raise requests.HTTPError('expired CDN URL')
                response.raise_for_status()
                expected=f'bytes {start+offset}-{end}/{size}'
                if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                    raise RuntimeError('CDN did not honor the exact requested range')
                with temporary.open('ab') as target:
                    for data in response.iter_content(1024*1024):
                        target.write(data)
            if temporary.stat().st_size!=end-start+1:
                raise RuntimeError('partial range response')
            temporary.replace(path)
            return index
        except requests.RequestException as error:
            # Do not print exception strings: they can expose signed URLs.
            log(range=index,attempt=attempt,retry=type(error).__name__)
            time.sleep(min(15,attempt*2))


count=(size+chunk-1)//chunk
started=time.monotonic()
log(status='resuming',workers=4,received_bytes=sum(p.stat().st_size for p in directory.glob('*.part'))+
    sum(p.stat().st_size for p in directory.glob('*.partial')),total=size)
with ThreadPoolExecutor(max_workers=4) as pool:
    order=[0,count-1,*range(1,count-1)]
    for future in as_completed([pool.submit(receive,index) for index in order]):
        index=future.result()
        complete=sum(path.stat().st_size for path in directory.glob('*.part'))
        log(completed_range=index,bytes=complete,total=size,elapsed=round(time.monotonic()-started,1))

destination=Path('/data2/ls/checkpoints/ditic-50k-warmups')/record['filename']
destination.parent.mkdir(parents=True,exist_ok=True)
temporary=destination.with_suffix('.pt.partial')
digest=hashlib.sha256()
with temporary.open('wb') as target:
    for index in range(count):
        with (directory/f'{index:05d}.part').open('rb') as source:
            for data in iter(lambda:source.read(8*1024*1024),b''):
                target.write(data)
                digest.update(data)
    target.flush()
    os.fsync(target.fileno())
if temporary.stat().st_size!=size or digest.hexdigest()!=record['expected_sha256']:
    raise RuntimeError('checkpoint hash/size differs from pinned Hub metadata')
temporary.replace(destination)
record.update(status='verified',path=str(destination),actual_sha256=digest.hexdigest(),completed_at=time.time())
(ROOT/'download.json').write_text(json.dumps(record,indent=2)+'\n')
log(**record)
