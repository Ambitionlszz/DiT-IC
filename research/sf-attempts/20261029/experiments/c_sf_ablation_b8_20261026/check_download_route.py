import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from huggingface_hub import get_hf_file_metadata, hf_hub_url
import requests

token = Path('/home/ls/.cache/huggingface/token').read_text().strip()
url = hf_hub_url('Ambitionls/ditic-50k-warmups',
                'c_precision_v2_lambda16_C_joint10k_20261023/0010000.pt',
                revision='3e96cf9492dd5ce86a3fc72f6aa4c9269209f531')
response = requests.get(url, headers={'Authorization': 'Bearer '+token, 'Range': 'bytes=0-1023'},
                        stream=True, timeout=(15, 30))
response.raise_for_status()
location = response.url
print(json.dumps(dict(download_host=urlsplit(location).hostname, status=response.status_code,
                      content_range=response.headers.get('Content-Range'))), flush=True)
response.close()
session = requests.Session()
session.trust_env = False
started = time.monotonic()
try:
    response = session.get(location, headers={'Range': 'bytes=0-1048575'},
                           stream=True, timeout=(5, 15))
    response.raise_for_status()
    data = next(response.iter_content(1024*1024))
    print(json.dumps(dict(direct_status=response.status_code, bytes=len(data), seconds=time.monotonic()-started)), flush=True)
    response.close()
except requests.RequestException as error:
    # Exceptions can contain private signed URLs: print the class only.
    print(json.dumps(dict(direct_error=type(error).__name__, seconds=time.monotonic()-started)), flush=True)
