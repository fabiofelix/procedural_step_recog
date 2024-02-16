import os
import sys
import torch
from typing import Optional
from urllib.parse import urlparse  # noqa: F401
from torch.hub import download_url_to_file, get_dir


SUPPORTED_PROTOCOLS = ['http://', 'https://']

def cached_download_file(
    url: str,
    model_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    progress: bool = True
) -> str:
    '''Download and cache pytorch checkpoints (or any file, really) from a url.
    
    '''
    if not url or not any(url.startswith(prefix) for prefix in SUPPORTED_PROTOCOLS):
        return url
    model_dir = model_dir or os.path.join(get_dir(), 'checkpoints', __name__.split('.')[0])
    os.makedirs(model_dir, exist_ok=True)
    cached_file = os.path.join(model_dir, file_name or os.path.basename(urlparse(url).path))
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file