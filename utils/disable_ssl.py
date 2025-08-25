# utils/disable_ssl.py

import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

class UnsafeAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl._create_unverified_context()
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        context = ssl._create_unverified_context()
        kwargs['ssl_context'] = context
        return super().proxy_manager_for(*args, **kwargs)

def patch_global_requests():
    """Monkey-patch requests globally to disable SSL verification."""
    adapter = UnsafeAdapter()
    session = requests.Session()
    session.mount("https://", adapter)
    requests.Session = lambda: session  # Override the default Session constructor
