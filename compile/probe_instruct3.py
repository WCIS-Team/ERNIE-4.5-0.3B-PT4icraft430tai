"""List PaddlePaddle's HF models via author endpoint (no unreliable search param) and filter ERNIE."""
import json
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def get(url, proxy=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'python-urllib', 'Accept': 'application/json'})
    if proxy:
        req.set_proxy(proxy, 'https')
        req.set_proxy(proxy, 'http')
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 'ERR', repr(e).encode()


for proxy in [None, '127.0.0.1:7897']:
    url = 'https://huggingface.co/api/models?author=PaddlePaddle&full=false&limit=1000'
    st, b = get(url, proxy)
    print('proxy=%s HTTP=%s bytes=%d' % (proxy, st, len(b)))
    if st != 200:
        print('  %r' % b[:200])
        continue
    arr = json.loads(b.decode('utf-8', 'replace'))
    print('  total models by author=%d' % len(arr))
    ern = [m for m in arr if 'ERNIE' in (m.get('id') or '').upper()]
    print('  ids containing ERNIE=%d' % len(ern))
    for m in sorted(ern, key=lambda x: x.get('id')):
        print('    %-56s dl=%-8s tags=%s' % (
            m.get('id'), m.get('downloads'),
            [t for t in (m.get('tags') or []) if t.startswith(('license', 'arxiv', 'region'))][:3]))
    small = [m.get('id') for m in ern if '0.3B' in (m.get('id') or '')]
    print('  0.3B variants=%r' % small)
    break
