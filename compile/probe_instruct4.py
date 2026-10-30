"""Direct per-id lookups on HF + sample of author listing, to settle which ERNIE-4.5-0.3B variants exist."""
import json
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def get(url, proxy=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'python-urllib'})
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


st, b = get('https://huggingface.co/api/models?author=PaddlePaddle&limit=1000')
arr = json.loads(b.decode('utf-8', 'replace'))
ids = sorted(m.get('id') for m in arr)
print('author listing count=%d' % len(ids))
print('sample first 8: %s' % ids[:8])
print('any containing ernie (lower): %s' % [i for i in ids if 'ernie' in i.lower()])
print('any containing 4.5: %s' % [i for i in ids if '4.5' in i])

print('--- per-id direct lookups on HF ---')
CAND = [
    'PaddlePaddle/ERNIE-4.5-0.3B-PT',
    'PaddlePaddle/ERNIE-4.5-0.3B-Instruct',
    'PaddlePaddle/ERNIE-4.5-0.3B',
    'PaddlePaddle/ernie-4.5-0.3b-pt',
    'PaddlePaddle/ERNIE-4.5-2.6B-PT',
    'baidu/ERNIE-4.5-0.3B-Instruct',
]
for c in CAND:
    st, b = get('https://huggingface.co/api/models/' + c)
    line = '  %-46s HTTP=%s' % (c, st)
    if st == 200:
        d = json.loads(b.decode('utf-8', 'replace'))
        line += '  id=%s dl=%s up=%s' % (d.get('modelId'), d.get('downloads'), d.get('lastModified'))
    else:
        line += ' %r' % b[:60]
    print(line)
