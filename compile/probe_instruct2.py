"""Enumerate every ERNIE-4.5-0.3B variant on ModelScope (dolphin search) and HF (api/models)."""
import json
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def get(url, data=None, headers=None, proxy=None):
    h = {'User-Agent': 'python-urllib', 'Accept': 'application/json'}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
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


print('##### ModelScope dolphin search: ERNIE-4.5 #####')
body = json.dumps({'Query': 'ERNIE-4.5', 'Targets': ['Name', 'DatasetId', 'Description', 'FromHome', 'Path'],
                   'PageSize': 50, 'PageNumber': 1, 'Sort': {'is_weight': True}}).encode()
st, b = get('https://modelscope.cn/api/v1/dolphin/models', data=body,
            headers={'Content-Type': 'application/json'})
print('HTTP=%s' % st)
if st == 200:
    d = json.loads(b.decode('utf-8', 'replace'))
    items = (d.get('Data') or {}).get('Model', {}).get('Models') or []
    print('returned=%d' % len(items))
    for m in items:
        print('  %-40s %-38s tasks=%s' % (
            m.get('ModelName') or m.get('Name'), m.get('Path') or m.get('Organization'),
            [t.get('Name') for t in (m.get('Tasks') or [])][:2]))
else:
    print('  body[:200]=%r' % b[:200])

print('##### ModelScope: candidate ids #####')
for mid in ['PaddlePaddle/ERNIE-4.5-0.3B-Instruct', 'PaddlePaddle/ERNIE-4.5-0.3B-PT',
            'ERNIE-4.5-0.3B-Instruct', 'PaddlePaddle/ERNIE-4.5-2.6B-Instruct',
            'PaddlePaddle/ERNIE-4.5-0.3B', 'PaddlePaddle/ERNIE-4.5-0.3B-PT-4bit',
            'PaddlePaddle/ERNIE-4.5-0.3B-Instruct-4bit']:
    st, b = get('https://modelscope.cn/api/v1/models/%s' % mid)
    print('  %-46s HTTP=%s %s' % (mid, st, '' if st == 200 else b[:70]))

print('##### HF api: author=PaddlePaddle search=ERNIE-4.5-0.3B #####')
for proxy in [None, '127.0.0.1:7897']:
    st, b = get('https://huggingface.co/api/models?author=PaddlePaddle&search=ERNIE-4.5-0.3B&limit=100',
                proxy=proxy)
    print('  proxy=%s HTTP=%s' % (proxy, st))
    if st == 200:
        arr = json.loads(b.decode('utf-8', 'replace'))
        print('  count=%d' % len(arr))
        for m in arr:
            print('    %-52s downloads=%s pipeline=%s' % (
                m.get('id'), m.get('downloads'), (m.get('pipeline_tag'))))
        break
    print('  body[:140]=%r' % b[:140])
