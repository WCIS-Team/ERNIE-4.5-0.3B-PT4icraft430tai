"""Enumerate ERNIE repos under the PaddlePaddle org on ModelScope (official site API, anonymous)."""
import json
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def call(url, data=None, ctype=None):
    h = {'User-Agent': 'python-urllib'}
    if ctype:
        h['Content-Type'] = ctype
    req = urllib.request.Request(url, data=data, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 'ERR', repr(e).encode()


tries = [
    ('GET', 'https://modelscope.cn/api/v1/models?Owner=PaddlePaddle&PageSize=200&PageNumber=1', None, None),
    ('GET', 'https://modelscope.cn/api/v1/models/PaddlePaddle?PageSize=200', None, None),
    ('POST', 'https://modelscope.cn/api/v1/dolphin/models', json.dumps(
        {'Query': 'ERNIE', 'Targets': ['Name'], 'PageSize': 60, 'PageNumber': 1}).encode(),
     'application/json'),
    ('GET', 'https://modelscope.cn/api/v1/models?Query=ERNIE&PageSize=200&PageNumber=1', None, None),
]
for meth, url, body, ctype in tries:
    st, b = call(url, body, ctype)
    print('=== %s %s -> HTTP=%s len=%d' % (meth, url, st, len(b)))
    if st != 200:
        print('   %r' % b[:150])
        continue
    try:
        d = json.loads(b.decode('utf-8', 'replace'))
    except Exception as e:
        print('   not json: %r' % b[:150])
        continue
    data = d.get('Data')
    print('   top keys=%r success=%s' % (list(d.keys()), d.get('Success')))
    if isinstance(data, dict):
        print('   Data keys=%r' % list(data.keys()))
        models = data.get('Model') or data.get('Models') or data
        if isinstance(models, dict):
            print('   Model keys=%r' % list(models.keys()))
            lst = models.get('Models') or models.get('Model') or []
        else:
            lst = models
        if isinstance(lst, list):
            print('   list len=%d' % len(lst))
            for m in lst[:80]:
                if isinstance(m, dict):
                    print('     %-46s %s' % (m.get('Name') or m.get('ModelName'), m.get('Path') or m.get('Namespace')))
                else:
                    print('     %r' % m)
    elif isinstance(data, list):
        print('   list len=%d' % len(data))
        for m in data[:80]:
            print('     %s' % (m.get('Name') if isinstance(m, dict) else m))
