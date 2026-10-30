"""Probe ModelScope for the ERNIE-4.5-0.3B Instruct variant (existence + file list + sizes)."""
import json
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

IDS = [
    'PaddlePaddle/ERNIE-4.5-0.3B-Instruct',
    'PaddlePaddle/ERNIE-4.5-0.3B-PT',
    'PaddlePaddle/ERNIE-4.5-0.3B',
    'PaddlePaddle/ERNIE-4.5-0.3B-Base',
]
BASE = 'https://modelscope.cn/api/v1/models/%s'


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'python-urllib'})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 'ERR', repr(e).encode()


for mid in IDS:
    st, body = get(BASE % mid)
    print('=== %s  HTTP=%s' % (mid, st))
    if st != 200:
        print('   body[:160]=%r' % body[:160])
        continue
    d = json.loads(body.decode('utf-8', 'replace'))
    m = d.get('Data') or {}
    print('   Name=%r  CreatedTime=%r  LastUpdatedTime=%r' % (
        m.get('Name'), m.get('CreatedTime'), m.get('LastUpdatedTime')))
    print('   Architecture=%r  Downloads=%r  Tags=%r' % (
        m.get('Architecture'), m.get('Downloads'), [t.get('label') for t in (m.get('Tags') or [])]))
    st2, b2 = get(BASE % mid + '/repo/files?Revision=master')
    print('   files HTTP=%s' % st2)
    if st2 == 200:
        fl = (json.loads(b2.decode('utf-8', 'replace')).get('Data') or {}).get('Files') or []
        for f in sorted(fl, key=lambda x: x.get('Size') or 0, reverse=True)[:14]:
            print('     %-46s %14d' % (f.get('Path'), f.get('Size') or 0))
        print('     count=%d total=%d' % (len(fl), sum(f.get('Size') or 0 for f in fl)))
