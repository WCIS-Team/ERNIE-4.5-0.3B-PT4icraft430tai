"""Discriminate HF 401 vs 404: does an unauthenticated per-id lookup on a NON-EXISTENT repo 401 too?"""
import json
import re
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def get(url, proxy=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'python-urllib'})
    if proxy:
        req.set_proxy(proxy, 'https')
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 'ERR', repr(e).encode()


CONTROL = [
    # definitely not a real repo id, same org -> tells us what "absent" looks like here
    'PaddlePaddle/ZZQ-NoSuch-Model-9f3a1b',
    'PaddlePaddle/ERNIE-4.5-0.3B-Instruct-ZZQ',
    # definitely real and public -> tells us what "exists+public" looks like
    'PaddlePaddle/PP-Chart2Table',
    'bert-base-uncased',
    'Qwen/Qwen3-0.6B',
    # the ones we actually care about
    'PaddlePaddle/ERNIE-4.5-0.3B-PT',
    'PaddlePaddle/ERNIE-4.5-0.3B-Instruct',
    'baidu/ERNIE-4.5-0.3B-Instruct',
    'PaddlePaddle/ERNIE-4.5-2.6B-Instruct',
    'PaddlePaddle/ERNIE-4.5-21B-A3B-PT',
]
for proxy in [None]:
    print('proxy=%s' % proxy)
    for c in CONTROL:
        st, b = get('https://huggingface.co/api/models/' + c, proxy)
        print('  %-48s HTTP=%s %s' % (c, st, '' if st == 200 else b[:70]))

print('--- also: does the org model page render ids server-side? ---')
st, b = get('https://huggingface.co/PaddlePaddle/models?search=ERNIE')
t = b.decode('utf-8', 'replace')
print('HTTP=%s bytes=%d  count(ERNIE)=%d' % (st, len(b), t.count('ERNIE')))
for m in sorted(set(re.findall(r'ERNIE[-0-9A-Za-z._]*', t)))[:40]:
    print('   %s' % m)
