"""Authoritative check of the ERNIE 4.5 lineup: official GitHub README + baidu-org HF control test."""
import json
import re
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def get(url, proxy=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
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


print('##### 1. baidu org on HF: control test #####')
for c in ['baidu/ZZQ-NoSuch-Model-9f3a1b', 'baidu/ERNIE-4.5-0.3B-PT',
          'baidu/ERNIE-4.5-0.3B-Instruct', 'baidu/ERNIE-4.5-2.6B-PT',
          'baidu/ERNIE-4.5-21B-A3B-PT', 'baidu/ERNIE-4.5-300B-A47B-PT']:
    st, b = get('https://huggingface.co/api/models/' + c)
    print('  %-38s HTTP=%s %s' % (c, st, '' if st == 200 else b[:60]))

print('##### 2. baidu org listing (does it enumerate ERNIE?) #####')
for proxy in [None, '127.0.0.1:7897']:
    st, b = get('https://huggingface.co/api/models?author=baidu&limit=1000', proxy)
    print('  proxy=%s HTTP=%s bytes=%d' % (proxy, st, len(b)))
    if st == 200:
        arr = json.loads(b.decode('utf-8', 'replace'))
        ids = sorted(m.get('id') for m in arr)
        print('  author=baidu count=%d  containing ERNIE=%d' % (len(ids), len([i for i in ids if 'ERNIE' in i.upper()])))
        for i in ids:
            if 'ERNIE' in i.upper():
                print('    %s' % i)
        break

print('##### 3. official repo: github.com/PaddlePaddle/ERNIE README #####')
for proxy in [None, '127.0.0.1:7897']:
    st, b = get('https://raw.githubusercontent.com/PaddlePaddle/ERNIE/main/README.md', proxy)
    print('  proxy=%s HTTP=%s bytes=%d' % (proxy, st, len(b)))
    if st == 200:
        t = b.decode('utf-8', 'replace')
        open(r'S:\work\WCIS\workenv\ernie\gh_ernie_readme.md', 'w', encoding='utf-8').write(t)
        print('  count(ERNIE-4.5-0.3B)=%d count(Instruct)=%d count(-PT)=%d'
              % (t.count('ERNIE-4.5-0.3B'), len(re.findall(r'(?i)instruct', t)), t.count('-PT')))
        ids = sorted(set(re.findall(r'(?:baidu|PaddlePaddle)/ERNIE[-0-9A-Za-z._]*', t)))
        print('  distinct model ids (%d):' % len(ids))
        for i in ids:
            print('    %s' % i)
        print('  --- lines with 0.3B ---')
        for ln in t.splitlines():
            if '0.3B' in ln:
                print('    %s' % ln.strip()[:200])
        break
    print('  %r' % b[:100])
