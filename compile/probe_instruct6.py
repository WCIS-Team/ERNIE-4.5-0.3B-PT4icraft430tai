"""Settle whether an ERNIE-4.5-0.3B Instruct/sibling exists on ModelScope:
raw search-page HTML + candidate per-id probes + the PT model card text. Counts are done by us."""
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def get(url, data=None, ctype=None, timeout=30):
    h = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    if ctype:
        h['Content-Type'] = ctype
    req = urllib.request.Request(url, data=data, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 'ERR', repr(e).encode()


print('##### A. search page HTML (raw, counted here) #####')
for q in ['ERNIE-4.5', 'ERNIE']:
    st, b = get('https://modelscope.cn/search?search=%s' % urllib.parse.quote(q))
    print('GET /search?search=%s HTTP=%s bytes=%d' % (q, st, len(b)))
    if st == 200:
        t = b.decode('utf-8', 'replace')
        open(r'S:\work\WCIS\workenv\ernie\ms_search_%s.html' % q.replace('.', '_'), 'w',
             encoding='utf-8').write(t)
        hits = sorted(set(re.findall(r'ERNIE[-0-9A-Za-z._]*', t)))
        print('  distinct ERNIE substrings=%d -> %s' % (len(hits), hits[:60]))
        print('  count("ERNIE-4.5-0.3B")=%d count("Instruct")=%d count("ernie")=%d'
              % (t.count('ERNIE-4.5-0.3B'), t.count('Instruct'), t.lower().count('ernie')))

print('##### B. candidate per-id probes #####')
CAND = [
    'PaddlePaddle/ERNIE-4.5-0.3B-PT',
    'PaddlePaddle/ERNIE-4.5-0.3B-Instruct',
    'PaddlePaddle/ERNIE-4.5-0.3B-Base',
    'PaddlePaddle/ERNIE-4.5-0.3B-Chat',
    'PaddlePaddle/ERNIE-4.5-0.3B',
    'PaddlePaddle/ERNIE-4.5-2.6B-PT',
    'PaddlePaddle/ERNIE-4.5-2.6B-Instruct',
    'PaddlePaddle/ERNIE-4.5-21B-A3B-PT',
    'PaddlePaddle/ERNIE-4.5-0.3B-PT-GGUF',
    'PaddlePaddle/ERNIE-4.5-0.3B-PT-online',
    'PaddlePaddle/ERNIE-4.5-0.3B-Instruct-GGUF',
    'onnxruntime-genai/ERNIE-4.5-0.3B-PT-onnx',
    'PaddlePaddle/ERNIE-Speed-0.3B',
    'PaddlePaddle/ERNIE-CT-0.3B',
]
for c in CAND:
    st, b = get('https://modelscope.cn/api/v1/models/%s' % c)
    mark = 'EXISTS' if st == 200 else 'absent'
    print('  %-50s HTTP=%s %s' % (c, st, mark))

print('##### C. PT model card (README.md from repo) #####')
st, b = get('https://modelscope.cn/api/v1/models/PaddlePaddle/ERNIE-4.5-0.3B-PT/repo'
            '?Revision=master&FilePath=README.md')
print('HTTP=%s bytes=%d' % (st, len(b)))
if st == 200:
    t = b.decode('utf-8', 'replace')
    open(r'S:\work\WCIS\workenv\ernie\ms_pt_readme.md', 'w', encoding='utf-8').write(t)
    sib = sorted(set(re.findall(r'PaddlePaddle/ERNIE[-0-9A-Za-z._]*', t)))
    print('  sibling ids mentioned: %s' % sib)
    print('  lines mentioning Instruct: %d' % len(re.findall(r'(?i)instruct', t)))
    for m in re.finditer(r'(?i).{0,90}instruct.{0,90}', t):
        print('    | %s' % m.group(0).replace('\n', ' ')[:180])
