"""Find which shipped binary actually recognises the past_key_values / LLM-cache input names.

The onnx parser DLL contains no 'past_key_values' string at all, so the name convention must be matched
somewhere else -- and the string right next to it tells us what the code believes about it. Walk every
DLL/EXE under the CLI bin dir, pull the printable runs, and print any run containing one of the probe
substrings together with the file that carried it. Also print format-string-looking runs ('%d', '%s')
that mention cache/input so a hardcoded count or index range shows up.

Usage: python -X utf8 dll_strings.py past_key_values network_input
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BIN = r'C:\Icraft\CLI v3.39.0\bin'
probes = sys.argv[1:] or ['past_key_values', 'network_input', 'llm_mode']
RUN = re.compile(rb'[\x20-\x7e]{5,}')

files = [f for f in sorted(os.listdir(BIN))
         if os.path.isfile(os.path.join(BIN, f)) and f.lower().endswith(('.dll', '.exe'))
         and (f.lower().startswith(('icraft', 'xir', 'ox', 'parse')) or 'parse' in f.lower())]
print('%d icraft dll/exe under %s' % (len(files), BIN))
for f in files:
    if 'cuda' in f.lower() or 'cudnn' in f.lower() or f.lower().startswith(('npp', 'cublas', 'cusparse',
                                                                           'cufft', 'curand', 'cudart')):
        continue
    p = os.path.join(BIN, f)
    try:
        data = io.open(p, 'rb').read()
    except Exception as e:
        print('  skip %s (%s)' % (f, e))
        continue
    for probe in probes:
        hits = set()
        for r in RUN.findall(data):
            s = r.decode('ascii', 'replace')
            if probe in s.lower() or probe.lower() in s:
                hits.add(s[:220])
        if hits:
            print('== %s  [%s]  %d hits' % (f, probe, len(hits)))
            for h in sorted(hits)[:20]:
                print('     ', h)
