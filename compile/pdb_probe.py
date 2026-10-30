"""Read the vendor parser's own debug symbols to find where LLM cache inputs get registered.

The parse logs show the whole discriminator in one number: the op-0 InputNode emits 58 network tensors
for a 28-layer graph and 2 for an 18-layer one, and then the lowering dies entering the first
GroupQueryAttention. Nothing in the public docs explains what decides that, and NetParam (docs
parse/_cppapi) carries no layer-count field. But the install ships .pdb files, so the object-file list
and mangled symbol names are readable: which source file owns InputNode lowering, which owns the
GroupQueryAttention expansion, and what functions they call. Print the non-opset parser sources and any
symbol mentioning cache / llm / input / kv.

Usage: python -X utf8 pdb_probe.py [pdb_path]
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BIN = r'C:\Icraft\CLI v3.39.0\bin'
p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BIN, 'icraft_onnx_parser.pdb')
data = io.open(p, 'rb').read()
print('== %s  %d bytes' % (os.path.basename(p), len(data)))

runs = re.findall(rb'[\x20-\x7e]{6,}', data)
uniq = sorted({r.decode('ascii', 'replace') for r in runs})

objs = sorted({s.rsplit('\\', 1)[-1] for s in uniq if s.endswith('.cpp.obj')})
special = [o for o in objs if 'opset' not in o]
print('cpp objects %d ; non-opset %d' % (len(objs), len(special)))
for o in special:
    print('   ', o)

print('== symbols mentioning cache / kv / llm / networkinput / past ==')
pat = re.compile(r'(?i)(cache|llm|network_?input|past|kv|decoder|prefill)')
seen = set()
for s in uniq:
    if not pat.search(s):
        continue
    if s.startswith('.?AV') or s.startswith('??'):
        pass
    key = s[:200]
    if key in seen:
        continue
    seen.add(key)
    if len(s) < 240:
        print('   ', s)
print('   (%d distinct)' % len(seen))
