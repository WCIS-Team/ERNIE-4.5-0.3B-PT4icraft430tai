"""Print the code-literal strings inside the vendor onnx parser DLL.

The parse log proves the discriminator is which graph inputs get registered as network inputs, and the
dll has no 'past_key_values' literal at all, so whatever the parser matches on must be one of its other
string literals -- operand-slot keys, pattern names, attribute names, or a hardcoded index/count in an
error message. Dump every printable run that looks like a source-level literal (short, contains an
underscore or a space or a printf spec) and no C++ mangling, so the GQA expansion's vocabulary is on one
screen.

Usage: python -X utf8 dll_literals.py [dll_path]
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    r'C:', os.sep, 'Icraft', 'CLI v3.39.0', 'bin', 'icraft_onnx_parser.dll')
data = io.open(p, 'rb').read()
runs = sorted({r.decode('ascii', 'replace') for r in re.findall(rb'[\x20-\x7e]{4,}', data)})
print('%s : %d bytes, %d distinct printable runs' % (os.path.basename(p), len(data), len(runs)))


def literal(x):
    if x.startswith('.?') or x.startswith('??') or x.startswith('@'):
        return False
    if 'protobuf' in x or 'std@@' in x or 'onnx@@' in x or 'icraft@@' in x:
        return False
    if x.startswith('auto __cdecl') or x.startswith('type_info'):
        return False
    if re.search(r'%[a-z]', x):
        return True
    if ('_' in x or ' ' in x) and len(x) < 62:
        return True
    return False


out = [x for x in runs if literal(x)]
print('literal-like: %d' % len(out))
for x in out:
    print('   ', x)
