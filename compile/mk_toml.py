"""Clone a parse toml for another fmodel dir under a new net_name.

The five sections carry the net name and the jr_path in their file names, so a plain text rewrite
of the base toml is enough. Diagnostic variants (e_v2_ri0 etc.) each need their own toml because
icraft derives the log/artifact names from net_name.
"""
import argparse
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.join('S:', os.sep, 'apps', 'llm', 'agentwork', 'ernie', 'compile')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='E2_parse.toml')
    ap.add_argument('--name', required=True)
    ap.add_argument('--fm', required=True, help='fmodel subdir, e.g. e_v2_ri0')
    ap.add_argument('--prompt', type=int, default=0, help='override max_prompt_length')
    ap.add_argument('--tokens', type=int, default=0, help='override max_tokens_length')
    args = ap.parse_args()

    base = os.path.join(ROOT, args.base)
    t = open(base, encoding='utf-8').read()
    old_name = [l.split("'")[1] for l in t.splitlines() if l.startswith('net_name')][0]
    old_fm = os.path.dirname([l.split("'")[1] for l in t.splitlines() if l.startswith('network =')][0]) + '/'
    t = t.replace("net_name = '%s'" % old_name, "net_name = '%s'" % args.name)
    t = t.replace('/%s-' % old_name, '/%s-' % args.name)
    t = t.replace(old_fm, './fmodel/%s/' % args.fm)
    t = t.replace("./jr_path/%s/" % old_name, "./jr_path/%s/" % args.name)
    if args.prompt:
        t = re.sub(r'(?m)^max_prompt_length = \d+$', 'max_prompt_length = %d' % args.prompt, t)
    if args.tokens:
        t = re.sub(r'(?m)^max_tokens_length = \d+$', 'max_tokens_length = %d' % args.tokens, t)
    out = os.path.join(ROOT, '%s_parse.toml' % args.name)
    open(out, 'w', encoding='utf-8', newline='\n').write(t)
    print('base=%s name=%s->%s fm=%s->%s' % (args.base, old_name, args.name, old_fm, args.fm))
    print('wrote', out, os.path.getsize(out), 'bytes')


if __name__ == '__main__':
    main()
