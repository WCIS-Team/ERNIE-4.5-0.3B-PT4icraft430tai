"""Print the feed contract read off an adapted json, so host_gen can be pointed at a new net.

host_gen.py asserts len(pair) == len(past) over every v_id except 0 and 1. The padded ERNIE graph
carries four *unconsumed* cache declarations on top of the 36 real ones, so 40 - 36 would abort the
driver even though the forward is fine. This probe shows exactly which v_ids are producers, which are
pairable, and which are dead weight that still has to be fed.

Usage: python -X utf8 probe_feed_contract.py <adapted.json>
"""
import json
import math
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DT = {'@fp(32)': 4, '@fp(16)': 2, '@fp(64)': 8, '@si(32)': 4, '@si(64)': 8}

path = sys.argv[1]
d = json.load(open(path, encoding='utf-8'))
ops = d['ops']
print('name', d['name'], 'ops', len(ops))
first, last = ops[0], ops[-1]
print('first %s outputs %d | last %s inputs %d'
      % (first['_type_key'], len(first['outputs']), last['_type_key'], len(last['inputs'])))


def desc(o):
    return 'v%-4d %-10s %-22s %d B' % (
        o['v_id'], o['dtype']['element_dtype'], str(o['dtype']['shape']),
        DT[o['dtype']['element_dtype']] * math.prod(o['dtype']['shape']))


consumed = {i['v_id'] for op in ops[1:] for i in (op.get('inputs') or [])}
print('\n-- inputs (v_id order) --')
for o in sorted(first['outputs'], key=lambda x: x['v_id']):
    print(('   ' if o['v_id'] in consumed else 'DEAD') + ' ' + desc(o))
print('\n-- outputs --')
for o in sorted(last['inputs'], key=lambda x: x['v_id']):
    print('   ' + desc(o))
producer = {}
for op in ops:
    for o in op.get('outputs') or []:
        producer[o['v_id']] = op
print('\n-- output producers of the non-ctrl outputs --')
for o in sorted(last['inputs'], key=lambda x: x['v_id']):
    cur, depth, chain = o['v_id'], 0, []
    while depth < 30:
        op = producer.get(cur)
        if op is None:
            break
        chain.append(op['_type_key'])
        if op['_type_key'].endswith('SliceDynamic'):
            break
        if not op.get('inputs'):
            break
        cur = op['inputs'][0]['v_id']
        depth += 1
    print('  v%-4d %s -> %s' % (o['v_id'], ','.join(chain[-3:]), cur))
