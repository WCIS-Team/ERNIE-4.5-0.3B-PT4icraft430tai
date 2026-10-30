import os, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('MODELSCOPE_CACHE', r'D:\yzd\WCIS\MOD\ernie\ms_cache')
from modelscope import snapshot_download

t0 = time.time()
dst = r'D:\yzd\WCIS\MOD\ernie\ERNIE-4.5-0.3B-PT'
p = snapshot_download('PaddlePaddle/ERNIE-4.5-0.3B-PT', local_dir=dst, allow_patterns=[
    'config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json',
    'tokenizer.model', 'vocab.json', 'added_tokens.json', 'special_tokens_map.json',
    'chat_template.jinja', 'chat_template.json',
    '*.safetensors', 'model.safetensors.index.json', 'README.md', 'LICENSE', 'configuration.json',
])
print('DOWNLOADED_TO', p)
tot = 0
for root, dirs, files in os.walk(dst):
    for f in files:
        fp = os.path.join(root, f)
        sz = os.path.getsize(fp)
        tot += sz
        print('FILE', os.path.relpath(fp, dst).replace('\\', '/'), sz)
print('TOTAL_MB', round(tot / 2**20, 3), 'SECS', round(time.time() - t0, 1))
