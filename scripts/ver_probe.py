import sys, importlib.metadata as md
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print('python=%s' % sys.version.split()[0])
for p in ('transformers','torch','onnx','onnxruntime','onnxruntime-genai','modelscope','numpy'):
    try:
        print('%s=%s' % (p, md.version(p)))
    except Exception as e:
        print('%s=ABSENT(%s)' % (p, type(e).__name__))
import onnxruntime as ort
print('ort_providers=%s' % ort.get_available_providers())
