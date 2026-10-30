$ErrorActionPreference = 'Continue'
$py  = 'D:\yzd\WCIS\MOD\qwen3\venv\Scripts\python.exe'
$log = 'D:\yzd\WCIS\MOD\ernie\export_a.log'
Set-Location 'D:\yzd\WCIS\MOD\ernie'
& $py -m onnxruntime_genai.models.builder `
  -i 'D:\yzd\WCIS\MOD\ernie\ERNIE-4.5-0.3B-PT' `
  -o 'D:\yzd\WCIS\MOD\ernie\ernie_fp16_cuda' `
  -p fp16 -e cuda `
  -c 'D:\yzd\WCIS\MOD\ernie\cache_dir' 2>&1 | Tee-Object -FilePath $log
"EXIT=$LASTEXITCODE" | Out-File -FilePath $log -Append -Encoding utf8
