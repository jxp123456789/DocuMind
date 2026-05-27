Set-Location $PSScriptRoot
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_OFFLINE = "1"
& "C:\Users\jin\miniconda3\envs\env_1\python.exe" main.py
