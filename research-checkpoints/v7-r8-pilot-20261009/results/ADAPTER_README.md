Experimental final-step v7 r8 pilot adapter; behavioral quality on unseen users has not been established.

Base: Qwen/Qwen3.5-2B, revision 15852e8c16360a2fea060d615a32b45270f8a8fc. Dataset: v7-20261009-r8-dev, 200-conversation pilot.

Dependencies: existing CUDA torch 2.11.0+cu128, transformers 5.15.0, peft 0.20.0, safetensors 0.8.0; pinned r8 application/data dependencies. Extract into a new directory. Use the verified r8 code bundle.

PowerShell (replace the adapter directory with the extracted path):
```powershell
$env:HF_HOME="D:\SLM\hf-cache"
$env:SLM_FPY_SRC="D:\SLM\v7-approved-policy-20261009-r8-review\runtime\fpy-v6-pinned\src"
& D:\SLM\SLM-FineTuning-Testing\.venv\Scripts\python.exe D:\SLM\v7-approved-policy-20261009-r8-review\scripts\chat-peft.py --variant custom --adapter .\best-adapter --adapter-manifest .\adapter-files.json --base-model Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --prompt-version v7 --device cuda --dtype bfloat16 --max-new-tokens 1024
```
