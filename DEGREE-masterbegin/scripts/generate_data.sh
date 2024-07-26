#!/bin/bash

# 运行大模型生成样本
python preprocessing/prompt.py

# 初步处理
python preprocessing/process.py

# 将生成的样本格式换成我们需要的
# python preprocessing/process_databyLLM.py

bash scripts/process_ace05e_process.sh

