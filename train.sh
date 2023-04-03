# #!/bin/bash
export CUDA_VISIBLE_DEVICES=1
nohup python -u src/main.py set.train=true set.eval=true > src/train.log 2>&1 &