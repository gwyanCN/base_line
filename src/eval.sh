# #!/bin/bash
export CUDA_VISIBLE_DEVICES=0
nohup python -u main.py set.train=false set.eval=true  > eval1.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=0
# nohup python -u main2.py set.train=false set.eval=true > eval2.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=1
# nohup python -u main3.py set.train=false set.eval=true  > eval3.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=1
# nohup python -u main4.py set.train=false set.eval=true  > eval4.log 2>&1 &