
import numpy as np
import pandas as pd
from glob import glob
import os.path as osp
import os

base_path = os.getcwd()
print(base_path)
def pd_concat():
    ori_template = f"{base_path}/../src/output_csv/ori/Eval_out_ori_?.csv"
    tim_template = f"{base_path}/../src/output_csv/tim/Eval_out_tim_?.csv"
    ori_paths = [i for i in glob(ori_template)]
    tim_paths = [i for i in glob(tim_template)]
    
    chart_list = []
    for p in ori_paths:
        chart_list.append(pd.read_csv(p))
    chart = pd.concat(chart_list,axis=0).reset_index(drop=True)
    chart.to_csv(f"{base_path}/../src/output_csv/ori/Eval_out_ori.csv",index=False)
    
    chart_list = []
    for p in tim_paths:
        chart_list.append(pd.read_csv(p))
    chart = pd.concat(chart_list,axis=0).reset_index(drop=True)
    chart.to_csv(f"{base_path}/../src/output_csv/tim/Eval_out_tim.csv",index=False)

pd_concat()