
import numpy as np
import pandas as pd
from glob import glob
def pd_concat():
    ori_template = "/media/b227/ygw/Dcase2023/baseline/src/output_csv/ori/Eval_out_ori_*.csv"
    tim_template = "/media/b227/ygw/Dcase2023/baseline/src/output_csv/tim/Eval_out_tim_?.csv"
    ori_paths = [i for i in glob(ori_template)]
    tim_paths = [i for i in glob(tim_template)]
    
    chart_list = []
    for p in ori_paths:
        chart_list.append(pd.read_csv(p))
    chart = pd.concat(chart_list,axis=0).reset_index(drop=True)
    chart.to_csv("/media/b227/ygw/Dcase2023/baseline/src/output_csv/ori/Eval_out_ori.csv",index=False)
    
    chart_list = []
    for p in tim_paths:
        chart_list.append(pd.read_csv(p))
    chart = pd.concat(chart_list,axis=0).reset_index(drop=True)
    chart.to_csv("/media/b227/ygw/Dcase2023/baseline/src/output_csv/tim/Eval_out_tim.csv",index=False)

pd_concat()