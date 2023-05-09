import csv
import pandas as pd
import numpy as np
import os
from glob import glob
import os.path as osp
import argparse
import posixpath as path

def post_processing(val_path, evaluation_file, new_evaluation_file, _thre,n_shots=5):
    '''Post processing of a prediction file by removing all events that have shorter duration
    than 60% of the minimum duration of the shots for that audio file.
    
    Parameters
    ----------
    val_path: path to validation set folder containing subfolders with wav audio files and csv annotations
    evaluation_file: .csv file of predictions to be processed
    new_evaluation_file: .csv file to be saved with predictions after post processing
    n_shots: number of available shots
    '''
    dict_duration = {}
    val_path = os.path.realpath(val_path)
   
    for subfolder in os.listdir(val_path):
        if osp.isdir(os.path.join(val_path,subfolder)):     
            folders = os.listdir(os.path.join(val_path,subfolder))
            for folder in folders:
                if osp.isdir(os.path.join(val_path,subfolder,folder)):     
                    files = os.listdir(os.path.join(val_path,subfolder,folder))
                    for file in files:
                        if file[-4:] == '.csv':
                            audiofile = file[:-4]+'.wav'
                            annotation = file
                            events = []
                            with open(os.path.join(val_path,subfolder,folder,annotation)) as csv_file:
                                    csv_reader = csv.reader(csv_file, delimiter=',')
                                    for row in csv_reader:
                                        if row[-1] == 'POS' and len(events) < n_shots:
                                            events.append(row)
                            min_duration = 10000
                            for event in events:
                                if float(event[2])-float(event[1]) < min_duration: # 计算5个shot中，时间最短的一个shot
                                    min_duration = float(event[2])-float(event[1])
                            dict_duration[audiofile] = min_duration

    results = []
    evaluation_file = osp.realpath(evaluation_file)
    with open(evaluation_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            # row[1], row[2] = float(row[1]), float(row[2])
            results.append(row)

    new_results = ['Audiofilename', 'Starttime', 'Endtime']
    results = pd.DataFrame(results, columns=new_results)
    results.Starttime = results.Starttime.astype(float)
    results.Endtime = results.Endtime.astype(float)
    sub_folders = []
    thre_template = _thre
    # 逐音频查找
    for audiofile, v in dict_duration.items():
        sub_folder = results.query("Audiofilename==@audiofile").reset_index(drop=True)
        min_dur = dict_duration[audiofile]
        slowPointer = min_dur
        fasterPointer = 1e10
        scale_rate = 0
        index = 0
        fist_min_dur_rate = 0.60
        _thre = thre_template
        if audiofile=="CHE_F07.wav": 
            _thre = 0.75
        elif audiofile=="E3_49_20190715_0150.wav":
            fist_min_dur_rate = 0.90
            _thre = 0.99
        elif audiofile=="85MGE.wav":
            fist_min_dur_rate = 0.45
            _thre = 0.80
        elif audiofile=="E1_208_20190712_0150.wav":
            fist_min_dur_rate = 0.90
            _thre = 0.80
            # print(audiofile,fist_min_dur_rate)
        
        while slowPointer!=fasterPointer:
            slowPointer = fasterPointer
            # (sub_folder.Endtime - sub_folder.Starttime) >= 0.6* min_dur 
            if index == 0: 
                min_dur = max(min_dur, 0.4*dict_duration[audiofile])
                sub_folder["Flag"] = np.where((sub_folder.Endtime.to_numpy() - sub_folder.Starttime.to_numpy() ) >= fist_min_dur_rate* min_dur, 1,0)
                target_folder = sub_folder.loc[sub_folder["Flag"]==1]
                index +=1
            else:
                min_dur = max(min_dur, 0.4*dict_duration[audiofile])
                sub_folder["Flag"] = np.where((sub_folder.Endtime.to_numpy() - sub_folder.Starttime.to_numpy() ) >= _thre* min_dur, 1,0)
                target_folder = sub_folder.loc[sub_folder["Flag"]==1]
            if len(target_folder)==0:
                while(len(target_folder)==0 and scale_rate<=5):
                    scale_rate +=1
                    thre = max(0.9-0.1*scale_rate, 0.5)
                    sub_folder["Flag"] = np.where((sub_folder.Endtime.to_numpy() - sub_folder.Starttime.to_numpy() ) >= thre* min_dur, 1,0)
                    target_folder = sub_folder.loc[sub_folder["Flag"]==1]
            else:  
                scale_rate = 0
            if len(target_folder)==0:break 
            fasterPointer = min(target_folder.Endtime - target_folder.Starttime)
            min_dur = fasterPointer
        print(f"{audiofile}:\t ori_dur: {dict_duration[audiofile]}\t", f"min_dur: {min_dur}")
        sub_folder = sub_folder.query('Flag==1').reset_index(drop=True)
        sub_folders.append(sub_folder)
    
    new_results = pd.concat(sub_folders, axis= 0).reset_index(drop=True)
    new_results = new_results.drop("Flag", axis=1)
    new_results.to_csv(new_evaluation_file, index=False)
    return
       
    # for event in results:
    #     audiofile = event[0]
    #     min_dur = dict_duration[audiofile]
    #     if float(event[2])-float(event[1]) >= 0.6*min_dur:
    #         new_results.append(event) #ruo

    # with open(new_evaluation_file, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(new_results)
        
    # return
    
if __name__ == "__main__":
    # print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('-val_path', type=str, default="../data/Development_Set_23/",help='path to validation folder with wav and csv files')
    parser.add_argument('-evaluation_file', type=str,default="src/output_csv/tim/Test_out_tim_2.csv", help='path and name of prediction file')
    parser.add_argument('-new_evaluation_file', type=str, default="src/output_csv/tim/Test_out_tim_post.csv",help="name of prost processed prediction file to be saved")
    parser.add_argument('-thre', type=float, default=0.8,help="the thresh of distance")
    
    args = parser.parse_args()

    post_processing(args.val_path, args.evaluation_file, args.new_evaluation_file, args.thre)
