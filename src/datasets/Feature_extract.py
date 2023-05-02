import os
import librosa
import h5py
import pandas as pd
import numpy as np
from scipy import signal
from glob import glob
from itertools import chain
from torchaudio import functional as F
import torchaudio.transforms as T   
import torch

pd.options.mode.chained_assignment = None

def create_dataset(df_pos,pcen,glob_cls_name,file_name,hf,seg_len,hop_seg,fps):

    '''Chunk the time-frequecy representation to segment length and store in h5py dataset

    Args:
        -df_pos : dataframe
        -log_mel_spec : log mel spectrogram
        -glob_cls_name: Name of the class used in audio files where only one class is present
        -file_name : Name of the csv file
        -hf: h5py object
        -seg_len : fixed segment length
        -fps: frame per second
    Out:
        - label_list: list of labels for the extracted mel patches'''

    label_list = []
    if len(hf['features'][:]) == 0:
        file_index = 0  # 类似于指针的作用
    else:
        file_index = len(hf['features'][:])


    start_time,end_time = time_2_frame(df_pos,fps) # 将开始时间，结束时间，转为开始为第几frame ,结束为第几frame
    # print('start_time, end_time ',start_time,end_time)

    'For csv files with a column name Call, pick up the global class name'

    if 'CALL' in df_pos.columns:
        cls_list = [glob_cls_name] * len(start_time)
    else:
        cls_list = [df_pos.columns[(df_pos == 'POS').loc[index]].values for index, row in df_pos.iterrows()]
        cls_list = list(chain.from_iterable(cls_list))

    assert len(start_time) == len(end_time)
    assert len(cls_list) == len(start_time)

    for index in range(len(start_time)):

        str_ind = max(0,start_time[index])
        end_ind = end_time[index]
        label = cls_list[index]  # 保证了不在同一时间发生两件事？
        # print('str_ind ',str_ind)
        # print('end_ind ',end_ind)
        # print('label ',label)
        'Extract segment and move forward with hop_seg'

        if end_ind - str_ind > seg_len: # 开始帧和结束帧之间的间隔大于 17
            shift = 0
            while end_ind - (str_ind + shift) > seg_len:

                pcen_patch = pcen[int(str_ind + shift):int(str_ind + shift + seg_len)]
                # print('int(str_ind + shift) ',int(str_ind + shift))
                # print('int(str_ind + shift + seg_len) ',int(str_ind + shift + seg_len))
                hf['features'].resize((file_index + 1, pcen_patch.shape[0], pcen_patch.shape[1]))
                hf['features'][file_index] = pcen_patch
                label_list.append(label)
                file_index += 1
                shift = shift + hop_seg # 隔 4 frame 取

            pcen_patch_last = pcen[end_ind - seg_len:end_ind] # 最后一段


            hf['features'].resize((file_index+1 , pcen_patch.shape[0], pcen_patch.shape[1]))
            hf['features'][file_index] = pcen_patch_last
            label_list.append(label)
            file_index += 1
        else: # 若间隔小于17 frame, 则需要重复，让它的长度达到17

            'If patch length is less than segment length then tile the patch multiple times till it reaches the segment length'

            pcen_patch = pcen[str_ind:end_ind]
            if pcen_patch.shape[0] == 0:
                print(pcen_patch.shape[0])
                print("The patch is of 0 length")
                continue

            repeat_num = int(seg_len / (pcen_patch.shape[0])) + 1 # 看需要重复多少次
            pcen_patch_new = np.tile(pcen_patch, (repeat_num, 1))
            pcen_patch_new = pcen_patch_new[0:int(seg_len)]
            hf['features'].resize((file_index+1, pcen_patch_new.shape[0], pcen_patch_new.shape[1]))
            hf['features'][file_index] = pcen_patch_new
            label_list.append(label)
            file_index += 1

    print("Total files created : {}".format(file_index))
    return label_list

def create_dataset_v2(df_pos,pcen,glob_cls_name,file_name,hf,seg_len,hop_seg,fps):

    '''Chunk the time-frequecy representation to segment length and store in h5py dataset

    Args:
        -df_pos : dataframe
        -log_mel_spec : log mel spectrogram
        -glob_cls_name: Name of the class used in audio files where only one class is present
        -file_name : Name of the csv file
        -hf: h5py object
        -seg_len : fixed segment length
        -fps: frame per second
    Out:
        - label_list: list of labels for the extracted mel patches'''

    label_list = []
    if len(hf['features'][:]) == 0:
        file_index = 0  # 类似于指针的作用
    else:
        file_index = len(hf['features'][:])


    start_time,end_time = time_2_frame(df_pos,fps) # 将开始时间，结束时间，转为开始为第几frame ,结束为第几frame
    # print('start_time, end_time ',start_time,end_time)

    'For csv files with a column name Call, pick up the global class name'

    if 'CALL' in df_pos.columns:
        cls_list = [glob_cls_name] * len(start_time)
    else:
        cls_list = [df_pos.columns[(df_pos == 'POS').loc[index]].values for index, row in df_pos.iterrows()]
        cls_list = list(chain.from_iterable(cls_list))

    assert len(start_time) == len(end_time)
    assert len(cls_list) == len(start_time)

    nframe_label = ['0']*pcen.shape[0]
    for index in range(len(start_time)):
        str_ind = start_time[index]
        end_ind = end_time[index]
        label = cls_list[index]  # 保证了不在同一时间发生两件事？
        for ind in range(str_ind,end_ind):
            nframe_label[ind]=label
        # print('str_ind ',str_ind)
        # print('end_ind ',end_ind)
        # print('label ',label)
        # 'Extract segment and move forward with hop_seg'
    cur_time = 0
    while cur_time+seg_len<=pcen.shape[0]:#不足部分丢弃
        pcen_patch = pcen[int(cur_time):int(cur_time+seg_len)]
        hf['features'].resize((file_index+1,pcen_patch.shape[0],pcen_patch.shape[1]))
        hf['features'][file_index] = pcen_patch
        file_index +=1
       
        # 开始端点截断部分不训练
        middle_idx = 0
        label_path = nframe_label[int(cur_time):int(cur_time+seg_len)]
        if cur_time and nframe_label[int(cur_time)-1] !='0' and nframe_label[int(cur_time)+1] !='0':
            try:
                middle_idx = label_path.index('0')
            except:
                middle_idx = seg_len
        for idx in range(0,middle_idx):
            label_list.append('-1')
        # 结束端点截断部分不训练
        end_idx = seg_len
        if cur_time+seg_len< pcen.shape[0] and nframe_label[int(cur_time+seg_len)-2] !='0' \
            and nframe_label[int(cur_time+seg_len)] !='0':
            temp_label = label_path[middle_idx:]
            temp_label.reverse()
            try:
                end_idx = seg_len-temp_label.index('0')
            except:
                end_idx = seg_len
        for idx in range(middle_idx,end_idx):
            label_list.append(label_path[idx])
        for idx in range(end_idx,seg_len):
            label_list.append('-1')
        cur_time = cur_time + hop_seg

    assert len(label_list) == (cur_time//hop_seg)*seg_len

    print('Total files created:{}'.format(file_index))
    return label_list

class Feature_Extractor():

       def __init__(self, conf):
           self.sr =conf.features.sr
           self.n_fft = conf.features.n_fft
           self.hop = conf.features.hop_mel
           self.n_mels = conf.features.n_mels
           self.fmax = conf.features.fmax
           
       def extract_feature(self,audio):

           mel_spec = librosa.feature.melspectrogram(audio,sr=self.sr, n_fft=self.n_fft,
                                                     hop_length=self.hop,n_mels=self.n_mels,fmax=self.fmax)
           pcen = librosa.core.pcen(mel_spec,sr=22050)
           pcen = pcen.astype(np.float32)

           return pcen

def extract_feature(audio_path,feat_extractor,conf,aug=False):

    y,fs = librosa.load(audio_path,sr=conf.features.sr)

    'Scaling audio as per suggestion in librosa documentation'

    y = y * (2**32)
    pcen = feat_extractor.extract_feature(y)
    # if aug:
    #     pcen = torch.from_numpy(pcen)
    #     pcen = frequencyMask(pcen)
    #     pcen = TimeMask(pcen)
    #     pcen = pcen.numpy()
    return pcen.T

def add_gussionNoise(data,sigma=0.4):# 添加高斯噪音
    mean = torch.mean(data.cpu()).numpy()
    var = torch.std(data.cpu()).numpy()
    noise = np.random.normal(mean,var**2,data.shape)
    noise = sigma*torch.from_numpy(noise).to(data.device).float()
    aug_data = noise+data
    return aug_data

# FilterAugment
def filt_aug(features, db_range=[-8, 8], n_band=[3, 6], min_bw=6, filter_type="linear"):
    '''Feature augmentation method
    '''
    # this is updated FilterAugment algorithm used for ICASSP 2022
    if not isinstance(filter_type, str):
        if torch.rand(1).item() < filter_type:
            filter_type = "step"
            n_band = [2, 5]
            min_bw = 4
        else:
            filter_type = "linear"
            n_band = [3, 6]
            min_bw = 6 # filter band mini-width

    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()   # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        if filter_type == "step":
            band_factors = torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

        elif filter_type == "linear":
            band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                for j in range(batch_size):
                    freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                        torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                       band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
            freq_filt = 10 ** (freq_filt / 20)
        return features * freq_filt
    else:
        return features

def frequencyMask(data,freq_mask=20):
    data_mask_list =[]
    freq_mask = freq_mask
    frequency_mask = T.FrequencyMasking(freq_mask_param=freq_mask)
    for i in range(len(data)):
        sub_data = data[i]
        data_mask_list.append(frequency_mask(sub_data))
    data_mask = torch.stack(data_mask_list)
    return data_mask

def TimeMask(data, Time_mask=10):
    Time_mask = Time_mask
    time_maskData_list = []
    Timeing_mask = T.TimeMasking(time_mask_param=Time_mask)
    for i in range(len(data)):
        sub_data  = data[i]
        mask_sub_data = Timeing_mask(sub_data)  
        mask_data = torch.stack([mask_sub_data, sub_data])
        mask_data = torch.permute(mask_data,[1,2,0])
        mask_data =  torch.mean(mask_data,dim=-1)
        time_maskData_list.append(mask_data)
    mask_data = torch.stack(time_maskData_list)
    return mask_data

def time_2_frame(df,fps):
    'Margin of 25 ms around the onset and offsets'

    df.loc[:,'Starttime'] = df['Starttime'] - 0.025
    df.loc[:,'Endtime'] = df['Endtime'] + 0.025

    'Converting time to frames'

    start_time = [int(np.floor(start * fps)) for start in df['Starttime']] # fps 即 1s多少个frame， start*fps 即可得出开始的帧

    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]

    return start_time,end_time


def time_2_frame2(df,fps):
    'Margin of 25 ms around the onset and offsets'

    df.loc[:,'Starttime'] = df['Starttime'] 
    df.loc[:,'Endtime'] = df['Endtime']

    'Converting time to frames'

    start_time = [int(np.floor(start * fps)) for start in df['Starttime']] # fps 即 1s多少个frame， start*fps 即可得出开始的帧

    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]

    return start_time,end_time


def feature_transform(conf=None,mode=None,aug=False):
    '''
       Training:
          Extract mel-spectrogram/PCEN and slice each data sample into segments of length conf.seg_len.
          Each segment inherits clip level label. The segment length is kept same across training
          and validation set.
       Evaluation:
           Currently using the validation set for evaluation.
           
           For each audio file, extract time-frequency representation and create 3 subsets:
           a) Positive set - Extract segments based on the provided onset-offset annotations.
           b) Negative set - Since there is no negative annotation provided, we consider the entire
                         audio file as the negative class and extract patches of length conf.seg_len
           c) Query set - From the end time of the 5th annotation to the end of the audio file.
                          Onset-offset prediction is made on this subset.

       Args:
       - config: config object
       - mode: train/valid

       Out:
       - Num_extract_train/Num_extract_valid - Number of samples in training/validation set
                                                                                              '''
    label_tr = []
    pcen_extractor = Feature_Extractor(conf)

    fps =  conf.features.sr / conf.features.hop_mel  # 22050/256=86
    'Converting fixed segment legnth to frames'
    # print('fps ',fps)
    seg_len = int(round(conf.features.seg_len * fps)) # 86*0.200= 17
    # print('seg_len ',seg_len)
    hop_seg = int(round(conf.features.hop_seg * fps)) # 86*0.05= 4
    # print('hop_seg ',hop_seg)
    extension = "*.csv"

    if mode == 'train':

        print("=== Processing training set ===")
        meta_path = conf.path.train_dir # 训练数据路径
        all_csv_files = [file
                         for path_dir, subdir, files in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, extension))] #  获得所有的cvs文件路径
        hdf_tr = os.path.join(conf.path.feat_train,'Mel_train.h5') # 存放mel的路径
        hf = h5py.File(hdf_tr,'w')
        hf.create_dataset('features', shape=(0, seg_len, conf.features.n_mels),
                          maxshape=(None, seg_len, conf.features.n_mels))
        num_extract = 0
        for file in all_csv_files:
            
            split_list = file.split('/')
            glob_cls_name = split_list[split_list.index('Training_Set') + 1]
            file_name = split_list[split_list.index('Training_Set') + 2]
            df = pd.read_csv(file, header=0, index_col=False)
            audio_path = file.replace('csv', 'wav')
            print("Processing file name {}".format(audio_path))
            pcen = extract_feature(audio_path, pcen_extractor,conf,aug=aug)
            df_pos = df[(df == 'POS').any(axis=1)] # 找到任何一列含有POS的行
            label_list = create_dataset_v2(df_pos,pcen,glob_cls_name,file_name,hf,seg_len,hop_seg,fps) # 只处理 含有pos的段
            label_tr.append(label_list)
            # break

        print(" Feature extraction for training set complete")
        num_extract = len(hf['features'])
        flat_list = [item for sublist in label_tr for item in sublist] # 将多个list 合并成一个
        hf.create_dataset('labels', data=[s.encode() for s in flat_list], dtype='S20') # 保存下来
        data_shape = hf['features'].shape
        hf.close()
        return num_extract,data_shape

    elif mode=='eval':

        print("=== Processing Validation set ===")

        meta_path = conf.path.eval_dir

        all_csv_files = [file
                         for path_dir, subdir, files in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, extension))]

        num_extract_eval = 0

        for file in all_csv_files:

            idx_pos = 0
            idx_neg = 0
            start_neg = 0
            hop_neg = 0
            idx_query = 0
            hop_query = 0
            strt_index = 0

            split_list = file.split('/')
            name = str(split_list[-1].split('.')[0])
            
            feat_name = name + '.h5'
            audio_path = file.replace('csv', 'wav')
            feat_info = []
            hdf_eval = os.path.join(conf.path.feat_eval,feat_name)
            hf = h5py.File(hdf_eval,'w')

            hf.create_dataset('feat_pos', shape=(0, seg_len, conf.features.n_mels),
                              maxshape= (None, seg_len, conf.features.n_mels))
            hf.create_dataset('feat_query',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('feat_neg',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('start_index_query',shape=(1,),maxshape=(None))

            "In case you want to use the statistics of each file to normalize"
            hf.create_dataset('mean_global',shape=(1,),maxshape=(None))
            hf.create_dataset('std_dev_global',shape=(1,),maxshape=(None))
            df_eval = pd.read_csv(file, header=0, index_col=False)
            Q_list = df_eval['Q'].to_numpy() # Q 列

            start_time,end_time = time_2_frame(df_eval,fps) # 时间转 frame
            index_sup = np.where(Q_list == 'POS')[0][:conf.features.n_shot] # 查找前n_shot个有POS标签的 片段
             
            pcen = extract_feature(audio_path, pcen_extractor,conf) # 提取feature
            mean = np.mean(pcen)
            std = np.std(pcen) # ？？ 有问题
            hf['mean_global'][:] = mean
            hf['std_dev_global'][:] = std

            strt_indx_query = end_time[index_sup[-1]] # 开始查询时间， 即为 最后一个support的结束
            end_idx_neg = pcen.shape[0] - 1 
            hf['start_index_query'][:] = strt_indx_query

            # ----------------------------------------------------------------------------------------
            print("Creating Positive dataset from {}".format(file))
            idx_pos = 0
            for index in index_sup:
                str_ind = max(0,int(start_time[index]))
                end_ind = int(end_time[index])

                patch_pos = pcen[int(str_ind):int(end_ind)]

                hf.create_dataset('feat_pos_%s'%idx_pos,shape=(0,patch_pos.shape[0],patch_pos.shape[1]),maxshape=(None,patch_pos.shape[0],patch_pos.shape[1]))
                hf['feat_pos_%s'%idx_pos].resize((0+1,patch_pos.shape[0],patch_pos.shape[1]))
                hf['feat_pos_%s'%idx_pos][0]=patch_pos
                idx_pos +=1

            print('index_Pos: ', idx_pos)

            print("Creating Negative dataset from {}".format(file))
            start_time,end_time = time_2_frame2(df_eval,fps)
            idx_neg = 0
            str_ind = 0
            for i in range(0,index_sup.shape[0]):
                index = index_sup[i]
                end_ind = max(0,int(start_time[index]))
                patch_pos = pcen[int(str_ind):int(end_ind)]
                hf.create_dataset('feat_neg_%s'%idx_neg,shape=(0,patch_pos.shape[0],patch_pos.shape[1]),maxshape=(None,patch_pos.shape[0],patch_pos.shape[1]))
                hf['feat_neg_%s'%idx_neg].resize((0+1,patch_pos.shape[0],patch_pos.shape[1]))
                hf['feat_neg_%s'%idx_neg][0] = patch_pos
                idx_neg +=1
                str_ind = int(end_time[index])

            print("Creating query dataset from {}".format(file))
            while end_idx_neg - (strt_indx_query+hop_query) > seg_len:
                patch_query = pcen[int(strt_indx_query+hop_query):int(strt_indx_query+hop_query+seg_len)] 
                hf['feat_query'].resize((idx_query+1,patch_query.shape[0],patch_query.shape[1]))
                hf['feat_query'][idx_query]=patch_query
                idx_query +=1
                hop_query += hop_seg
            print("index_Query", idx_query)
            last_patch_query = pcen[end_idx_neg-seg_len:end_idx_neg]
            hf['feat_query'].resize((idx_query+1,last_patch_query.shape[0],last_patch_query.shape[1]))
            hf['feat_query'][idx_query] = last_patch_query
            num_extract_eval += len(hf['feat_query'])
            
            hf.close()
        return num_extract_eval
    elif mode=='test':

        print("=== Processing Validation set ===")

        meta_path = conf.path.eval_dir

        all_csv_files = [file
                         for path_dir, subdir, files in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, extension))]

        num_extract_eval = 0

        for file in all_csv_files:

            idx_pos = 0
            idx_neg = 0
            start_neg = 0
            hop_neg = 0
            idx_query = 0
            hop_query = 0
            strt_index = 0

            split_list = file.split('/')
            name = str(split_list[-1].split('.')[0])
            feat_name = name + '.h5'
            audio_path = file.replace('csv', 'wav')
            feat_info = []
            hdf_eval = os.path.join(conf.path.feat_eval,feat_name)
            hf = h5py.File(hdf_eval,'w')

            hf.create_dataset('feat_pos', shape=(0, seg_len, conf.features.n_mels),
                              maxshape= (None, seg_len, conf.features.n_mels))
            hf.create_dataset('feat_query',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('feat_neg',shape=(0,seg_len,conf.features.n_mels),maxshape=(None,seg_len,conf.features.n_mels))
            hf.create_dataset('start_index_query',shape=(1,),maxshape=(None))
            hf.create_dataset('mean_global',shape=(1,),maxshape=(None))
            hf.create_dataset('std_dev_global',shape=(1,),maxshape=(None))
            df_eval = pd.read_csv(file, header=0, index_col=False)
            Q_list = df_eval['Q'].to_numpy() # Q 列

            start_time,end_time = time_2_frame(df_eval,fps) # 时间转 frame
            index_sup = np.where(Q_list == 'POS')[0][:conf.feature.n_shot] # 查找前n_shot个有POS标签的 片段
             
            pcen = extract_feature(audio_path, pcen_extractor,conf) # 提取feature
            mean = np.mean(pcen)
            std = np.std(pcen) # ？？ 有问题
            hf['mean_global'][:] = mean
            hf['std_dev_global'][:] = std

            strt_indx_query = end_time[index_sup[-1]] # 开始查询时间， 即为 最后一个support的结束
            end_idx_neg = pcen.shape[0] - 1 
            hf['start_index_query'][:] = strt_indx_query

            log_file = open("Feature_Creation.txt",'a')            

            print("Creating Positive dataset from {}".format(file))
            idx_pos = 0
            for index in index_sup:
                str_ind = max(0,int(start_time[index]))
                end_ind = int(end_time[index])

                patch_pos = pcen[int(str_ind):int(end_ind)]

                hf.create_dataset('feat_pos_%s'%idx_pos,shape=(0,patch_pos.shape[0],patch_pos.shape[1]),maxshape=(None,patch_pos.shape[0],patch_pos.shape[1]))
                hf['feat_pos_%s'%idx_pos].resize(0+1,patch_pos.shape[0],patch_pos.shape[1])
                hf['feat_pos_%s'%idx_pos][0]=patch_pos
                idx_pos +=1

            print('index_Pos: ', idx_pos)
            print("Creating Negative dataset from {}".format(file))
            start_time,end_time = time_2_frame2(df_eval,fps)
            idx_neg = 0
            str_ind = 0
            for i in range(0,index_sup.shape[0]):
                index = index_sup[i]
                end_idx = max(0,int(start_time[index]))
                patch_pos = pcen[int(str_ind):int(end_ind)]
                hf.create_dataset('feat_neg_%s'%idx_neg,shape=(0,patch_pos.shape[0],patch_pos.shape[1]),maxshape=(None,patch_pos.shape[0],patch_pos.shape[1]))
                hf['feat_neg_%s'%idx_neg].resize(0+1,patch_pos.shape[0],patch_pos.shape[1])
                hf['feat_neg_%s'%idx_neg][0] = patch_pos
                idx_neg +=1
                str_ind = int(end_time[index])

            print("Creating query dataset from {}".format(file))
            while end_idx_neg - (strt_indx_query+hop_query) > seg_len:
                patch_query = pcen[int(strt_indx_query+hop_query):int(strt_indx_query+hop_query+seg_len)] 
                hf['feat_query'].resize((idx_query+1,patch_query.shape[0],patch_query.shape[1]))
                hf['feat_query'][idx_query]=patch_query
                idx_query +=1
                hop_query += hop_seg
            print("index_Query", idx_query)
            last_patch_query = pcen[end_idx_neg-seg_len:end_idx_neg]
            hf['feat_query'].resize((idx_query+1,last_patch_query.shape[0],last_patch_query.shape[1]))
            num_extract_eval += len(hf['feat_query'])
            
            hf.close()
        return num_extract_eval









