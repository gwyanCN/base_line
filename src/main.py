import yaml
import argparse
import pandas as pd
import random
import csv
import os
import h5py
import pandas as pd
from glob import glob
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from tqdm import tqdm
from collections import Counter
from datasets.batch_sampler import EpisodicBatchSampler
from utils.trainer import Trainer
from utils.util import warp_tqdm, save_checkpoint
from models import __dict__
from datasets.Feature_extract import feature_transform
from datasets.Datagenerator import Datagen
from utils.eval import Evaluator
import time
import librosa
import pdb 
import json
import os
import os.path as osp
import torch.nn as nn
from utils.tensorbord_ import Summary

def get_model(arch, num_classes,ema=False):
    if arch == 'resnet10' or arch == 'resnet18':
        model = __dict__[arch](num_classes=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
            return model
        return model
    else:
        model = __dict__[arch](num_classes=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
            return model
        return model

def train_protonet(model,train_loader,valid_loader,conf):
    arch = 'Protonet'
    alpha = 0.0  
    disable_tqdm = False 
    ckpt_path = conf.eval.ckpt_path
    pretrain = True
    resume = False
    if conf.train.device == 'cuda':
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    optim = torch.optim.Adam(model.parameters(), lr=conf.train.lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=conf.train.scheduler_gamma,
                                                   step_size=conf.train.scheduler_step_size)
    num_epochs = conf.train.epochs

    if pretrain: 
        pretrain = os.path.join(ckpt_path, 'checkpoint.pth.tar')
        if os.path.isfile(pretrain):
            print("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            print('[Warning]: Did not find pretrained model {}'.format(pretrain))

    if resume:
        resume_path = ckpt_path + '/checkpoint.pth.tar'
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print('[Warning]: Did not find checkpoint {}'.format(resume_path))
    else:
        start_epoch = 0
        best_prec1 = -1

    # cudnn.benchmark = True
    # model = nn.DataParallel(model,device_ids=device_ids)
    model.to(device) # cuda
    trainer = Trainer(device=device,num_class=conf.train.num_classes, train_loader=train_loader,val_loader=valid_loader,conf=conf)
    time_start=time.time()
    for epoch in range(num_epochs):
        # trainer.do_epoch2(epoch=epoch,scheduler=lr_scheduler,disable_tqdm=disable_tqdm,model=model,alpha=alpha,optimizer=optim)
        trainer.do_epoch(epoch=epoch,scheduler=lr_scheduler,disable_tqdm=disable_tqdm,model=model,alpha=alpha,optimizer=optim)
        # Evaluation on validation set
        prec1 = trainer.meta_val(epoch=epoch,model=model, disable_tqdm=disable_tqdm)
        print('Meta Val {}: {}'.format(epoch, prec1))
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if not disable_tqdm:
            print('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # Save checkpoint
        save_checkpoint(state={'epoch': epoch + 1,
                               'arch': arch,
                               'state_dict': model.state_dict(),
                               'best_prec1': best_prec1,
                               'optimizer': optim.state_dict()},
                        is_best=is_best,
                        folder=ckpt_path)
        if lr_scheduler is not None:
            lr_scheduler.step()
    time_end=time.time()
    print('totally cost',time_end-time_start)
    print('model_paramiter...............')
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
        # if epoch == num_epochs-1:
        #     trainer.met_plot(epoch, model, disable_tqdm)

def train_protonet2(model,train_loader,valid_loader,conf):
    arch = 'Protonet'
    alpha = 0.0  
    disable_tqdm = False 
    ckpt_path = conf.eval.ckpt_path
    is_pretrain = False
    resume = False
    if conf.train.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optim = torch.optim.Adam(model.parameters(), lr=conf.train.lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=conf.train.scheduler_gamma,
                                                   step_size=conf.train.scheduler_step_size)
    num_epochs = conf.train.epochs

    if is_pretrain: 
        pretrain = os.path.join(conf.path.work_path, 'check_point/checkpoint.pth.tar')
        if os.path.isfile(pretrain):
            print("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            print('[Warning]: Did not find pretrained model {}'.format(pretrain))
            
    if resume:
        resume_path = ckpt_path + '/checkpoint.pth.tar'
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print('[Warning]: Did not find checkpoint {}'.format(resume_path))
    else:
        start_epoch = 0
        best_prec1 = -1
    #cudnn.benchmark = True
    model.to(device,non_blocking=True) # cuda
    trainer = Trainer(device=device,num_class=conf.train.num_classes, train_loader=train_loader,val_loader=valid_loader,conf=conf)
    time_start=time.time()
    for epoch in range(num_epochs):
        trainer.do_epoch2(epoch=epoch,scheduler=lr_scheduler,disable_tqdm=disable_tqdm,model=model,alpha=alpha,optimizer=optim)
        # Evaluation on validation set
        prec1, prec1_1 = trainer.meta_val2(epoch=epoch,model=model, disable_tqdm=disable_tqdm)
        print('Meta Val {}: {}'.format(epoch, prec1)) # 2分类
        print('Meta Val_1 {}: {}'.format(epoch, prec1_1)) #20分类
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if not disable_tqdm:
            print('Best Acc {:.2f}'.format(best_prec1 * 100.))
        # Save checkpoint
        save_checkpoint(state={'epoch': epoch + 1,
                               'arch': arch,
                               'state_dict': model.state_dict(),
                               'best_prec1': best_prec1,
                               'optimizer': optim.state_dict()},
                        is_best=is_best,
                        folder=ckpt_path)
        if lr_scheduler is not None:
            lr_scheduler.step()
    time_end=time.time()
    print('totally cost',time_end-time_start)
    print('model_paramiter...............')
    
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)

def train_sep(model,model_sep,train_loader,valid_loader,conf):
    arch = 'Protonet'
    alpha = 0.0  
    disable_tqdm = False 
    ckpt_path = conf.eval.ckpt_sep_path
    is_pretrain = False
    resume = False
    if conf.train.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optim = torch.optim.Adam([{'params':model.parameters()},{'params':model_sep.parameters()}], lr=conf.train.lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=conf.train.scheduler_gamma,
                                                   step_size=conf.train.scheduler_step_size)
    num_epochs = conf.train.epochs

    if is_pretrain: 
        pretrain = os.path.join(conf.path.work_path, 'check_point/checkpoint.pth.tar')
        if os.path.isfile(pretrain):
            print("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            print('[Warning]: Did not find pretrained model {}'.format(pretrain))

    if resume:
        resume_path = ckpt_path + '/checkpoint.pth.tar'
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print('[Warning]: Did not find checkpoint {}'.format(resume_path))
    else:
        start_epoch = 0
        best_prec1 = -1

    #cudnn.benchmark = True
    model.to(device) # cuda
    model_sep.to(device)
    trainer = Trainer(device=device,num_class=conf.train.num_classes, train_loader=train_loader,val_loader=valid_loader,conf=conf)
    aug = conf.train.train_aug
    time_start=time.time()
    for epoch in range(num_epochs):
        trainer.do_epoch_sep(epoch=epoch,scheduler=lr_scheduler,disable_tqdm=disable_tqdm,model=model,model_sep=model_sep,alpha=alpha,optimizer=optim,aug=aug)
        # Evaluation on validation set
        prec1,prec1_1,prec1_2 = trainer.meta_sep_val(epoch=epoch,model=model,model_sep=model_sep, disable_tqdm=disable_tqdm)
        print('Meta Val {}: {}'.format(epoch, prec1)) # 2分类
        print('Meta Val_1 {}: {}'.format(epoch, prec1_1)) #20分类
        print('Meta Val_2 {}: {}'.format(epoch, prec1_2)) #分离20分类
        is_best = prec1_2 > best_prec1
        best_prec1 = max(prec1_2, best_prec1)
        if not disable_tqdm:
            print('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # Save checkpoint
        save_checkpoint(state={'epoch': epoch + 1,
                               'arch': arch,
                               'state_dict': model.state_dict(),
                               'state_dict_sep': model_sep.state_dict(),
                               'best_prec1': best_prec1,
                               'optimizer': optim.state_dict()},
                        is_best=is_best,
                        folder=ckpt_path)
        if lr_scheduler is not None:
            lr_scheduler.step()
    time_end=time.time()
    fp=open("/home/b227/ygw/Dcase2022/new_frame_level/DCASE2021Task5-main/src/loss_record.json",'w')
    json.dump(trainer.loss_record,fp=fp,indent=3)
    fp.close()
    print('totally cost',time_end-time_start)
    print('model_paramiter...............')
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
        # if epoch == num_epochs-1:
        #     trainer.met_plot(epoch, model, disable_tqdm)

def eval_only(model,train_loader,valid_loader,conf):  # what this function means?
    arch = 'Protonet'
    alpha = 0.0  # mixpu
    disable_tqdm = False # 
    ckpt_path = '/home/ydc/DACSE2021/sed-tim-base/check_point'
    pretrain = True
    resume = False
    if conf.train.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optim = torch.optim.Adam(model.parameters(), lr=conf.train.lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=conf.train.scheduler_gamma,
                                                   step_size=conf.train.scheduler_step_size)
    num_epochs = conf.train.epochs

    if pretrain:
        pretrain = os.path.join(ckpt_path, 'checkpoint.pth.tar')
        if os.path.isfile(pretrain):
            print("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            print('[Warning]: Did not find pretrained model {}'.format(pretrain))

    if resume:
        resume_path = ckpt_path + '/checkpoint.pth.tar'
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print('[Warning]: Did not find checkpoint {}'.format(resume_path))
    else:
        start_epoch = 0
        best_prec1 = -1

    #cudnn.benchmark = True
    model.to(device) # 
    trainer = Trainer(device=device,num_class=19, train_loader=train_loader,val_loader=valid_loader,conf=conf)
    for epoch in range(num_epochs):
        #trainer.do_epoch(epoch=epoch,scheduler=lr_scheduler,disable_tqdm=disable_tqdm,model=model,alpha=alpha,optimizer=optim)
        # Evaluation on validation set
        prec1 = trainer.met_plot(epoch=epoch,model=model, disable_tqdm=disable_tqdm)
        print('Meta Val {}: {}'.format(epoch, prec1))
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if not disable_tqdm:
            print('Best Acc {:.2f}'.format(best_prec1 * 100.))

def get_name2wav(wav_dir):
    hash_name2wav={}
    for subdir in os.listdir(wav_dir):
        sub_wav_dir = os.path.join(wav_dir,subdir)
        for name in os.listdir(sub_wav_dir):
            if ".wav" not in name:
                continue
            wav_path = os.path.join(sub_wav_dir,name)
            hash_name2wav[name] = wav_path
    return hash_name2wav

def post_contact(mean_pos_len,predict,offset,onset,thre):
    for i in range(len(offset)-1):
        start = offset[i]
        end = onset[i+1]
        if end - start < thre*mean_pos_len:
            predict[start:end]=1
    return predict

@hydra.main(config_name="config")
def main(conf : DictConfig):
    seed = 2021
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
    if not os.path.isdir(conf.path.feat_path):
        os.makedirs(conf.path.feat_path)
    if not os.path.isdir(conf.path.feat_train):
        os.makedirs(conf.path.feat_train)
    if not os.path.isdir(conf.path.feat_eval):
        os.makedirs(conf.path.feat_eval)
        
    if conf.set.features:
        print(" --Feature Extraction Stage--")
        Num_extract_train,data_shape = feature_transform(conf=conf,mode="train") # train data
        print("Shape of dataset is {}".format(data_shape))
        print("Total training samples is {}".format(Num_extract_train))
        
        Num_extract_eval = feature_transform(conf=conf,mode='eval',aug=False)
        print("Total number of samples used for evaluation: {}".format(Num_extract_eval)) # validate data
        print(" --Feature Extraction Complete--")

        # Num_extract_test = feature_transform(conf=conf,mode='test')
        # print("Total number of samples used for evaluation: {}".format(Num_extract_test)) # test data
        # print(" --Feature Extraction Complete--")

    if conf.set.train: # train
        print("============> start training!")
        meta_learning = False # wether use meta learing ways to train
        # import pdb 
        # pdb.set_trace()
        if meta_learning:
            gen_train = Datagen(conf) 
            X_train,Y_train,X_val,Y_val = gen_train.generate_train() # 
            X_tr = torch.tensor(X_train) 
            Y_tr = torch.LongTensor(Y_train)
            X_val = torch.tensor(X_val)
            Y_val = torch.LongTensor(Y_val)

            samples_per_cls =  conf.train.n_shot * 2 

            batch_size_tr = samples_per_cls * conf.train.k_way # the batch size of training 
            batch_size_vd = batch_size_tr # 

            num_batches_tr = len(Y_train)//batch_size_tr # num of batch
            num_batches_vd = len(Y_val)//batch_size_vd

            samplr_train = EpisodicBatchSampler(Y_train,num_batches_tr,conf.train.k_way,samples_per_cls) # batch_size_tr = samples_per_cls * conf.train.k_way
            samplr_valid = EpisodicBatchSampler(Y_val,num_batches_vd,conf.train.k_way,samples_per_cls)

            train_dataset = torch.utils.data.TensorDataset(X_tr,Y_tr) # 利用torch 的 dataset,整合X,Y
            valid_dataset = torch.utils.data.TensorDataset(X_val,Y_val)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_sampler=samplr_train,shuffle=False)
            # batch_sampler 批量采样，默认设置为None。但每次返回的是一批数据的索引,每次输入网络的数据是随机采样模式，这样能使数据更具有独立性质
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_sampler=samplr_valid,shuffle=False)
        else:
            try:
                train_datasets = torch.load(conf.train.train_data)
                X_tr=train_datasets['X_tr']
                Y_tr=train_datasets['Y_tr']
                Y2_tr=train_datasets['Y2_tr']
                
                X_val=train_datasets['X_val']
                Y_val=train_datasets['Y_val']
                Y2_val=train_datasets['Y2_val']
                
            except:
                gen_train = Datagen(conf) 
                X_train,Y_train,Y2_train,X_val,Y_val,Y2_val = gen_train.generate_train() 
                X_tr = torch.tensor(X_train) 
                Y_tr = torch.LongTensor(Y_train)
                Y2_tr = torch.LongTensor(Y2_train)
                
                X_val = torch.tensor(X_val)
                Y_val = torch.LongTensor(Y_val)
                Y2_val = torch.LongTensor(Y2_val)
                state = {
                    'X_tr':X_tr,
                    'Y_tr':Y_tr,
                    'Y2_tr':Y2_tr,
                    'X_val':X_val,
                    
                    'Y_val':Y_val,
                    'Y2_val':Y2_val        
                }
                torch.save(state,'/media/b227/ygw/Dcase2023/baseline/src/train_data/train_datasets_21.pth')

            # print('X_tr ',X_tr.shape)
            # print('X_val ',X_val.shape)
            samples_per_cls =  conf.train.n_shot 
            batch_size_tr =  conf.train.n_shot* conf.train.k_way #64 # the batch size of training 
           
            train_dataset = torch.utils.data.TensorDataset(X_tr,Y_tr) # 利用torch 的 dataset,整合X,Y
            valid_dataset = torch.utils.data.TensorDataset(X_val,Y_val)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=conf.train.batch_size,num_workers=10,shuffle=False)
            # batch_sampler 批量采样，默认设置为None。但每次返回的是一批数据的索引,每次输入网络的数据是随机采样模式，这样能使数据更具有独立性质
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_size=64,num_workers=10,shuffle=False)
            
            train_dataset2 = torch.utils.data.TensorDataset(X_tr,Y2_tr,Y_tr) # 利用torch 的 dataset,整合X,Y, Y2∈{0,1}, Y∈{0,1,2....,19}
            valid_dataset2 = torch.utils.data.TensorDataset(X_val,Y2_val,Y_val)
            train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2,batch_size=conf.train.batch_size,num_workers=10,pin_memory=True,shuffle=False)
            valid_loader2 = torch.utils.data.DataLoader(dataset=valid_dataset2,batch_size=64,batch_sampler=None,num_workers=10,pin_memory=True,shuffle=False)

        #model = get_model('Protonet',19)
        model = get_model('TSVAD1',conf.train.num_classes)
        # train_protonet(model,train_loader,valid_loader,conf)
        train_protonet2(model,train_loader2,valid_loader2,conf)

    if conf.set.eval: # eval
        device = 'cuda'
        # init_seed()
        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])

        name_arr_ori = np.array([])
        onset_arr_ori = np.array([])
        offset_arr_ori = np.array([])

        all_feat_files = sorted([file for file in glob(os.path.join(conf.path.feat_eval,'*.h5'))])
        evaluator = Evaluator(device=device)
        model = get_model('TSVAD1',conf.train.num_classes).cuda()

        # model = nn.DataParallel(model,device_ids=device_ids)
        student_model = get_model('TSVAD1',conf.train.num_classes,ema=False).cuda()
      
        ckpt_path = conf.eval.ckpt_path
        hash_name2wav = get_name2wav(conf.path.eval_dir)
        hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel) # 0.05*22050//256 == 86
        k_q = 128
        iter_num = conf.eval.iter_num # change wether use ML framework
        CURRENT_TIMES = time.strftime("%Y-%m-%d %H-%M",time.localtime())
        logger_writer = Summary(path=osp.join(conf.eval.tensorboard_path,CURRENT_TIMES))
        
        TOTAL_LENGTH = len(all_feat_files)
        for i in range(TOTAL_LENGTH//4):
            feat_file = all_feat_files[i]
            feat_name = feat_file.split('/')[-1]   
            file_name = feat_name.replace('.h5','')
            # if file_name!="n1":continue
            audio_name = feat_name.replace('h5','wav')
            print("Processing audio file : {}".format(audio_name))
            # if audio_name!="n1.wav":continue
            wav_path = hash_name2wav[audio_name]

            ori_path = os.path.join(conf.path.work_path,'src','output_csv','ori')
            if not os.path.exists(ori_path):
                os.makedirs(ori_path)
            tim_path = os.path.join(conf.path.work_path,'src','output_csv','tim')
            if not os.path.exists(tim_path):
                os.makedirs(tim_path)   
            
            if os.path.exists(os.path.join(conf.path.work_path,'waveFrame',file_name,"waveFrame.json")):
                reader = open(os.path.join(conf.path.work_path,'waveFrame',file_name,"waveFrame.json"),'r')
                waveData = json.load(fp=reader)
                nframe = waveData['nframe']
            else:
               if not os.path.exists(os.path.join(conf.path.work_path,'waveFrame',file_name)):
                    os.makedirs(os.path.join(conf.path.work_path,'waveFrame',file_name))
               writer = open(os.path.join(conf.path.work_path,'waveFrame',file_name,'waveFrame.json'),'w')
               y,fs = librosa.load(wav_path,sr= conf.features.sr)
               nframe = len(y)//conf.features.hop_mel
               json.dump({'nframe':nframe},fp=writer)
               writer.close()

            hdf_eval = h5py.File(feat_file,'r')
            start_index_query =  hdf_eval['start_index_query'][:][0]
            
            fileDict = {}
            fileDict['nframes']=nframe
            fileDict['start_index_query']=start_index_query
            mask_path = conf.eval.mask_path
            result, num,logs= evaluator.run_full_evaluation(test_file=audio_name[:-4],model=model, student_model=student_model,
                                                         model_path=ckpt_path,hdf_eval=hdf_eval,conf=conf,k_q=k_q,iter_num=iter_num) # only update W
          
            predict = result[0]
            MFL = result[1]
            thre = max(result[2]-0.05,0.5)
            mean_pos_len = result[3]
            
            krn = np.array([1,-1])
            prob_thresh = np.where(predict>thre,1,0)

            prob_middle = flatten_res(prob_thresh,hop_seg,int(nframe-start_index_query))
            predict_middle = flatten_res(predict,hop_seg,int(nframe-start_index_query))

            changes_smooth = np.convolve(krn, prob_middle)
            onset_frames_smooth = np.where(changes_smooth==1)[0] #事件开始点
            offset_frames_smooth = np.where(changes_smooth==-1)[0] #事件结束点

            # prob_middle=post_contact(mean_pos_len=mean_pos_len,offset=offset_frames_smooth,onset=offset_frames_smooth,predict=prob_middle,thre=0.4)

            for i in range(onset_frames_smooth.shape[0]-1):
                start = offset_frames_smooth[i]
                end = onset_frames_smooth[i+1]
                if mean_pos_len > 2*87 and end-start<87 and predict_middle[start:end].mean()>0.5:
                    prob_middle[start:end]=1
                # elif 2*87>mean_pos_len>2*80 and end-start < int(0.3*mean_pos_len) and predict_middle[start:end].mean()>0.5:
                #     prob_middle[start:end]=1
                    
            prob_med_filt = medFilt(prob_middle,5)
            start_index_query = start_index_query*conf.features.hop_mel / conf.features.sr
            print('start_index_query',start_index_query)

            ###############################ORI############################
            changes_ori = np.convolve(krn,prob_middle) # 原数据
            onset_frames_ori = np.where(changes_ori==1)[0]
            offset_frames_ori = np.where(changes_ori==-1)[0]

            onset_ori = (onset_frames_ori+1) *conf.features.hop_mel / conf.features.sr
            onset_ori = onset_ori + start_index_query
            offset_ori = (offset_frames_ori+1) *conf.features.hop_mel / conf.features.sr
            offset_ori = offset_ori + start_index_query

            name_ori = np.repeat(audio_name,len(onset_ori))
            name_arr_ori = np.append(name_arr_ori,name_ori)
            onset_arr_ori = np.append(onset_arr_ori,onset_ori)
            offset_arr_ori = np.append(offset_arr_ori,offset_ori)
            #############################################################

            changes = np.convolve(krn, prob_med_filt) #中值滤波
            onset_frames = np.where(changes==1)[0]
            print("onset_frames", onset_frames.shape)
            offset_frames = np.where(changes==-1)[0]

            onset = (onset_frames+1) * conf.features.hop_mel / conf.features.sr
            onset = onset + start_index_query 
            offset = (offset_frames+1) * conf.features.hop_mel / conf.features.sr
            offset = offset + start_index_query

            name = np.repeat(audio_name,len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)

        df_out_ori = pd.DataFrame({'Audiofilename':name_arr_ori,'Starttime':onset_arr_ori,'Endtime':offset_arr_ori})
        csv_path_ori = os.path.join(conf.path.work_path,'src','output_csv','ori','Eval_out_ori_1.csv')
        df_out_ori.to_csv(csv_path_ori,index=False)

        df_out_tim = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        csv_path_tim = os.path.join(conf.path.work_path,'src','output_csv','tim','Eval_out_tim_1.csv')
        df_out_tim.to_csv(csv_path_tim,index=False)

    if conf.set.test: # It only be used when test the final dataset of DCASE2021 task5

        device = 'cuda'

        # init_seed()
        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])
        all_feat_files = sorted([file for file in glob(os.path.join(conf.path.feat_test,'*.h5'))])
        evaluator = Evaluator(device=device)
        model = get_model('Protonet',19).cuda()
        student_model = get_model('Protonet',19).cuda()
        ckpt_path = '/home/ydc/DACSE2021/sed-tim-base/check_point'
        #ckpt_path = '/home/ydc/DACSE2021/task5/best2'
        #ckpt_path = '/home/ydc/DACSE2021/sed-tim-base/pre_best'
        hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel) # 0.05*22050//256 == 4
        k_q = 128
        iter_num = 0
        for feat_file in all_feat_files[:1]:
            print('file name ',feat_file)
            feat_name = feat_file.split('/')[-1]
            audio_name = feat_name.replace('h5','wav')
            print("Processing audio file : {}".format(audio_name))
            hdf_eval = h5py.File(feat_file,'r')
            strt_index_query =  hdf_eval['start_index_query'][:][0]
            result, num= evaluator.run_full_evaluation(test_file=audio_name[:-4],model=model,student_model=student_model,
                                                        model_path=ckpt_path,hdf_eval=hdf_eval,conf=conf,k_q=k_q,iter_num=iter_num) # only update W
            # result, num= evaluator.run_full_evaluation_model_w(test_file=audio_name[:-4],model=model,student_model=student_model,
            #                                             model_path=ckpt_path,hdf_eval=hdf_eval,conf=conf,k_q=k_q,iter_num=iter_num) # updata model and W
            # num 返回query 的长度
            predict = result[0]
            if predict.shape[0]>num:
                n_ = predict.shape[0]//k_q
                print('n_ ',n_)
                prob_final = predict[:(n_-1)*k_q]
                n_last = num - prob_final.shape[0]
                print('n_last ',n_last)
                prob_final = np.concatenate((prob_final,predict[-n_last:]))
                print('prob_final ',prob_final.shape)
            else:
                prob_final = predict
            
            assert num == prob_final.shape[0]
            krn = np.array([1, -1])
            prob_thresh = np.where(prob_final > 0.5, 1, 0) # 70572
            prob_pos_final = prob_final * prob_thresh
            changes = np.convolve(krn, prob_thresh) # 70573
            onset_frames = np.where(changes == 1)[0]
            print('onset_frames ',onset_frames.shape)
            offset_frames = np.where(changes == -1)[0]
            str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr # 转时间？
            print('str_time_query ',str_time_query) # 322.5
            onset = (onset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
            onset = onset + str_time_query
            offset = (offset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
            offset = offset + str_time_query
            assert len(onset) == len(offset)
            name = np.repeat(audio_name,len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)

        df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        csv_path = os.path.join(conf.path.work_path,'Eval_out_tim_test.csv')
        df_out.to_csv(csv_path,index=False)

def medFilt(detections, median_window):

    if median_window %2==0:
        median_window-=1
    x = detections
    k = median_window
    assert k%2 == 1, "Median filter length must be odd"
    assert x.ndim == 1, "Input must be one dimensional"
    k2 = (k - 1) // 2
    y = np.zeros((len(x),k),dtype=x.dtype)
    y[:,k2]=x
    for i in range(k2):
        j = k2 -1
        y[j:,i]=x[:-j]
        y[:j,i]=x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]

    return np.median(y,axis=1)

def medFilt0(detections, median_window,hop_seg):

    if median_window %2==0:
        median_window-=1
    x = detections
    k = median_window

    seg_num, seg_len = x.shape
   
    if k%2==0:
        k-=1
    k2 = (k-1)//2
    y = np.zeros((seg_num*hop_seg+seg_len-hop_seg,k), dtype=x.dtype)

    for j in range(k):
        for i in range(seg_num):
            start_y = i+j
            if i<j:
                y[i*hop_seg:(i+1)*hop_seg,j] = y[i*hop_seg:(i+1)*hop_seg,j-1]
            elif start_y >=seg_num:
                continue
            else:
                y[start_y*hop_seg:(start_y+1)*hop_seg,j] = x[i,j*hop_seg:(j+1)*hop_seg]
        y[seg_num*hop_seg:,j]=x[-1,-(seg_len-hop_seg):] #按照尾部补齐方法，会影响尾部效果
    return np.median(y,axis=1)

def flatten_res(detections,hop_seg,nframe):
    x = detections
    seg_num,seg_len = x.shape

    y = np.zeros(nframe,dtype=x.dtype)
    for i in range(seg_num-2):
        y[(i+1)*hop_seg:(i+2)*hop_seg] = x[i,hop_seg:2*hop_seg]
    y[:hop_seg] = x[0,:hop_seg]
    y[(seg_num-1)*hop_seg:nframe] = x[seg_num-1,-(nframe-(seg_num-1)*hop_seg):]
    return y

def flatten_res_melt(detections,hop_seg,nframe):
    x= detections
    seg_num,seg_len = x.shape

    y = np.zeros((5,nframe),dtype=x.dtype)
    for i in range(seg_num-1):
        y[0,i*hop_seg:(i+1)*hop_seg] = x[i,:hop_seg]
        y[1,(i+1)*hop_seg:(i+2)*hop_seg] = x[i,hop_seg:2*hop_seg]
        y[2,(i+2)*hop_seg:(i+3)*hop_seg] = x[i,2*hop_seg:3*hop_seg]
        y[3,(i+3)*hop_seg:(i+4)*hop_seg] = x[i,3*hop_seg:4*hop_seg]
        y[4,(i+4)*hop_seg:(i+5)*hop_seg] = x[i,4*hop_seg:5*hop_seg]
    
    for i in range(5):
        y[i,:i*hop_seg] = x[0,:i*hop_seg]
    
    for i in range(5):
        start = i*hop_seg
        end = start+hop_seg*(seg_num-1)
        bu = nframe-end
        y[i,-bu:]=x[seg_num-1,-bu:]
    y = np.mean(y,axis=0)
    new_y = np.where(y>0.5,1,0)
    return new_y 

if __name__ == '__main__':   
    main()


