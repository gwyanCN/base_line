import numpy as np
# from sacred import Ingredient
from utils.util import warp_tqdm, compute_confidence_interval, load_checkpoint,load_checkpoint_sep
from utils.util import load_pickle, save_pickle
from utils.util import save_checkpoint
import os
import torch
import collections
import torch.nn.functional as F
from utils.tim import TIM, TIM_GD
from datasets.Datagenerator import Datagen_test,Datagen_train_select
from datasets.batch_sampler import EpisodicBatchSampler
from utils.util import warp_tqdm, get_metric, AverageMeter,euclidean_dist,save_plot_data
import torch.nn as nn
import time
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
import torchvision

def config():
    number_tasks = 10000
    n_ways = 5
    query_shots = 15
    method = 'baseline'
    model_tag = 'best'
    target_data_path = None  # Only for cross-domain scenario
    target_split_dir = None  # Only for cross-domain scenario
    plt_metrics = ['accs']
    shots = [1, 5]
    used_set = 'test'  # can also be val for hyperparameter tuning
    fresh_start = False

# seed = 2021
# if seed is not None:
#     random.seed(seed)
#     torch.manual_seed(seed)
#     cudnn.deterministic = True

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.number_tasks = 10000
        self.n_ways = 2
        self.query_shots = 15
        self.method = 'tim_gd' #tim_gd
        self.model_tag = 'best'
        self.plt_metrics = ['accs']
        self.shots = [5]
        self.used_set = 'test'
        self.fresh_start = True

    def align_predict_and_query(self,pre_predict,num_query,k_q): # align predict and true the number of query
        if pre_predict.shape[0]>num_query: # deal predict length
            n_ = pre_predict.shape[0]//k_q
            prob_final = pre_predict[:(n_-1)*k_q]
            n_last = num_query - prob_final.shape[0]
            prob_final = np.concatenate((prob_final,pre_predict[-n_last:]))
        else:
            prob_final = pre_predict
        return prob_final

    def best_lower_bound_search(self,pos_num,x_query_labels):
        l = 0.0
        r=0.5
        iterate_num = 0
        ans = (l+r)/2.0
        while iterate_num<50:
            mid = (l+r)/2.0
            x_query_neg_index = torch.where(x_query_labels<mid,torch.ones(x_query_labels.shape[0]),torch.zeros(x_query_labels.shape[0])) # 选出预测为正的样本
            x_q_trian_label = x_query_labels[x_query_neg_index==1]
            if x_q_trian_label.shape[0]>pos_num:
                r = mid
            elif x_q_trian_label.shape[0]<pos_num:
                ans = mid
                l = mid
            else:
                return mid
            iterate_num +=1
        return ans
    def best_upper_bound_search(self,pos_num,x_query_labels):
        l = 0.5
        r=1.0
        iterate_num = 0
        ans = 0.5
        while iterate_num<50:
            mid = (l+r)/2.0
            x_query_pos_index = torch.where(x_query_labels>mid,torch.ones(x_query_labels.shape[0]),torch.zeros(x_query_labels.shape[0])) # 选出预测为正的样本
            x_q_trian_label = x_query_labels[x_query_pos_index==1]
            if x_q_trian_label.shape[0]>pos_num:
                ans = mid
                l = mid
            elif x_q_trian_label.shape[0]<pos_num:
                r = mid
            else:
                return mid
            iterate_num +=1
        return ans
        
    def append_data(self,neg_data,neg_mask,pos1):
        slice_len = pos1.shape[0]
        total_len = int(431/2*2/3)
        c1_repeat = torch.tile(pos1,[total_len//slice_len+1,1])[:total_len,:]
        slice_c1 = []
        for i in range(total_len//slice_len+1):
            slice_c1.append(c1_repeat[i*slice_len:(i+1)*slice_len])
        distance_num = len(slice_c1) #pos之间的间距

        total_distance = int(431/2/3)

        mean_slice_distance = total_distance//(distance_num)
        distance_list = [mean_slice_distance for i in range(distance_num)]
        shake_distance_list = []
        for i in range(len(distance_list)//2):
            shake_distance = np.random.randint(0,mean_slice_distance)
            shake_distance_list.append(shake_distance)
            shake_distance_list.append(-shake_distance)
        if (len(distance_list)-len(shake_distance_list))==1:
            shake_distance_list.append(0)
        assert len(shake_distance_list)==len(distance_list)
        random.shuffle(shake_distance_list)
        distance_list = [i+j for i,j in zip(distance_list,shake_distance_list)]
        start = 0
        end = 0
        for index,dis in enumerate(distance_list):
            start = end+dis
            end = start+slice_c1[index].shape[0]
            neg_data[start:end,:] = slice_c1[index][:,:]
            neg_mask[start:end,:] = torch.ones_like(slice_c1[index][:,:])
        return neg_data,neg_mask

    def from_teacher_to_student(self,model_path,student,task_dict,W,pre_predict,num_query,k_q,iter_num,test_file,loaders_dic,conf): # control the KD learning
        prob_final = self.align_predict_and_query(pre_predict,num_query,k_q)
        # x_query = save_dict['x_query']  
        # x_pos_train = save_dict['x_pos'] # support sample, thoes sapmle are very small, just 5 shots

        extracted_features_dic = self.extract_features(student,loaders_dic)

        z_q = extracted_features_dic['query_features']
        z_s = extracted_features_dic['pos_features']
        y_s = torch.ones(z_s.shape[0])
        z_t,y_t = extracted_features_dic['train_features'], extracted_features_dic['train_labels']

        mask = torch.where(y_t>0,1,0)
        mask = torch.cat([mask,y_s])

        x_query = torch.from_numpy(x_query)
        assert prob_final.shape[0]==x_query.shape[0]
        x_query_labels = torch.from_numpy(prob_final) 
        hyper_high_confident_num = 400 # this is a hyper-parameter, you can set it by yourself
        thres_pos = self.best_upper_bound_search(hyper_high_confident_num,x_query_labels) 
        x_query_pos_index = torch.where(x_query_labels>=thres_pos,torch.ones(x_query_labels.shape[0]),torch.zeros(x_query_labels.shape[0])) # 选出预测为正的样本
        x_query_tr_pos = x_query[x_query_pos_index==1] # the query predict as positive
        x_query_tr_pos_label = x_query_labels[x_query_pos_index==1] # get thier predict probability
        thresh_neg = self.best_lower_bound_search(x_query_tr_pos_label.shape[0]+x_pos_train.shape[0],x_query_labels)
        x_query_neg_index = torch.where(x_query_labels<thresh_neg,torch.ones(x_query_labels.shape[0]),torch.zeros(x_query_labels.shape[0])) # 选出预测为正的样本
        x_q_trian = x_query[x_query_neg_index==1] # get exaplem by index,note x_query represent mel spectrum
        x_q_fake_label = x_query_labels[x_query_neg_index==1] # thier predict label by previous model
        x_q_trian = torch.cat([x_q_trian,x_query_tr_pos],0) # mix up pos sample and random sample
        x_q_fake_label = torch.cat([x_q_fake_label,x_query_tr_pos_label],0) # label
        assert x_q_trian.shape[0]==x_q_fake_label.shape[0] # judge the number of label and train number is same
        x_pos_train = torch.from_numpy(x_pos_train) # convert numpy to tensor
        x_pos_label = torch.ones(x_pos_train.shape[0]) # thier label is certainty, 1
        assert x_pos_train.shape[0]==x_pos_label.shape[0] # judege
        x_train = torch.cat([x_q_trian,x_pos_train],0) # add support sample, we need those label,because thier label is true
        y_train = torch.cat([x_q_fake_label,x_pos_label],0) # finally, we have get all the sample for student to train.
        model_path = '/home/ydc/DACSE2021/sed-tim-base/check_point/' + str(iter_num)
        student = self.train_student(x_train,y_train,student,W,x_pos_train,x_pos_label,model_path) #  get student model. note, we need x_pos to updata student according to W.
        # torch.save(student,'/home/ydc/DACSE2021/task5/sed-tim/check_point/model/best_55per.pth')
        model_path = '/home/ydc/DACSE2021/sed-tim-base/check_point/' + str(iter_num)+'/'
        extracted_features_dic = self.extract_features(model=student, model_path=model_path, model_tag='student',
                                    used_set=test_file, fresh_start=True,loaders_dic=loaders_dic,test_student=1) # use student model to extract feature

        predict = None
        for shot in self.shots: # 5 shot
            tasks = self.generate_tasks(extracted_features_dic=extracted_features_dic,k_q=k_q)  
            logs = self.run_task(task_dic=tasks,
                                 model=student,test_file=test_file,first=iter_num)
            # l2n_mean, l2n_conf = compute_confidence_interval(logs['acc'][:, -1])
            predict = logs['test']
            W = logs['W']
        return predict,W,student
        
    def run_full_evaluation(self,test_file, model,model_path,student_model,hdf_eval,conf,k_q,iter_num, n_fold):
        """
        Run the evaluation over all the tasks in parallel
        inputs:
            model : The loaded model containing the feature extractor
            loaders_dic : Dictionnary containing training and testing loaders
            model_path : Where was the model loaded from
            model_tag : Which model ('final' or 'best') to load
            method : Which method to use for inference ("baseline", "tim-gd" or "tim-adm")
            shots : Number of support shots to try

        returns :
            results : List of the mean accuracy for each number of support shots
        """
        print("=> Runnning full evaluation with method: {}".format(self.method))
        print("=> Load model from: {}".format(model_path))
        load_checkpoint(model=model, model_path=model_path, type=self.model_tag)
        load_checkpoint(model=student_model,model_path=model_path,type=self.model_tag)
        # Get loaders

        loaders_dic,save_dict = self.get_loaders(hdf_eval=hdf_eval,conf=conf,n_fold=n_fold) 
        # Extract features (just load them if already in memory)
        extracted_features_dic = self.extract_features(model=model, model_path=model_path, model_tag=self.model_tag,
                                    used_set=test_file, fresh_start=self.fresh_start,loaders_dic=loaders_dic)
        results = []
        predict = None
        for shot in self.shots: # 5 shot
            tasks,_ = self.generate_tasks(extracted_features_dic=extracted_features_dic,k_q=k_q,conf=conf,
                                    model=student_model,loaders_dic=loaders_dic,n_fold=n_fold)  
            logs = self.run_task(task_dic=tasks,model_student=student_model,
                                 model=model,test_file=test_file,first=0)
            # l2n_mean, l2n_conf = compute_confidence_interval(logs['acc'][:, -1])
            predict = logs['test']
            W = logs['W']
            thre = logs['thre']
            results.append(predict)
        results.append(tasks['MFL'])
        results.append(thre)
        results.append(tasks['mean_pos_len'])
        if iter_num ==0:
            return results, self.number_tasks,logs

        #ML过程
        results = []
        for i in range(iter_num):
            predict,W,student_model= self.from_teacher_to_student(student_model,save_dict,W,predict,self.number_tasks,k_q,i+1,test_file,loaders_dic)
            if i == iter_num-1:
                results.append(predict)
        
        return results, self.number_tasks
    def run_full_evaluation_model_w(self,test_file, model, model_path,student_model,hdf_eval,conf,k_q,iter_num):
        print("=> Runnning full evaluation with method: {}".format(self.method))
        # Load pre-trained model
        load_checkpoint(model=model, model_path=model_path, type=self.model_tag)
        load_checkpoint(model=student_model,model_path=model_path,type=self.model_tag)
        # Get loaders
        loaders_dic,save_dict = self.get_loaders(hdf_eval=hdf_eval,conf=conf) 
        # Extract features (just load them if already in memory)
        extracted_features_dic = self.extract_features(model=model, model_path=model_path, model_tag=self.model_tag,
                                    used_set=test_file, fresh_start=self.fresh_start,loaders_dic=loaders_dic)
        results = []
        predict = None
        for shot in self.shots: # 5 shot
            tasks = self.generate_tasks(extracted_features_dic=extracted_features_dic,k_q=k_q)  
            logs = self.run_task_model_w(task_dic=tasks,
                                 model=model,test_file=test_file,first=0,loaders_dic=loaders_dic,k_q=k_q,model_path=model_path)
            # l2n_mean, l2n_conf = compute_confidence_interval(logs['acc'][:, -1])
            predict = logs['test']
            W = logs['W']
            results.append(predict)
        if iter_num ==0:
            return results, self.number_tasks
        results = []
        for i in range(iter_num):
            predict,W,student_model= self.from_teacher_to_student(student_model,save_dict,W,predict,self.number_tasks,k_q,i+1,test_file,loaders_dic)
            if i == iter_num-1:
                results.append(predict)
        
        return results, self.number_tasks
    def train_student(self,train_data,label,student,W,x_pos_train,x_pos_label,model_path,z_t,y_t,mask):
        # backbone 微调过程
        losses = AverageMeter()
        top1 = AverageMeter()
        device = 'cuda'
        lr = 0.001
        fc = nn.Linear(1024, 2)
        student.cuda()
        fc.cuda()
        train_dataset = torch.utils.data.TensorDataset(train_data, label)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=None,batch_size=128,shuffle=True) 
        student.train()
        fc.train()
        optimizer = torch.optim.Adam([
            {'params': student.encoder[2].parameters(),'lr':0.1*lr},
            {'params': student.encoder[3].parameters(),'lr':0.1*lr},
            {'params': fc.parameters(),'lr': lr*50}], 
            lr=lr) # {'params': student.encoder[2].parameters()},
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5,step_size=10)
        epoches = 5
        best_loss = 1000.0
        best_student = student
        for epoch in range(epoches):
            loss = self.do_epoch(epoch,lr_scheduler,student,train_loader,optimizer,W,fc,x_pos_train,x_pos_label,z_t,y_t,mask)
            is_best = loss.get_avg() < best_loss
            print('loss.get_avg() ',loss.get_avg())
            if is_best:
                best_student = student
            best_loss = min(loss.get_avg(), best_loss)
            # Save checkpoint
            save_checkpoint(state={'epoch': epoch + 1,
                                'arch': 'Protonet',
                                'state_dict': student.state_dict(),
                                'best_prec1': best_loss,
                                'optimizer': optimizer.state_dict()},
                            is_best=is_best,
                            folder=model_path)
        return best_student

    def cross_entropy(self, logits, one_hot_targets, reduction='batchmean'):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        return - (one_hot_targets * logsoftmax).sum(1).mean()

    def cross_entropy_t(self, logits, targets, mask, reduction='batchmean'):

        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)

        log_pos = torch.gather(logsoftmax, 1, (targets*mask).long())
        return - (log_pos * mask).sum() / mask.sum()

    def compute_train(self,model,sample):
        batch_size, win, ndim = sample.shape
        list_vec = []
        for i in np.arange(0,batch_size,64):
            list_vec.append(model(sample[i:i+64].cuda(),step=0))
            outputs_samples = torch.cat(list_vec, 0)
        logits = model.fc(outputs_samples)
        return logits

    def do_epoch(self, epoch, scheduler, model,train_loader,optimizer,W,fc,x_pos_train,x_pos_label,z_t,y_t,mask,disable_tqdm=False,device='cuda'):  # 可以看做基类训练，不需要划分支持集，查询集
        batch_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        W_mean = W.mean(0)
        x_pos_train = x_pos_train.cuda()
        tqdm_train_loader = warp_tqdm(train_loader, disable_tqdm) 
        for i, (input, target) in enumerate(tqdm_train_loader):

            choice_id = random.choice(range(4))
            select_sub_train = torch.cat((z_t[choice_id:24*19:4],z_t[24*19+choice_id*2::4*2]),0).contiguous().cuda()
            select_y_t = torch.cat((y_t[choice_id:24*19:4],y_t[24*19+choice_id*2::4*2]),0).contiguous().cuda()
            select_mask_t = torch.cat((mask[choice_id:24*19:4],mask[24*19+choice_id*2::4*2]),0).contiguous().cuda()
            logits_t = self.compute_train(model,select_sub_train).flatten(end_dim=1)
            select_y_t = select_y_t.reshape(-1,1)
            select_mask_t = select_mask_t.reshape(-1,1)
            ce_t = self.cross_entropy_t(logits_t,select_y_t,select_mask_t)

            #读取输入数据和标签(上一个循环预测的置信度)
            input, target = input.to(self.device), target.to(device, non_blocking=True) # move to cuda
            feature,_ = model(input,True)
            index = torch.randperm(len(x_pos_train))[:100]
            x_sub_pos_train = x_pos_train[index]
            x_sub_pos_label = x_pos_label[index]

            pos_feature = model(x_sub_pos_train,step=0)
            pos_feature = pos_feature[x_sub_pos_label==1]
            #此处设定阈值为0.5；因为上步搜索选择阈值时，负例阈值上限为0.5
            choose = torch.where(target>0.5,1,0).cuda()
            assert choose.shape[0] == target.shape[0]

            neg_w = feature[choose==0] # 68
            neg_mul = (1-target[choose==0]).view(-1,1)
            neg_mul = neg_mul.repeat(1,neg_w.shape[1])
            neg_w_wi = torch.mul(neg_w,neg_mul) # 

            pos_w = pos_feature.mean(0) # positive samples
            neg_w_wi = neg_w_wi.mean(0)
            target_neg = 1-target
            target_pos = target.view(-1,1)
            target_neg = target_neg.view(-1,1)
            target_one_hot = torch.cat([target_pos,target_neg],1)
            logits = fc(feature).flatten(end_dim=1)
            loss_ce = self.cross_entropy(logits,target_one_hot)
            loss_w = 0.6*torch.cosine_similarity(W_mean[0],pos_w,dim=0) - 0.4*torch.cosine_similarity(W_mean[0],neg_w_wi,dim=0)
            loss = 0.7*loss_ce + 0.3*loss_w + 0.1*ce_t

            p_t = torch.cosine_similarity(W_mean[0],pos_w,dim=0)
            ls = []
            for k in range(neg_w.shape[0]):
                ls.append(torch.cosine_similarity(W_mean[0],neg_w[k],dim=0))
            T_t = 1
            fenmu = 0
            for t in ls:
                fenmu+= torch.exp(t/T_t)
            loss_clr = -torch.log(torch.exp(p_t/T_t)/fenmu)
           
             # thoese hyper parameter you can set by your self
            #loss = 0.5*loss_ce + 0.5*loss_clr # thoese hyper parameter you can set by your self
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 20== 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       loss=losses))
        return losses
            
    def run_task(self, task_dic, model_student, model,test_file,first):
        # Build the TIM classifier builder
        tim_builder = self.get_tim_builder(model_student,model,self.method,test_file,first) # choose the update methods
        # Extract support and query
        y_s = task_dic['y_s']  # n_task*?*?
        z_s, z_q = task_dic['z_s'], task_dic['z_q']
        min_len = task_dic['MFL']
        mean_pos_len = task_dic['mean_pos_len']
        z_t,y_t = task_dic['z_t'], task_dic['y_t']
        mask = task_dic['mask']
        
        # merge_images= z_s.clone()
        # merge_images=torchvision.utils.make_grid(merge_images.unsqueeze(1), nrow=10,padding = 1)
        # self.writer.add_img(self.file_name+"support",merge_images,1)
        
        # Transfer tensors to GPU if needed
        support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
        sub_train = z_t.to(self.device)
        mask = mask.to(self.device)

        y_s = y_s.long().to(self.device) 
        y_t = y_t.long().to(self.device) 
      
        # Initialize weights
        tim_builder.compute_lambda(support=support, query=query, y_s=y_s) # lambda
        print('tim_builder.loss_weights',tim_builder.loss_weights[0])
        print('self.number_task ',self.number_tasks)
        tim_builder.init_weights(support=support, y_s=y_s, query=query, sub_train=sub_train, y_t=y_t) # init W
        tim_builder.compute_FB_param(query)
        # Run adaptation
        tim_builder.run_adaptation(support=support, query=query, y_s=y_s,min_len=mean_pos_len, sub_train=sub_train,y_t=y_t,mask_t=mask) # update
        # Extract adaptation logs
        logs = tim_builder.get_logs()
        return logs

    def run_task_model_w(self, task_dic, model,test_file,first,loaders_dic,k_q,model_path):
        # Build the TIM classifier builder
        tim_builder = self.get_tim_builder(model,self.method,test_file,first) 

        # Extract support and query
        y_s = task_dic['y_s']  # n_task*?*?
        z_s, z_q = task_dic['z_s'], task_dic['z_q']

        # Transfer tensors to GPU if needed
        support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
        # print('y_s ',y_s.shape)
        y_s = y_s.long().squeeze(2).to(self.device) #
        # Perform normalizations required
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)
        # Initialize weights
        tim_builder.compute_lambda(support=support, query=query, y_s=y_s) 
        print('tim_builder.loss_weights',tim_builder.loss_weights[0])
        print('self.number_task ',self.number_tasks)
        tim_builder.init_weights(support=support, y_s=y_s, query=query) 
        tim_builder.compute_FB_param(query)
        # Run adaptation
        model_new,nums,iters = tim_builder.run_adaptation_model_w(support=support, query=query, y_s=y_s,nums=1) # update
        while nums<iters:
            print('nums ',nums)
            extracted_features_dic = self.extract_features(model=model_new, model_path=model_path, model_tag=self.model_tag,
                                    used_set=test_file, fresh_start=1,loaders_dic=loaders_dic)
            tasks = self.generate_tasks(extracted_features_dic=extracted_features_dic,k_q=k_q)  # generate task
            y_s = tasks['y_s']  # n_task*?*?
            z_s, z_q = tasks['z_s'], tasks['z_q']
            # Transfer tensors to GPU if needed
            support = z_s.to(self.device)  # [ N * (K_s + K_q), d]
            query = z_q.to(self.device)  # [ N * (K_s + K_q), d]
            # print('y_s ',y_s.shape)
            y_s = y_s.long().squeeze(2).to(self.device) # 
            model_new,nums,iters = tim_builder.run_adaptation_model_w(support=support, query=query, y_s=y_s,nums=nums) 
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)
        # Extract adaptation logs
        logs = tim_builder.get_logs()
        return logs

    def get_tim_builder(self, model_student,model,method,test_file,first):
        # Initialize TIM classifier builder
        tim_info = {'model_student':model_student,'model': model,'test_file': test_file,'first': first}
        if method == 'tim_adm':
            tim_builder = TIM_ADM(**tim_info)
        elif method == 'tim_gd':
            tim_builder = TIM_GD(**tim_info)
        elif method == 'baseline':
            tim_builder = TIM(**tim_info)
        else:
            raise ValueError("Method must be in ['tim_gd', 'tim_adm', 'baseline']")
        return tim_builder

    def get_loaders(self, hdf_eval,conf, n_fold):
        # First, get loaders
        loaders_dic = {}
        gen_eval = Datagen_test(hdf_eval,conf)
        gen_eval.getMeanStd(path=f"%s/mean_var_fold{n_fold+1}"%conf.path.work_path)
        
        X_pos_1, X_pos_2, X_pos_3, X_pos_4, X_pos_5,X_neg_1, X_neg_2, X_neg_3, X_neg_4, X_neg_5, X_query = gen_eval.generate_eval()

        save_dict = {}
        save_dict['x_pos_1'] = X_pos_1
        save_dict['x_pos_2'] = X_pos_2
        save_dict['x_pos_3'] = X_pos_3
        save_dict['x_pos_4'] = X_pos_4
        save_dict['x_pos_5'] = X_pos_5
       
        save_dict.update({'x_query': X_query})
        self.number_tasks = X_query.shape[0]

        X_pos_1 = torch.tensor(X_pos_1)
        Y_pos_1 = torch.LongTensor(np.ones(X_pos_1.shape[0]))

        X_pos_2 = torch.tensor(X_pos_2)
        Y_pos_2 = torch.LongTensor(np.ones(X_pos_2.shape[0]))

        X_pos_3 = torch.tensor(X_pos_3)
        Y_pos_3 = torch.LongTensor(np.ones(X_pos_3.shape[0]))

        X_pos_4 = torch.tensor(X_pos_4)
        Y_pos_4 = torch.LongTensor(np.ones(X_pos_4.shape[0]))

        X_pos_5 = torch.tensor(X_pos_5)
        Y_pos_5 = torch.LongTensor(np.ones(X_pos_5.shape[0]))

        X_neg_1 = torch.tensor(X_neg_1)
        Y_neg_1 = torch.LongTensor(np.zeros(X_neg_1.shape[0]))

        X_neg_2 = torch.tensor(X_neg_2)
        Y_neg_2 = torch.LongTensor(np.zeros(X_neg_2.shape[0]))

        X_neg_3 = torch.tensor(X_neg_3)
        Y_neg_3 = torch.LongTensor(np.zeros(X_neg_3.shape[0]))

        X_neg_4 = torch.tensor(X_neg_4)
        Y_neg_4 = torch.LongTensor(np.zeros(X_neg_4.shape[0]))

        X_neg_5 = torch.tensor(X_neg_5)
        Y_neg_5 = torch.LongTensor(np.zeros(X_neg_5.shape[0]))

        X_query = torch.tensor(X_query)
        Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

        # for i in range(5):
        #     self.writer.add_img("X_pos",merge_image,0)
        
        query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
        q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=conf.eval.query_batch_size,shuffle=False) # 按顺序来
        loaders_dic['query'] = q_loader

        pos_dataset_1 = torch.utils.data.TensorDataset(X_pos_1, Y_pos_1)
        pos_dataset_2 = torch.utils.data.TensorDataset(X_pos_2, Y_pos_2)
        pos_dataset_3 = torch.utils.data.TensorDataset(X_pos_3, Y_pos_3)
        pos_dataset_4 = torch.utils.data.TensorDataset(X_pos_4, Y_pos_4)
        pos_dataset_5 = torch.utils.data.TensorDataset(X_pos_5, Y_pos_5)
        neg_dataset_1 = torch.utils.data.TensorDataset(X_neg_1, Y_neg_1)
        neg_dataset_2 = torch.utils.data.TensorDataset(X_neg_2, Y_neg_2)
        neg_dataset_3 = torch.utils.data.TensorDataset(X_neg_3, Y_neg_3)
        neg_dataset_4 = torch.utils.data.TensorDataset(X_neg_4, Y_neg_4)
        neg_dataset_5 = torch.utils.data.TensorDataset(X_neg_5, Y_neg_5)

        pos_loader_1 = torch.utils.data.DataLoader(dataset=pos_dataset_1,batch_sampler=None, batch_size=50,shuffle=False)
        pos_loader_2 = torch.utils.data.DataLoader(dataset=pos_dataset_2,batch_sampler=None, batch_size=50,shuffle=False)
        pos_loader_3 = torch.utils.data.DataLoader(dataset=pos_dataset_3,batch_sampler=None, batch_size=50,shuffle=False)
        pos_loader_4 = torch.utils.data.DataLoader(dataset=pos_dataset_4,batch_sampler=None, batch_size=50,shuffle=False)
        pos_loader_5 = torch.utils.data.DataLoader(dataset=pos_dataset_5,batch_sampler=None, batch_size=50,shuffle=False)
        neg_loader_1 = torch.utils.data.DataLoader(dataset=neg_dataset_1,batch_sampler=None, batch_size=50,shuffle=False)
        neg_loader_2 = torch.utils.data.DataLoader(dataset=neg_dataset_2,batch_sampler=None, batch_size=50,shuffle=False)
        neg_loader_3 = torch.utils.data.DataLoader(dataset=neg_dataset_3,batch_sampler=None, batch_size=50,shuffle=False)
        neg_loader_4 = torch.utils.data.DataLoader(dataset=neg_dataset_4,batch_sampler=None, batch_size=50,shuffle=False)
        neg_loader_5 = torch.utils.data.DataLoader(dataset=neg_dataset_5,batch_sampler=None, batch_size=50,shuffle=False)
        
        loaders_dic.update({'pos_loader_1': pos_loader_1})
        loaders_dic.update({'pos_loader_2': pos_loader_2})
        loaders_dic.update({'pos_loader_3': pos_loader_3})
        loaders_dic.update({'pos_loader_4': pos_loader_4})
        loaders_dic.update({'pos_loader_5': pos_loader_5})
        loaders_dic.update({'neg_loader_1': neg_loader_1})
        loaders_dic.update({'neg_loader_2': neg_loader_2})
        loaders_dic.update({'neg_loader_3': neg_loader_3})
        loaders_dic.update({'neg_loader_4': neg_loader_4})
        loaders_dic.update({'neg_loader_5': neg_loader_5})

        return loaders_dic,save_dict

    def extract_features(self, model, model_path, model_tag, used_set, fresh_start, loaders_dic,test_student=0):
        # Load features from memory if previously saved ...
        save_dir = os.path.join(model_path, model_tag, used_set)
        filepath = os.path.join(save_dir, 'output.plk')
        if os.path.isfile(filepath) and (not fresh_start):
            extracted_features_dic = load_pickle(filepath)
            print(" ==> Features loaded from {}".format(filepath))
            return extracted_features_dic

        # ... otherwise just extract them
        else:
            print(" ==> Beginning feature extraction")
            os.makedirs(save_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():
            all_features = []
            all_labels = []
            # print("===> Query feature extraction")
            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['query'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs)
                all_labels.append(labels)
            all_features = torch.cat(all_features, 0)
            all_labels = torch.cat(all_labels, 0)
            extracted_features_dic = {'query_features': all_features,
                                      'query_labels': all_labels}
            all_features = []
            all_labels = []

            # print("===> Pos feature extraction")
            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['pos_loader_1'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                all_labels.append(labels)

            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['pos_loader_2'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                all_labels.append(labels)

            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['pos_loader_3'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                all_labels.append(labels)

            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['pos_loader_4'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                all_labels.append(labels)
            
            for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['pos_loader_5'], True)):
                # inputs = inputs.to(self.device)
                # outputs, _ = model(inputs, True)
                all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                all_labels.append(labels)
            # all_features = torch.cat(all_features, 0)
            # all_labels = torch.cat(all_labels, 0)
            extracted_features_dic.update({'pos_features': all_features,
                                      'pos_labels': all_labels})

            all_features = []
            all_labels = []

            # print("===> Neg feature extraction")

            bad_cnt = 0
            try:
                for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['neg_loader_1'], True)):
                    # inputs = inputs.to(self.device)
                    # outputs, _ = model(inputs, True)
                    assert inputs.shape[-2]>0
                    all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                    all_labels.append(labels)
            except:
                bad_cnt +=1

            try:
                for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['neg_loader_2'], True)):
                    # inputs = inputs.to(self.device)
                    # outputs, _ = model(inputs, True)
                    assert inputs.shape[-2]>0
                    all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                    all_labels.append(labels)
            except:
                bad_cnt +=1

            try:
                for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['neg_loader_3'], True)):
                    # inputs = inputs.to(self.device)
                    # outputs, _ = model(inputs, True)
                    assert inputs.shape[-2]>0
                    all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                    all_labels.append(labels)
            except:
                bad_cnt +=1

            try:
                for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['neg_loader_4'], True)):
                    # inputs = inputs.to(self.device)
                    # outputs, _ = model(inputs, True)
                    assert inputs.shape[-2]>0
                    all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                    all_labels.append(labels)
            except:
                bad_cnt +=1

            try:
                for i, (inputs, labels) in enumerate(warp_tqdm(loaders_dic['neg_loader_5'], True)):
                    # inputs = inputs.to(self.device)
                    # outputs, _ = model(inputs, True)
                    assert inputs.shape[-2]>0
                    all_features.append(inputs.reshape(-1,inputs.shape[-1]))
                    all_labels.append(labels)
            except:
                bad_cnt +=1

            for i in range(bad_cnt):
                index = i%(5-bad_cnt)
                all_features.append(all_features[index])
                all_labels.append(all_labels[index])

            # all_features = torch.cat(all_features, 0)
            # all_labels = torch.cat(all_labels, 0)
            extracted_features_dic.update({'neg_features': all_features,
                                      'neg_labels': all_labels})
            extra_neg = self.get_extra_neg(model,extracted_features_dic['pos_features'], extracted_features_dic['query_features'])
            extracted_features_dic.update({'extra_neg':extra_neg})

        print(" ==> Saving features to {}".format(filepath))
        save_pickle(filepath, extracted_features_dic)
        return extracted_features_dic

    def get_extra_neg(self,model,pos_features,query_features):
        model.eval()
        # print("===> extra Neg feature extraction")
        with torch.no_grad():
            list_pos_vec = []
            for i in warp_tqdm(range(len(pos_features)), True):

                pos_input = torch.zeros_like(query_features[:1])
                start = max((query_features.shape[1]-pos_features[i].shape[0])//2,0)
                end = min(start+pos_features[i].shape[0],query_features.shape[1])
                pos_input[0,start:end] = pos_features[i][:(end-start)]

                pos_outputs = model.forward_encoder_test(pos_input.to(self.device))
                list_pos_vec.append(pos_outputs[:,start:end])
            
            pos_w = torch.cat(list_pos_vec,1).mean(1,keepdim=True)
            pos_w = F.normalize(pos_w,dim=2)

            list_neg = []
            for i in warp_tqdm(range(query_features.shape[0]),True):
                que_outputs = model.forward_encoder_test(query_features[i:(i+1)].to(self.device))
                _,neg_idx = (que_outputs*pos_w).sum(2).sort()
                list_neg.append(query_features[i, neg_idx[0,50:60],:])
            torch_neg = torch.cat(list_neg,0)

        return torch_neg
    def get_task(self, extracted_features_dic,index,k_q,conf,model,loaders_dic,n_fold):

        """
        inputs:
            extracted_features_dic : Dictionnary containing all extracted features and labels
            shot : Number of support shot per class
            n_ways : Number of ways for the task

        returns :
            task : Dictionnary : z_support : torch.tensor of shape [n_ways * shot, feature_dim]
                                 z_query : torch.tensor of shape [n_ways * query_shot, feature_dim]
                                 y_support : torch.tensor of shape [n_ways * shot]
                                 y_query : torch.tensor of shape [n_ways * query_shot]
        """
        query_features = extracted_features_dic['query_features']
        
        pos_features = extracted_features_dic['pos_features']
        # pos_labels = extracted_features_dic['pos_labels']

        neg_features = extracted_features_dic['neg_features']
        # neg_labels = extracted_features_dic['neg_labels']

        extra_neg = extracted_features_dic['extra_neg']

        query_samples = []
        query_samples.append(query_features)
        z_query = torch.cat(query_samples,0)

        support_samples = []
        
        y_list = []

        list_len_pos =[fea.shape[0] for fea in pos_features]
        list_len_neg =[fea.shape[0] for fea in neg_features]

        mean_pos_len = sum(list_len_pos)/len(list_len_pos)
        print('mean_pos_len:%s'%mean_pos_len)
        med_filter_len = min(list_len_pos)
        print('min_pos_len:%s'%med_filter_len)
        n_frame = 431
        max_seg_len = n_frame//2
        # print("====> Build features")
        for i in warp_tqdm(range(128),True):
            list_X = []
            list_Y = []
            len_cnt = 0
            while True:

                neg_id = random.choice(range(len(neg_features)))

                if sum(list_len_neg)<10 and i%2==1 and neg_features[neg_id].shape[0]<5:
                    neg_len = random.choice(range(10,50))
                    start = random.choice(range(extra_neg.shape[0]-neg_len))
                    list_X.append(extra_neg[start:start+neg_len])
                    list_Y.append(torch.zeros(neg_len))
                    len_cnt += neg_len
                else:
                    if neg_features[neg_id].shape[0] > max_seg_len:
                        start = random.choice(range(neg_features[neg_id].shape[0]- max_seg_len))
                        end = start+ max_seg_len
                    else:
                        start = 0
                        end = neg_features[neg_id].shape[0]
                    list_X.append(neg_features[neg_id][start:end])
                    list_Y.append(torch.zeros(end-start))
                    len_cnt +=end -start

                pos_id = random.choice(range(len(pos_features)))
                if pos_features[pos_id].shape[0] >max_seg_len:
                    start = random.choice(range(pos_features[pos_id].shape[0]-max_seg_len))
                    end = start+max_seg_len
                else:
                    start =0
                    end = pos_features[pos_id].shape[0]

                if len_cnt+end-start <=n_frame:
                    list_X.append(pos_features[pos_id][start:end])
                    list_Y.append(torch.ones(end-start))
                    len_cnt += end-start
                
                if len_cnt>n_frame:
                    break

            support_samples.append(torch.cat(list_X,0)[:n_frame])
            y_list.append(torch.cat(list_Y,0)[:n_frame])
            
        z_support = torch.stack(support_samples, 0)
        y_support = torch.stack(y_list, 0)
        #reload sub_train_datasets
        sub_train_dataset = conf.eval.trainDatasets + f"_{n_fold}.pth"
        try:
            sub_train_datasets  =torch.load(sub_train_dataset)
            z_sub_train = sub_train_datasets['z_sub_train']
            y_sub_train = sub_train_datasets['y_sub_train']
          
        except:
            meanVarPath = f"%s/mean_var_fold{n_fold+1}"%conf.path.work_path
            gen_sub_train = Datagen_train_select(conf)
            gen_sub_train.getMeanStd(meanVarPath)
            z_sub_train,y_sub_train = gen_sub_train.generate_eval()
            z_sub_train = torch.tensor(z_sub_train).type_as(z_support)
            y_sub_train = torch.tensor(y_sub_train).type_as(y_support)
           
            sub_train_datasets ={
                'z_sub_train': z_sub_train,
                'y_sub_train': y_sub_train
            }
            torch.save(sub_train_datasets,sub_train_dataset)
        mask = torch.where(y_sub_train>0,1,0)  
        z_train_dataset = torch.utils.data.TensorDataset(z_sub_train,y_sub_train)
        z_train_dataloader = torch.utils.data.DataLoader(dataset=z_train_dataset,batch_sampler=None,batch_size=64,num_workers=0,shuffle=True)
        loaders_dic['z_trainloader'] = z_train_dataloader
    
        sep_pos = []
        sep_pos_mask = []
        sep_neg = []
       
        hop_seg = 86
        seg_len = 431
        #先提neg段
        if sum(list_len_neg)<max_seg_len:
            seg_win_len = int(np.mean(list_len_neg))
            
            for i in range(128):
                random.shuffle(neg_features)
                negs = torch.cat(neg_features, dim=0)
                start = np.random.randint(0,sum(list_len_neg)-seg_win_len)
                end = start + seg_win_len
                neg = negs[start:end]
                repeatNum = seg_len // neg.shape[0] + 1
                sep_neg.append(torch.tile(neg,[repeatNum,1])[:seg_len])
        else:       
            for sub_neg in neg_features:
                start = 0
                end = start + seg_len
                nframes = sub_neg.shape[0]
                if nframes > seg_len:
                    while end<nframes and start<nframes:
                        sep_neg.append(sub_neg[start:end])
                        start +=hop_seg
                        end = start + seg_len
                    if end>nframes and start<nframes:
                        sep_neg.append(sub_neg[nframes-seg_len:nframes])
                elif nframes<=seg_len:
                    repeatNum = seg_len // nframes + 1
                    sep_neg.append(torch.tile(sub_neg,[repeatNum,1])[:seg_len])
        sep_neg  = torch.stack(sep_neg)
        if sep_neg.shape[0]<128:
            sep_neg = torch.tile(sep_neg,[128//sep_neg.shape[0]+1,1,1])[:128,:,:]
        index_neg = torch.randperm(len(sep_neg))
        sep_neg = sep_neg[index_neg]
        sep_neg = sep_neg[:128,:,:]
        # 再提pos段
        if mean_pos_len >=431:
            for i in range(128):
                pos_ = sep_neg[random.randint(0,len(sep_neg)-1)]
                pos1 = random.sample(pos_features,1)[0]
                if pos1.shape[0]>=431:
                    start = random.randint(0,pos1.shape[0]-431-1)
                    end = start+431
                    assert pos1[start:end].shape[0]==431
                    sep_pos.append(pos1[start:end])
                    sep_pos_mask.append(torch.ones_like(pos1[start:end]))
                else:
                    pos_mask = torch.zeros_like(pos_)
                    start = random.randint(0,431-pos1.shape[0]-1)
                    end = start + pos1.shape[0]
                    pos_[start:end] = pos1
                    pos_mask[start:end] = 1.0
                    assert pos_.shape[0]==431
                    sep_pos.append(pos_)
                    sep_pos_mask.append(pos_mask)
            sep_pos = torch.stack(sep_pos)
            sep_pos_mask = torch.stack(sep_pos_mask)
        else:
            for i in range(128):
                pos1,pos2 = random.sample(pos_features,2)
                mean_len = np.mean([pos1.shape[0],pos2.shape[0]])
                pos_ = sep_neg[i].clone()
                pos_mask = torch.zeros(431,128,dtype=torch.long)
                if 431>mean_len>431//2:
                    min_shape = min([pos1.shape[0],pos2.shape[0]])
                    max_shape = max([pos1.shape[0],pos2.shape[0]])
                    if min_shape>431:
                        pos_target = [pos1, pos2][np.argmin([pos1.shape[0],pos2.shape[0]])]
                        start = random.randint(0,pos_target.shape[0]-431-1)
                        end = start+431
                        pos_[:] = pos_target[start:end]
                        pos_mask[:] = 1
                    if max_shape<431//2:
                        pos_[:431//2,:], pos_mask[:431//2,:] = self.append_data(pos_[:431//2,:], pos_mask[:431//2,:],pos1)
                        pos_[431//2:,:], pos_mask[431//2:,:] = self.append_data(pos_[431//2:,:], pos_mask[431//2:,:],pos2)
                    elif sum([pos1.shape[0],pos2.shape[0]])<=431: 
                        if np.random.rand()<0.5:
                            num = 431 - sum([pos1.shape[0],pos2.shape[0]])
                            if num>0:
                                randnum =random.randint(0,num//2)
                            else:
                                randnum = 0                           
                            pos_[randnum:randnum+pos1.shape[0],:]=pos1
                            pos_[2*randnum+pos1.shape[0]:,:]=pos2
                            pos_mask[randnum:randnum+pos1.shape[0],:]=1
                            pos_mask[2*randnum+pos1.shape[0]:,:]=1
                        else:
                            num = 431 - sum([pos1.shape[0],pos2.shape[0]])
                            if num>0:
                                randnum = random.randint(0,num)
                            else:
                                randnum = 0
                            pos_[:pos1.shape[0],:]=pos1
                            pos_[randnum+pos1.shape[0]:,:]=pos2
                            pos_mask[:pos1.shape[0],:]=1
                            pos_mask[randnum+pos1.shape[0]:,:]=1
                    else:
                        sample = random.sample([pos1, pos2],1)[0]
                        if sample.shape[0]>431:
                            start = random.randint(0,sample.shape[0]-431-1)
                            end = start+431
                            pos_[:] = sample[start:end]
                            pos_mask[:] = 1
                        elif  sample.shape[0]==431:
                            start = 0
                            end = start+431
                            pos_[:] = sample[start:end]
                            pos_mask[:] = 1
                        else:
                            distance = random.randint(0,431 - sample.shape[0])
                            pos_[distance:distance+sample.shape[0],:]=sample
                            pos_mask[distance:distance+sample.shape[0],:]=1
                else:
                    pos_[:431//2,:], pos_mask[:431//2,:] = self.append_data(pos_[:431//2,:], pos_mask[:431//2,:],pos1)
                    pos_[431//2:,:], pos_mask[431//2:,:] = self.append_data(pos_[431//2:,:], pos_mask[431//2:,:],pos2)
                    
                sep_pos.append(pos_)
                sep_pos_mask.append(pos_mask)
            sep_pos = torch.stack(sep_pos)
            sep_pos_mask = torch.stack(sep_pos_mask)
            if sep_pos.shape[0]<128:
                sep_pos = torch.tile(sep_pos,[128//sep_pos.shape[0]+1,1,1])[:128,:,:]
            if sep_pos_mask.shape[0]<128:
                sep_pos_mask = torch.tile(sep_pos_mask,[128//sep_pos_mask.shape[0]+1,1,1])[:128,:,:]
        pos_index = torch.randperm(len(sep_pos))
        sep_pos = sep_pos[pos_index][:128,:,:]
        sep_pos_mask = sep_pos_mask[pos_index][:128,:,:]
        
        # torch.save(sep_pos,"/media/b227/ygw/Dcase2023/baseline/sep_pos.pth")
        
        z_support = torch.cat([z_support,sep_pos],dim=0)
        y_support = torch.cat([y_support,sep_pos_mask[...,0]],dim=0)
         
        z_sub_train = torch.cat((z_sub_train,z_support),0)
        y_sub_train = torch.cat((y_sub_train,y_support-1),0)
        
        mask = torch.cat((mask,y_support),0)
        
        task = {'z_s': z_support.contiguous(), 'y_s': y_support.contiguous(),
                'z_t': z_sub_train.contiguous(), 'y_t': y_sub_train.contiguous(),'mask':mask.contiguous(),
                'z_q': z_query.contiguous(),'MFL':med_filter_len, 'mean_pos_len':mean_pos_len}
        return task,loaders_dic 
    
    def generate_tasks(self, extracted_features_dic,k_q,conf,model,loaders_dic,n_fold):
        """
        inputs:
            extracted_features_dic :
            shot : Number of support shot per class
            number_tasks : Number of tasks to generate

        returns :
            merged_task : { z_support : torch.tensor of shape [number_tasks, n_ways * shot, feature_dim]
                            z_query : torch.tensor of shape [number_tasks, n_ways * query_shot, feature_dim]
                            y_support : torch.tensor of shape [number_tasks, n_ways * shot]
                            y_query : torch.tensor of shape [number_tasks, n_ways * query_shot] }
        """
        return self.get_task(extracted_features_dic,0,k_q,conf,model,loaders_dic,n_fold)

