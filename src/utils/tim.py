from utils.util import get_mi, get_cond_entropy, get_entropy, get_one_hot
from tqdm import tqdm
import torch
import time
import torch.nn.functional as F
import logging
import os
import math
import numpy as np
import torch.nn as nn 
import random

def config():
    temp = 15 # hyper-parameter 
    loss_weights = [0.7, 1.0, 0.1]  # [Xent, H(Y), H(Y|X)] # hyper-parameter 
    lr = 1e-4
    iter = 15
    alpha = 1.0

class TIM(object):
    def __init__(self, model_student,model,test_file,first):
        self.lr = 1e-3
        is_test = 0
        if is_test:
            self.test_init(test_file,first)
        else:
            self.eval_init(test_file,first)
        self.temp = 0.1 # different model may need different temp value
        self.loss_weights = [0.1, 0.1, 0.1] # [0.1, 0.1, 1]
        #self.loss_weights = [1, 1, 1] # [0.1, 0.1, 1]
        self.model = model
        self.model_student = model_student
        self.init_info_lists()
        self.alpha = 1.0

        self.m = 0.4
        self.s = 64
        self.cos_m =math.cos(self.m)
        self.sin_m =math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
    
    def sigmoid_ramup(self,current,rampup_length):
        if rampup_length==0:
            return 1.0
        else:
            current = np.clip(current,0.0,rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0*phase*phase))

    def get_current_consistency_weight(self,index):
        return self.consistence * self.sigmoid_ramup(index,self.consistence_ramup)

    def test_init(self,test_file,first):
        self.iter = 20
    def eval_init(self,test_file,first):
        self.iter =100
        
    def init_info_lists(self):
        self.timestamps = []
        self.mutual_infos = []
        self.entropy = []
        self.cond_entropy = []
        self.test_probs = []
        self.losses = []

    def process_sep(self,sample, embedding=None,train=False):
        b,_,_ = sample.shape
        if train:
            new_embedding = embedding.repeat_interleave(b,dim=0)
            mask = self.model_sep(sample,new_embedding)
        else:
            with torch.no_grad():
                new_embedding = embedding.repeat_interleave(b,dim=0)
                mask = self.model_sep(sample,new_embedding,step=1)
                sample_sep = sample*mask
                sample_sep = sample_sep.detach()
            return sample_sep,mask

    def get_logits(self, samples, is_train=False, is_class=False,label=None,embedding=False):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        sample_list = []
        mask_sample_list =[]
        mask_list = []
        model = self.model
        bias = self.bias[:,0]
        b_s = 32
        if is_train:
            model.train()
            batch_size,win,ndim = samples.shape
            list_vec = []
            for i in np.arange(0, batch_size,b_s):
                sample = samples[i:i+b_s]
                list_vec.append(model(sample,step=0))
                outputs_samples = torch.cat(list_vec,0)
            if is_class:
                logits = model.fc(outputs_samples)
                return logits
        else:
            list_vec = []
            model.eval()
            with torch.no_grad():
                batch_size, win, ndim = samples.shape
                for i in np.arange(0,batch_size,b_s):
                    sample = samples[i:i+b_s].reshape(-1,win,ndim)
                    list_vec.append(model.forward_encoder_test(sample))
                outputs_samples = torch.cat(list_vec,0)
        
        if None == label:
            logits0 = outputs_samples.matmul(self.weights[:,0:1].transpose(1,2)) + bias
            logits1 = outputs_samples.matmul(model.fc.weight[0].view(1,-1,1)) + model.fc.bias[0].view(1,1,-1)
            logits = torch.cat((logits0,logits1),-1)
        else:
            cosine = F.normalize(outputs_samples,dim=2).matmul(F.normalize(self.weights.transpose(1,2),dim=1))
            sine = torch.sqrt((1.0-torch.pow(cosine,2)).clamp(0,1))
            phi = cosine * self.cos_m - sine*self.sin_m
            phi - torch.where(cosine > self.th, phi, cosine-self.mm)
            one_hot = label
            output = (one_hot * phi) + ((1.0 - one_hot)*cosine)
            logits = output*self.s
        if embedding:
            return logits ,outputs_samples
        else:
            return logits

    def get_tsvad(self,ts_vectors,samples,is_train=False):
        lstm = self.model.Encoder_layers
        fc = self.model.decoder2
        b_s = 128
        batch_size,_,_ = samples.shape
        assert ts_vectors.shape[0]==batch_size
        out_list = []
        if is_train:  
            for i in range(0,batch_size,b_s):
                sample = samples[i:i+b_s]
                ts_vector = ts_vectors[i:i+b_s]
                d_vector = torch.cat([sample,ts_vector],dim=-1)
                ts_vad = lstm(d_vector)
                out = fc(ts_vad).softmax(-1)
                out_list.append(out)
        else:        
            with torch.no_grad():
                for i in range(0,batch_size,b_s):
                    sample = samples[i:i+b_s]
                    ts_vector = ts_vectors[i:i+b_s]
                    d_vector = torch.cat([sample,ts_vector],dim=-1)
                    ts_vad= lstm(d_vector)
                    out = fc(ts_vad).softmax(-1)
                    out_list.append(out)
        return torch.cat(out_list,dim=0)

    def get_preds(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, s_shot, feature_dim]
        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds

    def get_acc(self, preds,label,mask=None):
        if mask == None:
            acc = (preds.argmax(2)==label).int().sum()/label.numel()
        else:
            acc = (mask*(preds.argmax(1,keepdim=True)==label).int()).sum()/mask.sum()
        print('acc:%s'%acc)

    def compute_FB_param(self, features_q):
        logits_q = self.get_logits(features_q) # logits: according to W, calculate results
        logits_q = logits_q.detach()
        q_probs = logits_q.softmax(2) # predict probability
        #probas = self.get_probas(features_q).detach()
        b = q_probs[:,:,0]>0.5
        # print(1.0*b.sum(1)/a.shape[1])
        pos = 1.0*b.sum(1)/q_probs.shape[1]
        neg = 1.0 -pos
        pos = pos.unsqueeze(1)
        neg = neg.unsqueeze(1)
        self.FB_param2 = torch.cat([pos,neg],1)
        self.FB_param = (q_probs).mean(dim=1)

    def init_weights(self, support, query, y_s,sub_train,y_t): 
        self.model.eval()
        t0 = time.time()
        n_tasks = support.size(0)
        n,seq_len,_ = support.shape
        with torch.no_grad():
            outputs_support = self.model.forward_encoder_test(support)
            outputs_support = F.normalize(outputs_support, dim=2)
        # print('n_tasks ',n_tasks)
        one_hot = get_one_hot(y_s) # get one-hot vector
        counts = one_hot.sum(1).view(n_tasks, -1, 1) # 
        
        weights = one_hot.transpose(1, 2).matmul(outputs_support) 
        self.weights = weights.sum(0,keepdim=True) / counts.sum(0,keepdim=True)

        self.weights = F.normalize(self.weights,dim=2)
        self.bias  = torch.tensor([0.1]).reshape(1,1,1).type_as(self.weights)

        self.model.fc.weight[0].data.copy_(self.weights[0,1])
        # self.model_student.fc.weight[0].data.copy_(self.weights[0,1])
        self.weights = self.weights[:,0:1,:]

        self.record_info(new_time=time.time()-t0,
                         support=support,
                         query=query,
                         y_s=y_s)
        self.model.train()

    def init_weights2(self, support, query, y_s): 
        self.model.eval()
        t0 = time.time()
        n_tasks = support.size(0)

        with torch.no_grad():
            outputs_support = self.model.forward_encoder_test(support)
            outputs_query = F.normalize(query)
        # print('n_tasks ',n_tasks)
        one_hot = get_one_hot(y_s) # get one-hot vector
        counts = one_hot.sum(1).view(n_tasks, -1, 1) # 

        weights = one_hot.transpose(1, 2).matmul(outputs_support)
        self.weights = weights.sum(0,keepdim=True) / counts.sum(0,keepdim=True)

        self.weights = F.normalize(self.weights,dim=2)

        self.record_info(new_time=time.time()-t0,
                         support=support,
                         query=query,
                         y_s=y_s)
        self.model.train()
        _, neg_idx = (outputs_query * self.weights[:,1:]).sum(2).sort()

        list_neg = []
        for i in range(query.shape[0]):
            list_neg.append(query[i,neg_idx[i,100:106],:])
        torch_neg = torch.cat(list_neg,0)
        torch_neg = torch.cat((torch_neg,torch_neg),0)

        for i in range(support.shape[0]):
            for j in range(6):
                start = (j*60+20)
                end = start+30
                y_s [i,start:end] = 0
                support[i, start:end] = torch_neg[start:end]
        return support,y_s

    def compute_lambda(self, support, query, y_s): 
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]

        updates :
            self.loss_weights[0] : Scalar
        """
        self.N_s, self.N_q = support.size(1), query.size(1)
        self.num_classes = torch.unique(y_s).size(0) 
        if self.loss_weights[0] == 'auto':
            self.loss_weights[0] = (1 + self.loss_weights[2]) * self.N_s / self.N_q

    def record_info(self, new_time, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """
        with torch.no_grad():
           
            b_s,seq_len,_ = support.shape
            logits_s,s_embedding = self.get_logits(support,embedding=True)
            s_prototype = ((s_embedding*y_s.unsqueeze(-1)).sum(1,keepdim=True) / y_s.unsqueeze(-1).sum(1,keepdim=True)).mean(0,keepdim=True).repeat([1,seq_len,1])
            s_prototype_s = s_prototype[0:1].repeat_interleave(s_embedding.shape[0],dim=0)
            
            logits_s_2 = self.get_tsvad(s_prototype_s,s_embedding).unsqueeze(-1).detach()
            logits_s = logits_s.softmax(2).detach().unsqueeze(-1)
            logits_s = torch.cat([logits_s,logits_s_2],dim=-1).mean(-1)
            
            self.thre = ((logits_s[:,:,1]*y_s).sum(1)/y_s.sum(1)).min().item()
            print('thre:%s'%self.thre)
        
            logits_q,q_embedding = self.get_logits(query,embedding=True)
            logits_q = logits_q.softmax(2).detach()
            q_probs = logits_q.unsqueeze(-1) 

            s_prototype_q = s_prototype[0:1].repeat_interleave(q_embedding.shape[0],dim=0)
            q_probs_2 = self.get_tsvad(s_prototype_q,q_embedding).unsqueeze(-1).detach()
            q_probs = torch.cat([q_probs.cpu(),q_probs_2.cpu()],dim=-1).mean(dim=-1)
            
            self.timestamps.append(new_time) 
            self.mutual_infos.append(get_mi(probs=q_probs.detach().cpu())) 
            self.entropy.append(get_entropy(probs=q_probs.detach().cpu())) # # H(Y_q)
            self.cond_entropy.append(get_cond_entropy(probs=q_probs.detach().cpu())) # # H(Y_q | X_q)
            self.test_probs.append(q_probs[:,:,1].cpu()) 
            self.y_s = y_s.cpu().numpy()
            torch.cuda.empty_cache()

    def get_logs(self):
        self.test_probs = self.test_probs[-1].cpu().numpy() # use the last as results
        self.cond_entropy = torch.cat(self.cond_entropy, dim=1).cpu().numpy()
        self.entropy = torch.cat(self.entropy, dim=1).cpu().numpy()
        self.mutual_infos = torch.cat(self.mutual_infos, dim=1).cpu().numpy()
        self.W = self.weights
        return {'timestamps': self.timestamps, 'mutual_info': self.mutual_infos,
                'entropy': self.entropy, 'cond_entropy': self.cond_entropy, 'losses': self.losses,
                'test': self.test_probs,'W': self.W,'thre':self.thre,'y_s':self.y_s}


class TIM_GD(TIM):
    def __init__(self, model_student,model,test_file,first):
        super().__init__(model=model,model_student=model_student,test_file=test_file,first=first)

    def run_adaptation(self, support, query, y_s, min_len, sub_train, y_t, mask_t):
        t0 = time.time()
        self.min_len = min_len
        self.weights.requires_grad_() # W
        self.bias.requires_grad_() 
        optimizer = torch.optim.Adam([
            {'params':self.model.fc.parameters(),'lr':self.lr},
            {'params':self.model.encoder[2].parameters(),'lr':0.1*self.lr},
            {'params':self.model.encoder[3].parameters(),'lr':0.1*self.lr},
            {'params':self.model.Encoder_layers.parameters(),'lr':0.1*self.lr},
            {'params':self.model.decoder2.parameters(),'lr':0.1*self.lr},  
            {'params':self.weights},{'params':self.bias}], lr=self.lr)
        step_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)
        y_s_one_hot = get_one_hot(y_s)
        self.model.eval()
        
        l3 = 0.2
        self.iter=100
        b_s,seq_len,_ = support.shape
        for i in tqdm(range(self.iter)): # 
            self.model.train()
            
            choice_id = random.choice(range(4))
            select_sub_train = torch.cat((sub_train[choice_id:24*19:4],sub_train[24*19+choice_id*2::4*2]),0).contiguous().cuda()
            select_y_t = torch.cat((y_t[choice_id:24*19:4],y_t[24*19+choice_id*2::4*2]),0).contiguous().cuda()
            select_mask_t = torch.cat((mask_t[choice_id:24*19:4],mask_t[24*19+choice_id*2::4*2]),0).contiguous().cuda()

            select_y_t = select_y_t.reshape(-1,1)
            logits_t = self.get_logits(select_sub_train,is_train=True,is_class=True).reshape(-1,20)
            select_mask_t = select_mask_t.reshape(-1,1)
            
            ce_t = self.cross_entropy(logits_t,select_y_t,select_mask_t)

            # get_support_vec 
            logits_s,support_vec = self.get_logits(support,is_train=True,embedding=True)
            s_prototype = ((support_vec[:b_s//2]*y_s[:b_s//2].unsqueeze(-1)).sum(1,keepdim=True) / y_s[:b_s//2].unsqueeze(-1).sum(1,keepdim=True)).mean(0,keepdim=True).repeat([b_s//2,seq_len,1])
            logits_tsvad = self.get_tsvad(s_prototype,support_vec[b_s//2:],is_train=True)
            y_s_vad_one_hot = y_s_one_hot[b_s//2:]
            
            # ce = self.dice_loss(logits_s.softmax(2),y_s_one_hot)
            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0) 
            ce2 = - (y_s_vad_one_hot * torch.log(logits_tsvad + 1e-12)).sum(2).mean(1).sum(0) 
            
            # query
            logits_q, embedding_q = self.get_logits(query,embedding=True) #  
            q_probs = logits_q.softmax(2).unsqueeze(-1)
            s_prototype_q = s_prototype[0:1].repeat_interleave(embedding_q.shape[0],dim=0)
            q_probs_2 = self.get_tsvad(s_prototype_q,embedding_q,is_train=False).unsqueeze(-1)
            
            q_probs = torch.cat([q_probs.cpu(),q_probs_2.cpu()],dim=-1).mean(dim=-1)
            # get support vec
            
            self.select_query_data_v2(q_probs[:,:,1],query,self.thre)
            print(len(self.torch_q_x))
            
            #pseudo-label
            if  len(self.torch_q_x)>0 and i>=86: #   
                mask = torch.where(self.torch_q_y==-1,torch.zeros_like(self.torch_q_y),self.torch_q_y).cuda()
                y_qs_one_hot = get_one_hot(self.torch_q_y.cuda()*mask)
                
                logits_qs = self.get_logits(self.torch_q_x.cuda(),is_train=True,embedding=False)
                # s_prototype_q = s_prototype[0:1].repeat_interleave(embedding_qs.shape[0],dim=0)
                logits_qs = logits_qs.softmax(2)#.unsqueeze(-1)
                # q_probs_presudo = self.get_tsvad(s_prototype_q,embedding_qs,is_train=True).unsqueeze(-1)
                # logits_qs = torch.cat([logits_qs,q_probs_presudo],dim=-1).mean(dim=-1)

                ce_qs = -(mask.unsqueeze(2)*y_qs_one_hot * torch.log(logits_qs+1e-12)).sum(2).mean(1).sum(0)

                self.loss_weights[1]=0.1
                self.loss_weights[2]=0
            else:
                ce_qs = 0
                ce_qs2= 0
                self.loss_weights[1]=0 
                self.loss_weights[2]=0 
            self.loss_weights[2]=0.1  #if i <50 else 0.1
            loss = self.loss_weights[0] * ce + self.loss_weights[2] * ce2 +  self.loss_weights[0]*ce_t + self.loss_weights[1]*ce_qs    #  # + self.loss_weights[0]*ce_t # 不过分离网络  +  self.loss_weights[0]*ce_t + self.loss_weights[0]*ce_t2 
                
            optimizer.zero_grad()
            loss.backward()
            print(loss)
            print(ce2,'\n')
            optimizer.step()
            self.get_acc(logits_s,y_s)
            self.get_acc(logits_t.detach(),select_y_t,select_mask_t)  
            # if i>50:
            #     step_scheduler.step()
            if i > 2:
                self.compute_FB_param(query)
                l3 += 0.1
            t1 = time.time()
            self.model.eval()
            # if i >150:
            self.record_info(new_time=t1-t0,
                            support=support,
                            query=query,
                            y_s=y_s)
        
    def update_student_model(self,global_step):
        decay = min(1-1/(global_step+1),self.ema_decay)
        for ema_param,param in zip(self.model_student.parameters(),self.model.parameters()):
            ema_param.data.mul_(decay).add_(1-decay,param.data)
            ema_param.detach() 

    def update_thres(self,support,y_s):
        logits = self.get_logits(support,use_cnn1=True).softmax(2)
        self.thre = ((logits[:,:,1]*y_s).sum(1)/y_s.sum(1)).min().item()
        print("thre:%s"%self.thre)
    
    def add_gussionNoise(self,data,sigma):# 添加高斯噪音
        mean = torch.mean(data.cpu()).numpy()
        var = torch.std(data.cpu()).numpy()
        noise = np.random.normal(mean,var**2,data.shape)
        noise = sigma*torch.from_numpy(noise).to(data.device).float()
        aug_data = noise+data.clone()
        return aug_data

    def save_mask(self,support_mask_list,ori_support_mask_list,mask_support_list,query_mask_list,ori_query_mask_list,mask_query_list,save_path="",sep_dict=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        support_mask = support_mask_list[0].cpu().numpy()
        ori_support_mask = ori_support_mask_list[0].cpu().numpy()
        mask_support = mask_support_list[0].cpu().numpy()

        query_mask = query_mask_list[0].cpu().numpy()
        ori_query_mask = ori_query_mask_list[0].cpu().numpy()
        mask_query = mask_query_list[0].cpu().numpy()

        sep_pos = sep_dict["sep_pos"].cpu().numpy()
        sep_neg = sep_dict["sep_neg"].cpu().numpy()
        sep_query = sep_dict["sep_query"].cpu().numpy()
        sep_pos_mask = sep_dict["sep_pos_mask"].cpu().numpy()
        
        np.save(os.path.join(save_path,"sep_pos.npy"),sep_pos)
        np.save(os.path.join(save_path,"sep_neg.npy"),sep_neg)
        np.save(os.path.join(save_path,"sep_query.npy"),sep_query)
        np.save(os.path.join(save_path,"sep_pos_mask.npy"),sep_pos_mask)

        np.save(os.path.join(save_path,"support_mask.npy"),support_mask)
        np.save(os.path.join(save_path,"ori_support_mask.npy"),ori_support_mask)
        np.save(os.path.join(save_path,"mask_support.npy"),mask_support)

        np.save(os.path.join(save_path,"query_mask.npy"),query_mask)
        np.save(os.path.join(save_path,"ori_query_mask.npy"),ori_query_mask)
        np.save(os.path.join(save_path,"mask_query.npy"),mask_query)
        
    def get_embedding(self,sep_dict,train=False):
        pos_data = sep_dict['sep_pos'].clone()
        pos_mask_data  = sep_dict["sep_pos_mask"].clone()
        self.model_student.eval()

        pos_index = torch.randperm(len(pos_data))
        pos_data = pos_data[pos_index][:128,:,:]
        pos_mask_data = pos_mask_data[pos_index][:128,:,:]
        with torch.no_grad():
            feature_embedding = self.model_student(pos_data.cuda(),step=5)
        feature_embedding = feature_embedding.mean(1,keepdim=True).mean(0,keepdim=True)
        self.embedding = feature_embedding

    def train_sep(self,sep_dict,criterion_sep):
        self.model_sep.train()
        pos_data = sep_dict['sep_pos'].clone()
        neg_data = sep_dict['sep_neg'].clone()
        query_data = sep_dict['sep_query'].clone()
        sep_pos_mask = sep_dict['sep_pos_mask'].clone()

        pos_index = torch.randperm(len(pos_data))[:85]
        pos_data = pos_data[pos_index].cuda()
        sep_pos_mask = sep_pos_mask[pos_index].cuda()
        neg_index = torch.randperm(len(neg_data))[:80]
        neg_data = neg_data[neg_index].cuda()
        query_index = torch.randperm(len(query_data))[:80]
        query_data = query_data[query_index].cuda()


        enroll_input = pos_data[:5,:,:]
        enroll_mask = sep_pos_mask[:5,:,0].unsqueeze(-1)

        enroll_embedding = self.model_student(enroll_input,step=5)
        enroll_embedding = enroll_mask * enroll_embedding
        enroll_embedding = (enroll_embedding.sum(1,keepdim=True)/enroll_mask.sum(1,keepdim=True)).mean(0,keepdim=True)

        lam = np.random.rand()
        
        separation_sample_pos = pos_data[5:,:,:]
        separation_sample_pos_mask = sep_pos_mask[5:,:,:]

        separation_sample_neg = neg_data

        if separation_sample_pos.size(0)<query_data.size(0):
            repeat_num = query_data.size(0) // separation_sample_pos.size(0)+1
            new_separation_samples_pos = separation_sample_pos.repeat(repeat_num,1,1)[:query_data.size(0),:,:]

            new_separation_samples_pos_mask = separation_sample_pos_mask.repeat(repeat_num,1,1)[:query_data.size(0),:,:]
            separation_sample_mix = new_separation_samples_pos + query_data
        else:
            new_separation_samples_pos = separation_sample_pos[:query_data.size(0),:,:]
            new_separation_samples_pos_mask = separation_sample_pos_mask[:query_data.size(0),:,:]
            separation_sample_mix = new_separation_samples_pos + query_data
        
        if lam <0.5:
            input_sep = separation_sample_mix
            target_sep = new_separation_samples_pos*new_separation_samples_pos_mask
        elif lam<0.75:
            input_sep = separation_sample_pos
            target_sep = new_separation_samples_pos*new_separation_samples_pos_mask
        else:
            input_sep = separation_sample_neg
            target_sep = torch.zeros_like(separation_sample_neg)

        prototype_embedding_full = torch.repeat_interleave(enroll_embedding,input_sep.size(0),dim=0)
        mask = self.model_sep(input_sep.cuda(),prototype_embedding_full,step=1)
        sep_output = input_sep.cuda() * mask
        loss_sep = criterion_sep(sep_output,target_sep)
        return loss_sep

    def train_cnn1(self,sub_train,y_t,mask_t,support,y_s,y_s_one_hot,query,i):
        cnn1 =True
        choice_id = random.choice(range(4))
        select_sub_train = torch.cat((sub_train[choice_id:24*19:4],sub_train[24*19+choice_id*2::4*2]),0).contiguous().cuda()
        select_y_t = torch.cat((y_t[choice_id:24*19:4],y_t[24*19+choice_id*2::4*2]),0).contiguous().cuda()
        select_mask_t = torch.cat((mask_t[choice_id:24*19:4],mask_t[24*19+choice_id*2::4*2]),0).contiguous().cuda()
        target_decoder = select_mask_t

        logits_t = self.get_logits(select_sub_train,is_train=True,is_class=True,use_cnn1=cnn1).reshape(-1,20)
        select_y_t = select_y_t.reshape(-1,1)
        select_mask_t = select_mask_t.reshape(-1,1)

        logits_t2 = self.get_logits(select_sub_train,is_train=True,is_decoder2=True,use_cnn1=cnn1).reshape(-1,2)
        target_decoder = target_decoder.reshape(-1,1)
        mask = torch.ones_like(target_decoder)

        ce_t = self.cross_entropy(logits_t,select_y_t,select_mask_t)
        ce_t2 = self.cross_entropy(logits_t2,target_decoder,mask)

        logits_s = self.get_logits(support,is_train=True,use_cnn1=cnn1)  #
        logits_q = self.get_logits(query,use_cnn1=cnn1) # 

        ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0) 

        q_probs = logits_q.softmax(2)
        q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0) # H(Y|X)
        
        if len(self.torch_q_x)>0 and i>86:
            self.select_query_data_v2(q_probs[:,:,1],query,self.thre)
            print(len(self.torch_q_x))
            mask = torch.where(self.torch_q_y==-1,torch.zeros_like(self.torch_q_y),self.torch_q_y)
            y_qs_one_hot = get_one_hot(self.torch_q_y*mask)

            logits_qs = self.get_logits(self.torch_q_x,is_train=True,use_cnn1=cnn1)
            ce_qs = -(mask.unsqueeze(2)*y_qs_one_hot * torch.log(logits_qs.softmax(2)+1e-12)).sum(2).mean(1).sum(0)

            self.loss_weights[1]=0.1
            self.loss_weights[2]=0
        else:
            ce_qs = 0
            self.loss_weights[1]=0 
            self.loss_weights[2]=0 

        loss = self.loss_weights[0] * ce + self.loss_weights[1]*ce_qs +  self.loss_weights[0]*ce_t + self.loss_weights[0]*ce_t2

        self.get_acc(logits_s,y_s)
        self.get_acc(logits_t.detach(),select_y_t,select_mask_t)
        return loss

    def get_m(self,i):
        if i<1:
            m=0
        elif i<3:
            m=0.1
        elif i<6:
            m=0.2
        elif i<10:
            m=0.3
        else:
            m=0.4

    def cross_entropy(self,logits,targets,mask,reduction='batchmean'):
        
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        
        log_pos = torch.gather(logsoftmax,1,(targets*mask).long())
        return -(log_pos*mask).sum()/mask.sum()

    def dice_loss(self, preds,gts):
        preds = preds.permute([0,2,1])
        gts = gts.permute([0,2,1])
        num_cls = gts.shape[1]
        a = torch.sum((preds*gts),dim=-1)
        b = preds.sum(dim=-1) + gts.sum(dim=-1) #[b,c,H*W]
        c = a/b #[b,c]
        return (num_cls-c.sum(dim=-1)).sum()

    def update_one_hot(self,support,y_s_one_hot):
        updated_one_hot = torch.zeros_like(y_s_one_hot)
        logits_s = self.get_logits(support).softmax(2).detach()
        updated_one_hot[:,:,1] = torch.where(logits_s[:,:,1]>=0.5,y_s_one_hot[:,:,1],updated_one_hot[:,:,1])
        updated_one_hot[:,:,0] = torch.where(logits_s[:,:,0]<0.5,y_s_one_hot[:,:,0],updated_one_hot[:,:,0])
        return updated_one_hot


    def select_query_data(self,q_probs, query,thre):
        list_x = []
        list_y = []
        cnt_n=0
        if q_probs.shape[0]> 128:
            start_id = random.choice(range(q_probs.shape[0]-128))
            end = start_id + 128
        else:
            start_id = 0
            end_id = q_probs.shape[0]
        for i in range(start_id,end_id):
            p_index = (q_probs[i]>thre).long()
            n_index = (q_probs[i]<thre-0.2).long()

            p_index = self.medFilt(p_index,5)
            n_index = self.medFilt(n_index,5)
            if n_index.sum()>0 or p_index.sum()>0:
                np_index = (n_index+p_index-1)+p_index
                list_x.append(query[i])
                list_y.append(np_index)
                cnt_n+=1 
                
        if len(list_x) >0 and cnt_n<128:
            self.torch_q_x = torch.stack(list_x)
            self.torch_q_y = torch.stack(list_y)
        else:
            self.torch_q_x = []
            self.torch_q_y = []

    def select_query_data_v2(self,q_probs,query,thre):
        list_x_n = []
        list_y_n = []
        list_x_p = []
        list_y_p = []

        cnt_n = 0
        cnt_p = 0
        for i in range(0,q_probs.shape[0]):
            if self.min_len>2*87:
                sub_q_probs = self.meanFilt(q_probs[i],87)
            else:
                sub_q_probs = q_probs[i]

            p_index = (sub_q_probs>thre).long()
            n_index = (sub_q_probs<thre-0.2).long()

            p_index = self.medFilt(p_index,5)
            n_index = self.medFilt(n_index,5)
            np_index = (n_index+p_index-1)+p_index

            if n_index.sum()>0 and cnt_n<64:
                list_x_n.append(query[i])
                list_y_n.append(np_index)
                cnt_n +=1
            if p_index.sum()>0 and cnt_p<64:
                list_x_p.append(query[i])
                list_y_p.append(np_index)
                cnt_p +=1

        cnt = min(cnt_n,cnt_p)
        if cnt>0 and cnt==cnt_n:
            list_x = list_x_n[:cnt]
            list_y = list_y_n[:cnt]
        elif cnt>0 and cnt==cnt_p:
            list_x = list_x_p[:cnt]
            list_y = list_y_p[:cnt]
        elif cnt_n==0:
            list_x = list_x_p[:5]
            list_y = list_y_p[:5]
        else:
            list_x = list_x_n[:5]
            list_y = list_y_n[:5]

        if len(list_x)>0:
            self.torch_q_x = torch.stack(list_x)
            self.torch_q_y = torch.stack(list_y).long()
        else:
            self.torch_q_x = []
            self.torch_q_y = []

    def medFilt(self,detections, median_window):

        if median_window %2==0:
            median_window-=1

        x = detections
        k = median_window

        assert k%2 == 1, "Median filter length must be odd"
        assert x.ndim == 1, "Input must be one dimensional"
        k2 = (k - 1) // 2
        y = torch.zeros((len(x),k)).type_as(x)
        y[:,k2]=x
        for i in range(k2):
            j = k2 -1
            y[j:,i]=x[:-j]
            y[:j,i]=x[0]
            y[:-j,-(i+1)] = x[j:]
            y[-j:,-(i+1)] = x[-1]

        return torch.median(y,axis=1)[0]

    def meanFilt(self,detections, median_window):

        if median_window %2==0:
            median_window-=1

        x = detections
        k = median_window

        assert k%2 == 1, "Median filter length must be odd"
        assert x.ndim == 1, "Input must be one dimensional"
        k2 = (k - 1) // 2
        y = torch.zeros((len(x),k)).type_as(x)
        y[:,k2]=x
        for i in range(k2):
            j = k2 -1
            y[j:,i]=x[:-j]
            y[:j,i]=x[0]
            y[:-j,-(i+1)] = x[j:]
            y[-j:,-(i+1)] = x[-1]

        return torch.mean(y,axis=1)

    def run_adaptation_model_w(self, support, query, y_s,nums):
        t0 = time.time()
        self.weights.requires_grad_() 
        optimizer = torch.optim.Adam([{'params': self.model.encoder.parameters()},{'params': self.weights}], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        self.model.train()
        l3 = 0.2
        if nums<self.iter:
            logits_s = self.get_logits(support)  
            logits_q = self.get_logits(query) 
            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0) 
            q_probs = logits_q.softmax(2)
            q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0) # H(Y|X)
            q_ent = - (q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0) #H(Y)
            b = q_probs[:,:,0]>0.5
            pos = 1.0*b.sum(1)/q_probs.shape[1]
            neg = 1.0 -pos
            pos = pos.unsqueeze(1)
            neg = neg.unsqueeze(1)
            F2 = torch.cat([pos,neg],1)
            marginal = q_probs.mean(dim=1) # n_task,2
            div_kl = F.kl_div(marginal.softmax(dim=-1).log(), self.FB_param.softmax(dim=-1), reduction='sum')
            div_kl2 = F.kl_div(F2.softmax(dim=-1).log(),self.FB_param2.softmax(dim=-1),reduction='sum')
            loss = self.loss_weights[0] * ce - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent) + l3*div_kl + div_kl2  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if nums>2:
                self.compute_FB_param(query)
                l3 += 0.1
            t1 = time.time()
            self.model.eval()
            self.record_info(new_time=t1-t0,
                             support=support,
                             query=query,
                             y_s=y_s)
            
            self.model.train()
            t0 = time.time()
        return self.model,nums+1,self.iter
            