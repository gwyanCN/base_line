import torch
import time
import torch.nn as nn
import numpy as np
from utils.util import warp_tqdm, get_metric, AverageMeter,prototypical_loss
from sklearn import manifold, datasets
import h5py
from datasets.Feature_extract import frequencyMask, TimeMask, add_gussionNoise
import pdb

# setting
print_freq = 100
meta_val_way = 10
meta_val_shot = 5
meta_val_metric = 'cosine'  # ('euclidean', 'cosine', 'l1', l2')
meta_val_iter = 500
meta_val_query = 15
alpha = - 1.0
label_smoothing = 0.
class Trainer:
    def __init__(self, device,num_class,train_loader,val_loader, conf):
        self.train_loader,self.val_loader = train_loader,val_loader
        self.device = device
        self.num_classes = num_class # 
        self.alpha = -1.0
        self.label_smoothing = 0.1
        self.meta_val_metric = 'cosine'
        self.loss_record={
            "20sepclass_loss":[],
            "2class_loss":[],
            "sep_loss":[]
        }
    def cross_entropy(self, logits, targets,mask, reduction='batchmean'):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        log_pos = torch.gather(logsoftmax,1,targets*mask)
        return - (log_pos * mask).sum()/mask.sum()
   
    def cross_entropy2(self, logits, targets, reduction='batchmean'):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        log_pos = torch.gather(logsoftmax,1,targets)
        return -log_pos.mean()

    def do_epoch(self, epoch, scheduler, disable_tqdm, model,
                 alpha, optimizer):  
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        # switch to train mode
        model.train()
        steps_per_epoch = len(self.train_loader) 
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm) 
        for i, (input, target) in enumerate(tqdm_train_loader):

            input, target = input.to(self.device), target.to(self.device, non_blocking=True) 
            # smoothed_targets = self.smooth_one_hot(target,self.label_smoothing) 
            target = target.reshape(-1,1)
            mask = torch.where(target>=0,1,0)
            # assert (smoothed_targets.argmax(1) == target).float().mean() == 1.0
            # Forward pass
            if self.alpha > 0:  # Mixup augmentation
                # generate mixed sample and targets
                lam = np.random.beta(self.alpha, self.alpha)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = smoothed_targets
                target_b = smoothed_targets[rand_index]
                mixed_input = lam * input + (1 - lam) * input[rand_index]

                output = model(mixed_input)
                loss = self.cross_entropy(output, target_a) * lam + self.cross_entropy(output, target_b) * (1. - lam)
            else:
                output = model(input,step=1)
                loss = self.cross_entropy(output, target,mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = (output.argmax(1) == target.squeeze()).float().sum()/mask.sum()
            top1.update(prec1.item(), mask.sum().item())
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            # Measure accurac y and record loss
            losses.update(loss.item(), mask.sum().item())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    def DoAug(self,tensorInput):
        data = add_gussionNoise(tensorInput)
        data = frequencyMask(data)
        data = TimeMask(data)
        return data
    def inputAug(self,input):
        input1 = input[:,:431]
        input2 = input[:,431:]
        input1 = self.DoAug(input1)
        # input2 = self.DoAug(input2)
        output = torch.cat([input1,input2],dim=1)
        return output
    def do_epoch2(self, epoch, scheduler, disable_tqdm, model,
                 alpha, optimizer,aug,logger_writer):
        batch_time = AverageMeter()
        losses_1 = AverageMeter()
        top1_1 = AverageMeter()

        losses = AverageMeter()
        top1 = AverageMeter()
       
        model.train()
        steps_per_epoch = len(self.train_loader) 
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm) 

        for i, (input, target1, target2) in enumerate(tqdm_train_loader):
            # pdb.set_trace()
            ori_input, target1, target2 = input.to(self.device), target1.to(self.device, non_blocking=True), target2.to(self.device, non_blocking=True)        
            target1 = target1.reshape(-1,1)
            target2 = target2.reshape(-1,1)
            mask = torch.where(target2>=0,1,0)
            if aug:
                ori_input[:,:431,:] = self.DoAug(ori_input[:,:431,:])
            if i!=0 and i%10==0:
                with torch.no_grad():
                   
                    show_image1 = ori_input[:,:431,:]-ori_input[:,:431,:].flatten(start_dim=-2).min(dim=1,keepdim=True).values.unsqueeze(-1)
                    show_image1 = ((show_image1/show_image1.flatten(start_dim=-2).max(dim=1,keepdim=True).values.unsqueeze(-1))*255).unsqueeze(1).repeat_interleave(3,dim=1)
                  
                    show_image2 = ori_input[:,431:,:]-ori_input[:,431:,:].flatten(start_dim=-2).min(dim=1,keepdim=True).values.unsqueeze(-1)
                    show_image2 = ((show_image2/show_image2.flatten(start_dim=-2).max(dim=1,keepdim=True).values.unsqueeze(-1))*255).unsqueeze(1).repeat_interleave(3,dim=1)
                    
                    show_image = torch.cat([show_image1,show_image2])
                    logger_writer.add_img("Input Image",show_image.permute([0,1,3,2]).type(torch.uint8),epoch*len(tqdm_train_loader)+i)
                    
            if self.alpha > 0:  # Mixup augmentation
                # generate mixed sample and targets
                lam = np.random.beta(self.alpha, self.alpha)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = smoothed_targets
                target_b = smoothed_targets[rand_index]
                mixed_input = lam * input + (1 - lam) * input[rand_index]
                output = model(mixed_input)
                loss = self.cross_entropy(output, target_a) * lam + self.cross_entropy(output, target_b) * (1. - lam)
            else:
                output1 = model(ori_input,step=1)
                loss1 = self.cross_entropy(output1, target2,mask) # 20分类

                output = model(ori_input,step=3)
                loss = self.cross_entropy(output,target1,mask) # 增强2分类

            total_loss = loss1 + loss 
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            prec1 = ((output.argmax(1) == target1.squeeze()).float()*mask.squeeze()).sum()/mask.sum()
            
            top1.update(prec1.item(), mask.sum().item())
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            # Measure accurac y and record loss
            losses.update(loss.item(), mask.sum().item())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
            prec2 = (output1.argmax(1)==target2.squeeze()).float().sum()/mask.sum()

            top1_1.update(prec2.item(),mask.sum().item())
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            losses_1.update(loss1.item(),mask.sum().item())
            batch_time.update(time.time()-end)
            end = time.time()

            if i%print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses_1, top1=top1_1))              
    def do_epoch_PANN(self, epoch, scheduler, disable_tqdm, feature_extractor,model,
                    alpha, optimizer,aug):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        # switch to train mode
        model.train()
        steps_per_epoch = len(self.train_loader) 
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm) 
        for i, (input, target) in enumerate(tqdm_train_loader):
            input, target = input.to(self.device), target.to(self.device, non_blocking=True) 
            target = target.reshape(-1,1)
            mask = torch.where(target>=0,1,0)
           
            #compute loss
            feature = feature_extractor.extract_feature(input)
            output = model(feature,step=1)
            loss = self.cross_entropy(output, target,mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = (output.argmax(1) == target.squeeze()).float().sum()/mask.sum()
            top1.update(prec1.item(), mask.sum().item())
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            # Measure accurac y and record loss
            losses.update(loss.item(), mask.sum().item())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    def do_epoch_sep(self, epoch, scheduler, disable_tqdm, model,model_sep,
                 alpha, optimizer,aug):  
        batch_time = AverageMeter()
        losses_1 = AverageMeter()
        losses_2 = AverageMeter()
        top1_1 = AverageMeter()
        top1_2 = AverageMeter()

        losses = AverageMeter()
        top1 = AverageMeter()
        import pdb
        # switch to train mode
        model.train()
        steps_per_epoch = len(self.train_loader) 
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm) 
        n_way = 4
        loss_sep_fn = nn.MSELoss()
        loss_dict={
            "sep":[],
            "2class":[],
            "20class":[]
        }
        for i, (input, target1, target2) in enumerate(tqdm_train_loader): # 5个类，每类15个样本
            # pdb.set_trace()

            ori_input, target1, target2 = input.to(self.device), target1.to(self.device, non_blocking=True), target2.to(self.device, non_blocking=True)
            ori_target = target1.to(self.device, non_blocking=True)
            ori_target2 = target2.to(self.device, non_blocking=True)
            total_index = torch.tensor([i for i in range(len(target2))]).to(self.device)

            # smoothed_targets = self.smooth_one_hot(target,self.label_smoothing) 
            if aug:
                if np.random.randn(1)>0.5:
                    input = self.inputAug(ori_input)
                else:
                    input = ori_input
            else:
                input = ori_input
            target1 = target1.reshape(-1,1)
            target2 = target2.reshape(-1,1)
            mask = torch.where(target2>=0,1,0)
            # assert (smoothed_targets.argmax(1) == target).float().mean() == 1.0
            # Forward pass
            if self.alpha > 0:  # Mixup augmentation
                # generate mixed sample and targets
                lam = np.random.beta(self.alpha, self.alpha)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = smoothed_targets
                target_b = smoothed_targets[rand_index]
                mixed_input = lam * input + (1 - lam) * input[rand_index]

                output = model(mixed_input)
                loss = self.cross_entropy(output, target_a) * lam + self.cross_entropy(output, target_b) * (1. - lam)
            else:
                output1 = model(input,step=1)
                loss1 = self.cross_entropy(output1, target2,mask) # 20分类

                output = model(ori_input,step=2)
                loss = self.cross_entropy(output,target1,mask) # 增强2分类

                output = model(ori_input,step=-1)
                loss2 = self.cross_entropy(output,target1,mask)  # 一般2分类

                cluster_index = torch.stack([total_index[i::n_way] for i in range(n_way)])

                embedding_index = cluster_index[:,:5].flatten()
                embedding_input = ori_input[embedding_index]
                # pdb.set_trace()
                embedding_feature = model(embedding_input,step=3) 
                mask_embedding = torch.where(ori_target2[embedding_index]>0,1,0)
                embedding_feature = (embedding_feature*mask_embedding.unsqueeze(-1)).sum(1,keepdim=True)/mask_embedding.unsqueeze(-1).sum(1,keepdim=True) 
                embedding_index = embedding_index.reshape(n_way,-1)
                embedding_feature = embedding_feature[embedding_index].mean(1) #(b,1,1024)

                separate_index = cluster_index[:,5:]
                batch,num = separate_index.shape
                separate_input = ori_input[:,:431,:][separate_index.flatten()]
                flip_input = torch.flip(separate_input,dims=[0])
                mix_separate_input = separate_input+flip_input
                separate_target = separate_input
                lam = np.random.rand()
                if lam<0.33:
                    input_sep = mix_separate_input
                    target_sep = separate_target
                    sepClass_target = ori_target[separate_index.flatten()].reshape(-1,1)
                elif lam<0.66:
                    input_sep = separate_input
                    target_sep = separate_input
                    sepClass_target = ori_target[separate_index.flatten()].reshape(-1,1)
                else:
                    input_sep = flip_input
                    target_sep = torch.zeros_like(input_sep)
                    sepClass_target = ori_target[separate_index.flatten()]
                    sepClass_target = torch.flip(sepClass_target,dims=[0]).reshape(-1,1)
                out_put = model_sep(input_sep,embedding_feature,step=0)
                out_target = out_put*input_sep
                
                loss_sep = loss_sep_fn(out_target,target_sep)

                # sep分类损失
                sep_input = out_put*separate_input
                
                sepClass_mask = torch.where(sepClass_target>=0,1,0)
                out_put = model(sep_input,step=4)
                loss_sep_out = self.cross_entropy(out_put,sepClass_target,sepClass_mask)

            total_loss = loss1 + loss + loss2+loss_sep+loss_sep_out
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_dict['sep'].append(loss_sep.cpu().item())
            loss_dict['2class'].append(loss.cpu().item())
            loss_dict['20class'].append(loss_sep_out.cpu().item())

            prec1 = ((output.argmax(1) == target1.squeeze()).float()*mask.squeeze()).sum()/mask.sum() #20分类
            
            top1.update(prec1.item(), mask.sum().item())
            # if not disable_tqdm:
            #     tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            # Measure accurac y and record loss
            losses.update(loss.item(), mask.sum().item())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

            prec2 = (output1.argmax(1)==target2.squeeze()).float().sum()/mask.sum()
            top1_1.update(prec2.item(),mask.sum().item())
            # if not disable_tqdm:
            #     tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            losses_1.update(loss1.item(),mask.sum().item())
            batch_time.update(time.time()-end)
            end = time.time()
            if i%print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses_1, top1=top1_1))

            prec3 = (out_put.argmax(1)==sepClass_target.squeeze()).float().sum()/sepClass_mask.sum()
            top1_2.update(prec3.item(),sepClass_mask.sum().item())
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f}'.format(top1_2.avg)) #分离20分类

            losses_2.update(loss_sep_out.item(),sepClass_mask.sum().item())
            batch_time.update(time.time()-end)
            end = time.time()
            if i%print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses_2, top1=top1_2))
        self.loss_record["20sepclass_loss"].append(loss_dict['20class'])
        self.loss_record["2class_loss"].append(loss_dict['2class'])
        self.loss_record["sep_loss"].append(loss_dict['sep'])     
    def do_epoch_meta_learning(self, epoch, scheduler, disable_tqdm, model,
                 alpha, optimizer): 
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()
        steps_per_epoch = len(self.train_loader) 
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm) 
        for i, (input, target) in enumerate(tqdm_train_loader):
            input, target = input.to(self.device), target.to(self.device, non_blocking=True) 
            smoothed_targets = self.smooth_one_hot(target,self.label_smoothing) 
            # assert (smoothed_targets.argmax(1) == target).float().mean() == 1.0
            # Forward pass
            if self.alpha > 0:  # Mixup augmentation
                # generate mixed sample and targets
                lam = np.random.beta(self.alpha, self.alpha)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = smoothed_targets
                target_b = smoothed_targets[rand_index]
                mixed_input = lam * input + (1 - lam) * input[rand_index]

                output = model(mixed_input)
                loss = self.cross_entropy(output, target_a) * lam + self.cross_entropy(output, target_b) * (1. - lam)
            else:
                output, feature = model(input,feature = True)
                loss_val, acc_val = prototypical_loss(feature,target,5)
                loss = loss_val

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = acc_val
            top1.update(prec1.item(), input.size(0))
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            # Measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    def smooth_one_hot(self, targets, label_smoothing):
        assert 0 <= label_smoothing < 1
        with torch.no_grad():
            new_targets = torch.empty(size=(targets.size(0), self.num_classes), device=self.device)
            new_targets.fill_(label_smoothing / (self.num_classes-1))
            new_targets.scatter_(1, targets.unsqueeze(1), 1. - label_smoothing)
        return new_targets
    def get_feature_by_y(self,feature_x,target_y,idx):
        new_feature_x = []
        for i  in range(feature_x.shape[0]):
            if target_y[i]==idx:
                new_feature_x.append(feature_x[i,:])
        new_feature_x = np.array(new_feature_x)
        new_feature_x = new_feature_x.mean(0)
        return new_feature_x
    def meta_val(self,epoch, model, disable_tqdm):
        top1 = AverageMeter()
        model.eval() 

        with torch.no_grad():
            tqdm_test_loader = warp_tqdm(self.val_loader, disable_tqdm)
            for i, (inputs, target) in enumerate(tqdm_test_loader):
                inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)
                target = target.reshape(-1,1)
                mask = torch.where(target>=0,1,0)
                
                output = model(inputs)
                acc = (output.argmax(1)==target.squeeze()).float().sum()/mask.sum()
                top1.update(acc.item(),mask.sum().item())

                if not disable_tqdm:
                    tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg*100))
        return top1.avg
    def PANNs_meta_val(self,epoch, PANNS_model,model, disable_tqdm):
        top1 = AverageMeter()
        model.eval() 
        with torch.no_grad():
            tqdm_test_loader = warp_tqdm(self.val_loader, disable_tqdm)
            for i, (inputs, target) in enumerate(tqdm_test_loader):
                inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)
                target = target.reshape(-1,1)
                mask = torch.where(target>=0,1,0) 
                feature = PANNS_model.extract_feature(inputs)
                output = model(feature)
                acc = (output.argmax(1)==target.squeeze()).float().sum()/mask.sum()
                top1.update(acc.item(),mask.sum().item())

                if not disable_tqdm:
                    tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg*100))
        return top1.avg
    def meta_val2(self,epoch, model, disable_tqdm):
        top1 = AverageMeter()
        top1_1 = AverageMeter()
        model.eval() 

        with torch.no_grad():
            tqdm_test_loader = warp_tqdm(self.val_loader, disable_tqdm)
            for i, (inputs, target1,target2) in enumerate(tqdm_test_loader): #
                inputs, target1, target2 = inputs.to(self.device), target1.to(self.device, non_blocking=True), target2.to(self.device, non_blocking=True)
                target1 = target1.reshape(-1,1)
                target2 = target2.reshape(-1,1)
                mask = torch.where(target2>=0,1,0)
                
                output = model(inputs,step=3)
                acc = ((output.argmax(1)==target1.squeeze()).float()*mask.squeeze()).sum()/mask.sum() # 2 分类
                
                top1.update(acc.item(),mask.sum().item())
                output1 = model(inputs)
                acc1 = (output1.argmax(1)==target2.squeeze()).float().sum() / mask.sum() #20分类
                
                top1_1.update(acc1.item(),mask.sum().item())
                if not disable_tqdm:
                    tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg*100))
                    tqdm_test_loader.set_description('Acc {:.2f}'.format(top1_1.avg*100))
        return top1.avg, top1_1.avg # 2分类，20分类
    def meta_sep_val(self,epoch, model,model_sep, disable_tqdm):
        top1 = AverageMeter()
        top1_1 = AverageMeter()
        top1_2 = AverageMeter()
        model.eval() 

        with torch.no_grad():
            tqdm_test_loader = warp_tqdm(self.val_loader, disable_tqdm)
            for i, (inputs, target1,target2) in enumerate(tqdm_test_loader):
                inputs, ori_target1, ori_target2 = inputs.to(self.device), target1.to(self.device, non_blocking=True), target2.to(self.device, non_blocking=True)
                target1 = ori_target1.reshape(-1,1)
                target2 = ori_target2.reshape(-1,1)
                mask_target = torch.where(target2>=0,1,0)
                
                output = model(inputs,step=2)
                acc = ((output.argmax(1)==target1.squeeze()).float()*mask_target.squeeze()).sum()/mask_target.sum() # 2 分类
                top1.update(acc.item(),mask_target.sum().item())
                output1 = model(inputs)
                acc1 = (output1.argmax(1)==target2.squeeze()).float().sum() / mask_target.sum() #20分类
                top1_1.update(acc1.item(),mask_target.sum().item())

                embedding_feature = model(inputs,step=3)  
                embedding_feature_mask = torch.where(ori_target2>0,1,0)
                embedding_feature = (embedding_feature*embedding_feature_mask.unsqueeze(-1)).sum(1,keepdim=True) / embedding_feature_mask.unsqueeze(-1).sum(1,keepdim=True)
                sep_input = inputs[:,:431]
                mask_sep = model_sep(sep_input,embedding_feature,step=1)
                sep_input = sep_input*mask_sep
                output3 = model(sep_input,step=4)
                acc2 = (output3.argmax(1)==target2.squeeze()).float().sum() / mask_target.sum() #20分类
                top1_2.update(acc2.item(),mask_target.sum().item())

                if not disable_tqdm:
                    # tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg*100))
                    # tqdm_test_loader.set_description('Acc {:.2f}'.format(top1_1.avg*100))
                    tqdm_test_loader.set_description('Acc {:.2f}'.format(top1_2.avg*100))
        return top1.avg, top1_1.avg,top1_2.avg # 2分类，20分类
    def save_plot_data(self,feature_x,target_y,model):
        feature_x = np.array(feature_x)
        target_y = np.array(target_y)
        new_feature_x = []
        new_feature_y = []
        for i in range(19):
            new_feature_x.append(self.get_feature_by_y(feature_x, target_y, i))
            new_feature_y.append(i)
        for i in range(model.state_dict()['fc.weight'].shape[0]):
            new_feature_x.append(model.state_dict()['fc.weight'][i,:].detach().cpu().numpy())
            new_feature_y.append(i+19)
        new_feature_x = np.array(new_feature_x)
        new_feature_y = np.array(new_feature_y)
        print('new_feature_x ',new_feature_x.shape)
        print('new_feature_y ',new_feature_y.shape)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(new_feature_x)
        hf = h5py.File('/home/ydc/DACSE2021/sed-tim-base/check_point/plot/visual_transductive.h5', 'w')
        X_shape = X_tsne.shape[1]
        hf.create_dataset(
                name='feature', 
                shape=(new_feature_y.shape[0], X_shape), 
                dtype=np.float32)
        hf.create_dataset(
                name='target', 
                shape=(new_feature_y.shape[0],), 
                dtype=np.float32)
        for n,u in enumerate(X_tsne):
            hf['feature'][n] = u
        for n,u in enumerate(new_feature_y):
            hf['target'][n] = u
        hf.close()     
    def met_plot(self,epoch, model, disable_tqdm):
        top1 = AverageMeter()
        model.eval() 
        feature_x = []
        target_y = []
        with torch.no_grad():
            tqdm_test_loader = warp_tqdm(self.train_loader, disable_tqdm)
            for i, (inputs, target) in enumerate(tqdm_test_loader):
                inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)
                classes = torch.unique(target) 
                n_classes = len(classes)

                def supp_idxs(c):
                    return target.eq(c).nonzero()[:meta_val_shot].squeeze(1) 
                output = model(inputs, feature=True)[0].cuda(0)
                for o in output:
                    feature_x.append(o.detach().cpu().numpy())
                for t in target:
                    target_y.append(t.detach().cpu().numpy())
        
        self.save_plot_data(feature_x,target_y,model)
    def metric_prediction(self, support, query, train_label, meta_val_metric): # meta_val_metric--> consin?
        support = support.view(support.shape[0], -1) # n_way* 
        # print('support ',support.shape)
        query = query.view(query.shape[0], -1) # n_query * ?
        # print('query ',query.shape)
        distance = get_metric(meta_val_metric)(support, query) #
        # print('distance ',distance.shape)
        predict = torch.argmin(distance, dim=1) 
        # print('predict ',predict)
        predict = torch.take(train_label, predict) 
        # print('predict2 ',predict)
        return predict