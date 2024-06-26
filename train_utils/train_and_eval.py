import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
import os
import torch.nn.functional as F
from .dice_loss import DiceLoss,GeneralizedSoftDiceLoss
from .floss import FocalLoss,mIoULoss
os.environ["CUDA_VISIBLE_DEVICES"] = "2,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#from .dice_loss import make_one_hot,DiceLoss
from tqdm import tqdm
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast
def criterion(inputs, target):
    
    #loss = nn.cross_entropy()(inputs,target)
    #loss =nn.CrossEntropyLoss(ignore_index=255,label_smoothing=0.05)(inputs,target)
    ignore_index = 255


    loss = nn.functional.cross_entropy(inputs,target,ignore_index=ignore_index)
    
    #loss = nn.functional.cross_entropy(inputs,target)
    
    #loss = FocalLoss(ignore_index=7)(inputs,target)
    #dice_target = build_target(target, 9,ignore_index=9)
    #dice_target = build_target(target, 7)
    #diceloss = dice_loss(inputs, dice_target, multiclass=True,ignore_index=7)
    diceloss = GeneralizedSoftDiceLoss(ignore_lb=ignore_index)(inputs, target)
    #diceloss = DiceLoss(ignore_index=7)(inputs, dice_target)
    #miouloss = mIoULoss(n_classes=6)(inputs,target)
    loss = loss+diceloss
    
    return loss
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'val:'
    
    # Use tqdm for progress bar
    with tqdm(total=len(data_loader), desc=header) as progress_bar:
        with torch.no_grad():
            for sample in data_loader:
                image = sample['img']
                target = sample['gt_semantic_seg']
                image, target = image.to(device), target.to(device)

                output = model(image)

                confmat.update(target.flatten(), output.argmax(dim=1).flatten())

                # Update tqdm progress bar
                progress_bar.update(1)

        confmat.reduce_from_all_processes()

    return confmat
'''
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'val:'
    with torch.no_grad():
        for sample in metric_logger.log_every(data_loader, 100, header):
        #print(sample)
            image = sample['img']
            target = sample['gt_semantic_seg']

            image, target = image.to(device), target.to(device)
            
            output = model(image)
            
            
            #print(target.flatten().shape, output.argmax(dim=1).flatten().shape)

            confmat.update(target.flatten(), output.argmax(dim=1).flatten())

        confmat.reduce_from_all_processes()

    return confmat'''

'''
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, num_classes=6 ,scaler=None):
    model.train()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample in metric_logger.log_every(data_loader, print_freq, header):
        #print(sample)
        image = sample['img']
        
        target = sample['gt_semantic_seg']
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            output = model(image)
           
                    
            
            loss = criterion(output, target)
        
        confmat.update(target.flatten(), output.argmax(dim=1).flatten())
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            #print(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
            scaler.step(optimizer)
            scaler.update()
        else:
          
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        #print(loss.item(),loss)
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr,confmat
'''
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, num_classes=6, scaler=None):
    model.train()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # Use tqdm for progress bar
    with tqdm(total=len(data_loader), desc=header) as progress_bar:
        for batch_idx, sample in enumerate(data_loader):
            image = sample['img']
            target = sample['gt_semantic_seg']
            image, target = image.to(device), target.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image)
                loss = criterion(output, target)

            confmat.update(target.flatten(), output.argmax(dim=1).flatten())

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            #lr_scheduler.step(epoch)
            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss.item(), lr=lr)

            # Update tqdm progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), lr=lr)

    confmat.reduce_from_all_processes()
    return metric_logger.meters["loss"].global_avg, lr, confmat

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


