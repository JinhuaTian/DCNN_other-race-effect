# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:22:20 2019

@author: Administrator

transfer learning using pretrained vggface
"""

#import os

#import csv
import torch
#import torchvision
from torchvision import transforms
from torch import nn
#import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sys
sys.path.remove('/usr/local/neurosoft/labtool/python/dnnbrain')
###Please load the old version cnnbrain
sys.path.append('/nfs/h1/workingshop/tianjinhua/vgg_train/code/dnnbrain/')
from dnnbrain.dnn.io import PicDataset
from dnnbrain.dnn import io as dnn_io
#from dnnbrain.dnn.models import dnn_transfer_learning_model, dnn_test_model #,Vgg_face
import copy
import time
from torch.optim import lr_scheduler
from torch import nn

def dnn_train_model(dataloaders_train, model, criterion, optimizer, num_epoches, train_method='tradition',
                    dataloaders_train_test=None, dataloaders_val_test=None):
    LOSS = []
    ACC_train_top1 = []
    # ACC_train_top5 = []
    ACC_val_top1 = []
    # ACC_val_top5 = []
    EPOCH = []

    time0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)

    #### lr decay
    scheduler = lr_scheduler.StepLR(optimizer, 25, 250 ** (-1 / 3), last_epoch=-1)

    #### save the best model (best epoch and best accuracy)
    best_epoch = 0
    best_acc = 0

    for epoch in range(num_epoches):
        EPOCH.append(epoch + 1)
        print('Epoch time {}/{}'.format(epoch + 1, num_epoches))
        print('-' * 10)
        time1 = time.time()
        running_loss = 0.0

        for inputs, targets in dataloaders_train:
            inputs.requires_grad_(True)
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if train_method == 'tradition':
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                elif train_method == 'inception':
                    # Google inception model
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, targets)
                    loss2 = criterion(aux_outputs, targets)
                    loss = loss1 + 0.4 * loss2
                else:
                    raise Exception('Not Support this method yet, please contact authors for implementation.')

                _, pred = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
            # Statistics loss in every batch
            running_loss += loss.item() * inputs.size(0)

        # Caculate loss in every epoch
        epoch_loss = running_loss / len(dataloaders_train.dataset)
        print('Loss: {}\n'.format(epoch_loss))
        LOSS.append(epoch_loss)

        # Caculate ACC_train every epoch
        if dataloaders_train_test:
            model_copy = copy.deepcopy(model)
            # _, _, train_acc_top1, train_acc_top5 = dnn_test_model(dataloaders_train_test, model_copy)
            _, _, train_acc_top1 = dnn_test_model(dataloaders_train_test, model_copy)
            print('top1_acc_train: {}\n'.format(train_acc_top1))
            # print('top5_acc_train: {}\n'.format(train_acc_top5))
            ACC_train_top1.append(train_acc_top1)
            # ACC_train_top5.append(train_acc_top5)

        # Caculate ACC_val every epoch
        if dataloaders_val_test:
            model_copy = copy.deepcopy(model)
            # _, _, val_acc_top1, val_acc_top5 = dnn_test_model(dataloaders_val_test, model_copy)
            _, _, val_acc_top1 = dnn_test_model(dataloaders_val_test, model_copy)
            print('top1_acc_test: {}\n'.format(val_acc_top1))
            # print('top5_acc_test: {}\n'.format(val_acc_top5))
            ACC_val_top1.append(val_acc_top1)
            # ACC_val_top5.append(val_acc_top5)

        # print time of a epoch
        time_epoch = time.time() - time1
        print('This epoch training complete in {:.0f}m {:.0f}s'.format(time_epoch // 60, time_epoch % 60))

        ### Set lr decay, note that learning rate scheduler was expected to be called before the optimizer's update
        scheduler.step()

        ##### save the best model while training
        if round(val_acc_top1, 4) > round(best_acc, 4):
            best_acc = val_acc_top1
            best_epoch = epoch
            torch.save(model, '/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_pretrained_tl/vgg_vgg16_300_pretrained_tl_bestmodel.pth')

    #### print the best validation accuray and epoch
    print("best epoch is " + str(best_epoch))
    print("best validation acc is " + str(best_acc))

    # store LOSS, ACC_train, ACC_val to a dict
    if dataloaders_train_test and dataloaders_val_test:
        # metric = zip(LOSS, ACC_train_top1, ACC_train_top5, ACC_val_top1, ACC_val_top5)
        metric = zip(LOSS, ACC_train_top1, ACC_val_top1)
        metric_dict = dict(zip(EPOCH, metric))
    else:
        metric_dict = dict(zip(EPOCH, LOSS))

    time_elapsed = time.time() - time0
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, metric_dict

def dnn_test_model(dataloaders, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    model_target = []
    # model_target_top5 = []
    actual_target = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloaders):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, outputs_label = torch.max(outputs, 1)
            # outputs_label_top5 = torch.topk(outputs, 5)

            model_target.extend(outputs_label.cpu().numpy())
            # model_target_top5.extend(outputs_label_top5[1].cpu().numpy())
            actual_target.extend(targets.numpy())

    model_target = np.array(model_target)
    # model_target_top5 = np.array(model_target_top5)
    actual_target = np.array(actual_target)

    # Caculate the top1 acc and top5 acc (exclude the top5 acc)
    test_acc_top1 = 1.0 * np.sum(model_target == actual_target) / len(actual_target)

    return model_target, actual_target, test_acc_top1


import torch
torch.cuda.empty_cache()
'''
#load the best model
vgg = torch.load('/nfs/h1/workingshop/tianjinhua/vgg/dnn_models/vgg_face_dag.pth')
#vgg = torch.load('/nfs/a2/userhome/tianjinhua/workingdir/model/vgg_face_dag.pth')
#vgg = torchvision.models.vgg16(pretrained=True)

###print the framework of the network
#for name, value in vgg.named_parameters():
#    print('name: {0},\t grad: {1}'.format(name, value.requires_grad))

#import torchvision
#vgg = torchvision.models.vgg16(pretrained=True)
vgg.fc8 = torch.nn.Linear(4096, 300, bias=True)

#in_features = vgg.fc8.in_features
#vgg.fc8 = nn.Linear(in_features,200,bias=True)

no_grad = ['fc8.weight','fc8.bias']
for name, value in vgg.parameters():
    if name in no_grad:
        value.requires_grad = True
    else:
        value.requires_grad = False
'''
#transfer learning fc8
from dnnbrain.dnn.models import Vgg_face

def vgg_face(weights_path = None,**kwargs):
    vgg = Vgg_face()
    if weights_path:
        state_dict = torch.load(weights_path)
        vgg.load_state_dict(state_dict)
    return vgg

#load the pretrained model
vgg = vgg_face('/nfs/h1/workingshop/tianjinhua/vgg/dnn_models/vgg_face_dag.pth')

for param in vgg.parameters():
    param.requires_grad = False
in_features = vgg.fc8.in_features
vgg.fc8 = nn.Linear(in_features,300)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg.parameters(),lr = 0.01)
#optimizer = torch.optim.SGD(vgg.parameters(), lr=0.01, momentum=0.9) #, weight_decay=0.0005

'''
loss_func = nn.TripletMarginLoss(margin=1.0, p=2)
anchor = torch.randn(300, 300, requires_grad=True)
positive = torch.randn(300, 300, requires_grad=True)
negative = torch.randn(300, 300, requires_grad=True)
output = loss_func(anchor, positive, negative)
output.backward()
'''
data_transforms = {
    'train': transforms.Compose([
		transforms.Resize((224,224)),
		transforms.RandomRotation(15),
		#transforms.RandomVerticalFlip(),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean = [0.483, 0.408, 0.375],
							std = [0.142, 0.137, 0.141])]),

    'val': transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.483, 0.408, 0.375],
                             std = [0.142, 0.137, 0.141])])
    }

#load the data
picdataset_train = dnn_io.PicDataset('/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_pretrained_tl/tl_train.csv', transform=data_transforms['train'])
picdataloader_train = DataLoader(picdataset_train, batch_size=64, shuffle=True, num_workers=10)

picdataset_train_val = dnn_io.PicDataset('/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_pretrained_tl/tl_train.csv', transform=data_transforms['val'])
dataloaders_train_test = DataLoader(picdataset_train_val, batch_size=16, shuffle=False, num_workers=10)

picdataset_train_val = dnn_io.PicDataset('/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_pretrained_tl/tl_test.csv', transform=data_transforms['val'])
dataloaders_val_test = DataLoader(picdataset_train_val, batch_size=16, shuffle=False, num_workers=10)

model,metric_dict = dnn_train_model(picdataloader_train,
                           vgg,
                           criterion,
                           optimizer,
                           90,
                           dataloaders_train_test = dataloaders_train_test,
                           dataloaders_val_test = dataloaders_val_test)

out_put = pd.DataFrame(metric_dict)
out_put.to_csv("/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_pretrained_tl/Procudure_vgg16_300_tl.csv",index=False,sep=',')

#torch.save(model, '/nfs/h1/workingshop/tianjinhua/vgg_train/transfer_learning/vgg_face_trained_vgg11_tl.pth')

### test the model
#torch.cuda.empty_cache()
#vgg = torch.load('/nfs/a2/userhome/tianjinhua/workingdir/train_model/vgg_face_trained_vgg11_test.pth')
torch.cuda.empty_cache()

transforms =transforms.Compose([
    transforms.Resize((224,224))
    ,transforms.ToTensor()
    ,transforms.Normalize(mean = [0.483, 0.408, 0.375],
                         std = [0.142, 0.137, 0.141])
    ])

#test the model
#test_path = '/nfs/a2/userhome/tianjinhua/workingdir/transfer_learning/tl_test.csv'
vgg = torch.load('/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_pretrained_tl/vgg_vgg16_300_pretrained_tl_bestmodel.pth')
picdataset_test_test = dnn_io.PicDataset('/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_pretrained_tl/tl_test.csv', transform=data_transforms['val'])
dataloaders_test_test = DataLoader(picdataset_test_test, batch_size=16, shuffle=False, num_workers=10)

#test_dataset = PicDataset(test_path,transform=train_transform['val'])
#test_dataloader = DataLoader(test_dataset, batch_size = 32,shuffle=False)
model_target, actual_target, test_acc_top1 = dnn_test_model(dataloaders_test_test,vgg)

print(test_acc_top1)

'''
best epoch is 82
best validation acc is 0.7766666666666666
Training complete in 334m 29s
0.7766666666666666
2.
This epoch training complete in 3m 43s
best epoch is 81
best validation acc is 0.776
Training complete in 343m 9s
0.776                                                   
'''

train = np.transpose(model_target)
test = np.transpose(actual_target)
data = pd.DataFrame({'train':train,'test':test})

data.to_csv("/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_pretrained_tl/results_vgg16_300_tl.csv",index=False,sep=',')
'''
adam
best epoch is 87
best validation acc is 0.8519333333333333
Training complete in 338m 46s
'''