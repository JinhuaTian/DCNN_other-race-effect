# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:22:20 2019

@author: Administrator

train model using focal loss
"""
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
#import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sys

import copy
import time
from torch.optim import lr_scheduler
from torch import nn

import os
import scipy.io
from PIL import Image

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
    scheduler = lr_scheduler.StepLR(optimizer, 10, 250 ** (-1 / 3), last_epoch=-1)

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
            torch.save(model, '/nfs/a2/userhome/tianjinhua/workingdir/trainMoedelMix/vgg16_304_cosface_bestModel_AW.pth')

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

class Vgg_face(nn.Module):
    """Vgg_face's model architecture"""

    def __init__(self):
        super(Vgg_face, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [3, 224, 224]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc8(x)
        return x


class PicDataset(Dataset):
    """
    Build a dataset to load pictures
    """

    def __init__(self, csv_file, transform=None, crop=None):
        """
        Initialize PicDataset

        Parameters:
        ------------
        csv_file[str]:  table contains picture names, conditions and picture onset time.
                        This csv_file helps us connect cnn activation to brain images.
                        Please organize your information as:

                        [PICDIR]
                        stimID          condition   onset(optional) measurement(optional)
                        download/face1  face        1.1             3
                        mgh/face2.png   face        3.1             5
                        scene1.png      scene       5.1             4

        transform[callable function]: optional transform to be applied on a sample.
        crop[bool]:crop picture optionally by a bounding box.
                   The coordinates of bounding box for crop pictures should be measurements in csv_file.
                   The label of coordinates in csv_file should be left_coord,upper_coord,right_coord,lower_coord.
        """
        self.csv_file = pd.read_csv(csv_file, skiprows=1)
        with open(csv_file, 'r') as f:
            self.picpath = f.readline().rstrip()
        self.transform = transform
        picname = np.array(self.csv_file['stimID'])
        condition = np.array(self.csv_file['condition'])
        self.picname = picname
        self.condition = condition
        self.crop = crop
        if self.crop:
            self.left = np.array(self.csv_file['left_coord'])
            self.upper = np.array(self.csv_file['upper_coord'])
            self.right = np.array(self.csv_file['right_coord'])
            self.lower = np.array(self.csv_file['lower_coord'])

    def __len__(self):
        """
        Return sample size
        """
        return self.csv_file.shape[0]

    def __getitem__(self, idx):
        """
        Get picture name, picture data and target of each sample

        Parameters:
        -----------
        idx: index of sample

        Returns:
        ---------
        picname: picture name
        picimg: picture data, save as a pillow instance
        target_label: target of each sample (label)
        """
        # load pictures
        target_name = np.unique(self.condition)
        picimg = Image.open(os.path.join(self.picpath, self.picname[idx])).convert('RGB')
        if self.crop:
            picimg = picimg.crop((self.left[idx], self.upper[idx], self.right[idx], self.lower[idx]))
        target_label = target_name.tolist().index(self.condition[idx])
        if self.transform:
            picimg = self.transform(picimg)
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
            picimg = self.transform(picimg)
        return picimg, target_label

    def get_picname(self, idx):
        """
        Get picture name and its condition (target condition)

        Parameters:
        -----------
        idx: index of sample

        Returns:
        ---------
        picname: picture name
        condition: target condition
        """
        return os.path.basename(self.picname[idx]), self.condition[idx]


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

#torch.cuda.empty_cache()
data_transforms = {
    'train': transforms.Compose([
		transforms.Resize((224,224)),
		transforms.RandomRotation(15),
		#transforms.RandomVerticalFlip(),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean = [0.485, 0.456, 0.406],
							std = [0.229, 0.224, 0.225])]),

    'val': transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
    }

picdataset_train = PicDataset('/nfs/a2/userhome/tianjinhua/workingdir/trainMoedelMix/mix_training.csv', transform=data_transforms['train'])
picdataloader_train = DataLoader(picdataset_train, batch_size=64, shuffle=True, num_workers=20)

picdataset_train_val = PicDataset('/nfs/a2/userhome/tianjinhua/workingdir/trainMoedelMix/mix_training.csv', transform=data_transforms['val'])
dataloaders_train_test = DataLoader(picdataset_train_val, batch_size=64, shuffle=False, num_workers=20)

picdataset_test_val = PicDataset('/nfs/a2/userhome/tianjinhua/workingdir/trainMoedelMix/mix_validating.csv', transform=data_transforms['val'])
dataloaders_val_test = DataLoader(picdataset_test_val, batch_size=64, shuffle=False, num_workers=20)

# train model
vggface = torchvision.models.vgg16(pretrained=False)
vggface.fc8 = torch.nn.Linear(4096, 304, bias=True)
criterion = FocalLoss(class_num=304)#class_num=304
#criterion = LMCL_loss(num_classes=64, feat_dim=1000)
#criterion = torch.nn.CosineEmbeddingLoss()   
optimizer = torch.optim.SGD(vggface.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

#path_loss_acc = '/nfs/a2/userhome/tianjinhua/workingdir/cosface/loss.csv'

model,metric_dict = dnn_train_model(picdataloader_train,  
                           vggface, 
                           criterion,
                           optimizer, 
                           40,
                           dataloaders_train_test = dataloaders_train_test, 
                           dataloaders_val_test = dataloaders_val_test)
out_put = pd.DataFrame(metric_dict)
#out_put = pd.DataFrame(metric_dict,index = [0])
out_put.to_csv("/nfs/a2/userhome/tianjinhua/workingdir/trainMoedelMix/results_vgg16_40_cosface_AW.csv",index=False,sep=',')
#no need to save the last model
#torch.save(model, '/nfs/a2/userhome/tianjinhua/workingdir/trainMoedelMix/vgg16_AW_cosface.pth')

'''
This epoch training complete in 2m 25s
best epoch is 30
best validation acc is 0.5535
Training complete in 96m 54s
'''