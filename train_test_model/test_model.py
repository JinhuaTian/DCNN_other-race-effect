# -*- coding: utf-8 -*-
"""
All codes used in this script were copied or modified based on dnnbarin https://github.com/BNUCNL/dnnbrain

Train a vgg11_bn model
"""
import numpy as np
import pandas as pd
import torch
from dnnbrain.dnn import io as dnn_io
from torch.utils.data import DataLoader

#load the model you trained
vgg = torch.load('/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_AW/vgg_vgg11bn_304_mix_tl_bestmodel.pth')

# function dnn_train_model and dnn_test_model were modified from the original dnnbrain.dnn
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

picdataset_test_test = dnn_io.PicDataset('/nfs/h1/workingshop/tianjinhua/vgg_train/transfer_learning/all_test.csv', transform=data_transforms['val'])
dataloaders_test_test = DataLoader(picdataset_test_test, batch_size=16, shuffle=False, num_workers=10)

#test the model
model_target, actual_target, test_acc_top1 = dnn_test_model(dataloaders_test_test,vgg)

print(test_acc_top1)

# save the test results
train = np.transpose(model_target)
test = np.transpose(actual_target)
data = pd.DataFrame({'train':train,'test':test})

data.to_csv("/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_AW/results_vgg11_304_AWtl.csv",index=False,sep=',')