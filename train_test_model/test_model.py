# -*- coding: utf-8 -*-
"""
All codes used in this script were copied or modified based on dnnbarin https://github.com/BNUCNL/dnnbrain

Test a vgg11_bn model.

Testing dataset and training dataset should included the same identities.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

# load the image dataset you prepared using a csv file
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
        with open(csv_file,'r') as f:
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
            picimg = picimg.crop((self.left[idx],self.upper[idx],self.right[idx],self.lower[idx]))
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

#load the model you trained
vgg = torch.load('/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_AW/vgg_vgg11bn_304_mix_tl_bestmodel.pth')

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

picdataset_test_test = PicDataset('/nfs/h1/workingshop/tianjinhua/vgg_train/transfer_learning/all_test.csv', transform=data_transforms['val'])
dataloaders_test_test = DataLoader(picdataset_test_test, batch_size=16, shuffle=False, num_workers=10)

#test the model
model_target, actual_target, test_acc_top1 = dnn_test_model(dataloaders_test_test,vgg)

# print the test accuracy
print(test_acc_top1)

# save the test results as a csv file
train = np.transpose(model_target)
test = np.transpose(actual_target)
data = pd.DataFrame({'train':train,'test':test})

data.to_csv("/nfs/h1/workingshop/tianjinhua/vgg_train/vgg_AW/results_vgg11_304_AWtl.csv",index=False,sep=',')