# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:18:26 2019

@author: Administrator

make a csv stimuli file

please add the file path in the first row manually
"""
import os
import shutil
import pandas as pd
import numpy as np
#input your the image path
filepath = (r'D:\Final_data\transfer_learning_pretrained\Black_100')
files = os.listdir(filepath)
files.sort()
picname = []
label = []

for file in files:
    picdir = os.path.join(filepath,file)
    pics = os.listdir(picdir)
    # data selection
    for pic in pics:
        label.append(file)
        #file_name 
        picname_now = file + pic
        picname.append(picname_now)
        #move file
        newpath = r'D:\Final_data\transfer_learning_pretrained\tl_validate'
        newfile = os.path.join(newpath,picname_now)
        oldpath = os.path.join(filepath,picdir,pic)
        shutil.copyfile(oldpath,newfile)

stimID = np.transpose(picname)
condition = np.transpose(label)
dataframe = pd.DataFrame({'stimID':stimID,'condition':condition})        

dataframe.to_csv("D:/Final_data/transfer_learning_pretrained/validate_Black.csv",index=False,sep=',')

print('All done')
