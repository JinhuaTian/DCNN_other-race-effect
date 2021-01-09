# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:50:13 2021

@author: clemensTian

calculate ROC 

input is activation matrix and its ground turth
1.calculate its cosine correlation matrix
2.compute their inner and outer correlation (average) 
3.plot ROC curve and compute ROC area and EER
"""

#compute cos similarity between two vector
from sklearn.metrics.pairwise import cosine_similarity
import h5py
import numpy as np
import pandas as pd
from sklearn import preprocessing

# load the representational data
ac_a = h5py.File(r'C:\Users\clemensTian\Desktop\tl2tl3results\svmResults\tl1\White_vgg16_fc7.act.h5','r')
ac_a = np.array(ac_a['fc7'])
ac_a = pd.DataFrame(np.squeeze(ac_a))

a_cosMatrix = cosine_similarity(ac_a)


# average the activation of 50 id 
def m_average(ac,pic_num=50):
    ac_avg = []
    rows = int(ac.shape[0]/pic_num)
    for i in list(range(0,rows)):
        i_start = i*pic_num
        i_end = (i+1)*pic_num
        newrow = np.mean(ac[i_start:i_end],axis=0)
        newrow = list(newrow)
        ac_avg.append(newrow)
    
    ac_avg = np.array(ac_avg)    
    return ac_avg


a_cosMatrix = cosine_similarity(ac_a)
a_cosMatrix = m_average(a_cosMatrix,pic_num=50)
a_cosMatrix = a_cosMatrix.T

#a_cosMatrix = np.array(pd.DataFrame(preprocessing.scale(a_cosMatrix)))
#Normalize to (0,1)
minMaxScaler = preprocessing.MinMaxScaler()
minMax = minMaxScaler.fit(a_cosMatrix)
nor_a = minMax.transform(a_cosMatrix)

#calculate 
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp

#produce a label matrix
labelMat = np.zeros([5000,100])
for i in range(labelMat.shape[1]):
    k = int((labelMat.shape[0]/labelMat.shape[1]) * i)
    for j in range(int(labelMat.shape[0]/labelMat.shape[1])):
        row = k+j
        labelMat[row,i] = 1

y_test = labelMat

#number of classification  
n_classes = a_cosMatrix.shape[1]

#probability matrix
y_score = nor_a

# Compute ROC of each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area (Method 2)
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area(Method 1)
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
#plt.legend(loc="lower right")
plt.show()

import numpy as np
import sklearn.metrics

"""
Python compute equal error rate (eer)
https://github.com/YuanGongND/python-compute-eer
:param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
:param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
:param positive_label: the class that is viewed as positive class when computing EER
:return: equal error rate (EER)
"""
def compute_eer(label, pred, positive_label=1,n_class=100):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    EER = []

    for i in range(n_classes):
        fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test[:, i], y_score[:, i], positive_label)
        fnr = 1 - tpr
        # the threshold of fnr == fpr
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
        eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

        # return the mean of eer from fpr and from fnr
        eer = (eer_1 + eer_2) / 2
        EER.append(eer)
        
    meanEER = np.mean(EER)
    stdEER = np.std(EER,ddof=1) 
    return EER,meanEER,stdEER

eer,meanEER,stdEER = compute_eer(y_test, y_score,positive_label=1, n_class=100)
