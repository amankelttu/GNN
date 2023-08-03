import pandas as pd
import numpy as np
#import awkward
import itertools
import copy

from PIL import Image, ImageChops, ImageOps
import matplotlib.pyplot as plt
import random
import math
import os,sys, glob
import scipy
from scipy import stats
import logging

from scipy.optimize import curve_fit
import tensorflow as tf
from tensorflow import keras
from tf_keras_model import get_particle_net, get_particle_net_lite
import matplotlib as mpl

def reso_func(x, A, B):
    y = (A/np.sqrt(x))+B
    return y
def get_resolution(Emin, Emax, particle):
    pred, act, rat, simp_sum ,simp_sum_ratio = get_err_Erange_new(Emin, Emax, particle)
    avg_pred = np.mean(pred)
    sig_pred = np.std(pred)
    avg_act  = np.mean(act)
    
    avg_s_sum = np.mean(simp_sum)
    
    reso_pred = np.divide(sig_pred, avg_pred)
    
    reso_s_sum = np.divide(np.std(simp_sum), np.mean(simp_sum))
    
    return reso_pred, avg_pred, avg_act, reso_s_sum, avg_s_sum
def get_err_Erange_new(Emin, Emax, data, label,simple_sum):    
    ratio = []
    pred = []
    act = []
    simp_sum = []
    y_batch_particle = np.array(label ,dtype='float32')
    pred_particle = data
    simple_sum_converted_particle = simple_sum
    idx = np.where((y_batch_particle >= Emin) & (y_batch_particle < Emax))[0]    
    for ex in idx:
        rt = pred_particle[ex]/y_batch_particle[ex]
        ratio.append(rt)
        pred.append(pred_particle[ex])
        act.append(y_batch_particle[ex])
        simp_sum.append(simple_sum_converted_particle[ex])            
    ratio = np.array(ratio)
    pred  = np.array(pred)
    act   = np.array(act)
    simp_sum = np.array(simp_sum)
    simp_sum_ratio=np.divide(simp_sum,act)
    return pred, act, ratio, simp_sum, simp_sum_ratio
def get_simpleSum_features_float(feature_list_particle):
    feat_sum = []
    for ft in range(len(feature_list_particle)):
        ssum = math.fsum(feature_list_particle[ft])
        feat_sum.append(ssum)
    feat_sum = np.array(feat_sum)
    return feat_sum
def get_Xbatch_ybatch(dff):
    f_feature = np.array(dff['Features'].to_list(),dtype='float32')
    p_points  = np.array(dff['Points'].to_list(),dtype='float32')
    m_mask    = np.array(dff['Mask'].to_list(),dtype='float32')
    l_label   = np.array(dff['Label'].to_list(),dtype='float64')
    x_X_batch = {'points': p_points,
               'features': f_feature,
               'mask': m_mask
              }
    y_Y_batch = l_label.copy()
    
    return x_X_batch, y_Y_batch


#load in all 5 predictions
#flatten all predictions

prediction_normal=np.array(pd.read_pickle('3D-Data/prediction/prediciton_pion_normal_new.pkl')).flatten()
prediction_adam=np.array(pd.read_pickle('3D-Data/prediction/prediciton_pion_adam_new.pkl')).flatten()
prediction_adam_batch=np.array(pd.read_pickle('3D-Data/prediction/prediciton_pion_adam_batch_new.pkl')).flatten()
prediction_batch=np.array(pd.read_pickle('3D-Data/prediction/prediciton_pion_batch_new.pkl')).flatten()

#load in all test arrays for simple sum

test_normal  = pd.read_pickle('3D-Data/GNN-3D-150Gev_pion.pkl')
test_adam  = pd.read_pickle('3D-Data/GNN-3D-150Gev_pion.pkl')
test_adam_batch  = pd.read_pickle('3D-Data/GNN-3D-150Gev_pion.pkl')
test_batch  =pd.read_pickle('3D-Data/GNN-3D-150Gev_pion.pkl')

#setup features and labels


features_list_normal = test_normal['Features'].to_list()
labels_normal=test_normal['Label'].to_list()
#get simple sums

sum_normal  =np.array( get_simpleSum_features_float(features_list_normal),dtype='float32')
# split data into different bins

# normal
normal_pred_1t5,normal_act_1t5,normal_rat_1t5,normal_simp_sum_1t5,normal_rat_sum_1t5= get_err_Erange_new(1.0,   5.0  , prediction_normal, labels_normal ,sum_normal  )
normal_pred_5t10,normal_act_5t10,normal_rat_5t10,normal_simp_sum_5t10,normal_rat_sum_5t10= get_err_Erange_new(5.0,   10.0 ,prediction_normal, labels_normal ,sum_normal  )
normal_pred_10t20,normal_act_10t20,normal_rat_10t20,normal_simp_sum_10t20,normal_rat_sum_10t20= get_err_Erange_new(10.0,  20.0 , prediction_normal, labels_normal ,sum_normal )
normal_pred_20t30,normal_act_20t30,normal_rat_20t30,normal_simp_sum_20t30,normal_rat_sum_20t30= get_err_Erange_new(20.0,  30.0 , prediction_normal, labels_normal ,sum_normal )
normal_pred_30t40,normal_act_30t40,normal_rat_30t40,normal_simp_sum_30t40,normal_rat_sum_30t40= get_err_Erange_new(30.0,  40.0 , prediction_normal, labels_normal ,sum_normal )
normal_pred_40t60,normal_act_40t60,normal_rat_40t60,normal_simp_sum_40t60,normal_rat_sum_40t60= get_err_Erange_new(40.0,  60.0 , prediction_normal, labels_normal ,sum_normal )
normal_pred_60t80,normal_act_60t80,normal_rat_60t80,normal_simp_sum_60t80,normal_rat_sum_60t80= get_err_Erange_new(60.0,  80.0 , prediction_normal, labels_normal ,sum_normal)
normal_pred_80t100,normal_act_80t100,normal_rat_80t100,normal_simp_sum_80t100,normal_rat_sum_80t100= get_err_Erange_new(80.0,  100.0, prediction_normal, labels_normal ,sum_normal)
normal_pred_100t120,normal_act_100t120,normal_rat_100t120,normal_simp_sum_100t120,normal_rat_sum_100t120= get_err_Erange_new(100.0, 120.0, prediction_normal, labels_normal ,sum_normal )
normal_pred_120t150,normal_act_120t150,normal_rat_120t150,normal_simp_sum_120t150,normal_rat_sum_120t150= get_err_Erange_new(120.0, 150.0,prediction_normal, labels_normal ,sum_normal )


# adam
adam_pred_1t5,adam_act_1t5,adam_rat_1t5,adam_simp_sum_1t5,adam_rat_sum_1t5= get_err_Erange_new(1.0,   5.0  , prediction_adam, labels_normal ,sum_normal  )
adam_pred_5t10,adam_act_5t10,adam_rat_5t10,adam_simp_sum_5t10,adam_rat_sum_5t10= get_err_Erange_new(5.0,   10.0 ,prediction_adam, labels_normal ,sum_normal  )
adam_pred_10t20,adam_act_10t20,adam_rat_10t20,adam_simp_sum_10t20,adam_rat_sum_10t20= get_err_Erange_new(10.0,  20.0 , prediction_adam, labels_normal ,sum_normal )
adam_pred_20t30,adam_act_20t30,adam_rat_20t30,adam_simp_sum_20t30,adam_rat_sum_20t30= get_err_Erange_new(20.0,  30.0 , prediction_adam, labels_normal ,sum_normal )
adam_pred_30t40,adam_act_30t40,adam_rat_30t40,adam_simp_sum_30t40,adam_rat_sum_30t40= get_err_Erange_new(30.0,  40.0 , prediction_adam, labels_normal ,sum_normal )
adam_pred_40t60,adam_act_40t60,adam_rat_40t60,adam_simp_sum_40t60,adam_rat_sum_40t60= get_err_Erange_new(40.0,  60.0 , prediction_adam, labels_normal ,sum_normal )
adam_pred_60t80,adam_act_60t80,adam_rat_60t80,adam_simp_sum_60t80,adam_rat_sum_60t80= get_err_Erange_new(60.0,  80.0 , prediction_adam, labels_normal ,sum_normal )
adam_pred_80t100,adam_act_80t100,adam_rat_80t100,adam_simp_sum_80t100,adam_rat_sum_80t100= get_err_Erange_new(80.0,  100.0, prediction_adam, labels_normal ,sum_normal )
adam_pred_100t120,adam_act_100t120,adam_rat_100t120,adam_simp_sum_100t120,adam_rat_sum_100t120= get_err_Erange_new(100.0, 120.0, prediction_adam, labels_normal ,sum_normal )
adam_pred_120t150,adam_act_120t150,adam_rat_120t150,adam_simp_sum_120t150,adam_rat_sum_120t150= get_err_Erange_new(120.0, 150.0,prediction_adam, labels_normal ,sum_normal )

# adam_batch_batch
adam_batch_pred_1t5,adam_batch_act_1t5,adam_batch_rat_1t5,adam_batch_simp_sum_1t5,adam_batch_rat_sum_1t5= get_err_Erange_new(1.0,   5.0  , prediction_adam_batch, labels_normal ,sum_normal  )
adam_batch_pred_5t10,adam_batch_act_5t10,adam_batch_rat_5t10,adam_batch_simp_sum_5t10,adam_batch_rat_sum_5t10= get_err_Erange_new(5.0,   10.0 ,prediction_adam_batch, labels_normal ,sum_normal  )
adam_batch_pred_10t20,adam_batch_act_10t20,adam_batch_rat_10t20,adam_batch_simp_sum_10t20,adam_batch_rat_sum_10t20= get_err_Erange_new(10.0,  20.0 , prediction_adam_batch, labels_normal ,sum_normal )
adam_batch_pred_20t30,adam_batch_act_20t30,adam_batch_rat_20t30,adam_batch_simp_sum_20t30,adam_batch_rat_sum_20t30= get_err_Erange_new(20.0,  30.0 , prediction_adam_batch, labels_normal ,sum_normal )
adam_batch_pred_30t40,adam_batch_act_30t40,adam_batch_rat_30t40,adam_batch_simp_sum_30t40,adam_batch_rat_sum_30t40= get_err_Erange_new(30.0,  40.0 , prediction_adam_batch, labels_normal ,sum_normal )
adam_batch_pred_40t60,adam_batch_act_40t60,adam_batch_rat_40t60,adam_batch_simp_sum_40t60,adam_batch_rat_sum_40t60= get_err_Erange_new(40.0,  60.0 , prediction_adam_batch, labels_normal ,sum_normal )
adam_batch_pred_60t80,adam_batch_act_60t80,adam_batch_rat_60t80,adam_batch_simp_sum_60t80,adam_batch_rat_sum_60t80= get_err_Erange_new(60.0,  80.0 , prediction_adam_batch, labels_normal ,sum_normal )
adam_batch_pred_80t100,adam_batch_act_80t100,adam_batch_rat_80t100,adam_batch_simp_sum_80t100,adam_batch_rat_sum_80t100= get_err_Erange_new(80.0,  100.0, prediction_adam_batch, labels_normal ,sum_normal )
adam_batch_pred_100t120,adam_batch_act_100t120,adam_batch_rat_100t120,adam_batch_simp_sum_100t120,adam_batch_rat_sum_100t120= get_err_Erange_new(100.0, 120.0, prediction_adam_batch, labels_normal ,sum_normal )
adam_batch_pred_120t150,adam_batch_act_120t150,adam_batch_rat_120t150,adam_batch_simp_sum_120t150,adam_batch_rat_sum_120t150= get_err_Erange_new(120.0, 150.0,prediction_adam_batch, labels_normal ,sum_normal )

# batch
batch_pred_1t5,batch_act_1t5,batch_rat_1t5,batch_simp_sum_1t5,batch_rat_sum_1t5= get_err_Erange_new(1.0,   5.0  , prediction_batch, labels_normal ,sum_normal  )
batch_pred_5t10,batch_act_5t10,batch_rat_5t10,batch_simp_sum_5t10,batch_rat_sum_5t10= get_err_Erange_new(5.0,   10.0 ,prediction_batch, labels_normal ,sum_normal  )
batch_pred_10t20,batch_act_10t20,batch_rat_10t20,batch_simp_sum_10t20,batch_rat_sum_10t20= get_err_Erange_new(10.0,  20.0 , prediction_batch, labels_normal ,sum_normal )
batch_pred_20t30,batch_act_20t30,batch_rat_20t30,batch_simp_sum_20t30,batch_rat_sum_20t30= get_err_Erange_new(20.0,  30.0 , prediction_batch, labels_normal ,sum_normal )
batch_pred_30t40,batch_act_30t40,batch_rat_30t40,batch_simp_sum_30t40,batch_rat_sum_30t40= get_err_Erange_new(30.0,  40.0 , prediction_batch, labels_normal ,sum_normal )
batch_pred_40t60,batch_act_40t60,batch_rat_40t60,batch_simp_sum_40t60,batch_rat_sum_40t60= get_err_Erange_new(40.0,  60.0 , prediction_batch, labels_normal ,sum_normal )
batch_pred_60t80,batch_act_60t80,batch_rat_60t80,batch_simp_sum_60t80,batch_rat_sum_60t80= get_err_Erange_new(60.0,  80.0 , prediction_batch, labels_normal ,sum_normal )
batch_pred_80t100,batch_act_80t100,batch_rat_80t100,batch_simp_sum_80t100,batch_rat_sum_80t100= get_err_Erange_new(80.0,  100.0, prediction_batch, labels_normal ,sum_normal )
batch_pred_100t120,batch_act_100t120,batch_rat_100t120,batch_simp_sum_100t120,batch_rat_sum_100t120= get_err_Erange_new(100.0, 120.0, prediction_batch, labels_normal ,sum_normal )
batch_pred_120t150,batch_act_120t150,batch_rat_120t150,batch_simp_sum_120t150,batch_rat_sum_120t150= get_err_Erange_new(120.0, 150.0,prediction_batch, labels_normal ,sum_normal )

def reso(pred,act,simp_sum):
    avg_pred = np.mean(pred)
    sig_pred = np.std(pred)
    avg_act  = np.mean(act)
    avg_s_sum = np.mean(simp_sum)
    reso_pred = np.divide(sig_pred, avg_pred)    
    reso_s_sum = np.divide(np.std(simp_sum), np.mean(simp_sum))
    return reso_pred,avg_pred,avg_act,reso_s_sum,avg_s_sum

#gather all ratios and ratio_sum


normal_y_rat      = [np.mean(normal_rat_1t5), np.mean(normal_rat_5t10), np.mean(normal_rat_10t20), np.mean(normal_rat_20t30), np.mean(normal_rat_30t40), np.mean(normal_rat_40t60), np.mean(normal_rat_60t80), np.mean(normal_rat_80t100), np.mean(normal_rat_100t120), np.mean(normal_rat_120t150)]
normal_y_rat_sum = [np.mean(normal_rat_sum_1t5),   np.mean(normal_rat_sum_5t10),  np.mean(normal_rat_sum_10t20),  np.mean(normal_rat_sum_20t30),   np.mean(normal_rat_sum_30t40), np.mean(normal_rat_sum_40t60), np.mean(normal_rat_sum_60t80), np.mean(normal_rat_sum_80t100), np.mean(normal_rat_sum_100t120), np.mean(normal_rat_sum_120t150)]

adam_y_rat      = [np.mean(adam_rat_1t5), np.mean(adam_rat_5t10), np.mean(adam_rat_10t20), np.mean(adam_rat_20t30), np.mean(adam_rat_30t40), np.mean(adam_rat_40t60), np.mean(adam_rat_60t80), np.mean(adam_rat_80t100), np.mean(adam_rat_100t120), np.mean(adam_rat_120t150)]

adam_batch_y_rat      = [np.mean(adam_batch_rat_1t5), np.mean(adam_batch_rat_5t10), np.mean(adam_batch_rat_10t20), np.mean(adam_batch_rat_20t30), np.mean(adam_batch_rat_30t40), np.mean(adam_batch_rat_40t60), np.mean(adam_batch_rat_60t80), np.mean(adam_batch_rat_80t100), np.mean(adam_batch_rat_100t120), np.mean(adam_batch_rat_120t150)]

batch_y_rat      = [np.mean(batch_rat_1t5), np.mean(batch_rat_5t10), np.mean(batch_rat_10t20), np.mean(batch_rat_20t30), np.mean(batch_rat_30t40), np.mean(batch_rat_40t60), np.mean(batch_rat_60t80), np.mean(batch_rat_80t100), np.mean(batch_rat_100t120), np.mean(batch_rat_120t150)]
#make reso data

act_= [np.mean(normal_act_1t5), np.mean(normal_act_5t10), np.mean(normal_act_10t20),np.mean(normal_act_20t30), np.mean(normal_act_30t40), np.mean(normal_act_40t60), np.mean(normal_act_60t80), np.mean(normal_act_80t100), np.mean(normal_act_100t120), np.mean(normal_act_120t150)]



act=np.array(act_)



#now take predictions and find the sigmas and then do sigma/act then do mean




normal_y_pred      = [np.mean(normal_pred_1t5), np.mean(normal_pred_5t10), np.mean(normal_pred_10t20), np.mean(normal_pred_20t30), np.mean(normal_pred_30t40), np.mean(normal_pred_40t60), np.mean(normal_pred_60t80), np.mean(normal_pred_80t100), np.mean(normal_pred_100t120), np.mean(normal_pred_120t150)]
normal_y_pred_sum = [np.mean(normal_simp_sum_1t5),   np.mean(normal_simp_sum_5t10),  np.mean(normal_simp_sum_10t20),  np.mean(normal_simp_sum_20t30),   np.mean(normal_simp_sum_30t40), np.mean(normal_simp_sum_40t60), np.mean(normal_simp_sum_60t80), np.mean(normal_simp_sum_80t100), np.mean(normal_simp_sum_100t120), np.mean(normal_simp_sum_120t150)]

adam_y_pred      = [np.mean(adam_pred_1t5), np.mean(adam_pred_5t10), np.mean(adam_pred_10t20), np.mean(adam_pred_20t30), np.mean(adam_pred_30t40), np.mean(adam_pred_40t60), np.mean(adam_pred_60t80), np.mean(adam_pred_80t100), np.mean(adam_pred_100t120), np.mean(adam_pred_120t150)]

adam_batch_y_pred      = [np.mean(adam_batch_pred_1t5), np.mean(adam_batch_pred_5t10), np.mean(adam_batch_pred_10t20), np.mean(adam_batch_pred_20t30), np.mean(adam_batch_pred_30t40), np.mean(adam_batch_pred_40t60), np.mean(adam_batch_pred_60t80), np.mean(adam_batch_pred_80t100), np.mean(adam_batch_pred_100t120), np.mean(adam_batch_pred_120t150)]

batch_y_pred      = [np.mean(batch_pred_1t5), np.mean(batch_pred_5t10), np.mean(batch_pred_10t20), np.mean(batch_pred_20t30), np.mean(batch_pred_30t40), np.mean(batch_pred_40t60), np.mean(batch_pred_60t80), np.mean(batch_pred_80t100), np.mean(batch_pred_100t120), np.mean(batch_pred_120t150)]
#now do the sigmas

normal_y_sigma      = [np.std(normal_pred_1t5), np.std(normal_pred_5t10), np.std(normal_pred_10t20), np.std(normal_pred_20t30), np.std(normal_pred_30t40), np.std(normal_pred_40t60), np.std(normal_pred_60t80), np.std(normal_pred_80t100), np.std(normal_pred_100t120), np.std(normal_pred_120t150)]
normal_y_sigma_sum = [np.std(normal_simp_sum_1t5),   np.std(normal_simp_sum_5t10),  np.std(normal_simp_sum_10t20),  np.std(normal_simp_sum_20t30),   np.std(normal_simp_sum_30t40), np.std(normal_simp_sum_40t60), np.std(normal_simp_sum_60t80), np.std(normal_simp_sum_80t100), np.std(normal_simp_sum_100t120), np.std(normal_simp_sum_120t150)]

adam_y_sigma      = [np.std(adam_pred_1t5), np.std(adam_pred_5t10), np.std(adam_pred_10t20), np.std(adam_pred_20t30), np.std(adam_pred_30t40), np.std(adam_pred_40t60), np.std(adam_pred_60t80), np.std(adam_pred_80t100), np.std(adam_pred_100t120), np.std(adam_pred_120t150)]

adam_batch_y_sigma      = [np.std(adam_batch_pred_1t5), np.std(adam_batch_pred_5t10), np.std(adam_batch_pred_10t20), np.std(adam_batch_pred_20t30), np.std(adam_batch_pred_30t40), np.std(adam_batch_pred_40t60), np.std(adam_batch_pred_60t80), np.std(adam_batch_pred_80t100), np.std(adam_batch_pred_100t120), np.std(adam_batch_pred_120t150)]
batch_y_sigma     = [np.std(batch_pred_1t5), np.std(batch_pred_5t10), np.std(batch_pred_10t20), np.std(batch_pred_20t30), np.std(batch_pred_30t40), np.std(batch_pred_40t60), np.std(batch_pred_60t80), np.std(batch_pred_80t100), np.std(batch_pred_100t120), np.std(batch_pred_120t150)]

#now divide the two


normal_response=[np.divide(normal_y_sigma[0],normal_y_pred[0]),np.divide(normal_y_sigma[1],normal_y_pred[1]),np.divide(normal_y_sigma[2],normal_y_pred[2]),np.divide(normal_y_sigma[3],normal_y_pred[3]),np.divide(normal_y_sigma[4],normal_y_pred[4]),np.divide(normal_y_sigma[5],normal_y_pred[5]),np.divide(normal_y_sigma[6],normal_y_pred[6]),np.divide(normal_y_sigma[7],normal_y_pred[7]),np.divide(normal_y_sigma[8],normal_y_pred[8]),np.divide(normal_y_sigma[9],normal_y_pred[9])]

adam_response=[np.divide(adam_y_sigma[0],adam_y_pred[0]),np.divide(adam_y_sigma[1],adam_y_pred[1]),np.divide(adam_y_sigma[2],adam_y_pred[2]),np.divide(adam_y_sigma[3],adam_y_pred[3]),np.divide(adam_y_sigma[4],adam_y_pred[4]),np.divide(adam_y_sigma[5],adam_y_pred[5]),np.divide(adam_y_sigma[6],adam_y_pred[6]),np.divide(adam_y_sigma[7],adam_y_pred[7]),np.divide(adam_y_sigma[8],adam_y_pred[8]),np.divide(adam_y_sigma[9],adam_y_pred[9])]

adam_batch_response=[np.divide(adam_batch_y_sigma[0],adam_batch_y_pred[0]),np.divide(adam_batch_y_sigma[1],adam_batch_y_pred[1]),np.divide(adam_batch_y_sigma[2],adam_batch_y_pred[2]),np.divide(adam_batch_y_sigma[3],adam_batch_y_pred[3]),np.divide(adam_batch_y_sigma[4],adam_batch_y_pred[4]),np.divide(adam_batch_y_sigma[5],adam_batch_y_pred[5]),np.divide(adam_batch_y_sigma[6],adam_batch_y_pred[6]),np.divide(adam_batch_y_sigma[7],adam_batch_y_pred[7]),np.divide(adam_batch_y_sigma[8],adam_batch_y_pred[8]),np.divide(adam_batch_y_sigma[9],adam_batch_y_pred[9])]

batch_response=[np.divide(batch_y_sigma[0],batch_y_pred[0]),np.divide(batch_y_sigma[1],batch_y_pred[1]),np.divide(batch_y_sigma[2],batch_y_pred[2]),np.divide(batch_y_sigma[3],batch_y_pred[3]),np.divide(batch_y_sigma[4],batch_y_pred[4]),np.divide(batch_y_sigma[5],batch_y_pred[5]),np.divide(batch_y_sigma[6],batch_y_pred[6]),np.divide(batch_y_sigma[7],batch_y_pred[7]),np.divide(batch_y_sigma[8],batch_y_pred[8]),np.divide(batch_y_sigma[9],batch_y_pred[9])]

#now do sum response


normal_response_sum=[np.divide(normal_y_sigma_sum[0],normal_y_pred_sum[0]),np.divide(normal_y_sigma_sum[1],normal_y_pred_sum[1]),np.divide(normal_y_sigma_sum[2],normal_y_pred_sum[2]),np.divide(normal_y_sigma_sum[3],normal_y_pred_sum[3]),np.divide(normal_y_sigma_sum[4],normal_y_pred_sum[4]),np.divide(normal_y_sigma_sum[5],normal_y_pred_sum[5]),np.divide(normal_y_sigma_sum[6],normal_y_pred_sum[6]),np.divide(normal_y_sigma_sum[7],normal_y_pred_sum[7]),np.divide(normal_y_sigma_sum[8],normal_y_pred_sum[8]),np.divide(normal_y_sigma_sum[9],normal_y_pred_sum[9])]

#begin plotting 
plt.figure(1)
fig,ax=plt.subplots()
ax.scatter(act,normal_y_rat,label='normal pred',marker='x',color='orange')
ax.scatter(act,normal_y_rat_sum,label='simple sum',color='purple')
ax.scatter(act,adam_y_rat,label='adam pred',marker='x',color='green')
ax.scatter(act,adam_batch_y_rat,label='adam_batch pred',marker='x',color='red')
ax.scatter(act,batch_y_rat,label='batch pred',marker='x',color='black')
ax.set_xlabel('Ebeam [GeV]')
ax.set_ylabel('avg/Ebeam avg')
ax.set_title('Response W Simple Sum and GNN model  pion')
ax.set_ylim(0,2)
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax.legend(loc='upper right')
plt.savefig('plots_800k/response_pion_normal.png')
plt.close()
plt.figure(2)
fig1,ax1=plt.subplots()

ax1.scatter(act,normal_response,label='normal pred',marker='x',color='orange')
ax1.scatter(act,normal_response_sum,label='simple sum',color='purple')
ax1.scatter(act,adam_response,label='adam pred',marker='x',color='green')
ax1.scatter(act,adam_batch_response,label='adam_batch pred',marker='x',color='red')
ax1.scatter(act,batch_response,label='batch pred',marker='x',color='black')
ax1.set_xlabel('Ebeam [GeV]')
ax1.set_ylabel('Sigma/ Ebeam')
ax1.set_title('Resoltuion W Simple Sum and GNN model pion')
ax1.set_ylim(0,1)
pos = ax1.get_position()
ax1.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax1.legend(loc='upper right')
plt.savefig('plots_800k/resoltuion_pion_normal.png')
plt.close()


plt.figure(13)
fig13,ax13=plt.subplots()
ax13.scatter(act,normal_y_rat,label='normal pred',marker='x',color='orange')
ax13.scatter(act,normal_y_rat_sum,label='simple sum',color='purple')
ax13.scatter(act,adam_y_rat,label='adam pred',marker='x',color='green')
ax13.scatter(act,adam_batch_y_rat,label='adam_batch pred',marker='x',color='red')
ax13.scatter(act,batch_y_rat,label='batch pred',marker='x',color='black')
ax13.set_xlabel('Ebeam [GeV]')
ax13.set_ylabel('avg/Ebeam avg')
ax13.set_title('Response W Simple Sum and GNN model  pion')
ax13.set_ylim(.85,1.25)
pos = ax13.get_position()
ax13.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax13.legend(loc='upper right')
plt.savefig('plots_800k/response_pion_scaled.png')
plt.close()
plt.figure(14)
fig14,ax14=plt.subplots()

ax14.scatter(act,normal_response,label='normal pred',marker='x',color='orange')
ax14.scatter(act,normal_response_sum,label='simple sum',color='purple')
ax14.scatter(act,adam_response,label='adam pred',marker='x',color='green')
ax14.scatter(act,adam_batch_response,label='adam_batch pred',marker='x',color='red')
ax14.scatter(act,batch_response,label='batch pred',marker='x',color='black')
ax14.set_xlabel('Ebeam [GeV]')
ax14.set_ylabel('Sigma/ Ebeam')
ax14.set_title('Resoltuion W Simple Sum and GNN model pion')
ax14.set_ylim(0,.55)
pos = ax14.get_position()
ax14.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax14.legend(loc='upper right')
plt.savefig('plots_800k/resoltuion_pion_scaled.png')

plt.close()

plt.figure(3)
fig2,ax2=plt.subplots()

ax2.hist(normal_rat_1t5,label='normal pred',bins=np.arange(0,3,.05), histtype='step', facecolor='None',color='orange',edgecolor='orange')
ax2.hist(normal_rat_sum_1t5,label='simple sum',bins=np.arange(0,3,.05), histtype='step',facecolor='None',color='purple',edgecolor='purple')
ax2.hist(adam_rat_1t5,label='adam pred',bins=np.arange(0,3,.05), histtype='step',facecolor='None',color='green',edgecolor='green')
ax2.hist(adam_batch_rat_1t5,label='adam_batch pred',bins=np.arange(0,3,.05), histtype='step',facecolor='None',color='red',edgecolor='red')
ax2.hist(batch_rat_1t5,label='batch pred',bins=np.arange(0,3,.05), histtype='step',facecolor='None',color='black',edgecolor='black')
ax2.set_ylabel('Predicted/ True Ebeam')
ax2.set_title('ratio GNN model 1t5  pion')
#ax2.set_xlim(-0.2,3)                                                                                                                                                   
plt.yscale('linear')
pos = ax2.get_position()
ax2.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax2.legend(loc='upper right')
plt.savefig('plots_800k/response_hist_1t5_pion_normal.png')
plt.close()

plt.figure(4)
fig3,ax3=plt.subplots()

ax3.hist(normal_rat_5t10,label='normal pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='orange',edgecolor='orange')
ax3.hist(normal_rat_sum_5t10,label='simple sum',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='purple',edgecolor='purple')
ax3.hist(adam_rat_5t10,label='adam pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='green',edgecolor='green')
ax3.hist(adam_batch_rat_5t10,label='adam_batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='red',edgecolor='red')
ax3.hist(batch_rat_5t10,label='batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='black',edgecolor='black')
ax3.set_ylabel('Predicted/ True Ebeam')
ax3.set_title('ratio GNN model 5t10 pion')
#ax3.set_xlim(0,2.5)                                                                                                                                                    
plt.yscale('linear')
pos = ax3.get_position()
ax3.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax3.legend(loc='upper right')
plt.savefig('plots_800k/response_hist_5t10_pion_normal.png')
plt.close()
plt.figure(5)
fig5,ax5=plt.subplots()

ax5.hist(normal_rat_10t20,label='normal pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='orange',edgecolor='orange')
ax5.hist(normal_rat_sum_10t20,label='simple sum',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='purple',edgecolor='purple')
ax5.hist(adam_rat_10t20,label='adam pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='green',edgecolor='green')
ax5.hist(adam_batch_rat_10t20,label='adam_batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='red',edgecolor='red')
ax5.hist(batch_rat_10t20,label='batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='black',edgecolor='black')
ax5.set_xlabel('Ebeam [GeV]')
ax5.set_ylabel('Predicted/ True Ebeam')
ax5.set_title('ratio GNN model 10t20 pion')
plt.yscale('linear')
#ax5.set_xlim(0,2.5)                                                                                                                                                    
pos = ax5.get_position()
ax5.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax5.legend(loc='upper right')
plt.savefig('plots_800k/response_hist_10t20_pion_normal.png')
plt.close()


plt.figure(6)
fig6,ax6=plt.subplots()

ax6.hist(normal_rat_20t30,label='normal pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='orange',edgecolor='orange')
ax6.hist(normal_rat_sum_20t30,label='simple sum',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='purple',edgecolor='purple')
ax6.hist(adam_rat_20t30,label='adam pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='green',edgecolor='green')
ax6.hist(adam_batch_rat_20t30,label='adam_batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='red',edgecolor='red')
ax6.hist(batch_rat_20t30,label='batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='black',edgecolor='black')
ax6.set_ylabel('Predicted/ True Ebeam')
ax6.set_title('ratio GNN model 20t30 pion')
plt.yscale('linear')
#ax6.set_xlim(.5,1.5)                                                                                                                                                    
pos = ax6.get_position()
ax6.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax6.legend(loc='upper right')
plt.savefig('plots_800k/response_hist_20t30_pion_normal.png')
plt.close()

plt.figure(7)
fig7,ax7=plt.subplots()
ax7.hist(normal_rat_30t40,label='normal pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='orange',edgecolor='orange')
ax7.hist(normal_rat_sum_30t40,label='simple sum',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='purple',edgecolor='purple')
ax7.hist(adam_rat_30t40,label='adam pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='green',edgecolor='green')
ax7.hist(adam_batch_rat_30t40,label='adam_batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='red',edgecolor='red')
ax7.hist(batch_rat_30t40,label='batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='black',edgecolor='black')
ax7.set_ylabel('Predicted/ True Ebeam')
ax7.set_title('ratio GNN model 30t40 pion')
plt.yscale('linear')
#ax7.set_xlim(.25,1.75)                                                                                                                                                 
pos = ax7.get_position()
ax7.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax7.legend(loc='upper right')
plt.savefig('plots_800k/response_hist_30t40_pion_normal.png')
plt.close()
plt.figure(8)
fig8,ax8=plt.subplots()


ax8.hist(normal_rat_40t60,label='normal pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='orange',edgecolor='orange')
ax8.hist(normal_rat_sum_40t60,label='simple sum',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='purple',edgecolor='purple')
ax8.hist(adam_rat_40t60,label='adam pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='green',edgecolor='green')
ax8.hist(adam_batch_rat_40t60,label='adam_batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='red',edgecolor='red')
ax8.hist(batch_rat_40t60,label='batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='black',edgecolor='black')
ax8.set_ylabel('Predicted/ True Ebeam')
ax8.set_title('ratio GNN model 40t60 pion')
plt.yscale('linear')
#ax8.set_xlim(.25,1.75)                                                                                                                                                 
pos = ax8.get_position()
ax8.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax8.legend(loc='upper right')
plt.savefig('plots_800k/response_hist_40t60_pion_normal.png')

plt.close()
plt.figure(9)
fig9,ax9=plt.subplots()

ax9.hist(normal_rat_60t80,label='normal pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='orange',edgecolor='orange')
ax9.hist(normal_rat_sum_60t80,label='simple sum',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='purple',edgecolor='purple')
ax9.hist(adam_rat_60t80,label='adam pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='green',edgecolor='green')
ax9.hist(adam_batch_rat_60t80,label='adam_batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='red',edgecolor='red')
ax9.hist(batch_rat_60t80,label='batch pred',bins=np.arange(.5,1.5,.05), histtype='step',facecolor='None',color='black',edgecolor='black')
ax9.set_ylabel('Predicted/ True Ebeam')
ax9.set_title('ratio GNN model 60t80 pion')
plt.yscale('linear')
#ax9.set_xlim(.25,1.75)                                                                                                                                                 
pos = ax9.get_position()
ax9.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax9.legend(loc='upper right')
plt.savefig('plots_800k/response_hist_60t80_pion_normal.png')
plt.close()
plt.figure(10)
fig10,ax10=plt.subplots()

ax10.hist(normal_rat_80t100,label='normal pred',bins=np.arange(.75,1.5,.05), histtype='step',facecolor='None',color='orange',edgecolor='orange')
ax10.hist(normal_rat_sum_80t100,label='simple sum',bins=np.arange(.75,1.5,.05), histtype='step',facecolor='None',color='purple',edgecolor='purple')
ax10.hist(adam_rat_80t100,label='adam pred',bins=np.arange(.75,1.5,.05), histtype='step',facecolor='None',color='green',edgecolor='green')
ax10.hist(adam_batch_rat_80t100,label='adam_batch pred',bins=np.arange(.75,1.5,.05), histtype='step',facecolor='None',color='red',edgecolor='red')
ax10.hist(batch_rat_80t100,label='batch pred',bins=np.arange(.75,1.5,.05), histtype='step',facecolor='None',color='black',edgecolor='black')
ax10.set_ylabel('Predicted/ True Ebeam')
ax10.set_title('ratio GNN model 80t100 pion')
plt.yscale('linear')
#ax10.set_xlim(.25,1.75)                                                                                                                                                
pos = ax10.get_position()
ax10.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax10.legend(loc='upper right')
plt.savefig('plots_800k/response_hist_80t100_pion_normal.png')
plt.close()

plt.figure(11)
fig11,ax11=plt.subplots()
ax11.hist(normal_rat_100t120,label='normal pred',bins=np.arange(0.75,1.25,.05), histtype='step',facecolor='None',color='orange',edgecolor='orange')
ax11.hist(normal_rat_sum_100t120,label='simple sum',bins=np.arange(0.75,1.25,.05), histtype='step',facecolor='None',color='purple',edgecolor='purple')
ax11.hist(adam_rat_100t120,label='adam pred',bins=np.arange(0.75,1.25,.05), histtype='step',facecolor='None',color='green',edgecolor='green')
ax11.hist(adam_batch_rat_100t120,label='adam_batch pred',bins=np.arange(0.75,1.25,.05), histtype='step',facecolor='None',color='red',edgecolor='red')
ax11.hist(batch_rat_100t120,label='batch pred',bins=np.arange(0.75,1.25,.05), histtype='step',facecolor='None',color='black',edgecolor='black')
ax11.set_ylabel('Predicted/ True Ebeam')
ax11.set_title('ratio GNN model 100t120 pion')
plt.yscale('linear')
pos = ax11.get_position()
ax11.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax11.legend(loc='upper right')
plt.savefig('plots_800k/response_hist_100t120_pion_normal.png')
plt.close()
plt.figure(12)
fig12,ax12=plt.subplots()

ax12.hist(normal_rat_120t150,label='normal pred',bins=np.arange(0.75,1.25,.05), histtype='step',facecolor='None',color='orange',edgecolor='orange')
ax12.hist(normal_rat_sum_120t150,label='simple sum',bins=np.arange(0.75,1.25,.05), histtype='step',facecolor='None',color='purple',edgecolor='purple')
ax12.hist(adam_rat_120t150,label='adam pred',bins=np.arange(0.75,1.25,.05), histtype='step',facecolor='None',color='green',edgecolor='green')
ax12.hist(adam_batch_rat_120t150,label='adam_batch pred',bins=np.arange(0.75,1.25,.05), histtype='step',facecolor='None',color='red',edgecolor='red')
ax12.hist(batch_rat_120t150,label='batch pred',bins=np.arange(0.75,1.25,.05), histtype='step',facecolor='None',color='black',edgecolor='black')
ax12.set_ylabel('Predicted/ True Ebeam')
ax12.set_title('ratio GNN model 120t150 pion')
#ax12.set_xlim(0,1.4)                                                                                                                                                  
plt.yscale('linear')
pos = ax12.get_position()
ax12.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
ax12.legend(loc='upper right')
plt.savefig('plots_800k/response_hist_120t150_pion_normal.png')
plt.close()
