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
def get_err_Erange_new(Emin, Emax, particle):    
    ratio = []
    pred = []
    act = []
    simp_sum = []
    if particle == "pi+":
        y_batch_particle = y_batch_test
        pred_particle = prediction_em
        simple_sum_converted_particle = simple_sum_converted_pip
    if particle == "e-":
        y_batch_particle = y_batch_test_em
        pred_particle = prediction_em
        simple_sum_converted_particle = sum_array
    
    idx = np.where((y_batch_particle >= Emin) & (y_batch_particle < Emax))[0]
    print(idx)
    
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
def get_simpleSum_features(feature_list_particle):
    feat_sum = []
    for ft in range(len(feature_list_particle)):
        ssum = feature_list_particle[ft].sum()
        feat_sum.append(ssum)
    feat_sum = np.array(feat_sum)
    return feat_sum
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
def get_err_Erange(Emin, Emax):    
    ratio = []
    pred = []
    act = []
    simp_sum = []
    idx = np.where((y_batch_test >= Emin) & (y_batch_test < Emax))[0]
    
    
    for ex in idx:
        rt = prediction[ex]/y_batch_test[ex]
        ratio.append(rt)
        pred.append(prediction[ex])
        act.append(y_batch_test[ex])
        simp_sum.append(simple_sum_converted_pip[ex])
    
    ratio = np.array(ratio)
    pred  = np.array(pred)
    act   = np.array(act)
    simp_sum = np.array(simp_sum)
    
    return pred, act, ratio, simp_sum
def get_err_Erange_em(Emin, Emax):    
    ratio = []
    pred = []
    act = []
    simp_sum = []
    idx = np.where((y_batch_test_em >= Emin) & (y_batch_test_em < Emax))[0]
    
    
    for ex in idx:
        rt = prediction_em[ex]/y_batch_test_em[ex]
        ratio.append(rt)
        pred.append(prediction_em[ex])
        act.append(y_batch_test_em[ex])
        simp_sum.append(simple_sum_converted_em[ex])
    
    ratio = np.array(ratio)
    pred  = np.array(pred)
    act   = np.array(act)
    simp_sum = np.array(simp_sum)
    
    return pred, act, ratio, simp_sum

df_test_photon = pd.read_pickle('3D-Data/GNN-3D-150Gev_photon.pkl')
X_batch_photon, y_batch_photon = get_Xbatch_ybatch(df_test_photon)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


strategy = tf.distribute.MirroredStrategy()

test_model_scaled = tf.keras.models.load_model('model_checkpoints/adam_batch/GNN_3D_RecoHits_pip_adam_batch_rerun3.loss_0.008574.ep003.h5')
test_model_adam = tf.keras.models.load_model('model_checkpoints/adam_models/GNN_3D_RecoHits_pip_adam_rerun3.loss_0.008860.ep002.h5')
test_model_batch = tf.keras.models.load_model('model_checkpoints/batch/GNN_3D_RecoHits_pip_batch_rerun3.loss_0.008311.ep002.h5')
test_model_normal = tf.keras.models.load_model('model_checkpoints/normal/GNN_3D_RecoHits_pip_normal_rerun3.loss_0.008930.ep003.h5')

#old models with 500k training events
# test_model_scaled = tf.keras.models.load_model('model_checkpoints/unkown/GNN_3D_RecoHits_pip.loss_0.021581.ep004.h5')
# test_model_adam = tf.keras.models.load_model('model_checkpoints/adam/GNN_3D_RecoHits_pip_adam.loss_0.011984.ep006.h5')
# test_model_batch = tf.keras.models.load_model('model_checkpoints/batch/GNN_3D_RecoHits_pip_batch.loss_0.012579.ep008.h5')
# test_model_normal = tf.keras.models.load_model('model_checkpoints/normal/GNN_3D_RecoHits_pip_normal.loss_0.013480.ep004.h5')



prediction_scaled_1= test_model_scaled.predict(X_batch_photon, batch_size=24)
prediction_adam_1= test_model_adam.predict(X_batch_photon, batch_size=24)
prediction_batch_1= test_model_batch.predict(X_batch_photon, batch_size=124)
prediction_normal_1= test_model_normal.predict(X_batch_photon, batch_size=24)

prediction_scaled=prediction_scaled_1.flatten()
prediction_adam=prediction_adam_1.flatten()
prediction_batch=prediction_batch_1.flatten()
prediction_normal=prediction_normal_1.flatten()


tag="adam_batch_800k_rerun3"
zipped_scaled=list(zip(prediction_scaled))
pred_save = pd.DataFrame(zipped_scaled, columns=['prediction'])
pred_save.to_pickle('3D-Data/prediction/prediciton_photon_{}.pkl'.format(tag))

tag="adam_800k_rerun3"
zipped_adam=list(zip(prediction_adam))
pred_save = pd.DataFrame(zipped_adam, columns=['prediction'])
pred_save.to_pickle('3D-Data/prediction/prediciton_photon_{}.pkl'.format(tag))

tag="batch_800k_rerun3"
zipped_batch=list(zip(prediction_batch))
pred_save = pd.DataFrame(zipped_batch, columns=['prediction'])
pred_save.to_pickle('3D-Data/prediction/prediciton_photon_{}.pkl'.format(tag))

tag="normal_800k_rerun3"
zipped_normal=list(zip(prediction_normal))
pred_save = pd.DataFrame(zipped_normal, columns=['prediction'])
pred_save.to_pickle('3D-Data/prediction/prediciton_photon_{}.pkl'.format(tag))
