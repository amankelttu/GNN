#!/usr/bin/env python
import pandas as pd
import numpy as np
import itertools
import copy
from PIL import Image, ImageChops, ImageOps
import matplotlib.pyplot as plt
import random
import math
import os,sys, glob
import logging
import tensorflow as tf
from tensorflow import keras
from tf_keras_model import get_particle_net, get_particle_net_lite
import datetime as dt 
def train_generator():
    feature = np.array(df['Features'].to_list(),dtype='float32')
    points  = np.array(df['Points'].to_list(),dtype='float32')
    mask    = np.array(df['Mask'].to_list(),dtype='float32')
    label   = np.array(df['Label'].to_list(),dtype='float32')
    X_batch = {'points': points,
               'features': feature,
               'mask': mask
              }
    y_batch = label.copy()
    yield X_batch, y_batch            
def get_Xbatch_ybatch(ddf):
    f_feature = np.array(ddf['Features'].to_list(),dtype='float32')
    p_points  = np.array(ddf['Points'].to_list(),dtype='float32')
    m_mask    = np.array(ddf['Mask'].to_list(),dtype='float32')
    l_label   = np.array(ddf['Label'].to_list(),dtype='float32')
    x_X_batch = {'points': p_points,
               'features': f_feature,
               'mask': m_mask
              }
    y_Y_batch = l_label.copy()
    return x_X_batch, y_Y_batch
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 5:
        lr *= 0.1
    elif epoch > 10:
        lr *= 0.01
    elif epoch > 15:
        lr *= 0.002
    logging.info('Learning rate: %f'%lr)
    return lr
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
tag="normal"
batch_size = 28
batch_sizeV = 6
epochs = 5
print("about to laod data")
df_train = pd.read_pickle('3D-Data/GNN-3D-150Gev-train_csv_800k.pkl')
df_val   = pd.read_pickle('3D-Data/GNN-3D-150Gev-val_csv_800k.pkl')
print("done loading data")
X_batch_train, y_batch_train = get_Xbatch_ybatch(df_train)
X_batch_val,   y_batch_val   = get_Xbatch_ybatch(df_val)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print("this is killing it")
        print(e)
print("strat is next")
strategy = tf.distribute.MirroredStrategy()
print("start is over")
#model_type = 'particle_net_lite' # choose between 'particle_net' and 'particle_net_lite'
model_type = 'particle_net' # choose between 'particle_net' and 'particle_net_lite'
num_classes = 1
input_shapes = {'points': (2000, 3), 
                'features': (2000, 1), 
                'mask': (2000, 1)}
if 'lite' in model_type:
    with strategy.scope():
        model = get_particle_net_lite(num_classes, input_shapes)
else:
    with strategy.scope():
        model = get_particle_net(num_classes, input_shapes)
model.compile(loss='mean_squared_logarithmic_error',
              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)))

#model.summary()
#model=tf.keras.models.load_model('/lustre/research/hep/hgcdpg/amankel/gnn/model_checkpoints/normal/GNN_3D_RecoHits_pip_normal_rerun2.loss_0.009122.ep001.h5')
training_size = len(y_batch_train)
val_size=len(y_batch_val)
#test_size = len(y_batch_test)
compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
steps_per_epoch = compute_steps_per_epoch(training_size)
val_steps = compute_steps_per_epoch(val_size)
# Prepare model model saving directory.
now=dt.datetime.now()
save_dir = 'model_checkpoints/{}'.format("normal")
model_name = 'GNN_3D_RecoHits_pip_normal_rerun3.loss_{val_loss:01.6f}.ep{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
progress_bar = keras.callbacks.ProgbarLogger()

early = keras.callbacks.EarlyStopping(monitor="val_loss",
                      mode="min", patience=12)
callbacks = [checkpoint]
#maincommand
print("about to fit data")
model.fit(
    X_batch_train, y_batch_train,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=(X_batch_val, y_batch_val),
    validation_steps=val_steps,
    callbacks=callbacks,
    use_multiprocessing=True, workers=4 ,
    max_queue_size=240
    )
#plot the training and validation accuracy and loss at each epoch
print("all 8 epochs were completed")
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('3D-Data/prediction/loss_model_'+str(tag)+str(now)+'.png')
print("loss plots were completed")
#begin running predicitons
# test_data = pd.read_pickle('3D-Data/GNN-3D-150Gev-test_csv_normal.pkl')
# X_batch_test, y_batch_test= get_Xbatch_ybatch(test_data)
# prediciton=model.predict(X_batch_test, batch_size=20)
# prediction1=prediction.copy()
# prediction1=prediction1.flatten()

# zipped = list(zip(prediction1))
# pred_save = pd.DataFrame(zipped, columns=['prediction'])
# pred_save.to_pickle('3D-Data/prediction/Prediction_{}.pkl'.format(tag))
