#============================
# Author: Sanjana Sekhar
# Date: 19 Oct 20
#============================

import h5py
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm
from sklearn.metrics import r2_score
import numpy as np
import time
from plotter import *

h5_date = "nov1"
img_ext = "dnn_nov16"

# Load data
f = h5py.File('h5_files/train_full_angle_scan_%s.hdf5'%(h5_date), 'r')
xpix_flat_train = f['train_x_flat'][...]
ypix_flat_train = f['train_y_flat'][...]
cota_train = f['cota'][...]
cotb_train = f['cotb'][...]
x_train = f['x'][...] 
y_train = f['y'][...]
clustersize_x_train = f['clustersize_x'][...]
clustersize_y_train = f['clustersize_y'][...]
inputs_x_train = np.hstack((xpix_flat_train,cota_train,cotb_train))
inputs_y_train = np.hstack((ypix_flat_train,cota_train,cotb_train))
angles_train = np.hstack((cota_train,cotb_train))
f.close()


f = h5py.File('h5_files/test_full_angle_scan_%s.hdf5'%(h5_date), 'r')
xpix_flat_test = f['test_x_flat'][...]
ypix_flat_test = f['test_y_flat'][...]
cota_test = f['cota'][...]
cotb_test = f['cotb'][...]
x_test = f['x'][...] 
y_test = f['y'][...]
clustersize_x_test = f['clustersize_x'][...]
clustersize_y_test = f['clustersize_y'][...]
inputs_x_test = np.hstack((xpix_flat_test,cota_test,cotb_test))
inputs_y_test = np.hstack((ypix_flat_test,cota_test,cotb_test))
angles_test = np.hstack((cota_test,cotb_test))
f.close()

# Model configuration
batch_size = 256
loss_function = 'mse'
n_epochs = 5
optimizer = Adam(lr=0.001)
validation_split = 0.3

train_time_x = time.clock()
#train flat x


inputs = Input(shape=(13,)) #13 in x dimension + 2 angles
angles = Input(shape=(2,))
concat_inputs = concatenate([inputs,angles])
x = Dense(64)(concat_inputs)
x = Activation("relu")(x)
x = Dense(128)(inputs)
x = Activation("relu")(x)
#x = BatchNormalization()(x)
x = Dense(256)(x)
x = Activation("relu")(x)
x = Dense(256)(x)
x = Activation("relu")(x)
x = Dense(128)(x)
x = Activation("relu")(x)
x = Dense(64)(x)
x = Activation("relu")(x)
x = Dense(1)(x)
x_position = Activation("linear", name="x")(x)

model = Model(inputs=[inputs,angles],
              outputs=[x_position]
              )

 # Display a model summary
model.summary()

#history = model.load_weights("checkpoints/cp_x%s.ckpt"%(img_ext))

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['mse','mse']
              )

callbacks = [
ModelCheckpoint(filepath="checkpoints/cp_x%s.ckpt"%(img_ext),
                save_weights_only=True,
                monitor='val_loss')
]

# Fit data to model
history = model.fit([xpix_flat_train,angles_train], [x_train],
                batch_size=batch_size,
                epochs=n_epochs,
                callbacks=callbacks,
                validation_split=validation_split)

plot_dnn_loss(history.history,'x',img_ext)

print("x training time for dnn",time.clock()-train_time_x)

start = time.clock()
x_pred = model.predict([xpix_flat_test,angles_test], batch_size=9000)
inference_time_x = time.clock() - start

train_time_y = time.clock()

#train flat y


inputs = Input(shape=(21,)) #21 in y dimension + 2 angles
angles = Input(shape=(2,))
concat_inputs = concatenate([inputs,angles])
y = Dense(64)(concat_inputs)
y = Activation("relu")(y)
y = Dense(128)(inputs)
y = Activation("relu")(y)
#y = BatchNormalization()(y)
y = Dense(256)(y)
y = Activation("relu")(y)
y = Dense(256)(y)
y = Activation("relu")(y)
y = Dense(128)(y)
y = Activation("relu")(y)
y = Dense(64)(y)
y = Activation("relu")(y)
y = Dense(1)(y)
y_position = Activation("linear", name="y")(y)

model = Model(inputs=[inputs,angles],
              outputs=[y_position]
              )

 # Display a model summary
model.summary()

#history = model.load_weights("checkpoints/cp_y%s.ckpt"%(img_ext))

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['mse','mse']
              )

callbacks = [
ModelCheckpoint(filepath="checkpoints/cp_y%s.ckpt"%(img_ext),
                save_weights_only=True,
                monitor='val_loss')
]

# Fit data to model
history = model.fit([ypix_flat_train,angles_train], [y_train],
                batch_size=batch_size,
                epochs=n_epochs,
                callbacks=callbacks,
                validation_split=validation_split)


plot_dnn_loss(history.history,'y',img_ext)

print("y training time for dnn",time.clock()-train_time_y)

start = time.clock()
y_pred = model.predict([ypix_flat_test,angles_test], batch_size=9000)
inference_time_y = time.clock() - start

print("inference_time for dnn= ",(inference_time_x+inference_time_y))

residuals_x = x_pred - x_test
RMS_x = np.sqrt(np.mean(residuals_x*residuals_x))
print(np.amin(residuals_x),np.amax(residuals_x))
print("RMS_x = %f\n"%(RMS_x))
residuals_y = y_pred - y_test
RMS_y = np.sqrt(np.mean(residuals_y*residuals_y))
print(np.amin(residuals_y),np.amax(residuals_y))
print("RMS_y = %f\n"%(RMS_y))


mean_x, sigma_x = norm.fit(residuals_x)
print("mean_x = %0.2f, sigma_x = %0.2f"%(mean_x,sigma_x))

plot_residuals(residuals_x,mean_x,sigma_x,RMS_x,'x',img_ext)

mean_y, sigma_y = norm.fit(residuals_y)
print("mean_y = %0.2f, sigma_y = %0.2f"%(mean_y,sigma_y))

plot_residuals(residuals_y,mean_y,sigma_y,RMS_y,'y',img_ext)

plot_by_clustersize(residuals_x,clustersize_x_test,'x',img_ext)
plot_by_clustersize(residuals_y,clustersize_y_test,'y',img_ext)
