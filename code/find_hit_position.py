#============================
# Author: Sanjana Sekhar
# Date: 13 Sep 20
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
from scipy.stats import norm
import numpy as np
import time
from plotter import *
from keras.callbacks import EarlyStopping

h5_date = "dec12"
h5_ext = "phase1"
img_ext = "cnn_phase1_dec15"

# Load data
f = h5py.File('h5_files/train_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
pix_train = f['train_hits'][...]
cota_train = f['cota'][...]
cotb_train = f['cotb'][...]
x_train = f['x'][...] 
y_train = f['y'][...]
clustersize_x_train = f['clustersize_x'][...]
clustersize_y_train = f['clustersize_y'][...]
angles_train = np.hstack((cota_train,cotb_train))
f.close()

f = h5py.File('h5_files/test_%s_%s.hdf5'%(h5_ext,h5_date), 'r')
pix_test = f['test_hits'][...]
cota_test = f['cota'][...]
cotb_test = f['cotb'][...]
x_test = f['x'][...]
y_test = f['y'][...]
clustersize_x_test = f['clustersize_x'][...]
clustersize_y_test = f['clustersize_y'][...]
angles_test = np.hstack((cota_test,cotb_test))
f.close()


# Model configuration
batch_size = 256
loss_function = 'mse'
n_epochs = 30
optimizer = Adam(lr=0.001)
validation_split = 0.2

train_time_s = time.clock()
#Conv2D -> BatchNormalization -> Pooling -> Dropout

inputs = Input(shape=(13,21,1))
angles = Input(shape=(2,))
x = Conv2D(64, (3, 3), padding="same")(inputs)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)
x = Dropout(0.25)(x)
x = Conv2D(128, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)
x = Dropout(0.25)(x)
'''
x = Conv2D(256, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)
x = Dropout(0.25)(x)
'''
x = Conv2D(128, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)
x = Dropout(0.25)(x)
x_cnn = Flatten()(x)
concat_inputs = concatenate([x_cnn,angles])

x = Dense(64)(concat_inputs)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(128)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
'''
x = Dense(256)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
'''
x = Dense(128)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(64)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(1)(x)
x_position = Activation("linear", name="x")(x)

y = Dense(64)(concat_inputs)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.25)(y)
y = Dense(128)(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.25)(y)
'''
y = Dense(256)(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.25)(y)
'''
y = Dense(128)(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.25)(y)
y = Dense(64)(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.25)(y)
y = Dense(1)(y)
y_position = Activation("linear", name="y")(y)

model = Model(inputs=[inputs,angles],
              outputs=[x_position,y_position]
              )

 # Display a model summary
model.summary()

#history = model.load_weights("checkpoints/cp_%s.ckpt"%(img_ext))

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['mse','mse']
              )

callbacks = [
EarlyStopping(patience=4),
ModelCheckpoint(filepath="checkpoints/cp_%s.ckpt"%(img_ext),
                save_weights_only=True,
                monitor='val_loss')
]

# Fit data to model
history = model.fit([pix_train, angles_train], [x_train, y_train],
                batch_size=batch_size,
                epochs=n_epochs,
                callbacks=callbacks,
                validation_split=validation_split)

plot_cnn_loss(history.history,"x",img_ext)
plot_cnn_loss(history.history,"y",img_ext)

# Generate generalization metrics
print("training time ",time.clock()-train_time_s)

start = time.clock()
x_pred, y_pred = model.predict([pix_test, angles_test], batch_size=9000)
inference_time = time.clock() - start

print("inference_time = ",inference_time)


residuals_x = x_pred - x_test
RMS_x = np.sqrt(np.mean(residuals_x*residuals_x))
print("RMS_x = %f\n"%(RMS_x))
residuals_y = y_pred - y_test
RMS_y = np.sqrt(np.mean(residuals_y*residuals_y))
print("RMS_y = %f\n"%(RMS_y))


mean_x, sigma_x = norm.fit(residuals_x)
print("mean_x = %0.2f, sigma_x = %0.2f"%(mean_x,sigma_x))

plot_residuals(residuals_x,mean_x,sigma_x,RMS_x,'x',img_ext)

mean_y, sigma_y = norm.fit(residuals_y)
print("mean_y = %0.2f, sigma_y = %0.2f"%(mean_y,sigma_y))

plot_residuals(residuals_y,mean_y,sigma_y,RMS_y,'y',img_ext)

plot_by_clustersize(residuals_x,clustersize_x_test,'x',img_ext)
plot_by_clustersize(residuals_y,clustersize_y_test,'y',img_ext)
