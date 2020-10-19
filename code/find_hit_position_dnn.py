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

h5_date = "oct18_nonoise"
img_ext = "dnn_oct18_nonoise"

# Load data
f = h5py.File('h5_files/train_d49301_d49341_%s.hdf5'%(h5_date), 'r')
xpix_flat_train = f['train_x_flat'][...]
ypix_flat_train = f['train_y_flat'][...]
cota_train = f['cota'][...]
cotb_train = f['cotb'][...]
x_train = f['x'][...] 
y_train = f['y'][...]
inputs_x_train = np.hstack((xpix_flat_train,cota_train,cotb_train))
inputs_y_train = np.hstack((ypix_flat_train,cota_train,cotb_train))
angles_train = np.hstack((cota_train,cotb_train))
f.close()

print(inputs_x_train.shape)
print(inputs_y_train.shape)

f = h5py.File('h5_files/test_d49350_%s.hdf5'%(h5_date), 'r')
xpix_flat_test = f['test_x_flat'][...]
ypix_flat_test = f['test_y_flat'][...]
cota_test = f['cota'][...]
cotb_test = f['cotb'][...]
x_test = f['x'][...] 
y_test = f['y'][...]
inputs_x_test = np.hstack((xpix_flat_test,cota_test,cotb_test))
inputs_y_test = np.hstack((ypix_flat_test,cota_test,cotb_test))
angles_test = np.hstack((cota_test,cotb_test))
f.close()

# Model configuration
batch_size = 32
loss_function = 'mse'
n_epochs = 5
optimizer = Adam(lr=0.001)
validation_split = 0.3

train_time_x = time.clock()
#train flat x


inputs = Input(shape=(13,)) #13 in x dimension + 2 angles
angles = Input(shape=(2,))
x = Dense(16)(inputs)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(32)(inputs)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
concat_inputs = concatenate([x,angles])
x = Dense(64)(concat_inputs)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(64)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(32)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1)(x)
x_position = Activation("linear", name="x")(x)

model = Model(inputs=[inputs,angles],
              outputs=[x_position]
              )

 # Display a model summary
model.summary()

history = model.load_weights("checkpoints/cp_x%s.ckpt"%(img_ext))

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['mse','mse']
              )
'''
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


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('x position - model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['x-train', 'x-validation'], loc='upper right')
#plt.show()
plt.savefig("plots/loss_x_%s.png"%(img_ext))
plt.close()
'''
print("x training time for dnn",time.clock()-train_time_x)

start = time.clock()
x_pred = model.predict([xpix_flat_test,angles_test], batch_size=9000)
inference_time_x = time.clock() - start

train_time_y = time.clock()

#train flat y


inputs = Input(shape=(21,)) #21 in y dimension + 2 angles
angles = Input(shape=(2,))
y = Dense(16)(inputs)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)
y = Dense(32)(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)
concat_inputs = concatenate([y,angles])
y = Dense(64)(concat_inputs)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)
y = Dense(64)(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)
y = Dense(32)(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)
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


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('y position - model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['y-train', 'y-validation'], loc='upper right')
#plt.show()
plt.savefig("plots/loss_y_%s.png"%(img_ext))
plt.close()


print("y training time for dnn",time.clock()-train_time_y)

start = time.clock()
y_pred = model.predict([ypix_flat_test,angles_test], batch_size=9000)
inference_time_y = time.clock() - start

print("inference_time for dnn= ",(inference_time_x+inference_time_y))

residuals_x = x_pred - x_test
RMS_x = np.std(residuals_x)
print("RMS_x = %f\n"%(RMS_x))
residuals_y = y_pred - y_test
RMS_y = np.std(residuals_y)
print("RMS_y = %f\n"%(RMS_y))

mean_x, sigma_x = norm.fit(residuals_x)
print("mean_x = %0.2f, sigma_x = %0.2f"%(mean_x,sigma_x))

plt.hist(residuals_x, bins=np.arange(-60,60,0.5), histtype='step', density=True,linewidth=2, label=r'$\vartriangle x$')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_x, sigma_x)
plt.title(r'$\vartriangle x = x_{pred} - x_{true}$')
#plt.ylabel('No. of samples')
plt.xlabel(r'$\mu m$')
plt.plot(x, p, 'k', linewidth=1,color='red',label='gaussian fit')
plt.legend()
plt.savefig("plots/residuals_x_%s.png"%(img_ext))
plt.close()

mean_y, sigma_y = norm.fit(residuals_y)
print("mean_y = %0.2f, sigma_y = %0.2f"%(mean_y,sigma_y))

plt.hist(residuals_y, bins=np.arange(-60,60,0.5), histtype='step', density=True,linewidth=2, label=r'$\vartriangle y$')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_y, sigma_y)
plt.title(r'$\vartriangle y = y_{pred} - y_{true}$')
#plt.ylabel('No. of samples')
plt.xlabel(r'$\mu m$')
plt.plot(x, p, 'k', linewidth=1, color='red',label='gaussian fit')
plt.legend()
plt.savefig("plots/residuals_y_%s.png"%(img_ext))
plt.close()


