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
from sklearn.metrics import r2_score
import numpy as np

# Load data
f = h5py.File('h5_files/train_d49301_d49341.hdf5', 'r')
pix_train = f['train_hits'][...]
cosx_train = f['cosx'][...]
cosy_train = f['cosy'][...]
cosz_train = f['cosz'][...]
x_train = f['x'][...]
y_train = f['y'][...]
f.close()
angles_train = np.hstack((cosx_train,cosy_train,cosz_train))

f = h5py.File('h5_files/test_d49350.hdf5', 'r')
pix_test = f['test_hits'][...]
cosx_test = f['cosx'][...]
cosy_test = f['cosy'][...]
cosz_test = f['cosz'][...]
x_test = f['x'][...]
y_test = f['y'][...]
f.close()
angles_test = np.hstack((cosx_test,cosy_test,cosz_test))

# Model configuration
batch_size = 128
loss_function = 'mean_squared_error'
n_epochs = 1
optimizer = Adam(lr=0.001)
validation_split = 0.3

#Conv2D -> BatchNormalization -> Pooling -> Dropout

inputs = Input(shape=(13,21,1))
angles = Input(shape=(3,))
x = Conv2D(32, (3, 3), padding="same")(inputs)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x_cnn = Flatten()(x)
concat_inputs = concatenate([x_cnn,angles])

x = Dense(64)(concat_inputs)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(64)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1)(x)
x_position = Activation("linear", name="x")(x)

y = Dense(64)(concat_inputs)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)
y = Dense(128)(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)
y = Dense(64)(y)
y = Activation("relu")(y)
y = BatchNormalization()(y)
y = Dropout(0.5)(y)
y = Dense(1)(y)
y_position = Activation("linear", name="y")(y)

model = Model(inputs=[inputs,angles],
              outputs=[x_position,y_position]
              )

# Display a model summary
model.summary()

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['mse','mse']
              )

callbacks = [
    ModelCheckpoint(filepath="checkpoints/cp.ckpt", monitor='val_loss')
]

# Fit data to model
history = model.fit([pix_train, angles_train], [x_train, y_train],
            batch_size=batch_size,
            epochs=n_epochs,
            callbacks=callbacks,
            validation_split=validation_split)


# Generate generalization metrics
x_pred, y_pred = model.predict([pix_test, angles_test], batch_size=batch_size)
residuals_x = x_pred - x_test
sigma_x = np.std(residuals_x)
print("sigma_x = %f\n"%(sigma_x))
residuals_y = y_pred - y_test
sigma_y = np.std(residuals_y)
print("sigma_y = %f\n"%(sigma_y))

plt.plot(history.history['x_loss'])
plt.plot(history.history['val_x_loss'])
plt.title('x position - model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['x-train', 'x-validation'], loc='upper right')
#plt.show()
plt.savefig("plots/loss_x_sep20.png")
plt.close()

plt.plot(history.history['y_loss'])
plt.plot(history.history['val_y_loss'])
plt.title('y position - model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['y-train', 'y-validation'], loc='upper right')
#plt.show()
plt.savefig("plots/loss_y_sep20.png")
plt.close()

plt.hist(residuals_x, bins=np.arange(-60,60,0.5), histtype='step', label=r'$\vartriangle x$')
plt.hist(residuals_y, bins=np.arange(-60,60,0.5), histtype='step', label=r'$\vartriangle y$')
plt.title(r'$\vartriangle x = x_{pred} - x_{true}, \vartriangle y = y_{pred} - y_{true}$')
plt.ylabel('No. of samples')
plt.xlabel(r'$\mu m$')
plt.legend(loc='upper right')
#plt.show()
plt.savefig("plots/residuals_sep20.png")
plt.close()

